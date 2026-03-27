"""
SMC STRATEGY CLEAN BACKTEST v5.4-CLEAN
========================================
Same v5.4 strategy params. Cleaned universe:
  MIN_VOL_USDT  = $5,000,000   (was $500k — removes meme/thin pairs)
  TOP_PAIRS     = 30            (hard cap enforced)
  MIN_TRADES_PAIR = 8           (exclude pairs with < 8 trades — too noisy)

This gives honest aggregate stats over liquid, meaningful pairs only.

Install:
  pip install ccxt pandas numpy tqdm colorama

Usage:
  python smc_backtest_clean.py
"""

import asyncio
import ccxt.async_support as ccxt
import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Dict
from datetime import datetime, timezone
import logging, json

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    class tqdm:
        def __init__(self, iterable=None, **kw): self._it = iterable
        def __iter__(self): return iter(self._it)

try:
    from colorama import Fore, Style, init as cinit
    cinit(autoreset=True)
    GREEN = Fore.GREEN; RED = Fore.RED; YELLOW = Fore.YELLOW
    CYAN  = Fore.CYAN;  BOLD = Style.BRIGHT; RESET = Style.RESET_ALL
except ImportError:
    GREEN = RED = YELLOW = CYAN = BOLD = RESET = ''

logging.basicConfig(level=logging.WARNING, format='%(levelname)s %(message)s')
logger = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════════
#  CONFIG  (v5.4 params — only universe filters changed)
# ════════════════════════════════════════════════════════════

CONFIG = dict(
    # ── Universe (cleaned) ───────────────────────────────
    TOP_PAIRS           = 30,
    MIN_VOL_USDT        = 5_000_000,   # $5M — liquid pairs only
    MIN_TRADES_PAIR     = 8,           # exclude thin results
    PAIR_BLACKLIST      = {'1000PEPE', 'XRP', 'HYPE'},
    MAX_ATR_PRICE_PCT   = 0.030,

    # ── v5.4 strategy params (unchanged) ────────────────
    BARS_PER_PAIR       = 1000,
    SIGNAL_TF           = '1h',
    ATR_PERIOD          = 14,
    SWING_N             = 3,
    OB_BASIL_MIN        = 3,
    OB_MIN_IMBAL_ATR    = 1.0,
    OB_MAX_AGE          = 300,
    BB_MAX_AGE          = 200,
    IMPULSE_SKIP        = 7,
    FIRST_TOUCH_ONLY    = True,
    MIN_VOL_RATIO       = 0.60,
    VOL_LOOKBACK        = 20,
    MIN_DEPART_ATR      = 1.0,
    REQUIRE_DIR_CANDLE  = True,
    TP1_R               = 1.5,
    TP2_R               = 3.0,
    TP1_SIZE            = 0.50,
    SL_ATR_BUFFER       = 0.15,
    TIMEOUT_BARS        = 72,
    MIN_RR_FILTER       = 1.5,
    COOLDOWN_BARS       = 8,
    USE_SESSIONS        = True,
    SESSION_HOURS       = {'london': (7, 10), 'ny': (13, 16)},
    TREND_EMA_FAST      = 21,
    TREND_EMA_SLOW      = 50,
)


# ════════════════════════════════════════════════════════════
#  MATH
# ════════════════════════════════════════════════════════════

def calc_atr(H, L, C, period=14):
    n = len(C); tr = np.zeros(n)
    tr[0] = H[0] - L[0]
    for i in range(1, n):
        tr[i] = max(H[i]-L[i], abs(H[i]-C[i-1]), abs(L[i]-C[i-1]))
    atr = np.zeros(n)
    if n >= period:
        atr[period-1] = np.mean(tr[:period])
        for i in range(period, n):
            atr[i] = (atr[i-1]*(period-1) + tr[i]) / period
        for i in range(period-1): atr[i] = atr[period-1]
    return atr

def calc_ema(arr, w):
    out = np.zeros(len(arr))
    if not len(arr): return out
    out[0] = arr[0]; k = 2.0/(w+1)
    for i in range(1, len(arr)): out[i] = arr[i]*k + out[i-1]*(1-k)
    return out

def rolling_vol_avg(V, i, lookback=20):
    s = max(0, i - lookback)
    vals = V[s:i]
    return float(np.mean(vals)) if len(vals) > 0 else float(V[i])

def resample_4h(df1h):
    n = len(df1h)
    C4 = np.array([df1h['close'].values[min(i+3,n-1)] for i in range(0,n,4)])
    e21 = calc_ema(C4, CONFIG['TREND_EMA_FAST'])
    e50 = calc_ema(C4, CONFIG['TREND_EMA_SLOW'])
    T = np.empty(n, dtype=object)
    for b in range(len(C4)):
        s, e = b*4, min(b*4+4, n)
        if C4[b] > e21[b] > e50[b]:   T[s:e] = 'bull'
        elif C4[b] < e21[b] < e50[b]: T[s:e] = 'bear'
        else:                          T[s:e] = 'neutral'
    return T


# ════════════════════════════════════════════════════════════
#  SWINGS + TREND
# ════════════════════════════════════════════════════════════

@dataclass
class SwingPoint:
    idx: int; price: float; kind: str

def find_swings(H, L, sw=3):
    n = len(H); pts = []
    for i in range(sw, n-sw):
        if H[i] == max(H[max(0,i-sw):i+sw+1]): pts.append(SwingPoint(i, H[i], 'high'))
        if L[i] == min(L[max(0,i-sw):i+sw+1]): pts.append(SwingPoint(i, L[i], 'low'))
    return sorted(pts, key=lambda p: p.idx)

def trend_1h(swings, up_to):
    pts   = [p for p in swings if p.idx < up_to]
    highs = [p for p in pts if p.kind == 'high']
    lows  = [p for p in pts if p.kind == 'low']
    if len(highs) < 2 or len(lows) < 2: return 'ranging'
    hh = highs[-1].price > highs[-2].price; hl = lows[-1].price > lows[-2].price
    lh = highs[-1].price < highs[-2].price; ll = lows[-1].price < lows[-2].price
    if hh and hl: return 'uptrend'
    if lh and ll: return 'downtrend'
    return 'ranging'


# ════════════════════════════════════════════════════════════
#  ORDER BLOCKS
# ════════════════════════════════════════════════════════════

@dataclass
class OrderBlock:
    top: float; bottom: float; formed: int; kind: str
    basil: int = 0; failed: bool = False; fail_at: int = -1

def _avg_vol(V, end):
    return float(np.mean(V[max(0,end-50):end])) + 1e-9

def find_bull_obs(O, H, L, C, V, ATR, swings, T4H, end):
    obs = []; av = _avg_vol(V, end)
    for i in range(5, end-3):
        if not (C[i] < O[i]): continue
        atr = max(ATR[i], 1e-9)
        imp_end = min(i+4, end-1)
        rally = max(C[i+1:imp_end+1]) - O[i+1]
        if rally < CONFIG['OB_MIN_IMBAL_ATR'] * atr: continue
        fvg   = any(L[j] > H[j-2] for j in range(i+1, imp_end+1) if j >= 2)
        pr_lo = [p for p in swings if p.idx < i and p.kind == 'low']
        swept = (len(pr_lo) > 0 and min(L[i:imp_end+1]) < pr_lo[-1].price) if pr_lo else False
        t4    = T4H[i] if i < len(T4H) else 'neutral'
        pr_hi = [p for p in swings if p.idx < i and p.kind == 'high']
        bos   = (len(pr_hi) > 0 and max(C[i+1:imp_end+1]) > pr_hi[-1].price) if pr_hi else False
        b     = sum([bos, t4 in ('bull','neutral'), swept, fvg, V[i] > av*1.2])
        if b >= CONFIG['OB_BASIL_MIN']:
            obs.append(OrderBlock(top=max(O[i],C[i]), bottom=L[i], formed=i, kind='bullish', basil=b))
    return obs

def find_bear_obs(O, H, L, C, V, ATR, swings, T4H, end):
    obs = []; av = _avg_vol(V, end)
    for i in range(5, end-3):
        if not (C[i] > O[i]): continue
        atr = max(ATR[i], 1e-9)
        imp_end = min(i+4, end-1)
        drop = O[i+1] - min(C[i+1:imp_end+1])
        if drop < CONFIG['OB_MIN_IMBAL_ATR'] * atr: continue
        fvg   = any(H[j] < L[j-2] for j in range(i+1, imp_end+1) if j >= 2)
        pr_hi = [p for p in swings if p.idx < i and p.kind == 'high']
        swept = (len(pr_hi) > 0 and max(H[i:imp_end+1]) > pr_hi[-1].price) if pr_hi else False
        t4    = T4H[i] if i < len(T4H) else 'neutral'
        pr_lo = [p for p in swings if p.idx < i and p.kind == 'low']
        bos   = (len(pr_lo) > 0 and min(C[i+1:imp_end+1]) < pr_lo[-1].price) if pr_lo else False
        b     = sum([bos, t4 in ('bear','neutral'), swept, fvg, V[i] > av*1.2])
        if b >= CONFIG['OB_BASIL_MIN']:
            obs.append(OrderBlock(top=H[i], bottom=min(O[i],C[i]), formed=i, kind='bearish', basil=b))
    return obs


# ════════════════════════════════════════════════════════════
#  QUALITY CHECKS
# ════════════════════════════════════════════════════════════

def zone_previously_touched_rob(ob, H, L, check_end):
    scan_from = ob.formed + CONFIG['IMPULSE_SKIP']
    for k in range(scan_from, check_end):
        if L[k] <= ob.top and H[k] >= ob.bottom:
            return True
    return False

def zone_previously_touched_bb(bb, H, L, check_end):
    for k in range(bb.fail_at + 1, check_end):
        if L[k] <= bb.top and H[k] >= bb.bottom:
            return True
    return False

def price_departed_zone(ob, H, L, ATR, check_from, check_end):
    for k in range(check_from, check_end):
        atr_k = max(ATR[k], 1e-9)
        if H[k] >= ob.top + CONFIG['MIN_DEPART_ATR'] * atr_k:
            return True
    return False

def vol_ok(V, i):
    return V[i] >= rolling_vol_avg(V, i, CONFIG['VOL_LOOKBACK']) * CONFIG['MIN_VOL_RATIO']

def in_session(ts):
    if not CONFIG['USE_SESSIONS']: return True
    h = ts.hour
    return (CONFIG['SESSION_HOURS']['london'][0] <= h < CONFIG['SESSION_HOURS']['london'][1] or
            CONFIG['SESSION_HOURS']['ny'][0]     <= h < CONFIG['SESSION_HOURS']['ny'][1])


# ════════════════════════════════════════════════════════════
#  TRADE
# ════════════════════════════════════════════════════════════

@dataclass
class Trade:
    symbol: str; direction: str; entry: float; sl: float
    tp1: float; tp2: float; open_bar: int; technique: str
    tp1_hit: bool = False; closed: bool = False; close_bar: int = -1
    close_px: float = 0.0; pnl_r: float = 0.0; exit_reason: str = ''


# ════════════════════════════════════════════════════════════
#  SIGNAL GENERATOR
# ════════════════════════════════════════════════════════════

def generate_signals(df: pd.DataFrame, symbol: str) -> List[Trade]:
    if len(df) < 80: return []

    O  = df['open'].values.astype(float)
    H  = df['high'].values.astype(float)
    L  = df['low'].values.astype(float)
    C  = df['close'].values.astype(float)
    V  = df['volume'].values.astype(float)
    TS = df['timestamp'].values

    ATR    = calc_atr(H, L, C, CONFIG['ATR_PERIOD'])
    T4H    = resample_4h(df)
    swings = find_swings(H, L, CONFIG['SWING_N'])

    trades: List[Trade] = []
    last_sig: Dict[str, int] = {}
    spent_obs = set()
    spent_bb  = set()
    n = len(C)

    bull_obs = find_bull_obs(O, H, L, C, V, ATR, swings, T4H, n-3)
    bear_obs = find_bear_obs(O, H, L, C, V, ATR, swings, T4H, n-3)
    bear_breakers: List[OrderBlock] = []

    for i in range(60, n-1):
        atr         = max(ATR[i], 1e-9)
        t4h         = T4H[i]
        ts          = pd.Timestamp(TS[i])
        is_bull_bar = C[i] > O[i]
        buf         = atr * 0.2

        if C[i] > 0 and atr / C[i] > CONFIG['MAX_ATR_PRICE_PCT']:
            continue

        for ob in bear_obs:
            if ob.failed or ob.formed >= i: continue
            if C[i] > ob.top + buf:
                ob.failed = True; ob.fail_at = i; bear_breakers.append(ob)

        # ── ROB ─────────────────────────────────────────────
        for ob in bull_obs:
            if id(ob) in spent_obs or ob.formed >= i: continue
            if not (L[i] <= ob.top + buf and H[i] >= ob.bottom - buf): continue
            if i - ob.formed > CONFIG['OB_MAX_AGE']: continue
            if t4h == 'bear': continue
            if CONFIG['REQUIRE_DIR_CANDLE'] and not is_bull_bar: continue
            if not vol_ok(V, i): continue
            if CONFIG['FIRST_TOUCH_ONLY'] and zone_previously_touched_rob(ob, H, L, i): continue
            if not price_departed_zone(ob, H, L, ATR, ob.formed+1, i): continue
            if i - last_sig.get('LONG_ROB', 0) < CONFIG['COOLDOWN_BARS']: continue
            if not in_session(ts): continue

            entry = ob.top
            sl    = ob.bottom - atr * CONFIG['SL_ATR_BUFFER']
            risk  = abs(entry - sl)
            if risk < 1e-9: continue
            tp1 = entry + risk * CONFIG['TP1_R']
            tp2 = entry + risk * CONFIG['TP2_R']
            if abs(tp1-entry)/risk < CONFIG['MIN_RR_FILTER']: continue

            trades.append(Trade(symbol=symbol, direction='LONG',
                entry=entry, sl=sl, tp1=tp1, tp2=tp2, open_bar=i, technique='ROB'))
            last_sig['LONG_ROB'] = i; spent_obs.add(id(ob))

        # ── BB ──────────────────────────────────────────────
        for bb in bear_breakers:
            if id(bb) in spent_bb or bb.fail_at < 0 or i <= bb.fail_at: continue
            if not (L[i] <= bb.top + buf and H[i] >= bb.bottom - buf): continue
            if i - bb.fail_at > CONFIG['BB_MAX_AGE']: continue
            if t4h == 'bear': continue
            if CONFIG['REQUIRE_DIR_CANDLE'] and not is_bull_bar: continue
            if not vol_ok(V, i): continue
            if CONFIG['FIRST_TOUCH_ONLY'] and zone_previously_touched_bb(bb, H, L, i): continue
            if not price_departed_zone(bb, H, L, ATR, bb.fail_at+1, i): continue
            if i - last_sig.get('LONG_BB', 0) < CONFIG['COOLDOWN_BARS']: continue
            if not in_session(ts): continue

            entry = bb.top
            sl    = bb.bottom - atr * CONFIG['SL_ATR_BUFFER']
            risk  = abs(entry - sl)
            if risk < 1e-9: continue
            tp1 = entry + risk * CONFIG['TP1_R']
            tp2 = entry + risk * CONFIG['TP2_R']
            if abs(tp1-entry)/risk < CONFIG['MIN_RR_FILTER']: continue

            trades.append(Trade(symbol=symbol, direction='LONG',
                entry=entry, sl=sl, tp1=tp1, tp2=tp2, open_bar=i, technique='BB'))
            last_sig['LONG_BB'] = i; spent_bb.add(id(bb))

    return trades


# ════════════════════════════════════════════════════════════
#  SIMULATOR
# ════════════════════════════════════════════════════════════

def simulate_trades(df: pd.DataFrame, trades: List[Trade]) -> List[Trade]:
    H = df['high'].values.astype(float)
    L = df['low'].values.astype(float)
    n = len(df)
    for t in trades:
        if t.open_bar + 1 >= n:
            t.closed = True; t.exit_reason = 'no_data'; continue
        be_sl = t.entry
        for j in range(t.open_bar+1, min(n, t.open_bar+CONFIG['TIMEOUT_BARS']+1)):
            h = H[j]; l = L[j]
            act_sl = be_sl if t.tp1_hit else t.sl
            if l <= act_sl:
                t.pnl_r = CONFIG['TP1_SIZE']*CONFIG['TP1_R'] if t.tp1_hit else -1.0
                t.exit_reason = 'BE' if t.tp1_hit else 'SL'
                t.closed = True; t.close_bar = j; t.close_px = act_sl; break
            if not t.tp1_hit and h >= t.tp1: t.tp1_hit = True
            if t.tp1_hit and h >= t.tp2:
                t.pnl_r = (CONFIG['TP1_SIZE']*CONFIG['TP1_R'] +
                           (1-CONFIG['TP1_SIZE'])*CONFIG['TP2_R'])
                t.exit_reason = 'TP2'; t.closed = True; t.close_bar = j; t.close_px = t.tp2; break
        if not t.closed:
            t.pnl_r = CONFIG['TP1_SIZE']*CONFIG['TP1_R'] if t.tp1_hit else 0.0
            t.exit_reason = 'Timeout+TP1' if t.tp1_hit else 'Timeout'
            t.closed = True; t.close_bar = min(t.open_bar+CONFIG['TIMEOUT_BARS'], n-1)
    return trades


# ════════════════════════════════════════════════════════════
#  METRICS
# ════════════════════════════════════════════════════════════

def compute_metrics(trades, label='', min_trades=0):
    if not trades: return {}
    pnls = [t.pnl_r for t in trades if t.closed]
    if len(pnls) < max(min_trades, 1): return {}
    wins   = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p < 0]
    be     = [p for p in pnls if p == 0]
    n      = len(pnls)
    wr     = len(wins)/n*100
    gp, gl = sum(wins), abs(sum(losses))
    pf     = gp/gl if gl > 0 else float('inf')
    eq     = np.cumsum(pnls)
    max_dd = float(np.max(np.maximum.accumulate(eq) - eq)) if len(eq) > 0 else 0
    tech_stats = {}
    for tech in ('ROB', 'BB'):
        tts = [t for t in trades if t.technique == tech and t.closed]
        if not tts: continue
        tpnl = [t.pnl_r for t in tts]; tw = [p for p in tpnl if p > 0]
        tech_stats[tech] = dict(count=len(tts), wr=round(len(tw)/len(tts)*100,1),
                                avg_r=round(float(np.mean(tpnl)),3), net_r=round(sum(tpnl),3))
    return dict(
        label=label, total=n, wins=len(wins), losses=len(losses), be=len(be),
        win_rate=round(wr,2), avg_win_r=round(float(np.mean(wins)) if wins else 0,3),
        avg_loss_r=round(float(np.mean(losses)) if losses else 0,3),
        profit_factor=round(pf,2) if pf != float('inf') else 'inf',
        net_r=round(sum(pnls),3), max_dd_r=round(max_dd,3),
        tp1_hits=sum(1 for t in trades if t.tp1_hit and t.closed),
        tp2_hits=sum(1 for t in trades if t.exit_reason=='TP2'),
        sl_hits=sum(1 for t in trades if t.exit_reason=='SL'),
        be_hits=sum(1 for t in trades if t.exit_reason=='BE'),
        timeouts=sum(1 for t in trades if 'Timeout' in (t.exit_reason or '')),
        by_technique=tech_stats,
    )


# ════════════════════════════════════════════════════════════
#  REPORT
# ════════════════════════════════════════════════════════════

def ps(c='═', w=70): print(BOLD + c*w + RESET)

def pf_str(pf): return f"{pf:.2f}" if isinstance(pf, float) else str(pf)

def print_metrics(m, indent=''):
    if not m: return
    wr = m['win_rate']; pf = m['profit_factor']; nr = m['net_r']
    wc = GREEN if wr >= 60 else (YELLOW if wr >= 52 else RED)
    pfc= GREEN if (pf == 'inf' or (isinstance(pf,float) and pf >= 2.0)) else \
        (YELLOW if isinstance(pf,float) and pf >= 1.5 else RED)
    nc = GREEN if nr >= 0 else RED
    print(f"{indent}Trades  : {BOLD}{m['total']}{RESET}  (W:{m['wins']} L:{m['losses']} BE:{m['be']})")
    print(f"{indent}Win Rate: {wc}{BOLD}{wr}%{RESET}  PF: {pfc}{BOLD}{pf_str(pf)}{RESET}")
    print(f"{indent}Net R   : {nc}{BOLD}{nr:+.2f}R{RESET}  MaxDD: {m['max_dd_r']:.2f}R")
    print(f"{indent}Avg Win : {GREEN}{m['avg_win_r']:+.3f}R{RESET}  "
          f"Avg Loss: {RED}{m['avg_loss_r']:+.3f}R{RESET}")
    print(f"{indent}TP1:{m['tp1_hits']}  TP2:{m['tp2_hits']}  "
          f"SL:{m['sl_hits']}  BE:{m['be_hits']}  Timeout:{m['timeouts']}")
    if m.get('by_technique'):
        print(f"{indent}┌─ By Technique ─────────────────────────────────────────")
        for tech, s in m['by_technique'].items():
            col = GREEN if s['wr'] >= 60 else (YELLOW if s['wr'] >= 52 else RED)
            print(f"{indent}│  {BOLD}{tech:6}{RESET} n={s['count']:3}  "
                  f"WR={col}{s['wr']:5.1f}%{RESET}  "
                  f"avg={s['avg_r']:+.3f}R  net={s['net_r']:+.3f}R")
        print(f"{indent}└────────────────────────────────────────────────────────")

def print_report(results, all_trades, shown_labels):
    print()
    ps('═')
    print(BOLD + CYAN +
          f"   SMC CLEAN BACKTEST v5.4 — LONG ONLY  (≥{CONFIG['MIN_TRADES_PAIR']} trades, $5M+ vol)" + RESET)
    ps('═')
    print(f"{'Pair':<14} {'#':>4} {'WR%':>7} {'PF':>7} {'NetR':>8} {'DD':>7}  Techniques")
    ps('─')

    shown    = [m for m in results if m and m.get('total',0) >= CONFIG['MIN_TRADES_PAIR']]
    excluded = [m for m in results if m and 0 < m.get('total',0) < CONFIG['MIN_TRADES_PAIR']]
    zero     = [m for m in results if m and m.get('total',0) == 0]

    for m in sorted(shown, key=lambda x: x.get('net_r',0), reverse=True):
        wr  = m['win_rate']; pf = m['profit_factor']; nr = m['net_r']
        wc  = GREEN if wr >= 60 else (YELLOW if wr >= 52 else RED)
        nc  = GREEN if nr >= 0 else RED
        pfc = GREEN if (pf == 'inf' or (isinstance(pf,float) and pf >= 2.0)) else \
              (YELLOW if isinstance(pf,float) and pf >= 1.5 else RED)
        tech_s = ' | '.join(f"{k}:{v['count']}" for k,v in m.get('by_technique',{}).items())
        print(f"{m['label']:<14} {m['total']:>4} {wc}{wr:>6.1f}%{RESET}"
              f" {pfc}{pf_str(pf):>7}{RESET} {nc}{nr:>+8.2f}R{RESET}"
              f" {m['max_dd_r']:>6.2f}R  {tech_s}")

    if excluded:
        print(f"\n  {YELLOW}Thin (<{CONFIG['MIN_TRADES_PAIR']} trades, excluded from aggregate):{RESET}")
        for m in sorted(excluded, key=lambda x: x.get('net_r',0), reverse=True):
            nc = GREEN if m['net_r'] >= 0 else RED
            print(f"  {m['label']:<12} {m['total']:>2} trades  "
                  f"WR {m['win_rate']:.0f}%  net {nc}{m['net_r']:+.2f}R{RESET}")
    if zero:
        print(f"\n  {YELLOW}Zero signals: {', '.join(m['label'] for m in zero if m)}{RESET}")

    ps('═')
    print(BOLD + CYAN +
          f"   AGGREGATE  (pairs ≥{CONFIG['MIN_TRADES_PAIR']} trades)" + RESET)
    ps('─')
    q_trades = [t for t in all_trades if t.symbol in shown_labels]
    ov = compute_metrics(q_trades, 'ALL')
    print_metrics(ov, indent='  ')

    if q_trades:
        pnls = [t.pnl_r for t in q_trades if t.closed]
        if pnls:
            eq = np.cumsum(pnls); cur = max_s = 0
            for p in pnls:
                cur = cur+1 if p < 0 else 0; max_s = max(max_s, cur)
            print(f"\n  Peak equity   : {GREEN}{max(eq):+.2f}R{RESET}")
            print(f"  Worst streak  : {RED}{max_s} consecutive losses{RESET}")
            print(f"  Expectancy    : {BOLD}{round(sum(pnls)/len(pnls), 3):+.3f}R per trade{RESET}")

    ps('═')
    if not ov:
        print(BOLD + RED + "\n  ❌ No qualifying pairs" + RESET); print(); return

    wr = ov.get('win_rate',0); nr = ov.get('net_r',0)
    pf = ov.get('profit_factor',0); total = ov.get('total',0)
    print()
    if total < 50:
        print(BOLD+YELLOW+f"  ⚠️  SMALL SAMPLE ({total} trades) — volume floor may be too high"+RESET)
    elif wr >= 60 and (pf == 'inf' or (isinstance(pf,float) and pf >= 2.0)) and nr > 0:
        print(BOLD+GREEN + "  ✅ STRONG — live bot confirmed" + RESET)
    elif wr >= 52 and isinstance(pf,float) and pf >= 1.5 and nr > 0:
        print(BOLD+YELLOW+ "  ⚠️  ACCEPTABLE — paper trade first" + RESET)
    elif nr > 0:
        print(BOLD+YELLOW+ "  ➡️  MARGINAL EDGE" + RESET)
    else:
        print(BOLD+RED   + "  ❌ WEAK" + RESET)
    print()


# ════════════════════════════════════════════════════════════
#  ASYNC RUNNER
# ════════════════════════════════════════════════════════════

async def fetch_top_pairs(ex):
    await ex.load_markets()
    tickers = await ex.fetch_tickers()
    pairs = [
        s for s in ex.symbols
        if s.endswith('/USDT:USDT') and 'PERP' not in s
        and tickers.get(s,{}).get('quoteVolume',0) > CONFIG['MIN_VOL_USDT']
        and s.replace('/USDT:USDT','') not in CONFIG['PAIR_BLACKLIST']
    ]
    pairs.sort(key=lambda x: tickers.get(x,{}).get('quoteVolume',0), reverse=True)
    top = pairs[:CONFIG['TOP_PAIRS']]
    print(f"  {CYAN}{len(top)} pairs  "
          f"(vol >${CONFIG['MIN_VOL_USDT']//1_000_000}M, top {CONFIG['TOP_PAIRS']}){RESET}")
    for p in top:
        vol = tickers.get(p,{}).get('quoteVolume',0)
        print(f"    {p.replace('/USDT:USDT',''):<14} ${vol/1e6:>7.1f}M")
    return top

async def fetch_ohlcv(ex, symbol, tf, limit):
    try:
        ohlcv = await ex.fetch_ohlcv(symbol, tf, limit=limit)
        df = pd.DataFrame(ohlcv, columns=['timestamp','open','high','low','close','volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
        return df
    except Exception as e:
        logger.warning(f"{symbol}: {e}"); return None

async def run_backtest():
    print()
    ps('═')
    print(BOLD + CYAN + "  SMC CLEAN BACKTEST v5.4  —  LONG ONLY" + RESET)
    print(f"  ROB + BB  |  $5M+ vol  |  Top {CONFIG['TOP_PAIRS']} pairs  |  "
          f"Min {CONFIG['MIN_TRADES_PAIR']} trades to count")
    print(f"  BASIL={CONFIG['OB_BASIL_MIN']}  ImpulseSkip={CONFIG['IMPULSE_SKIP']}  "
          f"Depart={CONFIG['MIN_DEPART_ATR']}xATR  Vol={CONFIG['MIN_VOL_RATIO']}x  "
          f"TP1={CONFIG['TP1_R']}R  TP2={CONFIG['TP2_R']}R  Bars={CONFIG['BARS_PER_PAIR']}")
    ps('═')

    ex = ccxt.binance({'enableRateLimit': True, 'options': {'defaultType': 'future'}})
    try:
        print(f"\n  {YELLOW}Fetching top pairs by volume...{RESET}\n")
        pairs = await fetch_top_pairs(ex)

        all_results, all_trades = [], []
        print(f"\n  {YELLOW}Running backtest ({CONFIG['BARS_PER_PAIR']} bars/pair)...{RESET}\n")
        pair_iter = tqdm(pairs, desc='  Pairs', ncols=72) if HAS_TQDM else pairs

        for symbol in pair_iter:
            try:
                df = await fetch_ohlcv(ex, symbol, CONFIG['SIGNAL_TF'], CONFIG['BARS_PER_PAIR'])
                if df is None or len(df) < 100:
                    await asyncio.sleep(0.3); continue
                label  = symbol.replace('/USDT:USDT', '')
                trades = generate_signals(df, label)
                trades = simulate_trades(df, trades)
                m      = compute_metrics(trades, label)
                if not m: m = {'label': label, 'total': 0, 'net_r': 0}
                all_results.append(m); all_trades.extend(trades)
                if not HAS_TQDM:
                    if m.get('total',0) > 0:
                        nc = GREEN if m['net_r'] > 0 else RED
                        print(f"  {label:<14} {m['total']:>3} trades  "
                              f"WR {m.get('win_rate',0):.0f}%  "
                              f"net {nc}{m['net_r']:+.2f}R{RESET}")
                    else:
                        print(f"  {label:<14}   0 trades")
                await asyncio.sleep(0.15)
            except Exception as e:
                logger.error(f"{symbol}: {e}"); await asyncio.sleep(1)

        shown_labels = {m['label'] for m in all_results
                        if m and m.get('total',0) >= CONFIG['MIN_TRADES_PAIR']}
        print_report(all_results, all_trades, shown_labels)

        out = {
            'generated': datetime.now(timezone.utc).isoformat(),
            'version': '5.4-clean',
            'config': {k: (list(v) if isinstance(v, set) else v) for k,v in CONFIG.items()},
            'pairs_tested': len(all_results),
            'pairs_qualifying': len(shown_labels),
            'total_trades': len(all_trades),
            'aggregate': compute_metrics(
                [t for t in all_trades if t.symbol in shown_labels], 'aggregate'),
            'per_pair': [m for m in all_results if m and m.get('total',0) > 0],
        }
        with open('smc_backtest_clean_results.json', 'w') as f:
            json.dump(out, f, indent=2, default=str)
        print(f"  {CYAN}Results → smc_backtest_clean_results.json{RESET}\n")
    finally:
        await ex.close()

if __name__ == '__main__':
    asyncio.run(run_backtest())
