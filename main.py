"""
ORDER BLOCKS SCANNER BOT v2.0
================================
Strategy: Smart Money Order Blocks (Market Mechanics)

Flow: 4H OB identification → 1H Market Structure Shift (MSS) → 15min sniper entry
  - Order Block: last candle BEFORE impulsive move (demand/supply)
  - Validity: FVG (imbalance) + BOS (structural break) + Unmitigated
  - Entry: 15min rejection candle at OB zone after 1H MSS
  - SL: below/above OB + 0.25 ATR buffer
  - TP1: 1R (50% close → SL to BE), TP2: 2R runner

Install:
  pip install ccxt ta pandas numpy python-telegram-bot
"""

import asyncio
import ccxt.async_support as ccxt
from telegram import Bot, Update
from telegram.ext import Application, CommandHandler, ContextTypes
from telegram.constants import ParseMode
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import List, Optional, Dict, Tuple
from collections import deque
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

import warnings
warnings.filterwarnings('ignore')

# ═══════════════════════════════════════════════════════
# ★  SETTINGS
# ═══════════════════════════════════════════════════════

# Order Block Detection (4H)
OB_LOOKBACK         = 120    # look back 120 4H bars (~20 days) — OBs stay valid for weeks
OB_MAX_AGE_BARS     = 120    # OB expires after 120 4H bars (~20 days)
IMPULSE_ATR_MULT    = 1.5    # impulse move > 1.5x ATR

# MSS Detection (1H)
MSS_LOOKBACK        = 40
SWING_PERIOD        = 2      # swing detection sensitivity (2 = more sensitive)

# Entry (15min)
MIN_BODY_ATR        = 0.10   # min candle body (relaxed for 15m)
MIN_WICK_ATR        = 0.25   # min wick for pin bar
OB_BUFFER_ATR       = 0.5    # wider zone buffer for realistic price touches

# Trade Management
ATR_SL_BUFFER       = 0.25
TP1_R               = 1.0
TP2_R               = 2.0
TP1_CLOSE_PCT       = 0.50
MAX_TRADE_HOURS     = 72
ATR_PERIOD          = 14

# Scanner
SCAN_INTERVAL_MIN   = 15
MIN_VOLUME_USDT     = 500_000
TOP_PAIRS_LIMIT     = 600
COOLDOWN_HOURS      = 4


# ═══════════════════════════════════════════════════════
# UTILITIES
# ═══════════════════════════════════════════════════════

def _calc_atr(H, L, C, period=14):
    n = len(C); tr = np.zeros(n); tr[0] = H[0]-L[0]
    for i in range(1, n):
        tr[i] = max(H[i]-L[i], abs(H[i]-C[i-1]), abs(L[i]-C[i-1]))
    atr = np.zeros(n)
    if n >= period:
        atr[period-1] = np.mean(tr[:period])
        for i in range(period, n):
            atr[i] = (atr[i-1]*(period-1)+tr[i])/period
        for i in range(period-1):
            atr[i] = atr[period-1]
    return atr


def _swing_highs(H, period=2):
    n = len(H); sh = []
    for i in range(period, n-period):
        if all(H[i] >= H[i-j] for j in range(1, period+1)) and \
           all(H[i] >= H[i+j] for j in range(1, period+1)):
            sh.append(i)
    return sh


def _swing_lows(L, period=2):
    n = len(L); sl = []
    for i in range(period, n-period):
        if all(L[i] <= L[i-j] for j in range(1, period+1)) and \
           all(L[i] <= L[i+j] for j in range(1, period+1)):
            sl.append(i)
    return sl


# ═══════════════════════════════════════════════════════
# ORDER BLOCK STRUCTURE
# ═══════════════════════════════════════════════════════

@dataclass
class OrderBlock:
    kind:        str    # 'demand' | 'supply'
    ob_high:     float
    ob_low:      float
    ob_close:    float
    formed_at:   int
    impulse_atr: float
    has_fvg:     bool
    fvg_size:    float
    bos_confirmed: bool
    mitigated:   bool = False
    mitigated_at: int = -1

    @property
    def mid(self):    return (self.ob_high + self.ob_low) / 2
    @property
    def height(self): return max(self.ob_high - self.ob_low, 1e-9)


# ═══════════════════════════════════════════════════════
# 4H ORDER BLOCK DETECTION
# ═══════════════════════════════════════════════════════

def detect_order_blocks(H, L, C, O, V, ATR, scan_start, scan_end):
    """
    Detect Order Blocks on 4H.
    OB = last candle BEFORE a strong impulsive move.
    Valid: FVG + BOS + Unmitigated.
    """
    blocks: List[OrderBlock] = []

    for i in range(scan_start+3, scan_end-6):
        atr = ATR[i]
        if atr < 1e-9:
            continue
        next_end = min(i+5, scan_end-1)

        # ── DEMAND OB: bullish impulse from bar i ──
        impulse_up = float(C[next_end]) - float(H[i])
        if impulse_up >= IMPULSE_ATR_MULT * atr:
            # OB candle: last bearish/neutral candle before impulse
            ob_c = i
            for back in range(i, max(scan_start+1, i-4), -1):
                if C[back] <= O[back]:
                    ob_c = back; break

            ob_high = float(H[ob_c]); ob_low = float(L[ob_c])

            # FVG: 3-candle imbalance OR classic gap
            fvg, fvg_sz = False, 0.0
            if ob_c+2 < scan_end:
                gap = float(H[ob_c]) - float(L[ob_c+2])
                if gap > 0: fvg, fvg_sz = True, gap
            if not fvg and ob_c > 0 and ob_c+1 < scan_end:
                gap2 = float(L[ob_c+1]) - float(H[ob_c-1])
                if gap2 > 0: fvg, fvg_sz = True, gap2

            # BOS: impulse closed above prior N-bar high
            lookback_s = max(0, i-30)
            prior_high = float(np.max(H[lookback_s:i]))
            bos = float(C[next_end]) > prior_high

            # Mitigation: body returned INSIDE OB zone
            mitigated = False; mit_at = -1
            for j in range(ob_c+2, scan_end):
                if float(C[j]) < ob_high and float(O[j]) < ob_high:
                    mitigated = True; mit_at = j; break

            blocks.append(OrderBlock(
                kind='demand', ob_high=ob_high, ob_low=ob_low,
                ob_close=float(C[ob_c]), formed_at=ob_c,
                impulse_atr=round(impulse_up/atr, 2),
                has_fvg=fvg, fvg_size=round(fvg_sz, 6),
                bos_confirmed=bos, mitigated=mitigated, mitigated_at=mit_at
            ))

        # ── SUPPLY OB: bearish impulse from bar i ──
        impulse_dn = float(L[i]) - float(C[next_end])
        if impulse_dn >= IMPULSE_ATR_MULT * atr:
            ob_c = i
            for back in range(i, max(scan_start+1, i-4), -1):
                if C[back] >= O[back]:
                    ob_c = back; break

            ob_high = float(H[ob_c]); ob_low = float(L[ob_c])

            fvg, fvg_sz = False, 0.0
            if ob_c+2 < scan_end:
                gap = float(L[ob_c]) - float(H[ob_c+2])
                if gap > 0: fvg, fvg_sz = True, gap
            if not fvg and ob_c > 0 and ob_c+1 < scan_end:
                gap2 = float(L[ob_c-1]) - float(H[ob_c+1])
                if gap2 > 0: fvg, fvg_sz = True, gap2

            lookback_s = max(0, i-30)
            prior_low = float(np.min(L[lookback_s:i]))
            bos = float(C[next_end]) < prior_low

            mitigated = False; mit_at = -1
            for j in range(ob_c+2, scan_end):
                if float(C[j]) > ob_low and float(O[j]) > ob_low:
                    mitigated = True; mit_at = j; break

            blocks.append(OrderBlock(
                kind='supply', ob_high=ob_high, ob_low=ob_low,
                ob_close=float(C[ob_c]), formed_at=ob_c,
                impulse_atr=round(impulse_dn/atr, 2),
                has_fvg=fvg, fvg_size=round(fvg_sz, 6),
                bos_confirmed=bos, mitigated=mitigated, mitigated_at=mit_at
            ))

    # Quality filter: unmitigated only; sort by FVG + impulse
    valid = [b for b in blocks if not b.mitigated]
    # Deduplicate: keep highest impulse per ob_high level
    seen = {}
    deduped = []
    for b in valid:
        key = (b.kind, round(b.ob_high, 4))
        if key not in seen or b.impulse_atr > seen[key].impulse_atr:
            seen[key] = b
    deduped = sorted(seen.values(), key=lambda b: (int(b.has_fvg), b.impulse_atr), reverse=True)
    return deduped


# ═══════════════════════════════════════════════════════
# 1H MARKET STRUCTURE SHIFT (MSS) DETECTION
# ═══════════════════════════════════════════════════════

def detect_mss_1h(H1, L1, C1, end_idx, lookback=40):
    """
    Detect Market Structure Shift.
    Bullish MSS: downtrend (LH/LL) → closes above last LH
    Bearish MSS: uptrend (HH/HL) → closes below last HL
    """
    s = max(0, end_idx - lookback)
    h = H1[s:end_idx]; l = L1[s:end_idx]; c = C1[s:end_idx]
    n = len(c)
    if n < 10:
        return None, 0.0

    sw = SWING_PERIOD
    sh_idx = _swing_highs(h, sw)
    sl_idx = _swing_lows(l, sw)
    curr_close = float(c[-1])

    if len(sh_idx) >= 2 and len(sl_idx) >= 2:
        last_sh = sh_idx[-1]; prev_sh = sh_idx[-2]
        last_sl = sl_idx[-1]; prev_sl = sl_idx[-2]

        # Downtrend → Bullish MSS
        was_down = h[last_sh] < h[prev_sh] and l[last_sl] < l[prev_sl]
        if was_down and curr_close > h[last_sh]:
            return 'bull', float(H1[s + last_sh])

        # Uptrend → Bearish MSS
        was_up = h[last_sh] > h[prev_sh] and l[last_sl] > l[prev_sl]
        if was_up and curr_close < l[last_sl]:
            return 'bear', float(L1[s + last_sl])

    # Fallback: momentum break of recent range
    if n >= 20:
        prior_high = float(np.max(h[:-5]))
        prior_low  = float(np.min(l[:-5]))
        recent_c   = c[-5:]
        if all(recent_c > prior_high): return 'bull', prior_high
        if all(recent_c < prior_low):  return 'bear', prior_low

    return None, 0.0


def get_1h_trend(H1, L1, C1, end_idx, lookback=40):
    s = max(0, end_idx - lookback)
    h = H1[s:end_idx]; l = L1[s:end_idx]
    sh = _swing_highs(h, SWING_PERIOD); sl = _swing_lows(l, SWING_PERIOD)
    if len(sh) < 2 or len(sl) < 2:
        return 'range'
    if h[sh[-1]] > h[sh[-2]] and l[sl[-1]] > l[sl[-2]]:
        return 'uptrend'
    if h[sh[-1]] < h[sh[-2]] and l[sl[-1]] < l[sl[-2]]:
        return 'downtrend'
    return 'range'


# ═══════════════════════════════════════════════════════
# 15MIN ENTRY DETECTION
# ═══════════════════════════════════════════════════════

def detect_15min_entry(O15, H15, L15, C15, ATR15, end_idx, ob: OrderBlock):
    """15min rejection candle at OB zone."""
    i = end_idx - 1
    if i < 1:
        return False, 'none'
    atr = ATR15[i]
    if atr < 1e-9:
        return False, 'none'

    buf = atr * OB_BUFFER_ATR
    o = O15[i]; h = H15[i]; l = L15[i]; c = C15[i]
    body = abs(c - o); rng = h - l

    if body < MIN_BODY_ATR * atr:
        return False, 'none'

    if ob.kind == 'demand':
        in_zone = l <= ob.ob_high + buf and h >= ob.ob_low - buf
        if not in_zone: return False, 'none'
        lower_wick = min(o, c) - l
        if lower_wick >= MIN_WICK_ATR * atr and lower_wick >= 1.5 * max(body, 1e-9):
            return True, 'pin_bar'
        if i >= 1 and c > o and C15[i-1] < O15[i-1] and c >= O15[i-1] and o <= C15[i-1]:
            return True, 'bull_engulf'
        if c > o and (c-l)/max(rng,1e-9) > 0.65 and body >= 0.3*atr:
            return True, 'strong_close'
    else:
        in_zone = h >= ob.ob_low - buf and l <= ob.ob_high + buf
        if not in_zone: return False, 'none'
        upper_wick = h - max(o, c)
        if upper_wick >= MIN_WICK_ATR * atr and upper_wick >= 1.5 * max(body, 1e-9):
            return True, 'pin_bar'
        if i >= 1 and c < o and C15[i-1] > O15[i-1] and c <= O15[i-1] and o >= C15[i-1]:
            return True, 'bear_engulf'
        if c < o and (h-c)/max(rng,1e-9) > 0.65 and body >= 0.3*atr:
            return True, 'strong_close'

    return False, 'none'


# ═══════════════════════════════════════════════════════
# BACKTESTER
# ═══════════════════════════════════════════════════════

@dataclass
class BacktestTrade:
    symbol:     str
    direction:  str
    entry:      float
    sl:         float
    tp1:        float
    tp2:        float
    entry_bar:  int
    entry_time: datetime
    exit_time:  datetime = None
    exit_price: float = 0.0
    outcome:    str = 'open'
    r_result:   float = 0.0
    ob_kind:    str = ''
    has_fvg:    bool = False
    entry_type: str = ''
    trend_1h:   str = ''
    mss_dir:    str = ''


def run_backtest(symbol: str, df4h: pd.DataFrame,
                 df1h: pd.DataFrame, df15m: pd.DataFrame) -> List[BacktestTrade]:
    """Walk-forward backtest: 4H OBs → 1H MSS → 15min entry."""
    trades: List[BacktestTrade] = []
    if len(df4h) < 60 or len(df1h) < 80 or len(df15m) < 100:
        return trades

    H4  = df4h['high'].values.astype(float)
    L4  = df4h['low'].values.astype(float)
    C4  = df4h['close'].values.astype(float)
    O4  = df4h['open'].values.astype(float)
    V4  = df4h['volume'].values.astype(float)
    T4  = df4h['timestamp'].values
    ATR4= _calc_atr(H4, L4, C4, ATR_PERIOD)

    H1  = df1h['high'].values.astype(float)
    L1  = df1h['low'].values.astype(float)
    C1  = df1h['close'].values.astype(float)
    T1  = df1h['timestamp'].values

    H15 = df15m['high'].values.astype(float)
    L15 = df15m['low'].values.astype(float)
    C15 = df15m['close'].values.astype(float)
    O15 = df15m['open'].values.astype(float)
    T15 = df15m['timestamp'].values
    ATR15 = _calc_atr(H15, L15, C15, ATR_PERIOD)

    active_trades: List[BacktestTrade] = []
    cooldown: Dict[str, int] = {}

    for bar15 in range(60, len(H15)-1):

        # ── Update active trades ──────────────────────────────
        for t in list(active_trades):
            price = float(C15[bar15])
            act_sl = t.entry if t.outcome == 'tp1' else t.sl
            age_h  = (pd.Timestamp(T15[bar15]) - t.entry_time).total_seconds() / 3600

            closed = False
            if t.direction == 'LONG':
                if t.outcome == 'open' and price >= t.tp1:
                    t.outcome = 'tp1'; t.r_result = TP1_R * TP1_CLOSE_PCT
                elif t.outcome == 'tp1' and price >= t.tp2:
                    t.outcome = 'tp2'; t.r_result = TP1_R*TP1_CLOSE_PCT + TP2_R*(1-TP1_CLOSE_PCT)
                    t.exit_price = t.tp2; t.exit_time = pd.Timestamp(T15[bar15])
                    active_trades.remove(t); trades.append(t); closed = True
                elif price <= act_sl:
                    if t.outcome == 'tp1':
                        t.outcome = 'be'; t.r_result = TP1_R * TP1_CLOSE_PCT
                    else:
                        t.outcome = 'sl'
                        risk = abs(t.entry - t.sl)
                        t.r_result = -(abs(price - t.entry)/risk) if risk > 0 else -1.0
                    t.exit_price = price; t.exit_time = pd.Timestamp(T15[bar15])
                    active_trades.remove(t); trades.append(t); closed = True
            else:
                if t.outcome == 'open' and price <= t.tp1:
                    t.outcome = 'tp1'; t.r_result = TP1_R * TP1_CLOSE_PCT
                elif t.outcome == 'tp1' and price <= t.tp2:
                    t.outcome = 'tp2'; t.r_result = TP1_R*TP1_CLOSE_PCT + TP2_R*(1-TP1_CLOSE_PCT)
                    t.exit_price = t.tp2; t.exit_time = pd.Timestamp(T15[bar15])
                    active_trades.remove(t); trades.append(t); closed = True
                elif price >= act_sl:
                    if t.outcome == 'tp1':
                        t.outcome = 'be'; t.r_result = TP1_R * TP1_CLOSE_PCT
                    else:
                        t.outcome = 'sl'
                        risk = abs(t.entry - t.sl)
                        t.r_result = -(abs(price - t.entry)/risk) if risk > 0 else -1.0
                    t.exit_price = price; t.exit_time = pd.Timestamp(T15[bar15])
                    active_trades.remove(t); trades.append(t); closed = True

            if not closed and t in active_trades and age_h > MAX_TRADE_HOURS:
                t.outcome = 'timeout'; t.exit_price = price
                risk = abs(t.entry - t.sl)
                if risk > 0:
                    t.r_result = (price-t.entry)/risk if t.direction=='LONG' else (t.entry-price)/risk
                t.exit_time = pd.Timestamp(T15[bar15])
                active_trades.remove(t); trades.append(t)

        if len(active_trades) >= 3:
            continue

        curr_ts = pd.Timestamp(T15[bar15])

        # Map 15m → 4H bar
        bar4 = int(np.searchsorted(T4, T15[bar15], side='right')) - 1
        if bar4 < 40: continue

        # Map 15m → 1H bar
        bar1 = int(np.searchsorted(T1, T15[bar15], side='right')) - 1
        if bar1 < 30: continue

        # 4H Order Blocks
        scan_s = max(0, bar4 - OB_LOOKBACK)
        obs = detect_order_blocks(H4, L4, C4, O4, V4, ATR4, scan_s, bar4)
        if not obs: continue

        for ob in obs[:6]:
            if (bar4 - ob.formed_at) > OB_MAX_AGE_BARS: continue

            # Cooldown
            ck = ob.kind
            if ck in cooldown and (bar15 - cooldown[ck]) < COOLDOWN_HOURS * 4:
                continue

            # 1H MSS
            mss_dir, _ = detect_mss_1h(H1, L1, C1, bar1, MSS_LOOKBACK)
            trend_1h   = get_1h_trend(H1, L1, C1, bar1)

            if ob.kind == 'demand' and mss_dir != 'bull': continue
            if ob.kind == 'supply' and mss_dir != 'bear': continue

            # 15min entry
            valid, entry_type = detect_15min_entry(O15, H15, L15, C15, ATR15, bar15+1, ob)
            if not valid: continue

            atr15 = ATR15[bar15]
            if ob.kind == 'demand':
                entry = ob.ob_high
                sl    = ob.ob_low - atr15 * ATR_SL_BUFFER
                risk  = abs(entry - sl)
                if risk < 1e-9: continue
                tp1 = entry + risk * TP1_R
                tp2 = entry + risk * TP2_R
                direction = 'LONG'
            else:
                entry = ob.ob_low
                sl    = ob.ob_high + atr15 * ATR_SL_BUFFER
                risk  = abs(sl - entry)
                if risk < 1e-9: continue
                tp1 = entry - risk * TP1_R
                tp2 = entry - risk * TP2_R
                direction = 'SHORT'

            t = BacktestTrade(
                symbol=symbol, direction=direction,
                entry=entry, sl=sl, tp1=tp1, tp2=tp2,
                entry_bar=bar15, entry_time=curr_ts,
                ob_kind=ob.kind, has_fvg=ob.has_fvg,
                entry_type=entry_type, trend_1h=trend_1h,
                mss_dir=mss_dir or ''
            )
            active_trades.append(t)
            cooldown[ck] = bar15
            break

    # Flush open trades
    for t in active_trades:
        t.outcome = 'timeout'; t.exit_price = float(C15[-1])
        t.exit_time = pd.Timestamp(T15[-1])
        risk = abs(t.entry - t.sl)
        if risk > 0:
            t.r_result = (t.exit_price-t.entry)/risk if t.direction=='LONG' \
                         else (t.entry-t.exit_price)/risk
        trades.append(t)

    return trades


def compute_backtest_stats(trades: List[BacktestTrade], symbol: str = 'ALL') -> dict:
    if not trades: return {}
    closed = [t for t in trades if t.outcome != 'open']
    wins   = [t for t in closed if t.outcome in ('tp1','tp2','be')]
    losses = [t for t in closed if t.outcome == 'sl']
    tp2s   = [t for t in closed if t.outcome == 'tp2']
    bes    = [t for t in closed if t.outcome == 'be']
    timeouts=[t for t in closed if t.outcome == 'timeout']

    n = len(closed)
    wr = round(len(wins)/n*100, 1) if n > 0 else 0
    r_vals = [t.r_result for t in closed]
    gross_profit = sum(r for r in r_vals if r > 0)
    gross_loss   = abs(sum(r for r in r_vals if r < 0))
    pf = round(gross_profit/gross_loss, 2) if gross_loss > 0 else 99.0

    equity = 0.0; peak = 0.0; max_dd = 0.0; eq_curve = []
    for r in r_vals:
        equity += r; eq_curve.append(equity)
        if equity > peak: peak = equity
        if (peak - equity) > max_dd: max_dd = peak - equity

    avg_rr = round(float(np.mean(r_vals)), 3) if r_vals else 0
    total_r= round(sum(r_vals), 2)

    span_days = 1
    if closed and closed[-1].exit_time and closed[0].entry_time:
        span_days = max(1, (closed[-1].exit_time - closed[0].entry_time).days)
    spd = round(n / span_days, 2)

    fvg_t = [t for t in closed if t.has_fvg]
    fvg_wr = round(sum(1 for t in fvg_t if t.outcome in ('tp1','tp2','be'))/
                   max(len(fvg_t),1)*100, 1)

    return {
        'symbol': symbol, 'total_trades': n,
        'wins': len(wins), 'losses': len(losses),
        'tp2_hits': len(tp2s), 'be_saves': len(bes), 'timeouts': len(timeouts),
        'win_rate': wr, 'profit_factor': pf,
        'max_dd_r': round(max_dd, 2), 'avg_rr': avg_rr,
        'total_r': total_r, 'signals_per_day': spd,
        'fvg_wr': fvg_wr, 'equity_curve': eq_curve,
    }


def format_backtest_report(results: List[dict]) -> str:
    if not results: return "❌ No backtest results."
    total_t  = sum(r['total_trades'] for r in results)
    total_w  = sum(r['wins'] for r in results)
    total_l  = sum(r['losses'] for r in results)
    total_tp2= sum(r['tp2_hits'] for r in results)
    total_be = sum(r['be_saves'] for r in results)
    total_to = sum(r['timeouts'] for r in results)
    overall_wr = round(total_w/max(total_t,1)*100, 1)
    all_pf = [r['profit_factor'] for r in results if r['profit_factor'] < 99]
    avg_pf = round(float(np.mean(all_pf)), 2) if all_pf else 0
    worst_dd = round(max(r['max_dd_r'] for r in results), 2)
    avg_rr   = round(float(np.mean([r['avg_rr'] for r in results])), 3)
    avg_spd  = round(float(np.mean([r['signals_per_day'] for r in results])), 2)
    total_r  = round(sum(r['total_r'] for r in results), 2)
    bar_wr   = '▰'*int(overall_wr/10)+'▱'*(10-int(overall_wr/10))

    m  = f"{'═'*40}\n📊 <b>ORDER BLOCKS BACKTEST REPORT</b>\n{'═'*40}\n\n"
    m += f"Pairs tested:  <b>{len(results)}</b>\n"
    m += f"Total trades:  <b>{total_t}</b>\n\n"
    m += f"<b>── Performance ──────────────────</b>\n"
    m += f"  Win Rate:        <b>{overall_wr}%</b>\n"
    m += f"  {bar_wr}\n"
    m += f"  Profit Factor:   <b>{avg_pf}</b>\n"
    m += f"  Avg R:R:         <b>{avg_rr}R</b>\n"
    m += f"  Total R gained:  <b>{total_r}R</b>\n\n"
    m += f"<b>── Outcomes ──────────────────────</b>\n"
    m += f"  ✅ Wins (TP1+):   {total_w}\n"
    m += f"  💰 TP2 runners:   {total_tp2}  ({round(total_tp2/max(total_w,1)*100)}% of wins)\n"
    m += f"  🔒 BE saves:      {total_be}\n"
    m += f"  ❌ Stop losses:   {total_l}\n"
    m += f"  ⏰ Timeouts:      {total_to}\n\n"
    m += f"<b>── Risk ────────────────────────</b>\n"
    m += f"  Max Drawdown:    <b>{worst_dd}R</b>\n"
    m += f"  Signals/day:     <b>{avg_spd}</b>\n\n"
    results_sorted = sorted(results, key=lambda x: x['total_r'], reverse=True)[:10]
    m += f"<b>── Top Pairs ──────────────────</b>\n"
    for r in results_sorted:
        pf_str = f"{r['profit_factor']:.1f}" if r['profit_factor'] < 99 else "∞"
        m += f"  #{r['symbol']}: {r['win_rate']}% WR | {r['total_r']}R | PF {pf_str} | {r['total_trades']}T\n"
    m += f"\n<i>Strategy: 4H OB → 1H MSS → 15m Rejection</i>\n"
    m += f"<i>⏰ {datetime.now().strftime('%d %b %Y %H:%M UTC')}</i>"
    return m


# ═══════════════════════════════════════════════════════
# MAIN SCANNER CLASS
# ═══════════════════════════════════════════════════════

class OBScanner:
    def __init__(self, telegram_token, telegram_chat_id,
                 binance_api_key=None, binance_secret=None):
        self.telegram_token = telegram_token
        self.telegram_bot   = Bot(token=telegram_token)
        self.chat_id        = telegram_chat_id
        self.exchange = ccxt.binance({
            'apiKey': binance_api_key, 'secret': binance_secret,
            'enableRateLimit': True, 'options': {'defaultType': 'future'},
        })
        self.active_trades: Dict[str, dict] = {}
        self.pair_cooldown: Dict[str, datetime] = {}
        self.signal_history: deque = deque(maxlen=300)
        self.is_scanning = False
        self.is_backtesting = False
        self.stats = {
            'total_signals':0, 'long_signals':0, 'short_signals':0,
            'tp1_hits':0, 'tp2_hits':0, 'sl_hits':0, 'be_saves':0, 'timeouts':0,
            'filtered':{'no_ob':0,'no_mss':0,'no_entry':0,'cooldown':0},
            'session_start': datetime.now(), 'last_scan': None, 'pairs_scanned':0,
        }

    async def _fetch_ohlcv(self, symbol, tf, limit=200):
        try:
            ohlcv = await self.exchange.fetch_ohlcv(symbol, tf, limit=limit)
            df = pd.DataFrame(ohlcv, columns=['timestamp','open','high','low','close','volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            return df
        except Exception as e:
            logger.error(f"fetch {symbol} {tf}: {e}"); return None

    def _detect_signal(self, symbol, df4h, df1h, df15m):
        if len(df4h)<60 or len(df1h)<60 or len(df15m)<60: return None
        H4=df4h['high'].values.astype(float); L4=df4h['low'].values.astype(float)
        C4=df4h['close'].values.astype(float); O4=df4h['open'].values.astype(float)
        V4=df4h['volume'].values.astype(float); ATR4=_calc_atr(H4,L4,C4,ATR_PERIOD)
        H1=df1h['high'].values.astype(float); L1=df1h['low'].values.astype(float)
        C1=df1h['close'].values.astype(float)
        H15=df15m['high'].values.astype(float); L15=df15m['low'].values.astype(float)
        C15=df15m['close'].values.astype(float); O15=df15m['open'].values.astype(float)
        ATR15=_calc_atr(H15,L15,C15,ATR_PERIOD)
        bar4=len(H4)-1; bar1=len(H1)-1

        scan_s = max(0, bar4-OB_LOOKBACK)
        obs = detect_order_blocks(H4,L4,C4,O4,V4,ATR4, scan_s, bar4)
        if not obs: self.stats['filtered']['no_ob']+=1; return None

        for ob in obs[:6]:
            if (bar4-ob.formed_at) > OB_MAX_AGE_BARS: continue
            mss_dir, _ = detect_mss_1h(H1,L1,C1, bar1, MSS_LOOKBACK)
            trend_1h   = get_1h_trend(H1,L1,C1, bar1)
            if ob.kind=='demand' and mss_dir!='bull': self.stats['filtered']['no_mss']+=1; continue
            if ob.kind=='supply' and mss_dir!='bear': self.stats['filtered']['no_mss']+=1; continue
            bar15 = len(H15)-1
            valid, entry_type = detect_15min_entry(O15,H15,L15,C15,ATR15, bar15+1, ob)
            if not valid: self.stats['filtered']['no_entry']+=1; continue
            atr15=ATR15[bar15]
            if ob.kind=='demand':
                entry=ob.ob_high; sl=ob.ob_low-atr15*ATR_SL_BUFFER; risk=abs(entry-sl)
                if risk<1e-9: continue
                tp1=entry+risk*TP1_R; tp2=entry+risk*TP2_R; direction='LONG'
            else:
                entry=ob.ob_low; sl=ob.ob_high+atr15*ATR_SL_BUFFER; risk=abs(sl-entry)
                if risk<1e-9: continue
                tp1=entry-risk*TP1_R; tp2=entry-risk*TP2_R; direction='SHORT'
            return self._build_signal(symbol,direction,entry,sl,tp1,tp2,atr15,ob,trend_1h,mss_dir or '',entry_type,risk)
        return None

    def _build_signal(self, symbol, direction, entry, sl, tp1, tp2,
                      atr, ob, trend_1h, mss_dir, entry_type, risk):
        pair=symbol.replace('/USDT:USDT','')
        tid=f"{pair}_{direction[0]}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        rr=round(abs(tp1-entry)/max(risk,1e-9),2)
        def pct(a,b): return round(abs(a-b)/max(abs(b),1e-9)*100,2)
        tags = []
        if ob.has_fvg: tags.append(f'⚡ FVG {ob.fvg_size:.4f}')
        if ob.impulse_atr>=2.5: tags.append(f'🚀 {ob.impulse_atr}x ATR impulse')
        if ob.bos_confirmed: tags.append('✅ BOS')
        rej={'pin_bar':'📍','bull_engulf':'🟢','bear_engulf':'🔴','strong_close':'💪'}
        tags.append(f"{rej.get(entry_type,'●')} {entry_type.replace('_',' ').title()}")
        return {
            'trade_id':tid,'symbol':pair,'full_symbol':symbol,'signal':direction,
            'entry':entry,'stop_loss':sl,'tp1':tp1,'tp1_pct':pct(tp1,entry),
            'tp2':tp2,'tp2_pct':pct(tp2,entry),'rr':rr,'risk_pct':pct(sl,entry),
            'atr':round(atr,6),'ob_high':round(ob.ob_high,6),'ob_low':round(ob.ob_low,6),
            'ob_kind':ob.kind,'has_fvg':ob.has_fvg,'impulse_atr':ob.impulse_atr,
            'trend_1h':trend_1h,'mss_dir':mss_dir,'entry_type':entry_type,'tags':tags,
            'close_plan':(f"📋 <b>Close plan:</b>\n  • TP1 → close <b>50%</b> → SL to BE\n  • TP2 → close remaining <b>50%</b>"),
            'tp1_hit':False,'tp2_hit':False,'sl_hit':False,'be_active':False,'partial_taken':False,
            'timestamp':datetime.now(),
        }

    def _fmt_signal(self, sig):
        e='🚀' if sig['signal']=='LONG' else '🔻'
        mss_arrow={'bull':'🔼 Bullish','bear':'🔽 Bearish'}.get(sig['mss_dir'],'➡️')
        m  = f"{'─'*38}\n{e} <b>{sig['signal']} — Order Block Signal</b>\n{'─'*38}\n\n"
        m += f"<b>Pair:</b>     #{sig['symbol']}\n"
        m += f"<b>MSS:</b>      {mss_arrow} | 1H: {sig['trend_1h']}\n"
        m += f"<b>OB:</b>       {'FVG ✅' if sig['has_fvg'] else 'No FVG'} | Impulse {sig['impulse_atr']}x ATR\n"
        m += f"<b>Entry:</b>    {' + '.join(sig['tags'])}\n\n"
        m += f"<b>Entry:</b>      <code>${sig['entry']:.6f}</code>\n"
        m += f"<b>TP1 (1R):</b>  <code>${sig['tp1']:.6f}</code>  +{sig['tp1_pct']:.2f}%\n"
        m += f"<b>TP2 (2R):</b>  <code>${sig['tp2']:.6f}</code>  +{sig['tp2_pct']:.2f}%\n"
        m += f"<b>Stop loss:</b> <code>${sig['stop_loss']:.6f}</code>  -{sig['risk_pct']:.2f}%\n"
        m += f"<b>RR (TP1):</b>  {sig['rr']}:1\n\n"
        m += f"<b>OB Zone:</b>   {sig['ob_low']:.6f} — {sig['ob_high']:.6f}\n\n"
        m += f"{sig['close_plan']}\n\n"
        m += f"<i>🆔 {sig['trade_id']}</i>\n"
        m += f"<i>⏰ {sig['timestamp'].strftime('%H:%M UTC')} | OB v2.0</i>"
        return m

    async def _send(self, text):
        try:
            await self.telegram_bot.send_message(chat_id=self.chat_id, text=text, parse_mode=ParseMode.HTML)
        except Exception as e: logger.error(f"Telegram: {e}")

    async def _tp1_alert(self, t, price):
        m  = f"✅ <b>TP1 HIT</b> ✅\n\n<b>{t['symbol']}</b> {t['signal']}\n"
        m += f"Entry: ${t['entry']:.6f}\nTP1:   ${price:.6f}  <b>+{abs((price-t['entry'])/t['entry']*100):.2f}%</b>\n\n"
        m += f"✂️ Close <b>50%</b>\n🔒 SL → BE (${t['entry']:.6f})\n🎯 Runner: ${t['tp2']:.6f}\n\n<i>{t['trade_id']}</i>"
        await self._send(m); t['tp1_hit']=True; t['be_active']=True; t['partial_taken']=True
        self.stats['tp1_hits']+=1

    async def _tp2_alert(self, t, price):
        m  = f"💰 <b>TP2 HIT — FULL TARGET!</b> 💰\n\n<b>{t['symbol']}</b> {t['signal']}\n"
        m += f"Entry: ${t['entry']:.6f}\nTP2:   ${price:.6f}  <b>+{abs((price-t['entry'])/t['entry']*100):.2f}%</b>\n\n"
        m += f"✅ Close remaining 50% — complete\n\n<i>{t['trade_id']}</i>"
        await self._send(m); t['tp2_hit']=True; self.stats['tp2_hits']+=1

    async def _sl_alert(self, t, price, be_save=False):
        if be_save:
            m=f"🔒 <b>BREAKEVEN CLOSE</b>\n\n<b>{t['symbol']}</b> {t['signal']}\nTP1 hit ✅ — runner closed at BE\n\n<i>{t['trade_id']}</i>"
            self.stats['be_saves']+=1
        else:
            m=f"⛔ <b>STOP LOSS</b>\n\n<b>{t['symbol']}</b> {t['signal']}\nEntry: ${t['entry']:.6f}\nSL: ${price:.6f}  <b>-{abs((price-t['entry'])/t['entry']*100):.2f}%</b>\n\n<i>Next OB incoming 🎯</i>"
            self.stats['sl_hits']+=1
        await self._send(m)

    async def _track_trades(self):
        logger.info("📡 Trade tracker started")
        while True:
            try:
                if not self.active_trades: await asyncio.sleep(30); continue
                done=[]
                for tid,t in list(self.active_trades.items()):
                    try:
                        if datetime.now()-t['timestamp']>timedelta(hours=MAX_TRADE_HOURS):
                            self.stats['timeouts']+=1; done.append(tid); continue
                        ticker=await self.exchange.fetch_ticker(t['full_symbol'])
                        price=ticker['last']; act_sl=t['entry'] if t['be_active'] else t['stop_loss']
                        if t['signal']=='LONG':
                            if not t['tp1_hit'] and price>=t['tp1']: await self._tp1_alert(t,price)
                            elif t['tp1_hit'] and not t['tp2_hit'] and price>=t['tp2']:
                                await self._tp2_alert(t,price); done.append(tid)
                            elif price<=act_sl: await self._sl_alert(t,price,t['be_active']); done.append(tid)
                        else:
                            if not t['tp1_hit'] and price<=t['tp1']: await self._tp1_alert(t,price)
                            elif t['tp1_hit'] and not t['tp2_hit'] and price<=t['tp2']:
                                await self._tp2_alert(t,price); done.append(tid)
                            elif price>=act_sl: await self._sl_alert(t,price,t['be_active']); done.append(tid)
                    except Exception as e: logger.error(f"Track {tid}: {e}")
                for tid in done: self.active_trades.pop(tid,None)
                await asyncio.sleep(30)
            except Exception as e: logger.error(f"Tracker: {e}"); await asyncio.sleep(60)

    async def _get_pairs(self):
        try:
            await self.exchange.load_markets()
            tickers=await self.exchange.fetch_tickers()
            pairs=[s for s in self.exchange.symbols
                   if s.endswith('/USDT:USDT') and 'PERP' not in s
                   and tickers.get(s,{}).get('quoteVolume',0)>MIN_VOLUME_USDT]
            pairs.sort(key=lambda x:tickers.get(x,{}).get('quoteVolume',0),reverse=True)
            return pairs[:TOP_PAIRS_LIMIT]
        except Exception as e: logger.error(f"get_pairs: {e}"); return []

    async def run_backtest_all(self, pairs=None, send_telegram=True):
        if self.is_backtesting: return
        self.is_backtesting=True
        if send_telegram:
            await self._send("🔬 <b>Running backtest...</b>\nFetching historical data. This takes ~2-3 minutes...")
        if pairs is None: pairs=await self._get_pairs()
        all_results=[]
        for symbol in pairs[:20]:
            try:
                df4h=await self._fetch_ohlcv(symbol,'4h',limit=300)
                df1h=await self._fetch_ohlcv(symbol,'1h',limit=500)
                df15m=await self._fetch_ohlcv(symbol,'15m',limit=800)
                if df4h is None or df1h is None or df15m is None:
                    await asyncio.sleep(0.5); continue
                pair_name=symbol.replace('/USDT:USDT','')
                trades=run_backtest(pair_name,df4h,df1h,df15m)
                if trades:
                    stats=compute_backtest_stats(trades,pair_name)
                    all_results.append(stats)
                    logger.info(f"BT {pair_name}: {stats['total_trades']}T WR={stats['win_rate']}% PF={stats['profit_factor']}")
                await asyncio.sleep(0.5)
            except Exception as e: logger.error(f"BT {symbol}: {e}")
        if send_telegram:
            await self._send(format_backtest_report(all_results))
        self.is_backtesting=False
        return all_results

    async def scan_all(self):
        if self.is_scanning: return []
        self.is_scanning=True; signals=[]
        pairs=await self._get_pairs()
        logger.info(f"🔍 Scanning {len(pairs)} pairs (4H OB → 1H MSS → 15m)...")
        for symbol in pairs:
            try:
                df4h=await self._fetch_ohlcv(symbol,'4h',limit=150)
                df1h=await self._fetch_ohlcv(symbol,'1h',limit=200)
                df15m=await self._fetch_ohlcv(symbol,'15m',limit=150)
                if df4h is None or df1h is None or df15m is None:
                    await asyncio.sleep(0.3); continue
                sig=self._detect_signal(symbol,df4h,df1h,df15m)
                if sig is None: await asyncio.sleep(0.2); continue
                ck=f"{sig['symbol']}_{sig['signal']}"
                last=self.pair_cooldown.get(ck)
                if last and (datetime.now()-last).total_seconds()<COOLDOWN_HOURS*3600:
                    self.stats['filtered']['cooldown']+=1; await asyncio.sleep(0.2); continue
                self.pair_cooldown[ck]=datetime.now()
                self.active_trades[sig['trade_id']]=sig; self.signal_history.append(sig)
                self.stats['total_signals']+=1
                if sig['signal']=='LONG': self.stats['long_signals']+=1
                else: self.stats['short_signals']+=1
                await self._send(self._fmt_signal(sig)); signals.append(sig)
                logger.info(f"✅ {sig['symbol']} {sig['signal']} {sig['entry_type']} OB={sig['ob_kind']} FVG={sig['has_fvg']}")
                await asyncio.sleep(1)
            except Exception as e: logger.error(f"Scan {symbol}: {e}")
            await asyncio.sleep(0.3)
        self.stats['last_scan']=datetime.now(); self.stats['pairs_scanned']=len(pairs)
        self.is_scanning=False; return signals

    async def _daily_report(self):
        while True:
            await asyncio.sleep(24*3600)
            try:
                s=self.stats; tp1=s['tp1_hits']; tp2=s['tp2_hits']; sl=s['sl_hits']; be=s['be_saves']
                tot=tp1+sl; wr=round(tp1/tot*100,1) if tot>0 else 0
                hrs=round((datetime.now()-s['session_start']).total_seconds()/3600,1)
                cutoff=datetime.now()-timedelta(hours=24)
                day_sigs=[t for t in self.signal_history if t['timestamp']>=cutoff]
                bar='▰'*int(wr/10)+'▱'*(10-int(wr/10))
                m=f"{'─'*36}\n📅 <b>24H REPORT — OB Scanner v2.0</b>\n{'─'*36}\n\n"
                m+=f"Session: {hrs}h | Signals today: <b>{len(day_sigs)}</b>\n\n"
                m+=f"<b>Performance:</b>\n  ✅ TP1: {tp1}  💰 TP2: {tp2}\n  🔒 BE: {be}  ❌ SL: {sl}\n\n"
                m+=f"<b>WR: {wr}%</b>\n{bar}\n\n"
                flt=s['filtered']
                m+=f"<b>Filters:</b> no_ob={flt.get('no_ob',0)} no_mss={flt.get('no_mss',0)} no_entry={flt.get('no_entry',0)}\n"
                m+=f"Tracking: {len(self.active_trades)} trades\n"
                m+=f"<i>⏰ {datetime.now().strftime('%d %b %Y %H:%M UTC')}</i>"
                await self._send(m)
            except Exception as e: logger.error(f"Daily report: {e}")

    async def run(self):
        logger.info("🚀 OB Scanner v2.0 | 4H OB → 1H MSS → 15min entry")
        asyncio.create_task(self.run_backtest_all())
        asyncio.create_task(self._track_trades())
        asyncio.create_task(self._daily_report())
        while True:
            try:
                await self.scan_all()
                await asyncio.sleep(SCAN_INTERVAL_MIN*60)
            except Exception as e:
                logger.error(f"Run loop: {e}"); await asyncio.sleep(60)

    async def close(self):
        await self.exchange.close()


# ═══════════════════════════════════════════════════════
# TELEGRAM COMMANDS
# ═══════════════════════════════════════════════════════

class BotCommands:
    def __init__(self, scanner: OBScanner):
        self.s = scanner

    async def cmd_start(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        await update.message.reply_text(
            "🏦 <b>Order Blocks Scanner v2.0</b>\n"
            "Smart Money · FVG + BOS · 4H→1H→15m\n\n"
            "/scan      — force scan now\n"
            "/backtest  — run backtest & show results\n"
            "/stats     — session performance\n"
            "/trades    — active trades\n"
            "/strategy  — explain the OB strategy\n"
            "/help      — this message",
            parse_mode=ParseMode.HTML)

    async def cmd_scan(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        if self.s.is_scanning: await update.message.reply_text("⚠️ Scan already running!"); return
        await update.message.reply_text("🔍 Scanning now...")
        asyncio.create_task(self.s.scan_all())

    async def cmd_backtest(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        if self.s.is_backtesting: await update.message.reply_text("⚠️ Backtest already running!"); return
        await update.message.reply_text("🔬 Starting backtest... (~2-3 mins)")
        asyncio.create_task(self.s.run_backtest_all())

    async def cmd_stats(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        s=self.s.stats; tp1=s['tp1_hits']; tp2=s['tp2_hits']; sl=s['sl_hits']; be=s['be_saves']
        tot=tp1+sl; wr=round(tp1/tot*100,1) if tot>0 else 0
        hrs=round((datetime.now()-s['session_start']).total_seconds()/3600,1)
        flt=s['filtered']
        m=f"📊 <b>OB SCANNER STATS</b>\n\nSession: {hrs}h\n\n"
        m+=f"<b>Signals:</b> {s['total_signals']}\n  🟢 Long: {s['long_signals']}  🔴 Short: {s['short_signals']}\n\n"
        m+=f"<b>Performance:</b>\n  ✅ TP1: {tp1}  ({wr}% WR)\n  💰 TP2: {tp2}  🔒 BE: {be}  ❌ SL: {sl}\n\n"
        m+=f"<b>Filters fired:</b>\n  No OB: {flt.get('no_ob',0)}  No MSS: {flt.get('no_mss',0)}\n"
        m+=f"  No entry: {flt.get('no_entry',0)}  Cooldown: {flt.get('cooldown',0)}\n\n"
        m+=f"Tracking: {len(self.s.active_trades)} trades"
        if s['last_scan']: m+=f"\nLast scan: {s['last_scan'].strftime('%H:%M')}"
        await update.message.reply_text(m, parse_mode=ParseMode.HTML)

    async def cmd_trades(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        trades=self.s.active_trades
        if not trades: await update.message.reply_text("📭 No active trades."); return
        m=f"📡 <b>ACTIVE TRADES ({len(trades)})</b>\n\n"
        for tid,t in list(trades.items())[:10]:
            age=int((datetime.now()-t['timestamp']).total_seconds()/3600)
            tp1s='✅' if t['tp1_hit'] else '⏳'; tp2s='✅' if t['tp2_hit'] else '⏳'
            be_s=' 🔒BE' if t['be_active'] else ''; fvg=' ⚡FVG' if t['has_fvg'] else ''
            m+=f"<b>{t['symbol']}</b> {t['signal']}{be_s}{fvg}\n"
            m+=f"  Entry: ${t['entry']:.6f} | OB: {t['ob_kind']}\n"
            m+=f"  MSS:{t['mss_dir']} · {t['entry_type']} · {age}h old\n"
            m+=f"  TP1:{tp1s} TP2:{tp2s}\n\n"
        await update.message.reply_text(m, parse_mode=ParseMode.HTML)

    async def cmd_strategy(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        m  = "🏦 <b>ORDER BLOCKS STRATEGY v2.0</b>\n\n"
        m += "<b>What is an Order Block?</b>\n"
        m += "The LAST candle BEFORE a strong impulsive move.\n"
        m += "Smart money placed large orders here. Price returns\n"
        m += "to fill remaining orders → your entry zone.\n\n"
        m += "<b>3 Validity Rules:</b>\n"
        m += "  1️⃣ <b>FVG</b> — imbalance/gap after the move (institutional)\n"
        m += "  2️⃣ <b>BOS</b> — the move broke a prior swing level\n"
        m += "  3️⃣ <b>Unmitigated</b> — price has NOT revisited OB yet\n\n"
        m += "<b>Top-Down Entry Flow:</b>\n"
        m += "  📊 4H → Identify valid Order Block\n"
        m += "  📈 1H → Wait for Market Structure Shift (MSS)\n"
        m += "         Downtrend breaks last LH → Bullish MSS\n"
        m += "         Uptrend breaks last HL  → Bearish MSS\n"
        m += "  🎯 15m → Sniper entry at OB zone\n"
        m += "         Pin bar / Engulf / Strong close\n\n"
        m += "<b>Trade Management:</b>\n"
        m += f"  SL  = OB edge ± {ATR_SL_BUFFER}x ATR\n"
        m += f"  TP1 = 1R → close 50%, SL → BE\n"
        m += f"  TP2 = 2R → close remaining 50%\n"
        m += f"  Timeout: {MAX_TRADE_HOURS}h max\n\n"
        m += "<b>Best setups:</b> FVG + BOS + flip zone"
        await update.message.reply_text(m, parse_mode=ParseMode.HTML)

    async def cmd_help(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        await self.cmd_start(update, ctx)


# ═══════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════

async def main():
    # ╔══════════════════════════════════════╗
    TELEGRAM_TOKEN   = "7957028587:AAE7aSYtE4hCxxTIPkAs_1ULJ9e8alkY6Ic"
    TELEGRAM_CHAT_ID = "-1003659830260"
    BINANCE_API_KEY  = None
    BINANCE_SECRET   = None
    # ╚══════════════════════════════════════╝

    scanner = OBScanner(
        telegram_token   = TELEGRAM_TOKEN,
        telegram_chat_id = TELEGRAM_CHAT_ID,
        binance_api_key  = BINANCE_API_KEY,
        binance_secret   = BINANCE_SECRET,
    )
    app = Application.builder().token(TELEGRAM_TOKEN).build()
    cmds = BotCommands(scanner)
    for cmd, fn in [
        ('start',    cmds.cmd_start),
        ('scan',     cmds.cmd_scan),
        ('backtest', cmds.cmd_backtest),
        ('stats',    cmds.cmd_stats),
        ('trades',   cmds.cmd_trades),
        ('strategy', cmds.cmd_strategy),
        ('help',     cmds.cmd_help),
    ]:
        app.add_handler(CommandHandler(cmd, fn))
    await app.initialize(); await app.start()
    logger.info("🤖 OB Scanner v2.0 online")
    try:
        await scanner.run()
    except KeyboardInterrupt:
        logger.info("Shutting down...")
    finally:
        await scanner.close(); await app.stop(); await app.shutdown()


if __name__ == '__main__':
    asyncio.run(main())
