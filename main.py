"""
SMC LIVE BOT v1.0
==================
Strategy: v5.4 backtest parameters
  - LONG ONLY (ROB + BB)
  - 67.1% WR | 3.36 PF | 7.25R max DD (1000-bar backtest)
  - BB: 82.7% WR — priority signal
  - ROB: 64.4% WR

Filters wired in:
  BASIL >= 3, first-touch only, vol >= 0.6x avg,
  depart >= 1.0x ATR, dir candle, sessions London/NY,
  4H not bear, impulse skip 7 bars, top-30 by volume

Install:
  pip install ccxt pandas numpy python-telegram-bot

Usage:
  Fill TELEGRAM_TOKEN and TELEGRAM_CHAT_ID below, then:
  python smc_live_bot.py
"""

import asyncio
import ccxt.async_support as ccxt
from telegram import Bot, Update
from telegram.ext import Application, CommandHandler, ContextTypes
from telegram.constants import ParseMode
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple
from collections import deque
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s'
)
logger = logging.getLogger(__name__)

import warnings
warnings.filterwarnings('ignore')


# ════════════════════════════════════════════════════════════
#  ★  CREDENTIALS  — fill these in
# ════════════════════════════════════════════════════════════

TELEGRAM_TOKEN   = "7957028587:AAE7aSYtE4hCxxTIPkAs_1ULJ9e8alkY6Ic"
TELEGRAM_CHAT_ID = "-1003659830260"
BINANCE_API_KEY  = None   # optional — only needed for private endpoints
BINANCE_SECRET   = None


# ════════════════════════════════════════════════════════════
#  CONFIG  (v5.4 params — do not change without re-backtesting)
# ════════════════════════════════════════════════════════════

CFG = dict(
    # Pair universe
    TOP_PAIRS         = 600,
    MIN_VOL_USDT      = 500_000,   # $5M 24H volume floor
    PAIR_BLACKLIST    = {'1000PEPE', 'XRP', 'HYPE'},
    MAX_ATR_PRICE_PCT = 0.030,       # skip if ATR/price > 3%

    # OB detection
    OB_BASIL_MIN      = 3,
    OB_MIN_IMBAL_ATR  = 1.0,
    ATR_PERIOD        = 14,

    # Zone quality filters (v5.4)
    OB_MAX_AGE        = 300,         # bars
    BB_MAX_AGE        = 200,
    IMPULSE_SKIP      = 7,           # skip first N bars after OB formation for first-touch check
    FIRST_TOUCH_ONLY  = True,
    MIN_VOL_RATIO     = 0.60,        # entry bar vol >= 0.60 × 20-bar avg
    VOL_LOOKBACK      = 20,
    MIN_DEPART_ATR    = 1.0,         # price must move 1.0xATR above OB before returning
    REQUIRE_DIR_CANDLE = True,       # entry bar must close bullish (LONG only)

    # Trade management
    TP1_R             = 1.5,
    TP2_R             = 3.0,
    TP1_SIZE          = 0.50,        # close 50% at TP1, SL → BE
    SL_ATR_BUFFER     = 0.15,
    MAX_TRADE_HOURS   = 72,

    # Sessions (UTC)
    USE_SESSIONS      = True,
    SESSION_HOURS     = {'london': (7, 10), 'ny': (13, 16)},

    # Scanner
    SCAN_INTERVAL_MIN = 15,
    BARS_FETCH        = 350,         # 1H bars per pair per scan
    SIGNAL_TF         = '1h',
    COOLDOWN_BARS     = 8,

    # Trend
    TREND_EMA_FAST    = 21,
    TREND_EMA_SLOW    = 50,
    SWING_N           = 3,
)


# ════════════════════════════════════════════════════════════
#  MATH HELPERS
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
    return float(np.mean(vals)) if len(vals) > 0 else float(V[max(0,i)])

def resample_4h(df1h: pd.DataFrame) -> np.ndarray:
    n = len(df1h)
    C4 = np.array([df1h['close'].values[min(i+3,n-1)] for i in range(0,n,4)])
    e21 = calc_ema(C4, CFG['TREND_EMA_FAST'])
    e50 = calc_ema(C4, CFG['TREND_EMA_SLOW'])
    T = np.empty(n, dtype=object)
    for b in range(len(C4)):
        s, e = b*4, min(b*4+4, n)
        if C4[b] > e21[b] > e50[b]:   T[s:e] = 'bull'
        elif C4[b] < e21[b] < e50[b]: T[s:e] = 'bear'
        else:                          T[s:e] = 'neutral'
    return T


# ════════════════════════════════════════════════════════════
#  SWING POINTS + TREND
# ════════════════════════════════════════════════════════════

@dataclass
class SwingPoint:
    idx: int; price: float; kind: str

def find_swings(H, L, sw=3) -> List[SwingPoint]:
    n = len(H); pts = []
    for i in range(sw, n-sw):
        if H[i] == max(H[max(0,i-sw):i+sw+1]): pts.append(SwingPoint(i, H[i], 'high'))
        if L[i] == min(L[max(0,i-sw):i+sw+1]): pts.append(SwingPoint(i, L[i], 'low'))
    return sorted(pts, key=lambda p: p.idx)

def trend_1h(swings: List[SwingPoint], up_to: int) -> str:
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
#  ORDER BLOCK
# ════════════════════════════════════════════════════════════

@dataclass
class OrderBlock:
    top: float; bottom: float; formed: int; kind: str
    basil: int = 0; failed: bool = False; fail_at: int = -1

def _avg_vol(V, end):
    return float(np.mean(V[max(0,end-50):end])) + 1e-9

def find_bull_obs(O, H, L, C, V, ATR, swings, T4H, end) -> List[OrderBlock]:
    obs = []; av = _avg_vol(V, end)
    for i in range(5, end-3):
        if not (C[i] < O[i]): continue
        atr = max(ATR[i], 1e-9)
        imp_end = min(i+4, end-1)
        rally = max(C[i+1:imp_end+1]) - O[i+1]
        if rally < CFG['OB_MIN_IMBAL_ATR'] * atr: continue
        fvg   = any(L[j] > H[j-2] for j in range(i+1, imp_end+1) if j >= 2)
        pr_lo = [p for p in swings if p.idx < i and p.kind == 'low']
        swept = (len(pr_lo) > 0 and min(L[i:imp_end+1]) < pr_lo[-1].price) if pr_lo else False
        t4    = T4H[i] if i < len(T4H) else 'neutral'
        pr_hi = [p for p in swings if p.idx < i and p.kind == 'high']
        bos   = (len(pr_hi) > 0 and max(C[i+1:imp_end+1]) > pr_hi[-1].price) if pr_hi else False
        b     = sum([bos, t4 in ('bull','neutral'), swept, fvg, V[i] > av*1.2])
        if b >= CFG['OB_BASIL_MIN']:
            obs.append(OrderBlock(top=max(O[i],C[i]), bottom=L[i], formed=i, kind='bullish', basil=b))
    return obs

def find_bear_obs(O, H, L, C, V, ATR, swings, T4H, end) -> List[OrderBlock]:
    obs = []; av = _avg_vol(V, end)
    for i in range(5, end-3):
        if not (C[i] > O[i]): continue
        atr = max(ATR[i], 1e-9)
        imp_end = min(i+4, end-1)
        drop = O[i+1] - min(C[i+1:imp_end+1])
        if drop < CFG['OB_MIN_IMBAL_ATR'] * atr: continue
        fvg   = any(H[j] < L[j-2] for j in range(i+1, imp_end+1) if j >= 2)
        pr_hi = [p for p in swings if p.idx < i and p.kind == 'high']
        swept = (len(pr_hi) > 0 and max(H[i:imp_end+1]) > pr_hi[-1].price) if pr_hi else False
        t4    = T4H[i] if i < len(T4H) else 'neutral'
        pr_lo = [p for p in swings if p.idx < i and p.kind == 'low']
        bos   = (len(pr_lo) > 0 and min(C[i+1:imp_end+1]) < pr_lo[-1].price) if pr_lo else False
        b     = sum([bos, t4 in ('bear','neutral'), swept, fvg, V[i] > av*1.2])
        if b >= CFG['OB_BASIL_MIN']:
            obs.append(OrderBlock(top=H[i], bottom=min(O[i],C[i]), formed=i, kind='bearish', basil=b))
    return obs


# ════════════════════════════════════════════════════════════
#  QUALITY CHECKS  (v5.4)
# ════════════════════════════════════════════════════════════

def zone_previously_touched_rob(ob: OrderBlock, H, L, check_end: int) -> bool:
    """Skip IMPULSE_SKIP bars after OB formation before checking prior touches."""
    scan_from = ob.formed + CFG['IMPULSE_SKIP']
    for k in range(scan_from, check_end):
        if L[k] <= ob.top and H[k] >= ob.bottom:
            return True
    return False

def zone_previously_touched_bb(bb: OrderBlock, H, L, check_end: int) -> bool:
    for k in range(bb.fail_at + 1, check_end):
        if L[k] <= bb.top and H[k] >= bb.bottom:
            return True
    return False

def price_departed_zone(ob: OrderBlock, H, L, ATR, check_from: int, check_end: int) -> bool:
    for k in range(check_from, check_end):
        atr_k = max(ATR[k], 1e-9)
        if H[k] >= ob.top + CFG['MIN_DEPART_ATR'] * atr_k:
            return True
    return False

def vol_ok(V, i: int) -> bool:
    return V[i] >= rolling_vol_avg(V, i, CFG['VOL_LOOKBACK']) * CFG['MIN_VOL_RATIO']

def in_session(ts: pd.Timestamp) -> bool:
    if not CFG['USE_SESSIONS']: return True
    h = ts.hour
    return (CFG['SESSION_HOURS']['london'][0] <= h < CFG['SESSION_HOURS']['london'][1] or
            CFG['SESSION_HOURS']['ny'][0]     <= h < CFG['SESSION_HOURS']['ny'][1])


# ════════════════════════════════════════════════════════════
#  SIGNAL DETECTOR  (v5.4 logic — bar i = latest closed bar)
# ════════════════════════════════════════════════════════════

def detect_signal(df: pd.DataFrame, symbol: str,
                  last_sig: Dict[str, int]) -> Optional[dict]:
    """
    Returns a signal dict if ROB or BB fires on the latest bar,
    else None. last_sig tracks cooldown per technique key.
    """
    if len(df) < 80: return None

    O  = df['open'].values.astype(float)
    H  = df['high'].values.astype(float)
    L  = df['low'].values.astype(float)
    C  = df['close'].values.astype(float)
    V  = df['volume'].values.astype(float)
    TS = df['timestamp'].values

    ATR    = calc_atr(H, L, C, CFG['ATR_PERIOD'])
    T4H    = resample_4h(df)
    swings = find_swings(H, L, CFG['SWING_N'])
    n      = len(C)
    i      = n - 1   # latest completed bar

    atr  = max(ATR[i], 1e-9)
    t4h  = T4H[i]
    ts   = pd.Timestamp(TS[i])
    buf  = atr * 0.2

    # Global filters
    if C[i] > 0 and atr / C[i] > CFG['MAX_ATR_PRICE_PCT']:
        return None
    if t4h == 'bear':
        return None
    if not in_session(ts):
        return None
    if not (C[i] > O[i]):   # must be bullish bar (direction filter)
        return None

    bull_obs = find_bull_obs(O, H, L, C, V, ATR, swings, T4H, n-1)
    bear_obs = find_bear_obs(O, H, L, C, V, ATR, swings, T4H, n-1)

    # Find breakers (bearish OBs that failed upward before bar i)
    bear_breakers: List[OrderBlock] = []
    for ob in bear_obs:
        if ob.formed >= i: continue
        # Check if price ever closed above the OB top before bar i
        for k in range(ob.formed+1, i):
            if C[k] > ob.top + buf:
                ob.failed = True; ob.fail_at = k
                bear_breakers.append(ob)
                break

    # ── ROB ─────────────────────────────────────────────────
    for ob in bull_obs:
        if ob.formed >= i: continue
        if not (L[i] <= ob.top + buf and H[i] >= ob.bottom - buf): continue
        if i - ob.formed > CFG['OB_MAX_AGE']: continue
        if not vol_ok(V, i): continue
        if CFG['FIRST_TOUCH_ONLY'] and zone_previously_touched_rob(ob, H, L, i): continue
        if not price_departed_zone(ob, H, L, ATR, ob.formed+1, i): continue
        if i - last_sig.get('ROB', 0) < CFG['COOLDOWN_BARS']: continue

        entry = ob.top
        sl    = ob.bottom - atr * CFG['SL_ATR_BUFFER']
        risk  = abs(entry - sl)
        if risk < 1e-9: continue
        tp1 = entry + risk * CFG['TP1_R']
        tp2 = entry + risk * CFG['TP2_R']

        last_sig['ROB'] = i
        return _build_signal(symbol, 'LONG', entry, sl, tp1, tp2,
                             atr, ob, 'ROB', T4H[i], trend_1h(swings, i), risk, ts)

    # ── BB ──────────────────────────────────────────────────
    for bb in bear_breakers:
        if bb.fail_at < 0 or i <= bb.fail_at: continue
        if not (L[i] <= bb.top + buf and H[i] >= bb.bottom - buf): continue
        if i - bb.fail_at > CFG['BB_MAX_AGE']: continue
        if not vol_ok(V, i): continue
        if CFG['FIRST_TOUCH_ONLY'] and zone_previously_touched_bb(bb, H, L, i): continue
        if not price_departed_zone(bb, H, L, ATR, bb.fail_at+1, i): continue
        if i - last_sig.get('BB', 0) < CFG['COOLDOWN_BARS']: continue

        entry = bb.top
        sl    = bb.bottom - atr * CFG['SL_ATR_BUFFER']
        risk  = abs(entry - sl)
        if risk < 1e-9: continue
        tp1 = entry + risk * CFG['TP1_R']
        tp2 = entry + risk * CFG['TP2_R']

        last_sig['BB'] = i
        return _build_signal(symbol, 'LONG', entry, sl, tp1, tp2,
                             atr, bb, 'BB', T4H[i], trend_1h(swings, i), risk, ts)

    return None


def _build_signal(symbol, direction, entry, sl, tp1, tp2,
                  atr, zone, technique, t4h, t1h, risk, ts) -> dict:
    pair = symbol.replace('/USDT:USDT', '').replace('/USDT', '')
    tid  = f"{pair}_{technique}_{ts.strftime('%Y%m%d%H%M')}"

    def pct(a, b): return round(abs(a-b)/max(abs(b),1e-9)*100, 2)

    basil_bar = '▰' * min(zone.basil, 5) + '▱' * max(0, 5-zone.basil)
    t4h_emoji = {'bull': '🐂', 'bear': '🐻', 'neutral': '➡️'}.get(t4h, '➡️')
    tech_emoji = {'ROB': '📦', 'BB': '🔄'}.get(technique, '●')
    tech_label = {'ROB': 'Order Block', 'BB': 'Breaker Block'}.get(technique, technique)

    return {
        'trade_id':    tid,
        'symbol':      pair,
        'full_symbol': symbol,
        'signal':      direction,
        'entry':       entry,
        'stop_loss':   sl,
        'tp1':         tp1,  'tp1_pct': pct(tp1, entry),
        'tp2':         tp2,  'tp2_pct': pct(tp2, entry),
        'risk_pct':    pct(sl, entry),
        'atr':         round(atr, 6),
        'zone_top':    round(zone.top, 6),
        'zone_bottom': round(zone.bottom, 6),
        'basil':       zone.basil,
        'basil_bar':   basil_bar,
        'technique':   technique,
        'tech_label':  tech_label,
        'tech_emoji':  tech_emoji,
        't4h':         t4h, 't4h_emoji': t4h_emoji,
        't1h':         t1h,
        'tp1_hit':     False, 'tp2_hit': False,
        'sl_hit':      False, 'be_active': False,
        'timestamp':   datetime.now(timezone.utc),
    }


# ════════════════════════════════════════════════════════════
#  TELEGRAM MESSAGE FORMATTER
# ════════════════════════════════════════════════════════════

def fmt_signal(sig: dict) -> str:
    tech_color = '🔵' if sig['technique'] == 'BB' else '🟢'
    wr_note    = '82.7% WR' if sig['technique'] == 'BB' else '64.4% WR'

    m  = f"{'─'*36}\n"
    m += f"🚀 <b>LONG — {sig['tech_label']}</b>  {tech_color}\n"
    m += f"{'─'*36}\n\n"
    m += f"<b>#{sig['symbol']}</b>  {sig['t4h_emoji']} 4H {sig['t4h']}  ·  1H {sig['t1h']}\n"
    m += f"Zone quality: {sig['basil_bar']}  (BASIL {sig['basil']}/5)\n"
    m += f"<i>{wr_note} backtest (v5.4)</i>\n\n"
    m += f"<b>Entry:    </b> <code>${sig['entry']:.6f}</code>\n"
    m += f"<b>TP1 (+{sig['tp1_pct']:.2f}%):</b> <code>${sig['tp1']:.6f}</code>\n"
    m += f"<b>TP2 (+{sig['tp2_pct']:.2f}%):</b> <code>${sig['tp2']:.6f}</code>\n"
    m += f"<b>Stop loss:</b> <code>${sig['stop_loss']:.6f}</code>  (-{sig['risk_pct']:.2f}%)\n\n"
    m += f"📋 <b>Plan:</b> Close <b>50%</b> at TP1 → SL to BE → Runner to TP2\n\n"
    m += f"<i>🆔 {sig['trade_id']}</i>\n"
    m += f"<i>⏰ {sig['timestamp'].strftime('%H:%M UTC')}</i>"
    return m


def fmt_tp1(t: dict, price: float) -> str:
    gain = abs((price - t['entry']) / t['entry'] * 100)
    m  = f"✅ <b>TP1 HIT</b>\n\n"
    m += f"<b>#{t['symbol']}</b>  {t['technique']}\n"
    m += f"Entry: ${t['entry']:.6f}\n"
    m += f"TP1:   ${price:.6f}  <b>+{gain:.2f}%</b>\n\n"
    m += f"✂️ Close <b>50%</b> of position\n"
    m += f"🔒 Move SL → breakeven (${t['entry']:.6f})\n"
    m += f"🎯 Runner to TP2: ${t['tp2']:.6f} (+{t['tp2_pct']:.2f}%)\n"
    m += f"\n<i>{t['trade_id']}</i>"
    return m

def fmt_tp2(t: dict, price: float) -> str:
    gain = abs((price - t['entry']) / t['entry'] * 100)
    m  = f"💰 <b>TP2 — FULL TARGET HIT!</b>\n\n"
    m += f"<b>#{t['symbol']}</b>  {t['technique']}\n"
    m += f"Entry: ${t['entry']:.6f}\n"
    m += f"TP2:   ${price:.6f}  <b>+{gain:.2f}%</b>\n"
    m += f"✅ Close remaining 50% — trade complete\n"
    m += f"\n<i>{t['trade_id']}</i>"
    return m

def fmt_sl(t: dict, price: float, be_save: bool) -> str:
    if be_save:
        m  = f"🔒 <b>BREAKEVEN CLOSE</b>\n\n"
        m += f"<b>#{t['symbol']}</b>  TP1 was hit ✅\n"
        m += f"Remainder closed at entry — zero loss\n"
    else:
        loss = abs((price - t['entry']) / t['entry'] * 100)
        m  = f"⛔ <b>STOP LOSS</b>\n\n"
        m += f"<b>#{t['symbol']}</b>  {t['technique']}\n"
        m += f"Entry: ${t['entry']:.6f}\n"
        m += f"SL:    ${price:.6f}  <b>-{loss:.2f}%</b>\n"
    m += f"\n<i>{t['trade_id']}</i>"
    return m


# ════════════════════════════════════════════════════════════
#  MAIN SCANNER CLASS
# ════════════════════════════════════════════════════════════

class SMCBot:
    def __init__(self):
        self.bot     = Bot(token=TELEGRAM_TOKEN)
        self.chat_id = TELEGRAM_CHAT_ID
        self.exchange = ccxt.binance({
            'apiKey': BINANCE_API_KEY,
            'secret': BINANCE_SECRET,
            'enableRateLimit': True,
            'options': {'defaultType': 'future'},
        })
        self.active_trades: Dict[str, dict] = {}
        self.signal_history: deque          = deque(maxlen=500)
        self.pair_last_sig: Dict[str, Dict[str, int]] = {}  # symbol → {ROB: bar_i, BB: bar_i}
        self.is_scanning   = False
        self.stats = {
            'total': 0, 'rob': 0, 'bb': 0,
            'tp1': 0, 'tp2': 0, 'sl': 0, 'be': 0, 'timeout': 0,
            'start': datetime.now(timezone.utc),
            'last_scan': None, 'pairs_scanned': 0,
        }

    # ── Telegram ──────────────────────────────────────────

    async def send(self, text: str):
        try:
            await self.bot.send_message(
                chat_id=self.chat_id, text=text, parse_mode=ParseMode.HTML)
        except Exception as e:
            logger.error(f"Telegram send: {e}")

    # ── Data ──────────────────────────────────────────────

    async def _fetch_ohlcv(self, symbol: str) -> Optional[pd.DataFrame]:
        try:
            ohlcv = await self.exchange.fetch_ohlcv(
                symbol, CFG['SIGNAL_TF'], limit=CFG['BARS_FETCH'])
            df = pd.DataFrame(ohlcv,
                              columns=['timestamp','open','high','low','close','volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
            return df
        except Exception as e:
            logger.error(f"fetch_ohlcv {symbol}: {e}")
            return None

    async def _get_pairs(self) -> List[str]:
        try:
            await self.exchange.load_markets()
            tickers = await self.exchange.fetch_tickers()
            pairs = [
                s for s in self.exchange.symbols
                if s.endswith('/USDT:USDT') and 'PERP' not in s
                and tickers.get(s, {}).get('quoteVolume', 0) > CFG['MIN_VOL_USDT']
                and s.replace('/USDT:USDT','') not in CFG['PAIR_BLACKLIST']
            ]
            pairs.sort(
                key=lambda x: tickers.get(x, {}).get('quoteVolume', 0), reverse=True)
            return pairs[:CFG['TOP_PAIRS']]
        except Exception as e:
            logger.error(f"get_pairs: {e}")
            return []

    # ── Trade tracker ─────────────────────────────────────

    async def _track_trades(self):
        logger.info("📡 Trade tracker started")
        while True:
            try:
                if not self.active_trades:
                    await asyncio.sleep(30); continue

                done = []
                for tid, t in list(self.active_trades.items()):
                    try:
                        age = datetime.now(timezone.utc) - t['timestamp']
                        if age > timedelta(hours=CFG['MAX_TRADE_HOURS']):
                            logger.info(f"⏰ Timeout: {t['symbol']}")
                            self.stats['timeout'] += 1
                            done.append(tid); continue

                        ticker = await self.exchange.fetch_ticker(t['full_symbol'])
                        price  = ticker['last']
                        act_sl = t['entry'] if t['be_active'] else t['stop_loss']

                        # LONG only
                        if not t['tp1_hit'] and price >= t['tp1']:
                            await self.send(fmt_tp1(t, price))
                            t['tp1_hit'] = True; t['be_active'] = True
                            self.stats['tp1'] += 1

                        if t['tp1_hit'] and not t['tp2_hit'] and price >= t['tp2']:
                            await self.send(fmt_tp2(t, price))
                            t['tp2_hit'] = True; self.stats['tp2'] += 1
                            done.append(tid); continue

                        if price <= act_sl:
                            be = t['be_active']
                            await self.send(fmt_sl(t, price, be))
                            if be: self.stats['be'] += 1
                            else:  self.stats['sl'] += 1
                            done.append(tid); continue

                    except Exception as e:
                        logger.error(f"track {tid}: {e}")

                for tid in done:
                    self.active_trades.pop(tid, None)

                await asyncio.sleep(30)

            except Exception as e:
                logger.error(f"Tracker loop: {e}")
                await asyncio.sleep(60)

    # ── Main scan ─────────────────────────────────────────

    async def scan_all(self) -> List[dict]:
        if self.is_scanning: return []
        self.is_scanning = True
        signals = []

        pairs = await self._get_pairs()
        logger.info(f"🔍 Scanning {len(pairs)} pairs...")

        for symbol in pairs:
            try:
                df = await self._fetch_ohlcv(symbol)
                if df is None or len(df) < 80:
                    await asyncio.sleep(0.2); continue

                if symbol not in self.pair_last_sig:
                    self.pair_last_sig[symbol] = {}

                sig = detect_signal(df, symbol, self.pair_last_sig[symbol])
                if sig is None:
                    await asyncio.sleep(0.2); continue

                tid = sig['trade_id']
                self.active_trades[tid] = sig
                self.signal_history.append(sig)
                self.stats['total'] += 1
                if sig['technique'] == 'BB': self.stats['bb'] += 1
                else:                        self.stats['rob'] += 1

                await self.send(fmt_signal(sig))
                logger.info(
                    f"✅ {sig['symbol']} LONG {sig['technique']}  "
                    f"BASIL={sig['basil']}  4H={sig['t4h']}  1H={sig['t1h']}"
                )
                await asyncio.sleep(1)

            except Exception as e:
                logger.error(f"Scan {symbol}: {e}")
            await asyncio.sleep(0.3)

        self.stats['last_scan']     = datetime.now(timezone.utc)
        self.stats['pairs_scanned'] = len(pairs)
        self.is_scanning = False
        return signals

    # ── Daily report ──────────────────────────────────────

    async def _daily_report(self):
        while True:
            await asyncio.sleep(24 * 3600)
            try:
                s   = self.stats
                tot = s['tp1'] + s['sl']
                wr  = round(s['tp1']/tot*100, 1) if tot > 0 else 0
                hrs = round((datetime.now(timezone.utc)-s['start']).total_seconds()/3600, 1)
                bar = '▰'*int(wr/10) + '▱'*(10-int(wr/10))

                cutoff   = datetime.now(timezone.utc) - timedelta(hours=24)
                day_sigs = [t for t in self.signal_history if t['timestamp'] >= cutoff]

                m  = f"{'─'*34}\n📅 <b>24H REPORT — SMC Bot v1.0</b>\n{'─'*34}\n\n"
                m += f"Session: {hrs}h\n\n"
                m += f"<b>Today:</b> {len(day_sigs)} signals  "
                m += f"(ROB: {sum(1 for t in day_sigs if t['technique']=='ROB')}  "
                m += f"BB: {sum(1 for t in day_sigs if t['technique']=='BB')})\n\n"
                m += f"<b>Performance:</b>\n"
                m += f"  ✅ TP1: {s['tp1']}  💰 TP2: {s['tp2']}\n"
                m += f"  🔒 BE:  {s['be']}   ❌ SL:  {s['sl']}\n\n"
                m += f"<b>TP1 Win Rate: {wr}%</b>\n{bar}\n\n"

                if   wr >= 65: status = "🔥 Strong — strategy working"
                elif wr >= 55: status = "✅ Good — within backtest range"
                elif wr >= 45: status = "⚠️  Watch closely"
                else:          status = "🚨 Below target"
                m += f"{status}\n\n"
                m += f"Tracking: {len(self.active_trades)} open trades\n"
                m += f"<i>⏰ {datetime.now(timezone.utc).strftime('%d %b %Y %H:%M UTC')}</i>"
                await self.send(m)
            except Exception as e:
                logger.error(f"Daily report: {e}")

    # ── Run ───────────────────────────────────────────────

    async def run(self):
        logger.info(
            "🚀 SMC Bot v1.0 | LONG ONLY | ROB+BB | "
            f"TP1={CFG['TP1_R']}R TP2={CFG['TP2_R']}R | "
            f"Sessions={'ON' if CFG['USE_SESSIONS'] else 'OFF'}"
        )
        asyncio.create_task(self._track_trades())
        asyncio.create_task(self._daily_report())
        while True:
            try:
                await self.scan_all()
                await asyncio.sleep(CFG['SCAN_INTERVAL_MIN'] * 60)
            except Exception as e:
                logger.error(f"Run loop: {e}")
                await asyncio.sleep(60)

    async def close(self):
        await self.exchange.close()


# ════════════════════════════════════════════════════════════
#  TELEGRAM COMMANDS
# ════════════════════════════════════════════════════════════

class Commands:
    def __init__(self, bot: SMCBot):
        self.b = bot

    async def cmd_start(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        await update.message.reply_text(
            "📈 <b>SMC Bot v1.0</b>  —  LONG ONLY\n"
            "Strategy: Order Block + Breaker Block\n"
            "Backtest: 67.1% WR · 3.36 PF · 7.25R max DD\n\n"
            "/scan   — force scan now\n"
            "/stats  — session statistics\n"
            "/trades — active open trades\n"
            "/params — strategy parameters\n"
            "/help   — this message",
            parse_mode=ParseMode.HTML
        )

    async def cmd_scan(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        if self.b.is_scanning:
            await update.message.reply_text("⚠️ Scan already running...")
            return
        await update.message.reply_text("🔍 Scanning top 30 pairs...")
        asyncio.create_task(self.b.scan_all())

    async def cmd_stats(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        s   = self.b.stats
        tot = s['tp1'] + s['sl']
        wr  = round(s['tp1']/tot*100, 1) if tot > 0 else 0
        hrs = round((datetime.now(timezone.utc)-s['start']).total_seconds()/3600, 1)
        spd = round(s['total']/max(hrs, 0.1), 2)

        m  = f"📊 <b>SMC BOT STATS</b>\n\nSession: {hrs}h\n\n"
        m += f"<b>Signals:</b> {s['total']} ({spd}/h)\n"
        m += f"  📦 ROB: {s['rob']}  🔄 BB: {s['bb']}\n\n"
        m += f"<b>Performance:</b>\n"
        m += f"  ✅ TP1: {s['tp1']}  ({wr}% WR)\n"
        m += f"  💰 TP2: {s['tp2']}  ({round(s['tp2']/max(s['tp1'],1)*100)}% of TP1s extended)\n"
        m += f"  🔒 BE:  {s['be']}\n"
        m += f"  ❌ SL:  {s['sl']}\n\n"
        m += f"<b>Backtest targets:</b> WR 67% · PF 3.36 · DD 7.25R\n"
        m += f"Tracking: {len(self.b.active_trades)} open trades"
        if s['last_scan']:
            m += f"\nLast scan: {s['last_scan'].strftime('%H:%M UTC')}"
        await update.message.reply_text(m, parse_mode=ParseMode.HTML)

    async def cmd_trades(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        trades = self.b.active_trades
        if not trades:
            await update.message.reply_text("📭 No active trades.")
            return
        m = f"📡 <b>ACTIVE TRADES ({len(trades)})</b>\n\n"
        for tid, t in list(trades.items())[:10]:
            age   = round((datetime.now(timezone.utc)-t['timestamp']).total_seconds()/3600, 1)
            tp1_s = '✅' if t['tp1_hit'] else '⏳'
            be_s  = ' 🔒BE' if t['be_active'] else ''
            m += f"<b>#{t['symbol']}</b>  {t['technique']}{be_s}\n"
            m += f"  Entry: ${t['entry']:.6f}  ·  {age}h ago\n"
            m += f"  TP1:{tp1_s}  TP2:{'✅' if t['tp2_hit'] else '⏳'}\n"
            m += f"  Zone: {t['zone_bottom']:.6f}–{t['zone_top']:.6f}\n\n"
        await update.message.reply_text(m, parse_mode=ParseMode.HTML)

    async def cmd_params(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        m  = "⚙️ <b>STRATEGY PARAMS (v5.4)</b>\n\n"
        m += f"<b>Direction:</b> LONG only\n"
        m += f"<b>Techniques:</b> ROB (64.4% WR) · BB (82.7% WR)\n\n"
        m += f"<b>Entry filters:</b>\n"
        m += f"  BASIL ≥ {CFG['OB_BASIL_MIN']}  (OB quality score 0–5)\n"
        m += f"  First touch only  ·  Impulse skip: {CFG['IMPULSE_SKIP']} bars\n"
        m += f"  Price departed ≥ {CFG['MIN_DEPART_ATR']}× ATR before retest\n"
        m += f"  Vol ≥ {CFG['MIN_VOL_RATIO']}× rolling avg  ·  Bullish close bar\n"
        m += f"  4H trend: not bear  ·  Sessions: London + NY\n\n"
        m += f"<b>Trade management:</b>\n"
        m += f"  TP1 = {CFG['TP1_R']}R → close 50%, SL → BE\n"
        m += f"  TP2 = {CFG['TP2_R']}R → close remaining 50%\n"
        m += f"  SL  = zone bottom − {CFG['SL_ATR_BUFFER']}× ATR\n"
        m += f"  Timeout: {CFG['MAX_TRADE_HOURS']}h\n\n"
        m += f"<b>Universe:</b> Top {CFG['TOP_PAIRS']} pairs · $5M+ vol\n"
        m += f"<b>Scan:</b> every {CFG['SCAN_INTERVAL_MIN']} min\n\n"
        m += f"<b>Backtest results (1000 bars, 27 pairs):</b>\n"
        m += f"  WR: 67.1%  ·  PF: 3.36  ·  Max DD: 7.25R"
        await update.message.reply_text(m, parse_mode=ParseMode.HTML)

    async def cmd_help(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        await self.cmd_start(update, ctx)


# ════════════════════════════════════════════════════════════
#  ENTRY POINT
# ════════════════════════════════════════════════════════════

async def main():
    smc = SMCBot()

    app = Application.builder().token(TELEGRAM_TOKEN).build()
    cmds = Commands(smc)

    for name, fn in [
        ('start',  cmds.cmd_start),
        ('scan',   cmds.cmd_scan),
        ('stats',  cmds.cmd_stats),
        ('trades', cmds.cmd_trades),
        ('params', cmds.cmd_params),
        ('help',   cmds.cmd_help),
    ]:
        app.add_handler(CommandHandler(name, fn))

    await app.initialize()
    await app.start()

    await smc.send(
        "🟢 <b>SMC Bot v1.0 ONLINE</b>\n\n"
        "Strategy: Order Block + Breaker Block\n"
        "Direction: LONG only\n"
        "Backtest: 67.1% WR · 3.36 PF · 7.25R DD\n\n"
        f"Scanning top {CFG['TOP_PAIRS']} pairs every {CFG['SCAN_INTERVAL_MIN']} min\n"
        "Sessions: London (07–10 UTC) · NY (13–16 UTC)\n\n"
        "/stats /trades /params for details"
    )

    logger.info("🤖 SMC Bot v1.0 online")
    try:
        await smc.run()
    except KeyboardInterrupt:
        logger.info("Shutting down...")
    finally:
        await smc.close()
        await app.stop()
        await app.shutdown()


if __name__ == '__main__':
    asyncio.run(main())
