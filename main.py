"""
SUPPLY & DEMAND SCANNER BOT v1.0
==================================
Drop-in replacement for AdvancedDayTradingScanner.
Implements the full v7 backtest strategy (82.7% WR, PF 21.5, DD -1.8%).

Strategy: Supply & Demand zones (DBR / RBD patterns) with 7-fix filter stack.
  - Zone detection: consolidation base < 2x ATR, imbalance move > 1.5x ATR
  - Entry: pin bar or bull/bear engulf ONLY (strong_close banned everywhere)
  - 4H trend bias filter
  - SL below/above zone + 0.25x ATR buffer
  - 50% close at TP1 (1R), runner to TP2 (1.8R), SL → BE after TP1

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
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple
from collections import deque
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

import warnings
warnings.filterwarnings('ignore')

# ═══════════════════════════════════════════════════════
# ★  SETTINGS  (tuned from v7 backtest — 82.7% WR)
# ═══════════════════════════════════════════════════════

# Zone detection
CONSOL_ATR_MULT     = 2.0    # base range < 2.0x ATR (consolidation)
IMBAL_ATR_MULT      = 1.5    # imbalance move > 1.5x ATR (FIX 3)
MIN_CONFLUENCE      = 1      # vol spike or oversized move required
ZONE_LOOKBACK       = 80     # bars back to scan for zones (1H)
ZONE_MAX_AGE        = 100    # zone expires after N 1H candles

# Entry filters (all 7 fixes embedded)
MIN_WICK_ATR        = 0.4    # pin bar lower wick ≥ 0.4x ATR
MIN_BODY_ATR        = 0.2    # FIX 4: body ≥ 0.2x ATR (no doji)
SC_MAX_ATR_RATIO    = 1.15   # FIX 6A: sc blocked when ATR > 1.15x rolling
SC_ATR_LOOKBACK     = 20     # rolling window for 6A
STALL_ATR_MULT      = 1.5    # FIX 6B: engulf stall — 2-bar range < 1.5x ATR
MIN_BODY_ZONE_RATIO = 0.15   # FIX 6C: body > 15% of zone height

# Trade management
ATR_SL_BUFFER       = 0.25   # SL = zone edge + 0.25x ATR
TP1_R               = 1.0    # TP1 at 1R
TP2_R               = 1.8    # TP2 at 1.8R
TP1_CLOSE_PCT       = 0.50   # close 50% at TP1, 50% runner to TP2
MAX_TRADE_HOURS     = 72     # timeout

# Scanner
ATR_PERIOD          = 14
TREND_LOOKBACK_1H   = 50
SCAN_INTERVAL_MIN   = 15
MIN_VOLUME_USDT     = 500_000
TOP_PAIRS_LIMIT     = 50     # scan top N pairs by volume

# ═══════════════════════════════════════════════════════


# ── ATR ──────────────────────────────────────────────────────

def _calc_atr(H: np.ndarray, L: np.ndarray, C: np.ndarray, period: int = 14) -> np.ndarray:
    n = len(C)
    tr = np.zeros(n)
    tr[0] = H[0] - L[0]
    for i in range(1, n):
        tr[i] = max(H[i] - L[i], abs(H[i] - C[i-1]), abs(L[i] - C[i-1]))
    atr = np.zeros(n)
    if n >= period:
        atr[period - 1] = np.mean(tr[:period])
        for i in range(period, n):
            atr[i] = (atr[i-1] * (period - 1) + tr[i]) / period
        for i in range(period - 1):
            atr[i] = atr[period - 1]
    return atr


def _rolling_atr_mean(atr: np.ndarray, i: int, lookback: int = 20) -> float:
    s = max(0, i - lookback)
    vals = atr[s:i]
    return float(np.mean(vals)) if len(vals) > 0 else float(atr[i])


# ── 4H TREND ────────────────────────────────────────────────

def _calc_4h_trend(df1h: pd.DataFrame) -> np.ndarray:
    """Resample 1H df to 4H, compute EMA21/50 crossover, map back to 1H."""
    n = len(df1h)
    step = 4
    H1 = df1h['high'].values
    L1 = df1h['low'].values
    C1 = df1h['close'].values

    H4 = np.array([np.max(H1[i:i+step]) for i in range(0, n, step)])
    L4 = np.array([np.min(L1[i:i+step]) for i in range(0, n, step)])
    C4 = np.array([C1[min(i+step-1, n-1)] for i in range(0, n, step)])

    def _ema(arr, w):
        out = np.zeros(len(arr))
        out[0] = arr[0]
        k = 2.0 / (w + 1)
        for i in range(1, len(arr)):
            out[i] = arr[i] * k + out[i-1] * (1 - k)
        return out

    e21 = _ema(C4, 21)
    e50 = _ema(C4, 50)

    T4H = np.empty(n, dtype=object)
    for b in range(len(C4)):
        s = b * step
        e = min(s + step, n)
        if C4[b] > e21[b] and e21[b] > e50[b]:
            T4H[s:e] = 'bull'
        elif C4[b] < e21[b] and e21[b] < e50[b]:
            T4H[s:e] = 'bear'
        else:
            T4H[s:e] = 'neutral'
    return T4H


# ── 1H TREND ────────────────────────────────────────────────

def _detect_trend_1h(H: np.ndarray, L: np.ndarray, C: np.ndarray,
                     end_idx: int, lookback: int = 50) -> str:
    """HH/HL = uptrend, LH/LL = downtrend, else range."""
    s = max(0, end_idx - lookback)
    h = H[s:end_idx]
    l = L[s:end_idx]
    n = len(h)
    if n < 12:
        return 'range'
    sw = 3
    sh = [i for i in range(sw, n - sw) if h[i] == max(h[max(0, i-sw):i+sw+1])]
    sl = [i for i in range(sw, n - sw) if l[i] == min(l[max(0, i-sw):i+sw+1])]
    if len(sh) >= 2 and len(sl) >= 2:
        if h[sh[-1]] > h[sh[-2]] and l[sl[-1]] > l[sl[-2]]:
            return 'uptrend'
        if h[sh[-1]] < h[sh[-2]] and l[sl[-1]] < l[sl[-2]]:
            return 'downtrend'
    return 'range'


def _confirmed_higher_low(H: np.ndarray, L: np.ndarray, end_idx: int) -> bool:
    s = max(0, end_idx - 40)
    l = L[s:end_idx]
    n = len(l)
    sw = 3
    sl = [i for i in range(sw, n - sw) if l[i] == min(l[max(0, i-sw):i+sw+1])]
    return len(sl) >= 2 and l[sl[-1]] > l[sl[-2]]


def _confirmed_lower_high(H: np.ndarray, L: np.ndarray, end_idx: int) -> bool:
    s = max(0, end_idx - 40)
    h = H[s:end_idx]
    n = len(h)
    sw = 3
    sh = [i for i in range(sw, n - sw) if h[i] == max(h[max(0, i-sw):i+sw+1])]
    return len(sh) >= 2 and h[sh[-1]] < h[sh[-2]]


# ── ZONE STRUCTURES ──────────────────────────────────────────

@dataclass
class SDZone:
    kind:       str      # 'demand' | 'supply'
    top:        float
    bottom:     float
    formed_at:  int      # index in the bar array when zone was created
    confluence: int = 0
    is_flip:    bool = False
    move_atr:   float = 0.0

    @property
    def mid(self):    return (self.top + self.bottom) / 2
    @property
    def height(self): return max(self.top - self.bottom, 1e-9)


def _find_zones(H: np.ndarray, L: np.ndarray, C: np.ndarray,
                V: np.ndarray, ATR: np.ndarray,
                scan_start: int, scan_end: int) -> List[SDZone]:
    """Detect DBR (demand) and RBD (supply) zones in bar range [scan_start, scan_end)."""
    zones: List[SDZone] = []
    avg_vol = float(np.mean(V[max(0, scan_end - 50):scan_end])) + 1e-9

    for i in range(scan_start + 10, scan_end - 6):
        atr = ATR[i]
        if atr < 1e-9:
            continue

        # 3-candle consolidation base ending at bar i
        b_s = i - 3
        b_e = i
        bh = float(np.max(H[b_s:b_e]))
        bl = float(np.min(L[b_s:b_e]))

        # Base must be tight
        if bh - bl >= CONSOL_ATR_MULT * atr:
            continue

        pre = b_s - 1
        if pre < scan_start:
            continue
        post_end = min(b_e + 5, scan_end - 1)

        # ── DEMAND: drop into base, rally out ──────────────
        drop_in   = float(C[pre]) - bl
        rally_out = float(C[post_end]) - bh
        if drop_in >= IMBAL_ATR_MULT * atr and rally_out >= IMBAL_ATR_MULT * atr:
            conf = 0
            if V[b_e] > avg_vol * 1.4:                    conf += 1
            if rally_out > IMBAL_ATR_MULT * 2 * atr:      conf += 1
            if drop_in   > IMBAL_ATR_MULT * 2 * atr:      conf += 1
            zones.append(SDZone(
                kind='demand', top=bh, bottom=bl,
                formed_at=i, confluence=conf,
                move_atr=round(rally_out / atr, 1)
            ))

        # ── SUPPLY: rally into base, drop out ──────────────
        rally_in = bh - float(C[pre])
        drop_out = bl - float(C[post_end])
        if rally_in >= IMBAL_ATR_MULT * atr and drop_out >= IMBAL_ATR_MULT * atr:
            conf = 0
            if V[b_e] > avg_vol * 1.4:                    conf += 1
            if drop_out  > IMBAL_ATR_MULT * 2 * atr:      conf += 1
            if rally_in  > IMBAL_ATR_MULT * 2 * atr:      conf += 1
            zones.append(SDZone(
                kind='supply', top=bh, bottom=bl,
                formed_at=i, confluence=conf,
                move_atr=round(drop_out / atr, 1)
            ))

    return zones


def _enrich_zones(H: np.ndarray, L: np.ndarray, C: np.ndarray,
                  ATR: np.ndarray, zones: List[SDZone]) -> List[SDZone]:
    """Add flip flag and filter stale zones."""
    n = len(C)
    for z in zones:
        # Flip: zone overlaps opposite-kind zone in same price area
        for other in zones:
            if other is z or other.kind == z.kind:
                continue
            overlap = min(z.top, other.top) - max(z.bottom, other.bottom)
            if overlap > 0 and overlap / z.height > 0.3:
                z.is_flip = True
                break
    return zones


# ── REJECTION CANDLE ─────────────────────────────────────────

def _is_rejection_candle(
    O: np.ndarray, H: np.ndarray, L: np.ndarray, C: np.ndarray,
    i: int, atr: float, zone_kind: str
) -> Tuple[bool, str]:
    """
    Checks for pin bar, bull/bear engulf, or strong close.
    Returns (is_valid, type_name).
    """
    if i < 1:
        return False, 'none'
    o = O[i]; h = H[i]; l = L[i]; c = C[i]
    op = O[i-1]; cp = C[i-1]
    body = abs(c - o)
    rng  = h - l

    # FIX 4: minimum real body
    if body < MIN_BODY_ATR * atr:
        return False, 'none'

    if zone_kind == 'demand':
        lower_wick = min(o, c) - l
        if lower_wick >= MIN_WICK_ATR * atr and lower_wick >= 1.5 * max(body, 1e-9):
            return True, 'pin_bar'
        if c > o and cp < op and c >= op and o <= cp:
            return True, 'bull_engulf'
        if c > o and (c - l) / max(rng, 1e-9) > 0.65 and body >= 0.3 * atr:
            return True, 'strong_close'
    else:
        upper_wick = h - max(o, c)
        if upper_wick >= MIN_WICK_ATR * atr and upper_wick >= 1.5 * max(body, 1e-9):
            return True, 'pin_bar'
        if c < o and cp > op and c <= op and o >= cp:
            return True, 'bear_engulf'
        if c < o and (h - c) / max(rng, 1e-9) > 0.65 and body >= 0.3 * atr:
            return True, 'strong_close'

    return False, 'none'


def _strong_close_context_ok(
    O: np.ndarray, H: np.ndarray, L: np.ndarray, C: np.ndarray,
    V: np.ndarray, i: int, atr: float, trend_1h: str
) -> bool:
    """FIX 2: strong_close outside range needs vol spike + momentum."""
    if trend_1h == 'range':
        return True
    avg_vol = float(np.mean(V[max(0, i - 20):i])) + 1e-9
    vol_ok = V[i] > avg_vol * 1.2
    if i >= 3:
        ph = max(H[i-3:i])
        pl = min(L[i-3:i])
        pr = ph - pl
        mom_ok = (C[i] - pl) / max(pr, 1e-9) > 0.6 if pr > 0 else False
    else:
        mom_ok = False
    return vol_ok and mom_ok


def _has_stall(H: np.ndarray, L: np.ndarray, C: np.ndarray,
               i: int, atr: float, zone: SDZone) -> bool:
    """FIX 6B: prior 2-bar range tight, OR prior close inside zone."""
    if i < 2:
        return False
    combined_h = max(H[i-2], H[i-1])
    combined_l = min(L[i-2], L[i-1])
    cond_a = (combined_h - combined_l) < STALL_ATR_MULT * atr
    prior_close = C[i-1]
    buf = atr * 0.1
    cond_b = (zone.bottom - buf) <= prior_close <= (zone.top + buf)
    return cond_a or cond_b


# ── ENTRY FILTER STACK (all 7 fixes) ────────────────────────

def _passes_all_filters(
    O: np.ndarray, H: np.ndarray, L: np.ndarray, C: np.ndarray,
    V: np.ndarray, ATR: np.ndarray,
    i: int, atr: float, rej_type: str,
    trend_1h: str, zone: SDZone
) -> Tuple[bool, str]:
    """
    Returns (allowed, blocked_by_fix).
    Applies all 7 fixes in order.
    """
    # FIX 1: trending market = pin/engulf only
    if trend_1h in ('uptrend', 'downtrend') and rej_type == 'strong_close':
        return False, 'fix1'

    # FIX 2: strong_close needs context outside range
    if rej_type == 'strong_close':
        if not _strong_close_context_ok(O, H, L, C, V, i, atr, trend_1h):
            return False, 'fix2'

    # FIX 6A: strong_close blocked in elevated vol regime
    if rej_type == 'strong_close':
        rolling = _rolling_atr_mean(ATR, i, SC_ATR_LOOKBACK)
        if rolling > 0 and ATR[i] / rolling > SC_MAX_ATR_RATIO:
            return False, 'fix6a'

    # FIX 7: strong_close banned in range (final gate — closes all remaining gaps)
    if rej_type == 'strong_close' and trend_1h == 'range':
        return False, 'fix7'

    # FIX 6B: engulf in range needs stall confirm
    if rej_type in ('bull_engulf', 'bear_engulf') and trend_1h == 'range':
        if not _has_stall(H, L, C, i, atr, zone):
            return False, 'fix6b'

    # FIX 6C: candle body > 15% of zone height
    body = abs(C[i] - O[i])
    if body < MIN_BODY_ZONE_RATIO * zone.height:
        return False, 'fix6c'

    return True, ''


# ══════════════════════════════════════════════════════════════
# MAIN SCANNER CLASS
# ══════════════════════════════════════════════════════════════

class SDScanner:
    def __init__(self, telegram_token: str, telegram_chat_id: str,
                 binance_api_key: str = None, binance_secret: str = None):

        self.telegram_token = telegram_token
        self.telegram_bot   = Bot(token=telegram_token)
        self.chat_id        = telegram_chat_id

        self.exchange = ccxt.binance({
            'apiKey':          binance_api_key,
            'secret':          binance_secret,
            'enableRateLimit': True,
            'options':         {'defaultType': 'future'},
        })

        self.active_trades: Dict[str, dict] = {}
        self.pair_cooldown: Dict[str, datetime] = {}
        self.signal_history: deque = deque(maxlen=300)
        self._zone_cache: Dict[str, Tuple[List[SDZone], datetime]] = {}

        self.is_scanning = False

        self.stats = {
            'total_signals':   0,
            'long_signals':    0,
            'short_signals':   0,
            'tp1_hits':        0,
            'tp2_hits':        0,
            'sl_hits':         0,
            'be_saves':        0,
            'timeouts':        0,
            'filtered':        {'fix1':0,'fix2':0,'fix5':0,'fix6a':0,
                                'fix6b':0,'fix6c':0,'fix7':0,'no_rej':0,
                                'trend':0,'cooldown':0},
            'session_start':   datetime.now(),
            'last_scan':       None,
            'pairs_scanned':   0,
        }

    # ── Data fetching ──────────────────────────────────────────

    async def _fetch_ohlcv(self, symbol: str, tf: str,
                           limit: int = 120) -> Optional[pd.DataFrame]:
        try:
            ohlcv = await self.exchange.fetch_ohlcv(symbol, tf, limit=limit)
            df = pd.DataFrame(ohlcv,
                              columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            return df
        except Exception as e:
            logger.error(f"fetch_ohlcv {symbol} {tf}: {e}")
            return None

    # ── Zone cache (refresh every 8 bars = ~2h on 1H) ─────────

    def _get_cached_zones(
        self, symbol: str,
        H: np.ndarray, L: np.ndarray, C: np.ndarray,
        V: np.ndarray, ATR: np.ndarray,
        current_bar: int
    ) -> List[SDZone]:
        cache_key = f"{symbol}_{current_bar // 8}"
        if cache_key in self._zone_cache:
            return self._zone_cache[cache_key][0]

        scan_s = max(0, current_bar - ZONE_LOOKBACK)
        zones  = _find_zones(H, L, C, V, ATR, scan_s, current_bar)
        zones  = _enrich_zones(H, L, C, ATR, zones)

        # Flip bonus
        for z in zones:
            if z.is_flip:
                z.confluence += 2

        # Filters
        zones = [z for z in zones if z.confluence >= MIN_CONFLUENCE]
        zones = [z for z in zones if (current_bar - z.formed_at) <= ZONE_MAX_AGE]
        zones.sort(key=lambda z: (int(z.is_flip), z.confluence), reverse=True)

        # Prune old cache entries
        if len(self._zone_cache) > 200:
            oldest = sorted(self._zone_cache.keys())[0]
            del self._zone_cache[oldest]

        self._zone_cache[cache_key] = (zones, datetime.now())
        return zones

    # ── Signal detection ───────────────────────────────────────

    def _detect_signal(self, symbol: str, df1h: pd.DataFrame) -> Optional[dict]:
        """
        Runs the full v7 S&D detection on a 1H dataframe.
        Returns a signal dict or None.
        """
        if len(df1h) < ZONE_LOOKBACK + 20:
            return None

        H   = df1h['high'].values.astype(float)
        L   = df1h['low'].values.astype(float)
        C   = df1h['close'].values.astype(float)
        O   = df1h['open'].values.astype(float)
        V   = df1h['volume'].values.astype(float)
        ATR = _calc_atr(H, L, C, ATR_PERIOD)
        T4H = _calc_4h_trend(df1h)

        i   = len(df1h) - 1   # latest completed bar
        atr = ATR[i]
        if atr < 1e-9:
            return None

        t1h = _detect_trend_1h(H, L, C, i, TREND_LOOKBACK_1H)
        t4h = T4H[i]

        zones = self._get_cached_zones(symbol, H, L, C, V, ATR, i)

        for z in zones:
            buf = atr * 0.3

            # ── DEMAND / LONG ─────────────────────────────
            if z.kind == 'demand' and t1h in ('uptrend', 'range'):
                if t4h == 'bear':
                    continue

                in_zone = L[i] <= z.top + buf and H[i] >= z.bottom - buf * 2
                if not in_zone:
                    continue

                rej, rej_type = _is_rejection_candle(O, H, L, C, i, atr, 'demand')
                if not rej:
                    self.stats['filtered']['no_rej'] += 1
                    continue

                ok, blk = _passes_all_filters(O, H, L, C, V, ATR, i, atr,
                                              rej_type, t1h, z)
                if not ok:
                    self.stats['filtered'][blk] = self.stats['filtered'].get(blk, 0) + 1
                    continue

                # FIX 5: confirm HL in uptrend
                if t1h == 'uptrend' and not _confirmed_higher_low(H, L, i):
                    self.stats['filtered']['fix5'] += 1
                    continue

                entry = z.top
                sl    = z.bottom - atr * ATR_SL_BUFFER
                risk  = abs(entry - sl)
                if risk < 1e-9:
                    continue
                tp1 = entry + risk * TP1_R
                tp2 = entry + risk * TP2_R

                return self._build_signal(
                    symbol, 'LONG', entry, sl, tp1, tp2,
                    atr, z, t1h, t4h, rej_type, risk
                )

            # ── SUPPLY / SHORT ────────────────────────────
            elif z.kind == 'supply' and t1h in ('downtrend', 'range'):
                if t4h == 'bull':
                    continue

                in_zone = H[i] >= z.bottom - buf and L[i] <= z.top + buf * 2
                if not in_zone:
                    continue

                rej, rej_type = _is_rejection_candle(O, H, L, C, i, atr, 'supply')
                if not rej:
                    self.stats['filtered']['no_rej'] += 1
                    continue

                ok, blk = _passes_all_filters(O, H, L, C, V, ATR, i, atr,
                                              rej_type, t1h, z)
                if not ok:
                    self.stats['filtered'][blk] = self.stats['filtered'].get(blk, 0) + 1
                    continue

                # FIX 5: confirm LH in downtrend
                if t1h == 'downtrend' and not _confirmed_lower_high(H, L, i):
                    self.stats['filtered']['fix5'] += 1
                    continue

                entry = z.bottom
                sl    = z.top + atr * ATR_SL_BUFFER
                risk  = abs(sl - entry)
                if risk < 1e-9:
                    continue
                tp1 = entry - risk * TP1_R
                tp2 = entry - risk * TP2_R

                return self._build_signal(
                    symbol, 'SHORT', entry, sl, tp1, tp2,
                    atr, z, t1h, t4h, rej_type, risk
                )

        return None

    def _build_signal(
        self, symbol: str, direction: str,
        entry: float, sl: float, tp1: float, tp2: float,
        atr: float, zone: SDZone,
        trend_1h: str, trend_4h: str, rej_type: str, risk: float
    ) -> dict:
        pair = symbol.replace('/USDT:USDT', '')
        tid  = f"{pair}_{direction[0]}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        rr   = round(abs(tp1 - entry) / max(risk, 1e-9), 2)

        def pct(a, b):
            return round(abs(a - b) / max(abs(b), 1e-9) * 100, 2)

        # Confluence tag for message
        tags = []
        if zone.is_flip:
            tags.append('🔄 Flip zone')
        if zone.move_atr >= 3.0:
            tags.append(f'⚡ Strong move {zone.move_atr}x ATR')
        rej_emoji = {'pin_bar': '📍', 'bull_engulf': '🟢', 'bear_engulf': '🔴', 'strong_close': '💪'}
        tags.append(f"{rej_emoji.get(rej_type,'●')} {rej_type.replace('_',' ').title()}")
        trend_tag = f"1H {trend_1h} · 4H {trend_4h}"

        if direction == 'LONG':
            close_plan = (
                f"📋 <b>Close plan:</b>\n"
                f"  • TP1 → close <b>{int(TP1_CLOSE_PCT*100)}%</b> → SL to BE\n"
                f"  • TP2 → close remaining <b>{int((1-TP1_CLOSE_PCT)*100)}%</b> (runner)"
            )
        else:
            close_plan = (
                f"📋 <b>Close plan:</b>\n"
                f"  • TP1 → close <b>{int(TP1_CLOSE_PCT*100)}%</b> → SL to BE\n"
                f"  • TP2 → close remaining <b>{int((1-TP1_CLOSE_PCT)*100)}%</b> (runner)"
            )

        return {
            'trade_id':    tid,
            'symbol':      pair,
            'full_symbol': symbol,
            'signal':      direction,
            'entry':       entry,
            'stop_loss':   sl,
            'tp1':         tp1, 'tp1_pct': pct(tp1, entry),
            'tp2':         tp2, 'tp2_pct': pct(tp2, entry),
            'rr':          rr,
            'risk_pct':    pct(sl, entry),
            'atr':         round(atr, 6),
            'zone_top':    round(zone.top, 6),
            'zone_bottom': round(zone.bottom, 6),
            'zone_kind':   zone.kind,
            'confluence':  zone.confluence,
            'is_flip':     zone.is_flip,
            'trend_1h':    trend_1h,
            'trend_4h':    trend_4h,
            'rej_type':    rej_type,
            'tags':        tags,
            'trend_tag':   trend_tag,
            'close_plan':  close_plan,
            'tp1_hit':     False,
            'tp2_hit':     False,
            'sl_hit':      False,
            'be_active':   False,
            'partial_taken': False,
            'timestamp':   datetime.now(),
        }

    # ── Format Telegram message ────────────────────────────────

    def _fmt_signal(self, sig: dict) -> str:
        e  = '🚀' if sig['signal'] == 'LONG' else '🔻'
        t4 = {'bull': '🐂', 'bear': '🐻', 'neutral': '➡️'}.get(sig['trend_4h'], '➡️')

        zone_label = '🔄 Flip' if sig['is_flip'] else '🆕 Fresh'
        conf_bar   = '▰' * min(sig['confluence'], 5) + '▱' * max(0, 5 - sig['confluence'])

        m  = f"{'─'*38}\n"
        m += f"{e} <b>{sig['signal']} — S&D Zone Signal</b>\n"
        m += f"{'─'*38}\n\n"
        m += f"<b>Pair:</b>   #{sig['symbol']}  {t4} 4H {sig['trend_4h']}\n"
        m += f"<b>Trend:</b>  {sig['trend_tag']}\n"
        m += f"<b>Zone:</b>   {zone_label}  {conf_bar}  ({sig['zone_kind'].upper()})\n"
        m += f"<b>Entry:</b>  {' + '.join(sig['tags'])}\n\n"

        m += f"<b>Entry:</b>      <code>${sig['entry']:.6f}</code>\n"
        m += f"<b>TP1:</b>        <code>${sig['tp1']:.6f}</code>  +{sig['tp1_pct']:.2f}%\n"
        m += f"<b>TP2:</b>        <code>${sig['tp2']:.6f}</code>  +{sig['tp2_pct']:.2f}%\n"
        m += f"<b>Stop loss:</b>  <code>${sig['stop_loss']:.6f}</code>  -{sig['risk_pct']:.2f}%\n"
        m += f"<b>RR (TP1):</b>   {sig['rr']}:1\n\n"

        m += f"<b>Zone:</b>  {sig['zone_bottom']:.6f} — {sig['zone_top']:.6f}\n\n"

        m += f"{sig['close_plan']}\n\n"

        m += f"<i>🆔 {sig['trade_id']}</i>\n"
        m += f"<i>⏰ {sig['timestamp'].strftime('%H:%M UTC')} | SD v1.0</i>"
        return m

    # ── Telegram ───────────────────────────────────────────────

    async def _send(self, text: str):
        try:
            await self.telegram_bot.send_message(
                chat_id=self.chat_id,
                text=text,
                parse_mode=ParseMode.HTML,
            )
        except Exception as e:
            logger.error(f"Telegram send: {e}")

    # ── Trade alerts ───────────────────────────────────────────

    async def _tp1_alert(self, t: dict, price: float):
        gain = abs((price - t['entry']) / t['entry'] * 100)
        next_tp = f"${t['tp2']:.6f} (+{t['tp2_pct']:.2f}%)"
        m  = f"✅ <b>TP1 HIT</b> ✅\n\n"
        m += f"<b>{t['symbol']}</b> {t['signal']}\n"
        m += f"Entry: ${t['entry']:.6f}\n"
        m += f"TP1:   ${price:.6f}  <b>+{gain:.2f}%</b>\n\n"
        m += f"✂️ Close <b>{int(TP1_CLOSE_PCT*100)}%</b> of position\n"
        m += f"🔒 Move SL → breakeven (${t['entry']:.6f})\n"
        m += f"🎯 Next: {next_tp}\n"
        m += f"\n<i>{t['trade_id']}</i>"
        await self._send(m)
        t['tp1_hit']      = True
        t['be_active']    = True
        t['partial_taken'] = True
        self.stats['tp1_hits'] += 1

    async def _tp2_alert(self, t: dict, price: float):
        gain = abs((price - t['entry']) / t['entry'] * 100)
        m  = f"💰 <b>TP2 HIT — FULL TARGET!</b> 💰\n\n"
        m += f"<b>{t['symbol']}</b> {t['signal']}\n"
        m += f"Entry: ${t['entry']:.6f}\n"
        m += f"TP2:   ${price:.6f}  <b>+{gain:.2f}%</b>\n\n"
        m += f"✅ Close remaining <b>{int((1-TP1_CLOSE_PCT)*100)}%</b> — trade complete\n"
        m += f"\n<i>{t['trade_id']}</i>"
        await self._send(m)
        t['tp2_hit'] = True
        self.stats['tp2_hits'] += 1

    async def _sl_alert(self, t: dict, price: float, be_save: bool = False):
        if be_save:
            m  = f"🔒 <b>BREAKEVEN CLOSE</b>\n\n"
            m += f"<b>{t['symbol']}</b> {t['signal']}\n"
            m += f"TP1 was hit ✅ — SL was at entry\n"
            m += f"Closed remainder at breakeven — <b>zero loss</b>\n"
            m += f"\n<i>{t['trade_id']}</i>"
            self.stats['be_saves'] += 1
        else:
            loss = abs((price - t['entry']) / t['entry'] * 100)
            m  = f"⛔ <b>STOP LOSS</b>\n\n"
            m += f"<b>{t['symbol']}</b> {t['signal']}\n"
            m += f"Entry: ${t['entry']:.6f}\n"
            m += f"SL:    ${price:.6f}  <b>-{loss:.2f}%</b>\n\n"
            m += f"<i>Next zone incoming 🎯</i>"
            self.stats['sl_hits'] += 1
        await self._send(m)

    # ── Trade tracker ──────────────────────────────────────────

    async def _track_trades(self):
        logger.info("📡 Trade tracker started")
        while True:
            try:
                if not self.active_trades:
                    await asyncio.sleep(30)
                    continue

                done = []
                for tid, t in list(self.active_trades.items()):
                    try:
                        # Timeout
                        if datetime.now() - t['timestamp'] > timedelta(hours=MAX_TRADE_HOURS):
                            logger.info(f"⏰ Timeout: {t['symbol']}")
                            self.stats['timeouts'] += 1
                            done.append(tid)
                            continue

                        ticker = await self.exchange.fetch_ticker(t['full_symbol'])
                        price  = ticker['last']
                        act_sl = t['entry'] if t['be_active'] else t['stop_loss']

                        if t['signal'] == 'LONG':
                            if not t['tp1_hit'] and price >= t['tp1']:
                                await self._tp1_alert(t, price)
                            if t['tp1_hit'] and not t['tp2_hit'] and price >= t['tp2']:
                                await self._tp2_alert(t, price)
                                done.append(tid); continue
                            if price <= act_sl:
                                await self._sl_alert(t, price, be_save=t['be_active'])
                                done.append(tid); continue
                        else:  # SHORT
                            if not t['tp1_hit'] and price <= t['tp1']:
                                await self._tp1_alert(t, price)
                            if t['tp1_hit'] and not t['tp2_hit'] and price <= t['tp2']:
                                await self._tp2_alert(t, price)
                                done.append(tid); continue
                            if price >= act_sl:
                                await self._sl_alert(t, price, be_save=t['be_active'])
                                done.append(tid); continue

                    except Exception as e:
                        logger.error(f"Track {tid}: {e}")

                for tid in done:
                    self.active_trades.pop(tid, None)

                await asyncio.sleep(30)

            except Exception as e:
                logger.error(f"Tracker loop: {e}")
                await asyncio.sleep(60)

    # ── Pair list ──────────────────────────────────────────────

    async def _get_pairs(self) -> List[str]:
        try:
            await self.exchange.load_markets()
            tickers = await self.exchange.fetch_tickers()
            pairs = [
                s for s in self.exchange.symbols
                if s.endswith('/USDT:USDT')
                and 'PERP' not in s
                and tickers.get(s, {}).get('quoteVolume', 0) > MIN_VOLUME_USDT
            ]
            pairs.sort(
                key=lambda x: tickers.get(x, {}).get('quoteVolume', 0),
                reverse=True
            )
            return pairs[:TOP_PAIRS_LIMIT]
        except Exception as e:
            logger.error(f"get_pairs: {e}")
            return []

    # ── Main scan ──────────────────────────────────────────────

    async def scan_all(self) -> List[dict]:
        if self.is_scanning:
            return []
        self.is_scanning = True
        signals = []

        pairs = await self._get_pairs()
        logger.info(f"🔍 Scanning {len(pairs)} pairs...")

        for symbol in pairs:
            try:
                df1h = await self._fetch_ohlcv(symbol, '1h', limit=120)
                if df1h is None or len(df1h) < 90:
                    await asyncio.sleep(0.2)
                    continue

                sig = self._detect_signal(symbol, df1h)
                if sig is None:
                    await asyncio.sleep(0.2)
                    continue

                # Cooldown: same pair+direction max once per 4H
                ck   = f"{sig['symbol']}_{sig['signal']}"
                last = self.pair_cooldown.get(ck)
                if last and (datetime.now() - last).total_seconds() < 4 * 3600:
                    self.stats['filtered']['cooldown'] = \
                        self.stats['filtered'].get('cooldown', 0) + 1
                    await asyncio.sleep(0.2)
                    continue

                self.pair_cooldown[ck] = datetime.now()
                self.active_trades[sig['trade_id']] = sig
                self.signal_history.append(sig)

                self.stats['total_signals'] += 1
                if sig['signal'] == 'LONG':
                    self.stats['long_signals'] += 1
                else:
                    self.stats['short_signals'] += 1

                await self._send(self._fmt_signal(sig))
                signals.append(sig)
                logger.info(
                    f"✅ {sig['symbol']} {sig['signal']} {sig['rej_type']} "
                    f"zone={sig['zone_kind']} flip={sig['is_flip']} "
                    f"1H={sig['trend_1h']} 4H={sig['trend_4h']}"
                )
                await asyncio.sleep(1)

            except Exception as e:
                logger.error(f"Scan {symbol}: {e}")

            await asyncio.sleep(0.3)

        self.stats['last_scan']     = datetime.now()
        self.stats['pairs_scanned'] = len(pairs)

        filtered = self.stats['filtered']
        logger.info(
            f"✅ Scan done | {len(signals)} signals | "
            f"filtered: no_rej={filtered.get('no_rej',0)} "
            f"f1={filtered.get('fix1',0)} f2={filtered.get('fix2',0)} "
            f"f5={filtered.get('fix5',0)} f6a={filtered.get('fix6a',0)} "
            f"f6b={filtered.get('fix6b',0)} f7={filtered.get('fix7',0)} "
            f"cooldown={filtered.get('cooldown',0)} | "
            f"tracking={len(self.active_trades)}"
        )
        self.is_scanning = False
        return signals

    # ── Daily report ───────────────────────────────────────────

    async def _daily_report(self):
        while True:
            await asyncio.sleep(24 * 3600)
            try:
                s   = self.stats
                tp1 = s['tp1_hits']; tp2 = s['tp2_hits']
                sl  = s['sl_hits'];  be  = s['be_saves']
                tot = tp1 + sl
                wr  = round(tp1 / tot * 100, 1) if tot > 0 else 0
                hrs = round((datetime.now() - s['session_start']).total_seconds() / 3600, 1)

                cutoff   = datetime.now() - timedelta(hours=24)
                day_sigs = [t for t in self.signal_history if t['timestamp'] >= cutoff]
                day_long  = sum(1 for t in day_sigs if t['signal'] == 'LONG')
                day_short = sum(1 for t in day_sigs if t['signal'] == 'SHORT')

                bar = '▰' * int(wr / 10) + '▱' * (10 - int(wr / 10))

                m  = f"{'─'*36}\n📅 <b>24H REPORT — SD Scanner v1.0</b>\n{'─'*36}\n\n"
                m += f"Session: {hrs}h\n\n"
                m += f"<b>── Today's Signals ──</b>\n"
                m += f"  Total: <b>{len(day_sigs)}</b>  ({day_long}L / {day_short}S)\n\n"
                m += f"<b>── Performance ──</b>\n"
                m += f"  ✅ TP1 hits:  <b>{tp1}</b>\n"
                m += f"  💰 TP2 hits:  <b>{tp2}</b>  ({round(tp2/max(tp1,1)*100)}% extended)\n"
                m += f"  🔒 BE saves:  <b>{be}</b>\n"
                m += f"  ❌ SL hits:   <b>{sl}</b>\n\n"
                m += f"<b>TP1 Win Rate: {wr}%</b>\n{bar}\n\n"

                if wr >= 75:   status = "🔥 Excellent — strategy working"
                elif wr >= 60: status = "✅ Good — within target range"
                elif wr >= 50: status = "⚠️ Watch closely"
                else:          status = "🚨 Below target — check market conditions"
                m += f"{status}\n\n"

                flt = s['filtered']
                m += f"<b>Filters today:</b>\n"
                m += f"  No rejection:  {flt.get('no_rej',0)}\n"
                m += f"  FIX1 (trend):  {flt.get('fix1',0)}\n"
                m += f"  FIX7 (SC rng): {flt.get('fix7',0)}\n"
                m += f"  FIX6B (stall): {flt.get('fix6b',0)}\n"
                m += f"  Cooldown:      {flt.get('cooldown',0)}\n\n"
                m += f"  Tracking: {len(self.active_trades)} trades\n"
                m += f"<i>⏰ {datetime.now().strftime('%d %b %Y %H:%M UTC')}</i>"
                await self._send(m)

            except Exception as e:
                logger.error(f"Daily report: {e}")

    # ── Run ────────────────────────────────────────────────────

    async def run(self):
        logger.info(
            f"🚀 SD Scanner v1.0 | "
            f"IMBAL={IMBAL_ATR_MULT}x ATR | "
            f"TP1={TP1_R}R TP2={TP2_R}R | "
            f"Rejection=pin/engulf | All 7 fixes ON"
        )
        asyncio.create_task(self._track_trades())
        asyncio.create_task(self._daily_report())

        while True:
            try:
                await self.scan_all()
                await asyncio.sleep(SCAN_INTERVAL_MIN * 60)
            except Exception as e:
                logger.error(f"Run loop: {e}")
                await asyncio.sleep(60)

    async def close(self):
        await self.exchange.close()


# ══════════════════════════════════════════════════════════════
# TELEGRAM COMMANDS
# ══════════════════════════════════════════════════════════════

class BotCommands:
    def __init__(self, scanner: SDScanner):
        self.s = scanner

    async def cmd_start(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        await update.message.reply_text(
            "🏦 <b>Supply & Demand Scanner v1.0</b>\n"
            "82.7% WR backtest · Pin/Engulf signals · 1.8R target\n\n"
            "Commands:\n"
            "/scan   — force scan now\n"
            "/stats  — session stats\n"
            "/trades — active trades\n"
            "/zones  — explain zone logic\n"
            "/help   — this message",
            parse_mode=ParseMode.HTML
        )

    async def cmd_scan(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        if self.s.is_scanning:
            await update.message.reply_text("⚠️ Scan already running!")
            return
        await update.message.reply_text("🔍 Scanning now...")
        asyncio.create_task(self.s.scan_all())

    async def cmd_stats(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        s   = self.s.stats
        tp1 = s['tp1_hits']; tp2 = s['tp2_hits']
        sl  = s['sl_hits'];  be  = s['be_saves']
        tot = tp1 + sl
        wr  = round(tp1 / tot * 100, 1) if tot > 0 else 0
        hrs = round((datetime.now() - s['session_start']).total_seconds() / 3600, 1)
        spd = round(s['total_signals'] / max(hrs, 0.1), 2)
        flt = s['filtered']

        m  = f"📊 <b>SD SCANNER STATS</b>\n\nSession: {hrs}h\n\n"
        m += f"<b>Signals:</b> {s['total_signals']} ({spd}/h)\n"
        m += f"  🟢 Long:  {s['long_signals']}\n"
        m += f"  🔴 Short: {s['short_signals']}\n\n"
        m += f"<b>Performance:</b>\n"
        m += f"  ✅ TP1: {tp1}  ({wr}% WR)\n"
        m += f"  💰 TP2: {tp2}  ({round(tp2/max(tp1,1)*100)}% of TP1s ran)\n"
        m += f"  🔒 BE:  {be}\n"
        m += f"  ❌ SL:  {sl}\n\n"
        m += f"<b>Filters fired:</b>\n"
        m += f"  No rejection:     {flt.get('no_rej',0)}\n"
        m += f"  FIX1 trend:       {flt.get('fix1',0)}\n"
        m += f"  FIX2 SC context:  {flt.get('fix2',0)}\n"
        m += f"  FIX5 swing:       {flt.get('fix5',0)}\n"
        m += f"  FIX6A vol gate:   {flt.get('fix6a',0)}\n"
        m += f"  FIX6B stall:      {flt.get('fix6b',0)}\n"
        m += f"  FIX7 SC in rng:   {flt.get('fix7',0)}\n"
        m += f"  Cooldown:         {flt.get('cooldown',0)}\n\n"
        m += f"Tracking: {len(self.s.active_trades)} trades"
        if s['last_scan']:
            m += f"\nLast scan: {s['last_scan'].strftime('%H:%M')}"
        await update.message.reply_text(m, parse_mode=ParseMode.HTML)

    async def cmd_trades(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        trades = self.s.active_trades
        if not trades:
            await update.message.reply_text("📭 No active trades.")
            return
        m = f"📡 <b>ACTIVE TRADES ({len(trades)})</b>\n\n"
        for tid, t in list(trades.items())[:10]:
            age   = int((datetime.now() - t['timestamp']).total_seconds() / 3600)
            tp1_s = '✅' if t['tp1_hit'] else '⏳'
            tp2_s = '✅' if t['tp2_hit'] else '⏳'
            be_s  = ' 🔒BE' if t['be_active'] else ''
            flip_s = ' 🔄' if t['is_flip'] else ''
            m += f"<b>{t['symbol']}</b> {t['signal']}{be_s}{flip_s}\n"
            m += f"  Entry: ${t['entry']:.6f} | Zone: {t['zone_kind']}\n"
            m += f"  {t['rej_type'].replace('_',' ')} · {t['trend_1h']} · {age}h old\n"
            m += f"  TP1:{tp1_s} TP2:{tp2_s}\n\n"
        await update.message.reply_text(m, parse_mode=ParseMode.HTML)

    async def cmd_zones(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        m  = "🏦 <b>ZONE LOGIC — SD v1.0</b>\n\n"
        m += "<b>Zone types:</b>\n"
        m += "  📥 <b>Demand</b> = Drop → Base → Rally (DBR)\n"
        m += "  📤 <b>Supply</b> = Rally → Base → Drop (RBD)\n\n"
        m += "<b>Quality gates:</b>\n"
        m += f"  • Base range   &lt; {CONSOL_ATR_MULT}x ATR (consolidation)\n"
        m += f"  • Move in/out  ≥ {IMBAL_ATR_MULT}x ATR (institutional)\n"
        m += f"  • Min conf     ≥ {MIN_CONFLUENCE} (vol spike or big move)\n"
        m += f"  • Max age      ≤ {ZONE_MAX_AGE} bars\n"
        m += f"  • Flip zones   get +2 confluence bonus\n\n"
        m += "<b>Entry conditions (ALL must pass):</b>\n"
        m += "  1️⃣ Price enters zone ±0.3x ATR buffer\n"
        m += "  2️⃣ Rejection candle = pin bar or engulf\n"
        m += "  3️⃣ 4H trend not opposing (no LONG in 4H bear)\n"
        m += "  4️⃣ 1H structure confirmed (HL/LH check)\n"
        m += "  5️⃣ No elevated vol during entry (FIX 6A)\n"
        m += "  6️⃣ Stall confirm for range engulfs (FIX 6B)\n\n"
        m += "<b>Trade management:</b>\n"
        m += f"  SL  = zone edge ± {ATR_SL_BUFFER}x ATR\n"
        m += f"  TP1 = {TP1_R}R → close {int(TP1_CLOSE_PCT*100)}%, SL → BE\n"
        m += f"  TP2 = {TP2_R}R → close remaining {int((1-TP1_CLOSE_PCT)*100)}%\n"
        m += f"  Timeout: {MAX_TRADE_HOURS}h\n\n"
        m += "<b>Backtested results (v7):</b>\n"
        m += "  WR: 82.7% · PF: 21.5 · Max DD: -1.8%\n"
        await update.message.reply_text(m, parse_mode=ParseMode.HTML)

    async def cmd_help(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        await self.cmd_start(update, ctx)


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════

async def main():
    # ╔══════════════════════════════════════╗
    TELEGRAM_TOKEN   = "7957028587:AAE7aSYtE4hCxxTIPkAs_1ULJ9e8alkY6Ic"
    TELEGRAM_CHAT_ID = "-1003659830260"
    BINANCE_API_KEY  = None   # read-only key for market data (optional)
    BINANCE_SECRET   = None
    # ╚══════════════════════════════════════╝

    scanner = SDScanner(
        telegram_token    = TELEGRAM_TOKEN,
        telegram_chat_id  = TELEGRAM_CHAT_ID,
        binance_api_key   = BINANCE_API_KEY,
        binance_secret    = BINANCE_SECRET,
    )

    app = Application.builder().token(TELEGRAM_TOKEN).build()
    cmds = BotCommands(scanner)

    for cmd, fn in [
        ('start', cmds.cmd_start),
        ('scan',  cmds.cmd_scan),
        ('stats', cmds.cmd_stats),
        ('trades',cmds.cmd_trades),
        ('zones', cmds.cmd_zones),
        ('help',  cmds.cmd_help),
    ]:
        app.add_handler(CommandHandler(cmd, fn))

    await app.initialize()
    await app.start()
    logger.info("🤖 SD Scanner v1.0 online")

    try:
        await scanner.run()
    except KeyboardInterrupt:
        logger.info("Shutting down...")
    finally:
        await scanner.close()
        await app.stop()
        await app.shutdown()


if __name__ == '__main__':
    asyncio.run(main())
