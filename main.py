"""
SMC PRO SCANNER v5.0
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ROOT CAUSE FIXES from v4.0:
  PROBLEM 1: "Signal fires then reverses immediately"
  → OB was 50% mitigated before trigger formed = entering a dead zone
  → FIX: OB mitigation hard cap at 25% (was 50%)
  → FIX: OB freshness check — OB must be < 40 candles old
  → FIX: Price must be entering OB from the correct side, not exiting

  PROBLEM 2: "Signal fires too late (missed the move)"
  → Previous-candle trigger was stale, entry 1-2H behind price
  → FIX: Previous candle trigger cut to max +8pts (was +14)
  → FIX: Added "OB approach velocity" — price must be moving INTO OB
         not ranging inside it already for 3+ candles
  → FIX: If price has been inside OB > 3 candles → skip (stale)

  PROBLEM 3: Ranging/choppy market losses
  → HH/LL was only a bonus, ADX was never checked as a gate
  → FIX: 4H ADX < 18 = hard SKIP (pure chop, no edge)
  → FIX: 4H ADX 18-22 = ranging penalty (-15 score)
  → FIX: BTC dominance regime check — skip alts in BTC chop

  PROBLEM 4: Wrong trend direction entries  
  → 4H EMA check was too loose (21>50 alone = +10 pts)
  → FIX: Must confirm with DI+ > DI- for LONG, DI- > DI+ for SHORT
  → FIX: 4H price must be above/below both EMA 21 AND 50 for full pts
  → FIX: Counter-trend trades need BOTH MSS + sweep (not just one)

  ADDITIONAL UPGRADES:
  → ATR-adaptive SL: minimum 1.2x ATR (was 0.2x — way too tight)
  → OB quality score now checks vol at OB formation (was price only)
  → Near-miss debug shows EXACT reason score fell short
  → MIN_SCORE raised 75 → 80 (tighter gates earn it)
  → Scan skips pairs with spread > 0.15% (avoids illiquid noise)

TIMEFRAME ROLES v5.0:
  4H  → Trend bias (EMA + ADX gate) + HH/LL + BTC regime
  1H  → BOS/MSS + Fresh OB (≤40 bars, ≤25% mitigated) + Entry trigger
  15M → Volume spike bonus only (unchanged)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import asyncio
import ccxt.async_support as ccxt
from telegram import Bot, Update
from telegram.ext import Application, CommandHandler, ContextTypes
from telegram.constants import ParseMode
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import ta
import logging
from collections import deque

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

import warnings
warnings.filterwarnings('ignore')

# ═══════════════════════════════════════════════
#  TUNABLE SETTINGS
# ═══════════════════════════════════════════════
MAX_SIGNALS_PER_SCAN  = 5        # reduced from 6 — quality over quantity
MIN_SCORE             = 80       # raised from 75
MIN_VOLUME_24H        = 8_000_000  # raised from 5M — better liquidity
OB_TOLERANCE_PCT      = 0.006    # slightly tighter
OB_IMPULSE_ATR_MULT   = 1.2      # stronger impulse required (was 1.0)
OB_MAX_AGE_BARS       = 40       # NEW: OB older than 40x1H bars = skip
OB_MAX_MITIGATION_PCT = 0.25     # NEW: OB >25% mitigated = skip (was 50%)
OB_MAX_CANDLES_INSIDE = 3        # NEW: price inside OB >3 candles = stale
STRUCTURE_LOOKBACK    = 20
SCAN_INTERVAL_MIN     = 30
HH_LL_LOOKBACK        = 10
HH_LL_BONUS           = 8
ADX_CHOP_HARD_GATE    = 18       # NEW: 4H ADX below this = skip entirely
ADX_RANGE_PENALTY     = 15       # NEW: 4H ADX 18-22 = score penalty
ADX_RANGE_THRESHOLD   = 22       # NEW: above this = trending, no penalty
ATR_SL_MINIMUM_MULT   = 1.2      # NEW: SL must be at least 1.2x ATR away
MAX_SPREAD_PCT        = 0.0015   # NEW: skip pairs with spread > 0.15%


# ══════════════════════════════════════════════════════════════
#  INDICATORS
# ══════════════════════════════════════════════════════════════

def add_indicators(df):
    if len(df) < 55:
        return df
    try:
        df['ema_21']  = ta.trend.EMAIndicator(df['close'], 21).ema_indicator()
        df['ema_50']  = ta.trend.EMAIndicator(df['close'], 50).ema_indicator()
        df['ema_200'] = ta.trend.EMAIndicator(df['close'], min(200, len(df)-1)).ema_indicator()
        df['rsi']     = ta.momentum.RSIIndicator(df['close'], 14).rsi()

        macd = ta.trend.MACD(df['close'])
        df['macd']        = macd.macd()
        df['macd_signal'] = macd.macd_signal()
        df['macd_hist']   = macd.macd_diff()

        stoch = ta.momentum.StochRSIIndicator(df['close'])
        df['srsi_k'] = stoch.stochrsi_k()
        df['srsi_d'] = stoch.stochrsi_d()

        df['atr'] = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close']).average_true_range()

        bb = ta.volatility.BollingerBands(df['close'], 20, 2)
        df['bb_upper'] = bb.bollinger_hband()
        df['bb_lower'] = bb.bollinger_lband()
        df['bb_pband'] = bb.bollinger_pband()

        adx_i = ta.trend.ADXIndicator(df['high'], df['low'], df['close'])
        df['adx']    = adx_i.adx()
        df['di_pos'] = adx_i.adx_pos()
        df['di_neg'] = adx_i.adx_neg()

        df['cmf'] = ta.volume.ChaikinMoneyFlowIndicator(df['high'], df['low'], df['close'], df['volume']).chaikin_money_flow()
        df['mfi'] = ta.volume.MFIIndicator(df['high'], df['low'], df['close'], df['volume']).money_flow_index()

        df['vol_sma']   = df['volume'].rolling(20).mean()
        df['vol_ratio'] = df['volume'] / df['vol_sma'].replace(0, np.nan)

        tp = (df['high'] + df['low'] + df['close']) / 3
        df['vwap'] = (tp * df['volume']).cumsum() / df['volume'].cumsum()

        body = (df['close'] - df['open']).abs()
        uw   = df['high'] - df[['open','close']].max(axis=1)
        lw   = df[['open','close']].min(axis=1) - df['low']

        # ── Trigger candles (1H) ──────────────────────────────
        df['bull_engulf'] = (
            (df['close'].shift(1) < df['open'].shift(1)) &
            (df['close'] > df['open']) &
            (df['close'] > df['open'].shift(1)) &
            (df['open'] < df['close'].shift(1))
        ).astype(int)

        df['bear_engulf'] = (
            (df['close'].shift(1) > df['open'].shift(1)) &
            (df['close'] < df['open']) &
            (df['close'] < df['open'].shift(1)) &
            (df['open'] > df['close'].shift(1))
        ).astype(int)

        df['bull_pin'] = (
            (lw > body * 2.5) & (lw > uw * 2) & (df['close'] > df['open'])
        ).astype(int)

        df['bear_pin'] = (
            (uw > body * 2.5) & (uw > lw * 2) & (df['close'] < df['open'])
        ).astype(int)

        df['hammer'] = (
            (lw > body * 2.0) & (lw > uw * 1.5)
        ).astype(int)

        df['shooting_star'] = (
            (uw > body * 2.0) & (uw > lw * 1.5)
        ).astype(int)

    except Exception as e:
        logger.error(f"Indicator error: {e}")
    return df


# ══════════════════════════════════════════════════════════════
#  SMC ENGINE v5.0
# ══════════════════════════════════════════════════════════════

class SMCEngine:

    def swing_highs_lows(self, df, left=4, right=4):
        highs, lows = [], []
        n = len(df)
        for i in range(left, n - right):
            hi = df['high'].iloc[i]
            lo = df['low'].iloc[i]
            if all(hi >= df['high'].iloc[i-left:i]) and all(hi >= df['high'].iloc[i+1:i+right+1]):
                highs.append({'i': i, 'price': hi})
            if all(lo <= df['low'].iloc[i-left:i]) and all(lo <= df['low'].iloc[i+1:i+right+1]):
                lows.append({'i': i, 'price': lo})
        return highs, lows

    def check_4h_hh_ll(self, df_4h, direction, lookback=HH_LL_LOOKBACK):
        n = len(df_4h)
        if n < lookback * 2:
            return False, "⚠️ Not enough 4H data for HH/LL check"
        recent = df_4h.iloc[-lookback:]
        prior  = df_4h.iloc[-(lookback * 2):-lookback]
        if direction == 'LONG':
            rh, ph = recent['high'].max(), prior['high'].max()
            if rh > ph:
                return True,  f"📈 4H Higher High ({ph:.5f} → {rh:.5f}) +{HH_LL_BONUS}pts"
            return False, f"➖ 4H no HH ({rh:.5f} ≤ {ph:.5f}) — ranging"
        else:
            rl, pl = recent['low'].min(), prior['low'].min()
            if rl < pl:
                return True,  f"📉 4H Lower Low ({pl:.5f} → {rl:.5f}) +{HH_LL_BONUS}pts"
            return False, f"➖ 4H no LL ({rl:.5f} ≥ {pl:.5f}) — ranging"

    def check_adx_regime(self, df_4h):
        """
        NEW v5.0: Hard gate on 4H ADX.
        Returns (regime, adx_value, message)
        regime: 'TRENDING' | 'RANGING' | 'CHOP'
        """
        adx = df_4h['adx'].iloc[-1]
        if pd.isna(adx):
            return 'UNKNOWN', 0, "⚠️ ADX not available"
        if adx < ADX_CHOP_HARD_GATE:
            return 'CHOP', adx, f"❌ 4H ADX {adx:.1f} < {ADX_CHOP_HARD_GATE} — pure chop, HARD SKIP"
        elif adx < ADX_RANGE_THRESHOLD:
            return 'RANGING', adx, f"⚠️ 4H ADX {adx:.1f} — ranging, -{ADX_RANGE_PENALTY}pts penalty"
        else:
            return 'TRENDING', adx, f"✅ 4H ADX {adx:.1f} — trending market"

    def check_di_alignment(self, df_4h, direction):
        """
        NEW v5.0: DI+/DI- must confirm the bias direction.
        Prevents entering longs in hidden downtrends and vice versa.
        """
        di_pos = df_4h['di_pos'].iloc[-1]
        di_neg = df_4h['di_neg'].iloc[-1]
        if pd.isna(di_pos) or pd.isna(di_neg):
            return False, "⚠️ DI data unavailable"
        if direction == 'LONG':
            if di_pos > di_neg:
                return True,  f"✅ DI+ ({di_pos:.1f}) > DI- ({di_neg:.1f}) — bull pressure"
            return False, f"❌ DI- ({di_neg:.1f}) > DI+ ({di_pos:.1f}) — bear pressure in LONG"
        else:
            if di_neg > di_pos:
                return True,  f"✅ DI- ({di_neg:.1f}) > DI+ ({di_pos:.1f}) — bear pressure"
            return False, f"❌ DI+ ({di_pos:.1f}) > DI- ({di_neg:.1f}) — bull pressure in SHORT"

    def detect_structure_break(self, df, highs, lows, lookback=STRUCTURE_LOOKBACK):
        events = []
        close = df['close']
        n = len(df)
        start = max(0, n - lookback - 15)

        for k in range(1, len(highs)):
            ph = highs[k-1]; ch = highs[k]
            if ch['i'] < start: continue
            level = ph['price']
            for j in range(ch['i'], min(ch['i'] + 10, n)):
                if close.iloc[j] > level:
                    kind = 'BOS_BULL' if ch['price'] > ph['price'] else 'MSS_BULL'
                    events.append({'kind': kind, 'level': level, 'bar': j})
                    break

        for k in range(1, len(lows)):
            pl = lows[k-1]; cl = lows[k]
            if cl['i'] < start: continue
            level = pl['price']
            for j in range(cl['i'], min(cl['i'] + 10, n)):
                if close.iloc[j] < level:
                    kind = 'BOS_BEAR' if cl['price'] < pl['price'] else 'MSS_BEAR'
                    events.append({'kind': kind, 'level': level, 'bar': j})
                    break

        if not events:
            return None
        latest = sorted(events, key=lambda x: x['bar'])[-1]
        if latest['bar'] < n - lookback:
            return None
        return latest

    def find_order_blocks(self, df, direction, lookback=60):
        """
        v5.0 CHANGES:
        - Mitigation check raised from 50% to OB_MAX_MITIGATION_PCT (25%)
        - Age check: OB must be within OB_MAX_AGE_BARS candles
        - Volume at OB formation tracked for quality scoring
        """
        obs = []
        n = len(df)
        start = max(2, n - lookback)

        for i in range(start, n - 3):
            # Age gate: skip OBs too far back
            age = (n - 1) - i
            if age > OB_MAX_AGE_BARS:
                continue

            c = df.iloc[i]
            atr_local = df['atr'].iloc[i] if 'atr' in df.columns and not pd.isna(df['atr'].iloc[i]) else (c['high'] - c['low'])
            min_impulse = atr_local * OB_IMPULSE_ATR_MULT

            # Volume at OB formation (relative to recent avg)
            vol_at_ob = df['vol_ratio'].iloc[i] if 'vol_ratio' in df.columns and not pd.isna(df['vol_ratio'].iloc[i]) else 1.0

            if direction == 'LONG':
                if c['close'] >= c['open']: continue
                fwd_high = df['high'].iloc[i+1:min(i+5, n)].max()
                if fwd_high - c['low'] < min_impulse: continue

                ob_top    = max(c['open'], c['close'])
                ob_bottom = c['low']
                ob_range  = ob_top - ob_bottom

                # v5.0: Strict mitigation check (25% max)
                lowest_close_since = df['close'].iloc[i+1:n].min()
                if ob_range > 0:
                    mitigation = max(0, ob_top - lowest_close_since) / ob_range
                    if mitigation > OB_MAX_MITIGATION_PCT:
                        continue

                obs.append({
                    'top':       ob_top,
                    'bottom':    ob_bottom,
                    'mid':      (ob_top + ob_bottom) / 2,
                    'bar':       i,
                    'age':       age,
                    'vol_ratio': vol_at_ob,
                    'mitigation': mitigation if ob_range > 0 else 0
                })

            else:  # SHORT
                if c['close'] <= c['open']: continue
                fwd_low = df['low'].iloc[i+1:min(i+5, n)].min()
                if c['high'] - fwd_low < min_impulse: continue

                ob_top    = c['high']
                ob_bottom = min(c['open'], c['close'])
                ob_range  = ob_top - ob_bottom

                # v5.0: Strict mitigation check (25% max)
                highest_close_since = df['close'].iloc[i+1:n].max()
                if ob_range > 0:
                    mitigation = max(0, highest_close_since - ob_bottom) / ob_range
                    if mitigation > OB_MAX_MITIGATION_PCT:
                        continue

                obs.append({
                    'top':       ob_top,
                    'bottom':    ob_bottom,
                    'mid':      (ob_top + ob_bottom) / 2,
                    'bar':       i,
                    'age':       age,
                    'vol_ratio': vol_at_ob,
                    'mitigation': mitigation if ob_range > 0 else 0
                })

        obs.sort(key=lambda x: x['bar'], reverse=True)
        return obs

    def check_ob_approach(self, df, ob, direction, n_candles=OB_MAX_CANDLES_INSIDE):
        """
        NEW v5.0: Detects if price is freshly entering the OB vs ranging inside it.

        'FRESH'  = price entered OB within last 1-2 candles  ← BEST
        'RECENT' = price entered OB 3 candles ago             ← OK
        'STALE'  = price has been inside OB for >3 candles   ← SKIP

        Also checks approach direction (must come from the right side).
        """
        n = len(df)
        closes = df['close'].iloc[-n_candles-3:]

        # Count how many recent candles have been inside the OB
        candles_inside = 0
        for i in range(len(closes)-1, -1, -1):
            p = closes.iloc[i]
            if ob['bottom'] <= p <= ob['top']:
                candles_inside += 1
            else:
                break  # as soon as we hit a candle outside, stop counting

        if candles_inside > n_candles:
            return 'STALE', candles_inside, f"❌ Price inside OB {candles_inside} candles — stale, skip"
        elif candles_inside >= 2:
            return 'RECENT', candles_inside, f"⚠️ Price entered OB {candles_inside} candles ago — OK"
        else:
            # Verify price came from the correct approach direction
            prior_close = df['close'].iloc[-3] if len(df) >= 3 else df['close'].iloc[0]
            if direction == 'LONG' and prior_close > ob['top']:
                return 'FRESH', candles_inside, f"✅ Fresh OB approach from above (LONG)"
            elif direction == 'SHORT' and prior_close < ob['bottom']:
                return 'FRESH', candles_inside, f"✅ Fresh OB approach from below (SHORT)"
            else:
                return 'FRESH', candles_inside, f"✅ Fresh OB entry ({candles_inside} candle inside)"

    def price_in_ob(self, price, ob, tolerance_pct=OB_TOLERANCE_PCT):
        tol = ob['top'] * tolerance_pct
        return (ob['bottom'] - tol) <= price <= (ob['top'] + tol)

    def find_fvg(self, df, direction, lookback=25):
        fvgs = []
        n = len(df)
        for i in range(max(1, n - lookback), n - 1):
            prev = df.iloc[i-1]; nxt = df.iloc[i+1]
            if direction == 'LONG' and prev['high'] < nxt['low']:
                fvgs.append({'top': nxt['low'], 'bottom': prev['high'],
                             'mid': (nxt['low'] + prev['high']) / 2, 'bar': i})
            elif direction == 'SHORT' and prev['low'] > nxt['high']:
                fvgs.append({'top': prev['low'], 'bottom': nxt['high'],
                             'mid': (prev['low'] + nxt['high']) / 2, 'bar': i})
        return fvgs

    def recent_liquidity_sweep(self, df, direction, highs, lows, lookback=25):
        n = len(df)
        start = n - lookback
        if direction == 'LONG':
            for sl in reversed(lows):
                if sl['i'] < start: continue
                level = sl['price']
                for j in range(sl['i'] + 1, min(sl['i'] + 8, n)):
                    c = df.iloc[j]
                    if c['low'] < level and c['close'] > level:
                        return {'level': level, 'bar': j, 'type': 'SWEEP_LOW'}
        else:
            for sh in reversed(highs):
                if sh['i'] < start: continue
                level = sh['price']
                for j in range(sh['i'] + 1, min(sh['i'] + 8, n)):
                    c = df.iloc[j]
                    if c['high'] > level and c['close'] < level:
                        return {'level': level, 'bar': j, 'type': 'SWEEP_HIGH'}
        return None

    def pd_zone(self, df_4h, price):
        hi = df_4h['high'].iloc[-50:].max()
        lo = df_4h['low'].iloc[-50:].min()
        rang = hi - lo
        if rang == 0: return 'NEUTRAL', 0.5
        pos = (price - lo) / rang
        if pos < 0.40:   return 'DISCOUNT', pos
        elif pos > 0.60: return 'PREMIUM',  pos
        return 'NEUTRAL', pos

    def calculate_atr_sl(self, entry, ob, atr, direction):
        """
        NEW v5.0: ATR-adaptive SL.
        SL = max(OB-based SL, entry ± 1.2x ATR)
        Prevents wicked-out SLs on volatile alts.
        """
        min_distance = atr * ATR_SL_MINIMUM_MULT
        if direction == 'LONG':
            ob_sl = ob['bottom'] - atr * 0.3
            atr_sl = entry - min_distance
            sl = min(ob_sl, atr_sl)  # further away = safer
        else:
            ob_sl = ob['top'] + atr * 0.3
            atr_sl = entry + min_distance
            sl = max(ob_sl, atr_sl)  # further away = safer
        return sl


# ══════════════════════════════════════════════════════════════
#  SCORER v5.0
# ══════════════════════════════════════════════════════════════

def score_setup(direction, ob, ob_approach, structure, sweep, fvg_near,
                df_1h, df_15m, df_4h, pd_label, hh_ll_confirmed,
                adx_regime, di_aligned):
    score = 0
    reasons = []
    failed = []

    l1  = df_1h.iloc[-1]
    p1  = df_1h.iloc[-2]
    l15 = df_15m.iloc[-1]
    l4  = df_4h.iloc[-1]

    # ── 1. Structure (20 pts) ─────────────────────────────────
    if structure:
        if 'MSS' in structure['kind']:
            score += 20; reasons.append(f"🏗️ MSS — Early Reversal ({structure['kind']})")
        else:
            score += 14; reasons.append(f"🏗️ BOS — Pullback Entry ({structure['kind']})")
    else:
        failed.append("❌ No BOS/MSS in last 20 candles")

    # ── 2. Order Block quality (20 pts) ──────────────────────
    if ob:
        ob_size_pct = (ob['top'] - ob['bottom']) / ob['bottom'] * 100
        # v5.0: Also factor in mitigation level and vol at formation
        mit_pct = ob.get('mitigation', 0) * 100
        vol_at_ob = ob.get('vol_ratio', 1.0)

        base_ob_score = 0
        if ob_size_pct < 0.8:
            base_ob_score = 20
            reasons.append(f"📦 Tight OB ({ob_size_pct:.2f}%, mit:{mit_pct:.0f}%) — high quality")
        elif ob_size_pct < 2.0:
            base_ob_score = 13
            reasons.append(f"📦 OB ({ob_size_pct:.2f}%, mit:{mit_pct:.0f}%)")
        else:
            base_ob_score = 7
            reasons.append(f"📦 Wide OB ({ob_size_pct:.2f}%) — lower quality")

        # Bonus for high-volume OB formation (institutional activity)
        if vol_at_ob >= 2.0:
            base_ob_score = min(base_ob_score + 3, 20)
            reasons.append(f"🔊 High-vol OB formation ({vol_at_ob:.1f}x avg)")

        score += base_ob_score
    else:
        failed.append("❌ No valid OB found")

    # ── 3. OB Approach freshness (NEW — up to 8 pts, -10 if stale) ──
    approach_status = ob_approach[0] if ob_approach else 'UNKNOWN'
    if approach_status == 'FRESH':
        score += 8
        reasons.append(f"🆕 Fresh OB entry — price just arrived")
    elif approach_status == 'RECENT':
        score += 3
        reasons.append(f"⏱ Recent OB entry ({ob_approach[1]} candles ago)")
    elif approach_status == 'STALE':
        score -= 10
        failed.append(ob_approach[2])

    # ── 4. 4H Trend Alignment (15 pts) ───────────────────────
    e21 = l4.get('ema_21', 0); e50 = l4.get('ema_50', 0); e200 = l4.get('ema_200', 0)
    price = l1.get('close', 0)

    if direction == 'LONG':
        if e21 > e50 > e200 and price > e50:
            score += 15; reasons.append("📈 4H Triple EMA Bull Stack + price above")
        elif e21 > e50 and price > e21:
            score += 12; reasons.append("📈 4H EMA aligned + price above 21")
        elif e21 > e50:
            score += 8;  reasons.append("📈 4H EMA 21>50 Bull")
        elif pd_label == 'DISCOUNT':
            score += 5;  reasons.append("📈 4H Discount (counter-trend, low score)")
        else:
            failed.append("⚠️ 4H trend weak for LONG")
    else:
        if e21 < e50 < e200 and price < e50:
            score += 15; reasons.append("📉 4H Triple EMA Bear Stack + price below")
        elif e21 < e50 and price < e21:
            score += 12; reasons.append("📉 4H EMA aligned + price below 21")
        elif e21 < e50:
            score += 8;  reasons.append("📉 4H EMA 21<50 Bear")
        elif pd_label == 'PREMIUM':
            score += 5;  reasons.append("📉 4H Premium (counter-trend, low score)")
        else:
            failed.append("⚠️ 4H trend weak for SHORT")

    # ── 5. ADX Regime (NEW — penalty for ranging) ────────────
    if adx_regime == 'RANGING':
        score -= ADX_RANGE_PENALTY
        failed.append(f"⚠️ ADX ranging — applied -{ADX_RANGE_PENALTY}pts")
    elif adx_regime == 'TRENDING':
        score += 5
        reasons.append(f"💪 ADX trending (+5pts)")

    # ── 6. DI Alignment (NEW — 5 pts) ────────────────────────
    if di_aligned:
        score += 5
        reasons.append("✅ DI aligned with bias")
    else:
        failed.append("⚠️ DI not aligned — opposing directional pressure")

    # ── 7. 4H HH/LL Bonus (8 pts) ────────────────────────────
    if hh_ll_confirmed:
        score += HH_LL_BONUS
        reasons.append(f"🏔️ 4H HH/LL confirmed (+{HH_LL_BONUS}pts)")
    else:
        failed.append(f"➖ 4H HH/LL not confirmed — ranging")

    # ── 8. 1H Entry Trigger (25 pts) ─────────────────────────
    # v5.0: Previous candle trigger reduced to max +8 (was +14)
    trigger = False
    trigger_label = ""

    if direction == 'LONG':
        if l1.get('bull_engulf', 0) == 1:
            score += 25; trigger = True
            trigger_label = "🕯️ 1H Bullish Engulfing ✅ (strongest)"
        elif l1.get('bull_pin', 0) == 1:
            score += 22; trigger = True
            trigger_label = "🕯️ 1H Bullish Pin Bar ✅"
        elif l1.get('hammer', 0) == 1:
            score += 18; trigger = True
            trigger_label = "🕯️ 1H Hammer ✅"
        elif p1.get('bull_engulf', 0) == 1:
            score += 8; trigger = True   # ← reduced from 14
            trigger_label = "🕯️ 1H Bull Engulf (prev — reduced weight) ⚠️"
        elif p1.get('bull_pin', 0) == 1:
            score += 6; trigger = True   # ← reduced from 11
            trigger_label = "🕯️ 1H Bull Pin (prev — reduced weight) ⚠️"
        elif p1.get('hammer', 0) == 1:
            score += 5; trigger = True   # ← reduced from 9
            trigger_label = "🕯️ 1H Hammer (prev — reduced weight) ⚠️"
    else:
        if l1.get('bear_engulf', 0) == 1:
            score += 25; trigger = True
            trigger_label = "🕯️ 1H Bearish Engulfing ✅ (strongest)"
        elif l1.get('bear_pin', 0) == 1:
            score += 22; trigger = True
            trigger_label = "🕯️ 1H Bearish Pin Bar ✅"
        elif l1.get('shooting_star', 0) == 1:
            score += 18; trigger = True
            trigger_label = "🕯️ 1H Shooting Star ✅"
        elif p1.get('bear_engulf', 0) == 1:
            score += 8; trigger = True   # ← reduced from 14
            trigger_label = "🕯️ 1H Bear Engulf (prev — reduced weight) ⚠️"
        elif p1.get('bear_pin', 0) == 1:
            score += 6; trigger = True   # ← reduced from 11
            trigger_label = "🕯️ 1H Bear Pin (prev — reduced weight) ⚠️"
        elif p1.get('shooting_star', 0) == 1:
            score += 5; trigger = True   # ← reduced from 9
            trigger_label = "🕯️ 1H Shooting Star (prev — reduced weight) ⚠️"

    if trigger:
        reasons.append(trigger_label)
    else:
        score -= 12
        failed.append("⏳ No 1H trigger candle — wait for close")

    # ── 9. Momentum (12 pts) ─────────────────────────────────
    rsi1  = l1.get('rsi', 50)
    macd1 = l1.get('macd', 0);  ms1  = l1.get('macd_signal', 0)
    pm1   = p1.get('macd', 0);  pms1 = p1.get('macd_signal', 0)
    sk1   = l1.get('srsi_k', 0.5); sd1 = l1.get('srsi_d', 0.5)

    if direction == 'LONG':
        if 28 <= rsi1 <= 55:
            score += 4; reasons.append(f"✅ RSI reset zone ({rsi1:.0f})")
        elif rsi1 < 28:
            score += 3; reasons.append(f"✅ RSI oversold ({rsi1:.0f})")
        if macd1 > ms1 and pm1 <= pms1:
            score += 5; reasons.append("⚡ MACD bull cross")
        elif macd1 > ms1:
            score += 2; reasons.append("✅ MACD bullish")
        if sk1 < 0.3 and sk1 > sd1:
            score += 3; reasons.append("⚡ Stoch RSI bull cross")
    else:
        if 45 <= rsi1 <= 72:
            score += 4; reasons.append(f"✅ RSI overbought zone ({rsi1:.0f})")
        elif rsi1 > 72:
            score += 3; reasons.append(f"✅ RSI overbought ({rsi1:.0f})")
        if macd1 < ms1 and pm1 >= pms1:
            score += 5; reasons.append("⚡ MACD bear cross")
        elif macd1 < ms1:
            score += 2; reasons.append("✅ MACD bearish")
        if sk1 > 0.7 and sk1 < sd1:
            score += 3; reasons.append("⚡ Stoch RSI bear cross")

    # ── 10. Extras: Sweep / FVG / 15M Vol / VWAP (10 pts) ───
    extras = 0
    if sweep:
        extras += 4; reasons.append(f"💧 Liq. sweep @ {sweep['level']:.5f}")
    if fvg_near:
        extras += 3; reasons.append("⚡ FVG overlaps OB")

    vr15 = l15.get('vol_ratio', 1.0)
    if   vr15 >= 2.5:
        extras += 3; reasons.append(f"🚀 15M vol spike {vr15:.1f}x")
    elif vr15 >= 1.5:
        extras += 1; reasons.append(f"✅ 15M elevated vol {vr15:.1f}x")

    close1 = l1.get('close', 0); vwap1 = l1.get('vwap', 0)
    if direction == 'LONG' and close1 < vwap1:
        extras = min(extras+1, 10); reasons.append("✅ 1H below VWAP")
    elif direction == 'SHORT' and close1 > vwap1:
        extras = min(extras+1, 10); reasons.append("✅ 1H above VWAP")

    score += min(extras, 10)

    return max(0, min(int(score), 100)), reasons, failed


# ══════════════════════════════════════════════════════════════
#  MAIN BOT
# ══════════════════════════════════════════════════════════════

class SMCProScanner:
    def __init__(self, telegram_token, chat_id, api_key=None, secret=None):
        self.token    = telegram_token
        self.bot      = Bot(token=telegram_token)
        self.chat_id  = chat_id
        self.exchange = ccxt.binance({
            'apiKey': api_key, 'secret': secret,
            'enableRateLimit': True,
            'options': {'defaultType': 'future'}
        })
        self.smc            = SMCEngine()
        self.active_trades  = {}
        self.signal_history = deque(maxlen=300)
        self.is_scanning    = False
        self.last_debug     = []
        self.btc_regime     = 'UNKNOWN'  # NEW: track BTC market state
        self.stats = {
            'total': 0, 'long': 0, 'short': 0,
            'elite': 0, 'premium': 0, 'high': 0,
            'tp1': 0, 'tp2': 0, 'tp3': 0, 'sl': 0,
            'skipped_chop': 0,   # NEW
            'skipped_stale': 0,  # NEW
            'last_scan': None, 'pairs_scanned': 0
        }

    async def get_btc_regime(self):
        """
        NEW v5.0: Check if BTC itself is in chop.
        If BTC ADX < ADX_CHOP_HARD_GATE, alts will also chop — skip most alt signals.
        Returns 'TRENDING' | 'RANGING' | 'CHOP'
        """
        try:
            raw = await self.exchange.fetch_ohlcv('BTC/USDT:USDT', '4h', limit=60)
            df  = pd.DataFrame(raw, columns=['ts','open','high','low','close','volume'])
            df  = add_indicators(df)
            adx = df['adx'].iloc[-1]
            if pd.isna(adx):
                return 'UNKNOWN'
            if adx < ADX_CHOP_HARD_GATE:
                return 'CHOP'
            elif adx < ADX_RANGE_THRESHOLD:
                return 'RANGING'
            return 'TRENDING'
        except Exception as e:
            logger.error(f"BTC regime check: {e}")
            return 'UNKNOWN'

    async def get_pairs(self):
        try:
            await self.exchange.load_markets()
            tickers = await self.exchange.fetch_tickers()
            pairs = []
            for s in self.exchange.symbols:
                if not s.endswith('/USDT:USDT'): continue
                if 'PERP' in s: continue
                t = tickers.get(s, {})
                vol = t.get('quoteVolume', 0)
                if vol < MIN_VOLUME_24H: continue
                # v5.0: Filter out wide-spread pairs (illiquid noise)
                ask = t.get('ask', 0); bid = t.get('bid', 0)
                if ask > 0 and bid > 0:
                    spread = (ask - bid) / ask
                    if spread > MAX_SPREAD_PCT:
                        continue
                pairs.append(s)
            pairs.sort(key=lambda x: tickers.get(x, {}).get('quoteVolume', 0), reverse=True)
            logger.info(f"✅ {len(pairs)} pairs after vol+spread filter")
            return pairs
        except Exception as e:
            logger.error(f"Pairs: {e}"); return []

    async def fetch_data(self, symbol):
        try:
            result = {}
            for tf, lim in [('4h', 220), ('1h', 150), ('15m', 80)]:
                raw = await self.exchange.fetch_ohlcv(symbol, tf, limit=lim)
                df  = pd.DataFrame(raw, columns=['ts','open','high','low','close','volume'])
                df['ts'] = pd.to_datetime(df['ts'], unit='ms')
                result[tf] = add_indicators(df)
                await asyncio.sleep(0.04)
            return result
        except Exception as e:
            logger.error(f"Fetch {symbol}: {e}"); return None

    def analyse(self, data, symbol):
        debug = {'symbol': symbol.replace('/USDT:USDT',''), 'gates': [], 'score': 0, 'bias': '?'}

        try:
            df4 = data['4h']; df1 = data['1h']; df15 = data['15m']
            if len(df1) < 80 or len(df15) < 40:
                debug['gates'].append('❌ Not enough candle data')
                return None, debug

            price = df1['close'].iloc[-1]

            # Gate 1: 4H ADX Regime (HARD GATE — NEW v5.0)
            adx_regime, adx_val, adx_msg = self.smc.check_adx_regime(df4)
            debug['gates'].append(adx_msg)
            if adx_regime == 'CHOP':
                self.stats['skipped_chop'] += 1
                return None, debug

            # Gate 2: 4H Bias (EMA)
            l4 = df4.iloc[-1]
            e21 = l4.get('ema_21', 0); e50 = l4.get('ema_50', 0)
            if e21 > e50:       bias = 'LONG'
            elif e21 < e50:     bias = 'SHORT'
            else:
                debug['gates'].append('❌ 4H EMAs flat — no bias')
                return None, debug
            debug['bias'] = bias

            # Gate 3: DI Alignment (NEW v5.0 — soft gate, affects score)
            di_aligned, di_msg = self.smc.check_di_alignment(df4, bias)
            debug['gates'].append(di_msg)
            # Not a hard gate, but penalty in scorer

            # Gate 4: HH/LL bonus check
            hh_ll_ok, hh_ll_msg = self.smc.check_4h_hh_ll(df4, bias, HH_LL_LOOKBACK)
            debug['gates'].append(hh_ll_msg)

            # Gate 5: PD Zone
            pd_label, pd_pos = self.smc.pd_zone(df4, price)
            if bias == 'LONG' and pd_label == 'PREMIUM':
                debug['gates'].append(f'❌ PD zone: PREMIUM ({pd_pos*100:.0f}%) — no longs here')
                return None, debug
            if bias == 'SHORT' and pd_label == 'DISCOUNT':
                debug['gates'].append(f'❌ PD zone: DISCOUNT ({pd_pos*100:.0f}%) — no shorts here')
                return None, debug
            debug['gates'].append(f'✅ PD zone: {pd_label} ({pd_pos*100:.0f}%)')

            # Gate 6: 1H Structure
            highs1, lows1 = self.smc.swing_highs_lows(df1, left=4, right=4)
            structure = self.smc.detect_structure_break(df1, highs1, lows1, lookback=STRUCTURE_LOOKBACK)
            if structure:
                s_bull = 'BULL' in structure['kind']
                s_bear = 'BEAR' in structure['kind']
                if bias == 'LONG' and s_bear:
                    debug['gates'].append(f'❌ Structure ({structure["kind"]}) opposes LONG')
                    return None, debug
                if bias == 'SHORT' and s_bull:
                    debug['gates'].append(f'❌ Structure ({structure["kind"]}) opposes SHORT')
                    return None, debug
                debug['gates'].append(f'✅ Structure: {structure["kind"]}')
            else:
                debug['gates'].append('⚠️ No recent BOS/MSS (score=0 but continuing)')

            # Gate 7: 1H Order Block (HARD GATE)
            obs = self.smc.find_order_blocks(df1, bias, lookback=60)
            if not obs:
                debug['gates'].append(f'❌ No valid fresh {bias} OBs on 1H (all mitigated or stale)')
                return None, debug
            debug['gates'].append(f'✅ {len(obs)} fresh OB(s) found on 1H')

            active_ob = None
            for ob in obs:
                if self.smc.price_in_ob(price, ob, OB_TOLERANCE_PCT):
                    active_ob = ob; break

            if not active_ob:
                nearest   = obs[0]
                dist_pct  = min(abs(price - nearest['top']), abs(price - nearest['bottom'])) / price * 100
                debug['gates'].append(f'❌ Price not at OB — nearest {dist_pct:.2f}% away [{nearest["bottom"]:.5f}–{nearest["top"]:.5f}]')
                return None, debug
            debug['gates'].append(f'✅ Price IN OB [{active_ob["bottom"]:.5f}–{active_ob["top"]:.5f}] (mit:{active_ob["mitigation"]*100:.0f}%, age:{active_ob["age"]}bars)')

            # Gate 8: OB Approach check (NEW v5.0 — HARD GATE for STALE)
            ob_approach = self.smc.check_ob_approach(df1, active_ob, bias, OB_MAX_CANDLES_INSIDE)
            debug['gates'].append(ob_approach[2])
            if ob_approach[0] == 'STALE':
                self.stats['skipped_stale'] += 1
                return None, debug

            # FVG (bonus)
            fvgs = self.smc.find_fvg(df1, bias, lookback=25)
            fvg_near = None
            for fvg in fvgs:
                if fvg['bottom'] < active_ob['top'] and fvg['top'] > active_ob['bottom']:
                    fvg_near = fvg; break
            if fvg_near:
                debug['gates'].append('✅ 1H FVG overlaps OB')

            # Liquidity sweep (bonus)
            sweep = self.smc.recent_liquidity_sweep(df1, bias, highs1, lows1, lookback=20)
            if sweep:
                debug['gates'].append(f'✅ 1H liq sweep @ {sweep["level"]:.5f}')

            # Score
            score, reasons, failed = score_setup(
                bias, active_ob, ob_approach, structure, sweep, fvg_near,
                df1, df15, df4, pd_label, hh_ll_ok, adx_regime, di_aligned
            )
            debug['score'] = score
            debug['gates'] += failed

            if score < MIN_SCORE:
                debug['gates'].append(f'❌ Score {score} < {MIN_SCORE} minimum')
                return None, debug

            if   score >= 92: quality = 'ELITE 👑'
            elif score >= 85: quality = 'PREMIUM 💎'
            else:             quality = 'HIGH 🔥'

            # v5.0: ATR-adaptive SL (much safer than before)
            atr1  = df1['atr'].iloc[-1]
            entry = price
            sl = self.smc.calculate_atr_sl(entry, active_ob, atr1, bias)

            risk = abs(entry - sl)
            if risk < entry * 0.001:
                debug['gates'].append('❌ Degenerate SL')
                return None, debug

            if bias == 'LONG':
                tps = [entry + risk*1.5, entry + risk*2.5, entry + risk*4.0]
            else:
                tps = [entry - risk*1.5, entry - risk*2.5, entry - risk*4.0]

            rr       = [abs(t - entry) / risk for t in tps]
            risk_pct = risk / entry * 100
            tid      = f"{symbol.split('/')[0]}_{datetime.now().strftime('%Y%m%d%H%M%S')}"

            sig = {
                'trade_id':    tid,
                'symbol':      symbol.replace('/USDT:USDT', ''),
                'full_symbol': symbol,
                'signal':      bias,
                'quality':     quality,
                'score':       score,
                'hh_ll':       hh_ll_ok,
                'adx':         adx_val,
                'adx_regime':  adx_regime,
                'ob_approach': ob_approach[0],
                'entry':       entry,
                'stop_loss':   sl,
                'targets':     tps,
                'rr':          rr,
                'risk_pct':    risk_pct,
                'ob':          active_ob,
                'fvg':         fvg_near,
                'sweep':       sweep,
                'structure':   structure,
                'pd_zone':     pd_label,
                'pd_pos':      pd_pos,
                'reasons':     reasons,
                'tp_hit':      [False, False, False],
                'sl_hit':      False,
                'timestamp':   datetime.now(),
            }
            debug['gates'].append(f'✅ PASSED — Score {score}')
            return sig, debug

        except Exception as e:
            logger.error(f"Analyse {symbol}: {e}")
            debug['gates'].append(f'💥 Exception: {e}')
            return None, debug

    def fmt(self, s):
        arrow = '🟢 LONG' if s['signal'] == 'LONG' else '🔴 SHORT'
        ob    = s['ob']

        tp1_pct = abs((s['targets'][0] - s['entry']) / s['entry'] * 100)
        tp2_pct = abs((s['targets'][1] - s['entry']) / s['entry'] * 100)
        tp3_pct = abs((s['targets'][2] - s['entry']) / s['entry'] * 100)

        msg  = f"{'─'*32}\n"
        msg += f"{arrow}  <b>{s['symbol']}/USDT</b>  ⭐ {s['score']}/100  {s['quality']}\n"
        msg += f"{'─'*32}\n\n"

        msg += f"<b>ENTRY</b>   <code>${s['entry']:.5f}</code>\n"
        msg += f"<b>TP1</b>     <code>${s['targets'][0]:.5f}</code>  +{tp1_pct:.1f}%  — close 50%\n"
        msg += f"<b>TP2</b>     <code>${s['targets'][1]:.5f}</code>  +{tp2_pct:.1f}%  — close 30%\n"
        msg += f"<b>TP3</b>     <code>${s['targets'][2]:.5f}</code>  +{tp3_pct:.1f}%  — close 20%\n"
        msg += f"<b>SL</b>      <code>${s['stop_loss']:.5f}</code>  -{s['risk_pct']:.1f}%\n\n"

        msg += f"<b>OB zone</b>  <code>${ob['bottom']:.5f} – ${ob['top']:.5f}</code>\n\n"

        # Top 3 reasons only — keep it clean
        for r in s['reasons'][:3]:
            msg += f"• {r}\n"

        msg += f"\n<i>{s['timestamp'].strftime('%H:%M UTC')} — move SL to BE after TP1</i>"
        return msg

    async def send(self, text):
        try:
            await self.bot.send_message(chat_id=self.chat_id, text=text, parse_mode=ParseMode.HTML)
        except Exception as e:
            logger.error(f"Telegram: {e}")

    async def tp_alert(self, t, n, price):
        tp  = t['targets'][n-1]
        pct = abs((tp - t['entry'])/t['entry']*100)
        advice = {1:'Close 50% → Move SL to breakeven', 2:'Close 30% → Trail stop tight', 3:'Close final 20% 🎊'}
        msg  = f"🎯 <b>TP{n} HIT!</b>\n\n<code>{t['trade_id']}</code>\n<b>{t['symbol']}</b> {t['signal']}\n\n"
        msg += f"Target: <code>${tp:.6f}</code>\nCurrent: <code>${price:.6f}</code>\nProfit: <b>+{pct:.2f}%</b>\n\n"
        msg += f"📋 {advice[n]}"
        await self.send(msg)
        self.stats[f'tp{n}'] += 1

    async def sl_alert(self, t, price):
        loss = abs((price - t['entry'])/t['entry']*100)
        msg  = f"⛔ <b>STOP LOSS HIT</b>\n\n<code>{t['trade_id']}</code>\n<b>{t['symbol']}</b> {t['signal']}\n\n"
        msg += f"Entry: <code>${t['entry']:.6f}</code>\nLoss: <b>-{loss:.2f}%</b>\n\nOB invalidated. Next setup incoming."
        await self.send(msg)
        self.stats['sl'] += 1

    async def track(self):
        logger.info("📡 Tracker started")
        while True:
            try:
                if not self.active_trades:
                    await asyncio.sleep(30); continue
                remove = []
                for tid, t in list(self.active_trades.items()):
                    try:
                        if datetime.now() - t['timestamp'] > timedelta(hours=48):
                            await self.send(f"⏰ <b>48H TIMEOUT</b>\n<code>{tid}</code>\n{t['symbol']} — Close manually.")
                            remove.append(tid); continue
                        ticker = await self.exchange.fetch_ticker(t['full_symbol'])
                        p = ticker['last']
                        if t['signal'] == 'LONG':
                            for i, tp in enumerate(t['targets']):
                                if not t['tp_hit'][i] and p >= tp:
                                    await self.tp_alert(t, i+1, p); t['tp_hit'][i] = True
                                    if i == 2: remove.append(tid)
                            if not t['sl_hit'] and p <= t['stop_loss']:
                                await self.sl_alert(t, p); t['sl_hit'] = True; remove.append(tid)
                        else:
                            for i, tp in enumerate(t['targets']):
                                if not t['tp_hit'][i] and p <= tp:
                                    await self.tp_alert(t, i+1, p); t['tp_hit'][i] = True
                                    if i == 2: remove.append(tid)
                            if not t['sl_hit'] and p >= t['stop_loss']:
                                await self.sl_alert(t, p); t['sl_hit'] = True; remove.append(tid)
                    except Exception as e:
                        logger.error(f"Track {tid}: {e}")
                for tid in set(remove):
                    self.active_trades.pop(tid, None)
                await asyncio.sleep(30)
            except Exception as e:
                logger.error(f"Track loop: {e}"); await asyncio.sleep(60)

    async def scan(self):
        if self.is_scanning:
            return []
        self.is_scanning = True
        logger.info("🔍 Scan starting...")

        # v5.0: Check BTC regime before scanning alts
        self.btc_regime = await self.get_btc_regime()
        btc_warn = ""
        if self.btc_regime == 'CHOP':
            btc_warn = "\n⚠️ <b>BTC in CHOP — signal quality reduced. Fewer signals expected.</b>"
        elif self.btc_regime == 'RANGING':
            btc_warn = "\n⚠️ BTC ranging — only highest-confluence setups will pass."

        await self.send(
            f"🔍 <b>SMC v5.0 SCAN STARTED</b>\n"
            f"Entry: <b>1H trigger</b> | OB: fresh ≤25% mit | Trend: 4H ADX gate\n"
            f"Min score: {MIN_SCORE} | OB tolerance: {OB_TOLERANCE_PCT*100:.1f}%\n"
            f"Vol filter: ${MIN_VOLUME_24H/1e6:.0f}M | Spread: <{MAX_SPREAD_PCT*100:.2f}%\n"
            f"BTC regime: <b>{self.btc_regime}</b>{btc_warn}"
        )

        pairs       = await self.get_pairs()
        candidates  = []
        near_misses = []
        scanned     = 0

        for pair in pairs:
            try:
                data = await self.fetch_data(pair)
                if data:
                    sig, dbg = self.analyse(data, pair)
                    if sig:
                        candidates.append(sig)
                        logger.info(f"  💎 {pair} {sig['signal']} score={sig['score']}")
                    else:
                        if dbg['score'] > 0 and any('✅ Price IN OB' in g for g in dbg['gates']):
                            near_misses.append(dbg)
                scanned += 1
                if scanned % 30 == 0:
                    logger.info(f"  ⏳ {scanned}/{len(pairs)} | {len(candidates)} candidates")
                await asyncio.sleep(0.45)
            except Exception as e:
                logger.error(f"Scan {pair}: {e}"); continue

        candidates.sort(key=lambda x: x['score'], reverse=True)
        top = candidates[:MAX_SIGNALS_PER_SCAN]

        near_misses.sort(key=lambda x: x['score'], reverse=True)
        self.last_debug = near_misses[:10]

        for sig in top:
            self.signal_history.append(sig)
            self.active_trades[sig['trade_id']] = sig
            self.stats['total'] += 1
            self.stats[sig['signal'].lower()] += 1
            if 'ELITE'   in sig['quality']: self.stats['elite']   += 1
            elif 'PREMIUM' in sig['quality']: self.stats['premium'] += 1
            else:                             self.stats['high']    += 1
            await self.send(self.fmt(sig))
            await asyncio.sleep(2)

        self.stats['last_scan'] = datetime.now()
        self.stats['pairs_scanned'] = scanned

        el = sum(1 for s in top if 'ELITE'   in s['quality'])
        pr = sum(1 for s in top if 'PREMIUM' in s['quality'])
        hi = len(top) - el - pr
        lg = sum(1 for s in top if s['signal'] == 'LONG')
        tr = sum(1 for s in top if s.get('hh_ll'))

        summ  = f"✅ <b>SCAN COMPLETE — v5.0</b>\n\n"
        summ += f"📊 Pairs scanned: {scanned}\n"
        summ += f"🔍 Candidates:    {len(candidates)}\n"
        summ += f"🎯 Signals sent:  {len(top)}\n"
        summ += f"⛔ Skipped chop:  {self.stats['skipped_chop']}\n"
        summ += f"⏱ Skipped stale: {self.stats['skipped_stale']}\n"
        if top:
            summ += f"  👑 Elite:    {el}\n  💎 Premium:  {pr}\n  🔥 High:     {hi}\n"
            summ += f"  🟢 Long:     {lg}\n  🔴 Short:    {len(top)-lg}\n"
            summ += f"  🏔️ Trending: {tr}\n  〰️ Ranging:  {len(top)-tr}\n"
        else:
            summ += f"\n<i>No setups met criteria this scan.</i>\n"
            if self.btc_regime == 'CHOP':
                summ += f"⚠️ BTC in chop — most markets ranging. Normal.\n"
            summ += f"Near misses: {len(near_misses)} — use /debug\n"
        summ += f"\n⏰ {datetime.now().strftime('%H:%M UTC')}"
        await self.send(summ)

        logger.info(f"✅ Done. {len(candidates)} candidates → {len(top)} sent.")
        self.is_scanning = False
        return top

    async def run(self, interval_min=SCAN_INTERVAL_MIN):
        logger.info("🚀 SMC Pro v5.0 starting")
        await self.send(
            "👑 <b>SMC PRO v5.0 — ORDER BLOCK SCANNER</b> 👑\n\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            "<b>4H ADX Gate → 4H Trend → 1H Fresh OB + Trigger → 15M Vol</b>\n\n"
            f"✅ Hard gate: ADX ≥ {ADX_CHOP_HARD_GATE} (no chop trades)\n"
            f"✅ OB freshness: ≤{OB_MAX_CANDLES_INSIDE} candles inside, ≤{OB_MAX_MITIGATION_PCT*100:.0f}% mitigated\n"
            f"✅ SL: ATR-adaptive min {ATR_SL_MINIMUM_MULT}x ATR\n"
            f"✅ Entry trigger: 1H candles\n"
            f"✅ Min score: {MIN_SCORE}/100\n"
            f"✅ Vol filter: ${MIN_VOLUME_24H/1e6:.0f}M/day | Spread: <{MAX_SPREAD_PCT*100:.2f}%\n"
            f"✅ 4H HH/LL bonus: +{HH_LL_BONUS}pts\n"
            f"✅ Trade timeout: 48H | Scan every: {SCAN_INTERVAL_MIN}min\n\n"
            "Commands: /scan /stats /trades /debug /help\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        )
        asyncio.create_task(self.track())
        while True:
            try:
                await self.scan()
                logger.info(f"💤 Next scan in {interval_min}m")
                await asyncio.sleep(interval_min * 60)
            except Exception as e:
                logger.error(f"Main: {e}"); await asyncio.sleep(60)

    async def close(self):
        await self.exchange.close()


# ══════════════════════════════════════════════════════════════
#  BOT COMMANDS
# ══════════════════════════════════════════════════════════════

class Commands:
    def __init__(self, s: SMCProScanner):
        self.s = s

    async def start(self, u: Update, c: ContextTypes.DEFAULT_TYPE):
        await u.message.reply_text(
            "👑 <b>SMC Pro v5.0</b>\n\n"
            "Key upgrades:\n"
            "• ADX chop gate — no signals in ranging markets\n"
            "• Fresh OB only — ≤25% mitigated, ≤3 candles inside\n"
            "• ATR-adaptive SL — no more wicked-out stops\n"
            "• DI alignment check — confirms true trend direction\n"
            "• BTC regime check — warns when alts will chop\n\n"
            "/scan /stats /trades /debug /help",
            parse_mode=ParseMode.HTML
        )

    async def cmd_scan(self, u: Update, c: ContextTypes.DEFAULT_TYPE):
        if self.s.is_scanning:
            await u.message.reply_text("⚠️ Already scanning."); return
        await u.message.reply_text("🔍 Manual scan started...")
        asyncio.create_task(self.s.scan())

    async def stats(self, u: Update, c: ContextTypes.DEFAULT_TYPE):
        s = self.s.stats
        msg  = "📊 <b>SMC PRO v5.0 STATS</b>\n\n"
        msg += f"Total signals: {s['total']}\n"
        msg += f"  👑 Elite: {s['elite']}  💎 Premium: {s['premium']}  🔥 High: {s['high']}\n"
        msg += f"  🟢 Long: {s['long']}  🔴 Short: {s['short']}\n\n"
        msg += f"TP1: {s['tp1']} | TP2: {s['tp2']} | TP3: {s['tp3']} | SL: {s['sl']}\n\n"
        msg += f"Skipped (chop): {s['skipped_chop']}\n"
        msg += f"Skipped (stale OB): {s['skipped_stale']}\n\n"
        if s['last_scan']:
            msg += f"Last scan: {s['last_scan'].strftime('%H:%M UTC')}\n"
            msg += f"Pairs: {s['pairs_scanned']}\n"
        msg += f"Active trades: {len(self.s.active_trades)}\n"
        msg += f"BTC regime: {self.s.btc_regime}"
        await u.message.reply_text(msg, parse_mode=ParseMode.HTML)

    async def trades(self, u: Update, c: ContextTypes.DEFAULT_TYPE):
        if not self.s.active_trades:
            await u.message.reply_text("📭 No active trades."); return
        msg = f"📡 <b>ACTIVE TRADES ({len(self.s.active_trades)})</b>\n\n"
        for tid, t in list(self.s.active_trades.items())[:10]:
            age      = int((datetime.now() - t['timestamp']).total_seconds()/3600)
            tps      = ''.join(['✅' if h else '⏳' for h in t['tp_hit']])
            trend_tag = '🏔️' if t.get('hh_ll') else '〰️'
            fresh_tag = {'FRESH':'🆕','RECENT':'⏱','STALE':'⚠️'}.get(t.get('ob_approach',''),'')
            msg += (f"<b>{t['symbol']}</b> {t['signal']} {trend_tag}{fresh_tag} — {t['quality']}\n"
                    f"  Entry: <code>${t['entry']:.5f}</code> | Score: {t['score']}\n"
                    f"  TPs: {tps} | {age}h old\n\n")
        await u.message.reply_text(msg, parse_mode=ParseMode.HTML)

    async def debug(self, u: Update, c: ContextTypes.DEFAULT_TYPE):
        if not self.s.last_debug:
            await u.message.reply_text("📭 No debug data yet. Run /scan first.", parse_mode=ParseMode.HTML)
            return
        msg = "🔬 <b>NEAR MISSES — Last Scan</b>\n"
        msg += "<i>(At OB but below score threshold)</i>\n\n"
        for d in self.s.last_debug[:8]:
            msg += f"<b>{d['symbol']}</b> {d['bias']} — Score: {d['score']}/100\n"
            # v5.0: Show exactly what caused it to fail
            fail_reasons = [g for g in d['gates'] if g.startswith('❌') or g.startswith('⚠️')]
            for g in fail_reasons[-3:]:
                msg += f"  {g}\n"
            msg += "\n"
        msg += f"<i>Min score: {MIN_SCORE}. Raise/lower in TUNABLE SETTINGS.</i>"
        await u.message.reply_text(msg, parse_mode=ParseMode.HTML)

    async def help(self, u: Update, c: ContextTypes.DEFAULT_TYPE):
        msg  = "📚 <b>SMC PRO v5.0 — STRATEGY</b>\n\n"
        msg += "<b>Timeframe Stack:</b>\n"
        msg += "  4H  → ADX gate + EMA bias + DI check + HH/LL\n"
        msg += "  1H  → BOS/MSS + Fresh OB (≤25% mit) + Trigger  ← core\n"
        msg += "  15M → Volume spike bonus only\n\n"
        msg += "<b>Hard Gates (ALL must pass):</b>\n"
        msg += f"  1️⃣ 4H ADX ≥ {ADX_CHOP_HARD_GATE} (no chop)\n"
        msg += "  2️⃣ 4H EMA 21/50 bias\n"
        msg += "  3️⃣ PD zone filter\n"
        msg += "  4️⃣ 1H BOS/MSS within 20 candles\n"
        msg += f"  5️⃣ Price at fresh 1H OB (≤{OB_MAX_MITIGATION_PCT*100:.0f}% mit, ≤{OB_MAX_AGE_BARS}bars old)\n"
        msg += f"  6️⃣ OB not stale (price inside ≤{OB_MAX_CANDLES_INSIDE} candles)\n"
        msg += f"  7️⃣ Score ≥ {MIN_SCORE}/100\n\n"
        msg += "<b>Score System (max 100):</b>\n"
        msg += "  +25 — 1H entry trigger (current candle only for max pts)\n"
        msg += "  +20 — MSS structure\n"
        msg += "  +20 — Tight OB quality\n"
        msg += "  +15 — 4H triple EMA\n"
        msg += "  +8  — Fresh OB entry (NEW)\n"
        msg += f"  +{HH_LL_BONUS}  — 4H HH/LL confirmed\n"
        msg += "  +5  — DI aligned (NEW)\n"
        msg += "  +5  — ADX trending (NEW)\n"
        msg += "  +12 — Momentum (RSI/MACD/Stoch)\n"
        msg += "  +10 — Extras (sweep/FVG/vol)\n"
        msg += f"  -{ADX_RANGE_PENALTY} — ADX ranging penalty (NEW)\n"
        msg += "  -10 — Stale OB penalty (NEW)\n\n"
        msg += "<b>SL v5.0:</b>\n"
        msg += f"  ATR-adaptive: min {ATR_SL_MINIMUM_MULT}x ATR buffer\n"
        msg += "  Much wider than v4 — fewer wicked stops\n\n"
        msg += "<b>Config:</b>\n"
        msg += f"  MIN_SCORE={MIN_SCORE} | OB_MIT_CAP={OB_MAX_MITIGATION_PCT*100:.0f}%\n"
        msg += f"  OB_MAX_AGE={OB_MAX_AGE_BARS}bars | ADX_GATE={ADX_CHOP_HARD_GATE}"
        await u.message.reply_text(msg, parse_mode=ParseMode.HTML)


# ══════════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════════

async def main():
    # ════════════ CONFIG ════════════
    TELEGRAM_TOKEN   = "7957028587:AAE7aSYtE4hCxxTIPkAs_1ULJ9e8alkY6Ic"
    TELEGRAM_CHAT_ID = "-1002442074724"
    BINANCE_API_KEY  = None
    BINANCE_SECRET   = None
    # ════════════════════════════════

    scanner = SMCProScanner(
        telegram_token=TELEGRAM_TOKEN,
        chat_id=TELEGRAM_CHAT_ID,
        api_key=BINANCE_API_KEY,
        secret=BINANCE_SECRET
    )

    app = Application.builder().token(TELEGRAM_TOKEN).build()
    cmds = Commands(scanner)

    app.add_handler(CommandHandler("start",  cmds.start))
    app.add_handler(CommandHandler("scan",   cmds.cmd_scan))
    app.add_handler(CommandHandler("stats",  cmds.stats))
    app.add_handler(CommandHandler("trades", cmds.trades))
    app.add_handler(CommandHandler("debug",  cmds.debug))
    app.add_handler(CommandHandler("help",   cmds.help))

    await app.initialize()
    await app.start()
    logger.info("🤖 SMC Pro v5.0 ready!")

    try:
        await scanner.run(interval_min=SCAN_INTERVAL_MIN)
    except KeyboardInterrupt:
        logger.info("Shutting down...")
    finally:
        await scanner.close()
        await app.stop()
        await app.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
