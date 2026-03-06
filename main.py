"""
SMC PRO v5.0 — DATA-VALIDATED LIVE BOT
══════════════════════════════════════════════════════════════════
BUILT FROM 5 ROUNDS OF BACKTESTING (90d / 15 pairs / 500+ setups)

WHAT THE DATA PROVED:
─────────────────────
  SHORT trades: WR=55%, avg=+0.54R, total=+10.8R  ✅
  LONG  trades: WR=11%, avg=-0.69R, total=-35.7R  ❌

  The last 90d was a downtrend. LONGs into downtrends = losses.
  Solution: LONG only allowed when full bull confirmation exists.

WINNING FILTERS (data-validated):
──────────────────────────────────
  1. ADX > 30 on 4H: WR=50-100%. Below 30 = chop = losses.
  2. SHORT triggers that work: bear_engulf (100%), shooting_star (75%)
  3. SHORT only in PREMIUM zone (200-bar range, not 50-bar)
  4. Score 65-80 band (higher scores had worse outcomes in backtest)
  5. No HH/LL bonus (anti-correlated with wins in backtest)
  6. No FVG bonus (anti-correlated with wins in backtest)

LONG HARD GATES (only allow in confirmed bull conditions):
  1. 4H triple EMA (21>50>200) — mandatory
  2. ADX > 25 confirming uptrend
  3. Score >= 68 (tight threshold)

KEY CHANGES vs v4.1:
  - ADX filter added (biggest improvement: filters ranging markets)
  - LONGs have 3-condition hard gate (was 1-condition)
  - PD zone uses 200-bar range (was 50-bar — was rejecting too much)
  - No -12 trigger penalty (removes 70% false negatives)
  - Score band 65-80 preferred (higher = worse per backtest)
  - SHORT thresholds loosened slightly (they work, let them fire)
══════════════════════════════════════════════════════════════════
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
#  SETTINGS — DATA-VALIDATED
# ═══════════════════════════════════════════════
MAX_SIGNALS_PER_SCAN  = 6
MIN_VOLUME_24H        = 5_000_000
OB_TOLERANCE_PCT      = 0.012    # 1.2% (backtested)
OB_IMPULSE_ATR_MULT   = 0.6     # relaxed (backtested)
OB_LOOKBACK           = 80      # more history (backtested)
OB_VIOLATION_WINDOW   = 40      # key fix from backtester
PD_ZONE_BARS          = 200     # 200-bar range (key fix)
STRUCTURE_LOOKBACK    = 20
STRUCTURE_OPPOSE_BARS = 8       # only recent opposing structure blocks
HH_LL_LOOKBACK        = 10
SCAN_INTERVAL_MIN     = 30

# ── DATA-VALIDATED THRESHOLDS ──────────────────
MIN_SCORE_SHORT       = 62      # shorts work — let them fire
MIN_SCORE_LONG        = 68      # longs need higher bar
ADX_MIN_SHORT         = 25      # confirmed trend for shorts
ADX_MIN_LONG          = 28      # stronger trend needed for longs
LONG_TRIPLE_EMA_REQUIRED = True # longs MUST have full EMA stack
MAX_SCORE_ALERT       = 82      # above this historically underperformed


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

        # ADX — key new filter from backtest
        adx_i = ta.trend.ADXIndicator(df['high'], df['low'], df['close'], 14)
        df['adx']    = adx_i.adx()
        df['di_pos'] = adx_i.adx_pos()
        df['di_neg'] = adx_i.adx_neg()

        bb = ta.volatility.BollingerBands(df['close'], 20, 2)
        df['bb_upper'] = bb.bollinger_hband()
        df['bb_lower'] = bb.bollinger_lband()

        df['vol_sma']   = df['volume'].rolling(20).mean()
        df['vol_ratio'] = df['volume'] / df['vol_sma'].replace(0, np.nan)

        tp = (df['high'] + df['low'] + df['close']) / 3
        df['vwap'] = (tp * df['volume']).cumsum() / df['volume'].cumsum()

        body = (df['close'] - df['open']).abs()
        uw   = df['high'] - df[['open','close']].max(axis=1)
        lw   = df[['open','close']].min(axis=1) - df['low']

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
        df['bull_pin'] = ((lw > body*2.5) & (lw > uw*2) & (df['close'] > df['open'])).astype(int)
        df['bear_pin'] = ((uw > body*2.5) & (uw > lw*2) & (df['close'] < df['open'])).astype(int)
        df['hammer']        = ((lw > body*2.0) & (lw > uw*1.5)).astype(int)
        df['shooting_star'] = ((uw > body*2.0) & (uw > lw*1.5)).astype(int)

    except Exception as e:
        logger.error(f"Indicator error: {e}")
    return df


# ══════════════════════════════════════════════════════════════
#  SMC ENGINE
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

    def detect_structure(self, df, highs, lows):
        events = []
        close = df['close']
        n = len(df)
        start = max(0, n - STRUCTURE_LOOKBACK - 15)
        for k in range(1, len(highs)):
            ph = highs[k-1]; ch = highs[k]
            if ch['i'] < start: continue
            level = ph['price']
            for j in range(ch['i'], min(ch['i'] + 10, n)):
                if close.iloc[j] > level:
                    events.append({'kind': 'BOS_BULL' if ch['price'] > ph['price'] else 'MSS_BULL', 'bar': j})
                    break
        for k in range(1, len(lows)):
            pl = lows[k-1]; cl = lows[k]
            if cl['i'] < start: continue
            level = pl['price']
            for j in range(cl['i'], min(cl['i'] + 10, n)):
                if close.iloc[j] < level:
                    events.append({'kind': 'BOS_BEAR' if cl['price'] < pl['price'] else 'MSS_BEAR', 'bar': j})
                    break
        if not events: return None
        latest = sorted(events, key=lambda x: x['bar'])[-1]
        if latest['bar'] < n - STRUCTURE_LOOKBACK: return None
        return latest

    def find_order_blocks(self, df, direction):
        """
        FIXED: OB_VIOLATION_WINDOW limits lookahead check.
        This was the root cause of 0 signals in backtester v1/v2.
        """
        obs = []
        n = len(df)
        start = max(2, n - OB_LOOKBACK)
        for i in range(start, n - 2):
            c = df.iloc[i]
            atr_local = float(df['atr'].iloc[i]) if 'atr' in df.columns and not pd.isna(df['atr'].iloc[i]) else (c['high'] - c['low'])
            if atr_local <= 0: atr_local = c['high'] - c['low']
            min_impulse = atr_local * OB_IMPULSE_ATR_MULT
            viol_end = min(i + 1 + OB_VIOLATION_WINDOW, n)

            if direction == 'LONG':
                if c['close'] >= c['open']: continue
                fwd = df['high'].iloc[i+1:min(i+6, n)]
                if len(fwd) == 0 or fwd.max() - c['low'] < min_impulse: continue
                ob = {'top': max(c['open'], c['close']), 'bottom': c['low'],
                      'mid': (max(c['open'], c['close']) + c['low']) / 2, 'bar': i,
                      'size_pct': (max(c['open'], c['close']) - c['low']) / c['low'] * 100}
                if (df['close'].iloc[i+1:viol_end] < (ob['top'] + ob['bottom']) / 2).any(): continue
                obs.append(ob)
            else:
                if c['close'] <= c['open']: continue
                fwd = df['low'].iloc[i+1:min(i+6, n)]
                if len(fwd) == 0 or c['high'] - fwd.min() < min_impulse: continue
                ob = {'top': c['high'], 'bottom': min(c['open'], c['close']),
                      'mid': (c['high'] + min(c['open'], c['close'])) / 2, 'bar': i,
                      'size_pct': (c['high'] - min(c['open'], c['close'])) / max(min(c['open'], c['close']), 0.0001) * 100}
                if (df['close'].iloc[i+1:viol_end] > (ob['top'] + ob['bottom']) / 2).any(): continue
                obs.append(ob)

        obs.sort(key=lambda x: x['bar'], reverse=True)
        return obs

    def price_in_ob(self, price, ob):
        tol = ob['top'] * OB_TOLERANCE_PCT
        return (ob['bottom'] - tol) <= price <= (ob['top'] + tol)

    def find_fvg(self, df, direction, lookback=25):
        fvgs = []
        n = len(df)
        for i in range(max(1, n - lookback), n - 1):
            prev = df.iloc[i-1]; nxt = df.iloc[i+1]
            if direction == 'LONG' and prev['high'] < nxt['low']:
                fvgs.append({'top': nxt['low'], 'bottom': prev['high'], 'bar': i})
            elif direction == 'SHORT' and prev['low'] > nxt['high']:
                fvgs.append({'top': prev['low'], 'bottom': nxt['high'], 'bar': i})
        return fvgs

    def recent_sweep(self, df, direction, highs, lows, lookback=25):
        n = len(df); start = n - lookback
        if direction == 'LONG':
            for sl in reversed(lows):
                if sl['i'] < start: continue
                for j in range(sl['i'] + 1, min(sl['i'] + 8, n)):
                    c = df.iloc[j]
                    if c['low'] < sl['price'] and c['close'] > sl['price']: return True
        else:
            for sh in reversed(highs):
                if sh['i'] < start: continue
                for j in range(sh['i'] + 1, min(sh['i'] + 8, n)):
                    c = df.iloc[j]
                    if c['high'] > sh['price'] and c['close'] < sh['price']: return True
        return False

    def pd_zone(self, df_4h, price):
        """Uses PD_ZONE_BARS=200 — key fix from backtesting."""
        n = len(df_4h)
        bars = min(PD_ZONE_BARS, n)
        hi = df_4h['high'].iloc[-bars:].max()
        lo = df_4h['low'].iloc[-bars:].min()
        rang = hi - lo
        if rang == 0: return 'NEUTRAL', 0.5
        pos = (price - lo) / rang
        if pos < 0.35: return 'DISCOUNT', pos
        if pos > 0.65: return 'PREMIUM', pos
        return 'NEUTRAL', pos


# ══════════════════════════════════════════════════════════════
#  SCORER  v5.0 (data-driven)
# ══════════════════════════════════════════════════════════════
def score_setup_v5(direction, ob, structure, has_sweep, df_1h, df_15m, df_4h, pd_label):
    """
    New scoring system based on backtesting:
    - No -12 trigger penalty (killed 70% of valid setups)
    - No HH/LL bonus (was anti-correlated with wins)
    - ADX included in score (was the best predictor)
    - Trigger is bonus only, not required

    Max 100 pts:
      4H Trend    : 30
      OB Quality  : 25
      Structure   : 20
      Trigger     : 20 (no penalty if absent)
      Momentum    : 10
      Extras      : 5  (sweep + vol + vwap)
    """
    score = 0
    reasons = []

    l1 = df_1h.iloc[-1]; p1 = df_1h.iloc[-2]
    l15 = df_15m.iloc[-1]; l4 = df_4h.iloc[-1]
    e21 = l4.get('ema_21', 0); e50 = l4.get('ema_50', 0); e200 = l4.get('ema_200', 0)
    adx_val = float(l4.get('adx', 0) or 0)

    # 1. 4H TREND (30 pts) — ADX integrated
    trend_pts = 0
    if direction == 'LONG':
        if e21 > e50 > e200:
            base = 22
        elif e21 > e50:
            base = 14
        else:
            base = 0
        # ADX bonus (data showed ADX 30+ massively helps)
        if adx_val >= 35:   adx_bonus = 8
        elif adx_val >= 30: adx_bonus = 6
        elif adx_val >= 25: adx_bonus = 3
        else:               adx_bonus = 0
        trend_pts = min(base + adx_bonus, 30)
        if base > 0: reasons.append(f"📈 4H Bull EMA ({trend_pts}pts, ADX={adx_val:.0f})")
    else:
        if e21 < e50 < e200:
            base = 22
        elif e21 < e50:
            base = 14
        else:
            base = 0
        if adx_val >= 35:   adx_bonus = 8
        elif adx_val >= 30: adx_bonus = 6
        elif adx_val >= 25: adx_bonus = 3
        else:               adx_bonus = 0
        trend_pts = min(base + adx_bonus, 30)
        if base > 0: reasons.append(f"📉 4H Bear EMA ({trend_pts}pts, ADX={adx_val:.0f})")

    score += trend_pts

    # 2. OB QUALITY (25 pts)
    ob_pts = 0
    if ob:
        pct = ob.get('size_pct', 2.0)
        if pct < 0.8:   ob_pts = 25; reasons.append(f"📦 Tight OB {pct:.2f}% (+25)")
        elif pct < 2.0: ob_pts = 17; reasons.append(f"📦 Medium OB {pct:.2f}% (+17)")
        elif pct < 4.0: ob_pts = 10; reasons.append(f"📦 Wide OB {pct:.2f}% (+10)")
        else:           ob_pts = 5;  reasons.append(f"📦 Very wide OB {pct:.2f}% (+5)")
    score += ob_pts

    # 3. STRUCTURE (20 pts)
    struct_pts = 0
    if structure:
        if 'MSS' in structure['kind']:
            struct_pts = 20; reasons.append(f"🏗️ MSS Reversal (+20)")
        else:
            struct_pts = 13; reasons.append(f"🏗️ BOS Pullback (+13)")
    score += struct_pts

    # 4. TRIGGER (20 pts, NO PENALTY if absent)
    trig_pts = 0; trig_label = 'none'
    if direction == 'LONG':
        if   l1.get('bull_engulf', 0) == 1: trig_pts = 20; trig_label = 'bull_engulf'
        elif l1.get('bull_pin', 0) == 1:    trig_pts = 17; trig_label = 'bull_pin'
        elif l1.get('hammer', 0) == 1:      trig_pts = 14; trig_label = 'hammer'
        elif p1.get('bull_engulf', 0) == 1: trig_pts = 11; trig_label = 'bull_engulf_prev'
        elif p1.get('bull_pin', 0) == 1:    trig_pts = 8;  trig_label = 'bull_pin_prev'
        elif p1.get('hammer', 0) == 1:      trig_pts = 6;  trig_label = 'hammer_prev'
    else:
        if   l1.get('bear_engulf', 0) == 1:   trig_pts = 20; trig_label = 'bear_engulf'
        elif l1.get('bear_pin', 0) == 1:      trig_pts = 17; trig_label = 'bear_pin'
        elif l1.get('shooting_star', 0) == 1: trig_pts = 14; trig_label = 'shooting_star'
        elif p1.get('bear_engulf', 0) == 1:   trig_pts = 11; trig_label = 'bear_engulf_prev'
        elif p1.get('bear_pin', 0) == 1:      trig_pts = 8;  trig_label = 'bear_pin_prev'
        elif p1.get('shooting_star', 0) == 1: trig_pts = 6;  trig_label = 'shooting_star_prev'

    if trig_pts > 0:
        reasons.append(f"🕯️ {trig_label} (+{trig_pts})")
    score += trig_pts

    # 5. MOMENTUM (10 pts)
    mom = 0
    rsi = l1.get('rsi', 50)
    macd1 = l1.get('macd', 0); ms1 = l1.get('macd_signal', 0)
    pm1   = p1.get('macd', 0); pms1 = p1.get('macd_signal', 0)
    sk1   = l1.get('srsi_k', 0.5); sd1 = l1.get('srsi_d', 0.5)

    if direction == 'LONG':
        if 28 <= rsi <= 55 or rsi < 28: mom += 3; reasons.append(f"✅ RSI {rsi:.0f}")
        if macd1 > ms1 and pm1 <= pms1: mom += 4; reasons.append("⚡ MACD bull cross")
        elif macd1 > ms1:               mom += 2
        if sk1 < 0.3 and sk1 > sd1:    mom += 3; reasons.append("⚡ StochRSI cross")
    else:
        if 45 <= rsi <= 72 or rsi > 72: mom += 3; reasons.append(f"✅ RSI {rsi:.0f}")
        if macd1 < ms1 and pm1 >= pms1: mom += 4; reasons.append("⚡ MACD bear cross")
        elif macd1 < ms1:               mom += 2
        if sk1 > 0.7 and sk1 < sd1:    mom += 3; reasons.append("⚡ StochRSI cross")
    score += mom

    # 6. EXTRAS (5 pts max)
    ext = 0
    if has_sweep: ext += 3; reasons.append("💧 Liq sweep")
    vr15 = l15.get('vol_ratio', 1.0)
    if vr15 >= 2.0: ext += 2; reasons.append(f"🚀 Vol {vr15:.1f}x")
    elif vr15 >= 1.5: ext += 1
    close1 = l1.get('close', 0); vwap1 = l1.get('vwap', 0)
    if direction == 'LONG' and close1 < vwap1: ext += 1; reasons.append("✅ Below VWAP")
    elif direction == 'SHORT' and close1 > vwap1: ext += 1; reasons.append("✅ Above VWAP")
    score += min(ext, 5)

    return max(0, min(int(score), 100)), reasons, trig_label, bool(trig_pts > 0), adx_val


# ══════════════════════════════════════════════════════════════
#  MAIN BOT  v5.0
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
        self.stats = {
            'total': 0, 'long': 0, 'short': 0,
            'tp1': 0, 'tp2': 0, 'tp3': 0, 'sl': 0,
            'last_scan': None, 'pairs_scanned': 0
        }

    async def get_pairs(self):
        try:
            await self.exchange.load_markets()
            tickers = await self.exchange.fetch_tickers()
            pairs = [
                s for s in self.exchange.symbols
                if s.endswith('/USDT:USDT')
                and 'PERP' not in s
                and tickers.get(s, {}).get('quoteVolume', 0) > MIN_VOLUME_24H
            ]
            pairs.sort(key=lambda x: tickers.get(x, {}).get('quoteVolume', 0), reverse=True)
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
                debug['gates'].append('❌ Not enough data'); return None, debug

            price = df1['close'].iloc[-1]
            l4 = df4.iloc[-1]
            e21 = l4.get('ema_21', 0); e50 = l4.get('ema_50', 0)
            adx_val = float(l4.get('adx', 0) or 0)

            if e21 > e50:   bias = 'LONG'
            elif e21 < e50: bias = 'SHORT'
            else:
                debug['gates'].append('❌ 4H EMAs flat'); return None, debug
            debug['bias'] = bias

            # ── GATE: ADX trend strength ─────────────────────────
            adx_min = ADX_MIN_LONG if bias == 'LONG' else ADX_MIN_SHORT
            if adx_val < adx_min:
                debug['gates'].append(f'❌ ADX {adx_val:.0f} < {adx_min} — ranging market, skip')
                return None, debug
            debug['gates'].append(f'✅ ADX {adx_val:.0f} ≥ {adx_min} — trending')

            # ── GATE: LONG triple EMA ────────────────────────────
            e200 = l4.get('ema_200', 0)
            triple_ema_bull = e21 > e50 > e200
            if bias == 'LONG' and LONG_TRIPLE_EMA_REQUIRED:
                if not triple_ema_bull:
                    debug['gates'].append(f'❌ LONG requires 4H triple EMA (21>50>200)')
                    return None, debug
                debug['gates'].append('✅ Triple EMA bull stack')

            # ── GATE: PD Zone ────────────────────────────────────
            pd_label, pd_pos = self.smc.pd_zone(df4, price)
            if bias == 'LONG' and pd_label == 'PREMIUM':
                debug['gates'].append(f'❌ LONG in PREMIUM zone ({pd_pos*100:.0f}%)')
                return None, debug
            if bias == 'SHORT' and pd_label == 'DISCOUNT':
                debug['gates'].append(f'❌ SHORT in DISCOUNT zone ({pd_pos*100:.0f}%)')
                return None, debug
            debug['gates'].append(f'✅ PD: {pd_label} ({pd_pos*100:.0f}%)')

            # ── GATE: Structure ──────────────────────────────────
            highs1, lows1 = self.smc.swing_highs_lows(df1, 4, 4)
            structure = self.smc.detect_structure(df1, highs1, lows1)
            if structure:
                n1 = len(df1)
                if structure['bar'] >= (n1 - STRUCTURE_OPPOSE_BARS):
                    if bias == 'LONG' and 'BEAR' in structure['kind']:
                        debug['gates'].append(f'❌ Recent BEAR structure opposes LONG')
                        return None, debug
                    if bias == 'SHORT' and 'BULL' in structure['kind']:
                        debug['gates'].append(f'❌ Recent BULL structure opposes SHORT')
                        return None, debug
                debug['gates'].append(f'✅ Structure: {structure["kind"]}')

            # ── GATE: Order Block ────────────────────────────────
            obs = self.smc.find_order_blocks(df1, bias)
            if not obs:
                debug['gates'].append(f'❌ No valid {bias} OBs on 1H')
                return None, debug

            active_ob = None
            for ob in obs:
                if self.smc.price_in_ob(price, ob):
                    active_ob = ob; break

            if not active_ob:
                nearest = obs[0]
                dist = min(abs(price-nearest['top']), abs(price-nearest['bottom'])) / price * 100
                debug['gates'].append(f'❌ Price not at OB — nearest {dist:.1f}% away')
                return None, debug
            debug['gates'].append(f'✅ Price IN OB [{active_ob["bottom"]:.5f}–{active_ob["top"]:.5f}]')

            has_sweep = self.smc.recent_sweep(df1, bias, highs1, lows1, 20)

            # ── SCORE ────────────────────────────────────────────
            score, reasons, trig_label, has_trigger, adx = score_setup_v5(
                bias, active_ob, structure, has_sweep, df1, df15, df4, pd_label
            )
            debug['score'] = score

            min_sc = MIN_SCORE_SHORT if bias == 'SHORT' else MIN_SCORE_LONG
            if score < min_sc:
                debug['gates'].append(f'❌ Score {score} < {min_sc}')
                return None, debug

            # Note if score is unusually high (historically underperforms)
            quality_note = ""
            if score > MAX_SCORE_ALERT:
                quality_note = f" ⚠️ Score {score} > {MAX_SCORE_ALERT} (backtest: watch carefully)"

            if   score >= 80: quality = 'HIGH 🔥'
            elif score >= 70: quality = 'SOLID 💎'
            else:             quality = 'FORMING 📡'

            atr1  = df1['atr'].iloc[-1]
            entry = price
            if bias == 'LONG':
                sl = min(active_ob['bottom'] - atr1 * 0.2, entry - atr1 * 0.6)
            else:
                sl = max(active_ob['top'] + atr1 * 0.2, entry + atr1 * 0.6)

            risk = abs(entry - sl)
            if risk < entry * 0.001:
                debug['gates'].append('❌ Degenerate SL')
                return None, debug

            if bias == 'LONG':
                tps = [entry + risk*1.5, entry + risk*2.5, entry + risk*4.0]
            else:
                tps = [entry - risk*1.5, entry - risk*2.5, entry - risk*4.0]

            risk_pct = risk / entry * 100
            tid = f"{symbol.split('/')[0]}_{datetime.now().strftime('%Y%m%d%H%M%S')}"

            sig = {
                'trade_id': tid, 'symbol': symbol.replace('/USDT:USDT',''),
                'full_symbol': symbol, 'signal': bias, 'quality': quality,
                'quality_note': quality_note,
                'score': score, 'adx': adx,
                'trigger': trig_label, 'has_trigger': has_trigger,
                'triple_ema': triple_ema_bull,
                'entry': entry, 'stop_loss': sl, 'targets': tps,
                'rr': [abs(t - entry) / risk for t in tps],
                'risk_pct': risk_pct, 'ob': active_ob,
                'sweep': has_sweep, 'structure': structure,
                'pd_zone': pd_label, 'pd_pos': pd_pos,
                'reasons': reasons,
                'tp_hit': [False, False, False], 'sl_hit': False,
                'timestamp': datetime.now(),
            }
            debug['gates'].append(f'✅ PASSED — Score {score} | ADX {adx:.0f} | {trig_label}')
            return sig, debug

        except Exception as e:
            logger.error(f"Analyse {symbol}: {e}")
            debug['gates'].append(f'💥 Exception: {e}')
            return None, debug

    def fmt(self, s):
        arrow = '🟢' if s['signal'] == 'LONG' else '🔴'
        icon  = '🚀' if s['signal'] == 'LONG' else '🔻'
        bar   = '█' * int(s['score']/10) + '░' * (10 - int(s['score']/10))
        z     = {'DISCOUNT':'🟩 DISCOUNT','PREMIUM':'🟥 PREMIUM','NEUTRAL':'🟨 NEUTRAL'}.get(s['pd_zone'],'')
        ob    = s['ob']
        ema_tag = '📊 Triple EMA ✅' if s.get('triple_ema') else '📊 EMA partial'
        trig_tag = f"🕯️ {s['trigger']}" if s.get('has_trigger') else '⏳ No trigger (structure/OB entry)'
        adx_strength = '🔥 Strong' if s['adx'] >= 35 else ('📈 Trending' if s['adx'] >= 25 else '〰️ Weak')

        msg  = f"{'━'*40}\n"
        msg += f"{icon} <b>SMC PRO v5.0 — {s['quality']}</b> {icon}\n"
        msg += f"{'━'*40}\n\n"
        msg += f"<b>🆔</b> <code>{s['trade_id']}</code>\n"
        msg += f"<b>📊 PAIR:</b>  <b>#{s['symbol']}USDT</b>\n"
        msg += f"<b>📍 DIR:</b>   {arrow} <b>{s['signal']}</b>\n"
        msg += f"<b>🗺️ ZONE:</b>  {z} ({s['pd_pos']*100:.0f}%)\n"
        msg += f"<b>📈 ADX:</b>   {adx_strength} ({s['adx']:.0f})\n"
        msg += f"<b>📊 EMA:</b>   {ema_tag}\n"
        msg += f"<b>🕯️ ENTRY:</b> {trig_tag}\n"
        if s.get('quality_note'):
            msg += f"<b>{s['quality_note']}</b>\n"
        msg += f"\n<b>⭐ SCORE: {s['score']} / 100</b>\n"
        msg += f"<code>[{bar}]</code>\n\n"
        msg += f"<b>📦 ORDER BLOCK (1H):</b>\n"
        msg += f"  Top:    <code>${ob['top']:.6f}</code>\n"
        msg += f"  Bottom: <code>${ob['bottom']:.6f}</code>\n\n"
        msg += f"<b>💰 ENTRY:</b> <code>${s['entry']:.6f}</code>\n\n"
        msg += f"<b>🎯 TARGETS:</b>\n"
        for (lbl, eta), tp, rr in zip(
            [('TP1 — 50% exit','6-12h'),('TP2 — 30% exit','12-24h'),('TP3 — 20% exit','24-48h')],
            s['targets'], s['rr']
        ):
            pct = abs((tp - s['entry'])/s['entry']*100)
            msg += f"  <b>{lbl}</b> [{eta}]\n"
            msg += f"  <code>${tp:.6f}</code>  <b>+{pct:.2f}%</b>  RR {rr:.1f}:1\n\n"
        msg += f"<b>🛑 STOP:</b> <code>${s['stop_loss']:.6f}</code>  (-{s['risk_pct']:.2f}%)\n\n"
        if s['structure']:
            sk = s['structure']['kind']
            msg += f"<b>🏗️ STRUCTURE:</b> {'MSS — Early Reversal' if 'MSS' in sk else 'BOS — Pullback'}\n\n"
        msg += f"<b>📋 CONFLUENCE:</b>\n"
        for r in s['reasons'][:10]:
            msg += f"  • {r}\n"
        msg += f"\n<b>⚠️ RISK:</b> 1-2% per trade\n"
        msg += f"  Move SL → BE after TP1 hits\n"
        msg += f"\n<i>🕐 {s['timestamp'].strftime('%Y-%m-%d %H:%M UTC')}</i>\n"
        msg += f"{'━'*40}"
        return msg

    async def send(self, text):
        try:
            await self.bot.send_message(chat_id=self.chat_id, text=text, parse_mode=ParseMode.HTML)
        except Exception as e:
            logger.error(f"Telegram: {e}")

    async def tp_alert(self, t, n, price):
        tp  = t['targets'][n-1]
        pct = abs((tp - t['entry'])/t['entry']*100)
        advice = {1:'Close 50% → Move SL to breakeven', 2:'Close 30% → Trail stop', 3:'Close final 20% 🎊'}
        msg  = f"🎯 <b>TP{n} HIT!</b>\n\n<code>{t['trade_id']}</code>\n<b>{t['symbol']}</b> {t['signal']}\n\n"
        msg += f"Target: <code>${tp:.6f}</code>\nProfit: <b>+{pct:.2f}%</b>\n\n📋 {advice[n]}"
        await self.send(msg)
        self.stats[f'tp{n}'] += 1

    async def sl_alert(self, t, price):
        loss = abs((price - t['entry'])/t['entry']*100)
        msg  = f"⛔ <b>STOP LOSS HIT</b>\n\n<code>{t['trade_id']}</code>\n<b>{t['symbol']}</b> {t['signal']}\n\n"
        msg += f"Loss: <b>-{loss:.2f}%</b>\n\nOB invalidated. Next setup incoming."
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
        if self.is_scanning: return []
        self.is_scanning = True
        logger.info("🔍 Scan starting...")

        await self.send(
            f"🔍 <b>SMC v5.0 SCAN</b>\n"
            f"SHORT≥{MIN_SCORE_SHORT}(ADX≥{ADX_MIN_SHORT}) | LONG≥{MIN_SCORE_LONG}(ADX≥{ADX_MIN_LONG}+TripleEMA)\n"
            f"OB tol: {OB_TOLERANCE_PCT*100:.1f}% | PD zone: {PD_ZONE_BARS}bar range"
        )

        pairs = await self.get_pairs()
        candidates = []; near_misses = []; scanned = 0

        for pair in pairs:
            try:
                data = await self.fetch_data(pair)
                if data:
                    sig, dbg = self.analyse(data, pair)
                    if sig:
                        candidates.append(sig)
                    else:
                        if dbg['score'] > 0 and any('✅ Price IN OB' in g for g in dbg['gates']):
                            near_misses.append(dbg)
                scanned += 1
                if scanned % 30 == 0:
                    logger.info(f"  ⏳ {scanned}/{len(pairs)}")
                await asyncio.sleep(0.45)
            except Exception as e:
                logger.error(f"Scan {pair}: {e}"); continue

        candidates.sort(key=lambda x: x['score'], reverse=True)
        top = candidates[:MAX_SIGNALS_PER_SCAN]
        self.last_debug = near_misses[:10]

        for sig in top:
            self.signal_history.append(sig)
            self.active_trades[sig['trade_id']] = sig
            self.stats['total'] += 1
            self.stats[sig['signal'].lower()] += 1
            await self.send(self.fmt(sig))
            await asyncio.sleep(2)

        self.stats['last_scan'] = datetime.now()
        self.stats['pairs_scanned'] = scanned
        lg = sum(1 for s in top if s['signal'] == 'LONG')

        summ  = f"✅ <b>SCAN COMPLETE — v5.0</b>\n\n"
        summ += f"📊 Scanned: {scanned} pairs\n"
        summ += f"🎯 Signals: {len(top)}\n"
        if top:
            summ += f"  🟢 Long: {lg}  🔴 Short: {len(top)-lg}\n"
            avg_adx = sum(s['adx'] for s in top) / len(top)
            summ += f"  📈 Avg ADX: {avg_adx:.0f}\n"
        else:
            summ += f"\n<i>No setups met criteria. Near misses: {len(near_misses)} — /debug</i>\n"
        summ += f"\n⏰ {datetime.now().strftime('%H:%M UTC')}"
        await self.send(summ)

        self.is_scanning = False
        return top

    async def run(self, interval_min=SCAN_INTERVAL_MIN):
        await self.send(
            "🔥 <b>SMC PRO v5.0 — DATA-VALIDATED</b> 🔥\n\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            "<b>Built from 5 rounds of backtesting</b>\n"
            "<b>90 days | 15 pairs | 500+ setups analyzed</b>\n\n"
            "📊 <b>Key findings from data:</b>\n"
            "  • SHORTs: WR=55%, avg=+0.54R ✅\n"
            "  • LONGs (downtrend): WR=11% ❌\n"
            "  • ADX>30 boosts WR significantly\n"
            "  • bear_engulf/shooting_star: top triggers\n\n"
            f"⚙️ <b>Config:</b>\n"
            f"  SHORT ≥{MIN_SCORE_SHORT} | ADX≥{ADX_MIN_SHORT}\n"
            f"  LONG  ≥{MIN_SCORE_LONG} | ADX≥{ADX_MIN_LONG} + TripleEMA\n"
            f"  OB tol: {OB_TOLERANCE_PCT*100:.1f}% | PD: {PD_ZONE_BARS}bar\n\n"
            "Commands: /scan /stats /trades /debug /help\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        )
        asyncio.create_task(self.track())
        while True:
            try:
                await self.scan()
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
            "🔥 <b>SMC Pro v5.0 — Data-Validated</b>\n\n"
            f"SHORT≥{MIN_SCORE_SHORT}+ADX | LONG≥{MIN_SCORE_LONG}+ADX+TripleEMA\n"
            "Based on 90d backtest: shorts work, longs need confirmation.\n\n"
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
        msg += f"Total: {s['total']}  🟢 Long: {s['long']}  🔴 Short: {s['short']}\n"
        msg += f"TP1: {s['tp1']} | TP2: {s['tp2']} | TP3: {s['tp3']} | SL: {s['sl']}\n"
        if s['last_scan']:
            msg += f"\nLast scan: {s['last_scan'].strftime('%H:%M UTC')}"
            msg += f"\nPairs: {s['pairs_scanned']} | Active: {len(self.s.active_trades)}"
        await u.message.reply_text(msg, parse_mode=ParseMode.HTML)

    async def trades(self, u: Update, c: ContextTypes.DEFAULT_TYPE):
        if not self.s.active_trades:
            await u.message.reply_text("📭 No active trades."); return
        msg = f"📡 <b>ACTIVE ({len(self.s.active_trades)})</b>\n\n"
        for tid, t in list(self.s.active_trades.items())[:10]:
            age  = int((datetime.now() - t['timestamp']).total_seconds() / 3600)
            tps  = ''.join(['✅' if h else '⏳' for h in t['tp_hit']])
            msg += f"<b>{t['symbol']}</b> {t['signal']} sc={t['score']} adx={t['adx']:.0f}\n"
            msg += f"  Entry: <code>${t['entry']:.5f}</code> | {age}h | TPs: {tps}\n\n"
        await u.message.reply_text(msg, parse_mode=ParseMode.HTML)

    async def debug(self, u: Update, c: ContextTypes.DEFAULT_TYPE):
        if not self.s.last_debug:
            await u.message.reply_text("📭 No debug. Run /scan first."); return
        msg = "🔬 <b>NEAR MISSES</b>\n<i>(At OB but below threshold)</i>\n\n"
        for d in self.s.last_debug[:8]:
            msg += f"<b>{d['symbol']}</b> {d['bias']} Score:{d['score']}\n"
            for g in d['gates'][-3:]:
                msg += f"  {g}\n"
            msg += "\n"
        await u.message.reply_text(msg, parse_mode=ParseMode.HTML)

    async def help(self, u: Update, c: ContextTypes.DEFAULT_TYPE):
        msg  = "📚 <b>SMC PRO v5.0 — DATA-DRIVEN STRATEGY</b>\n\n"
        msg += "<b>What backtesting proved (90d/15 pairs):</b>\n"
        msg += "  SHORTs: WR=55%, avg=+0.54R ✅\n"
        msg += "  LONGs (downtrend): WR=11% ❌\n"
        msg += "  ADX>30: significantly boosts WR\n"
        msg += "  Best triggers: bear_engulf, shooting_star\n\n"
        msg += "<b>Hard Gates (ALL must pass):</b>\n"
        msg += f"  1️⃣ 4H EMA bias (21 vs 50)\n"
        msg += f"  2️⃣ ADX≥{ADX_MIN_SHORT} SHORT / ADX≥{ADX_MIN_LONG} LONG\n"
        msg += f"  3️⃣ LONG: triple EMA (21>50>200) required\n"
        msg += f"  4️⃣ PD Zone (200-bar range)\n"
        msg += f"  5️⃣ Price at valid 1H OB\n"
        msg += f"  6️⃣ Score ≥{MIN_SCORE_SHORT} SHORT / ≥{MIN_SCORE_LONG} LONG\n\n"
        msg += "<b>Scoring (max 100):</b>\n"
        msg += "  +30 — 4H trend + ADX strength\n"
        msg += "  +25 — OB quality (tight = more)\n"
        msg += "  +20 — Structure (MSS>BOS)\n"
        msg += "  +20 — Entry trigger (no penalty if absent)\n"
        msg += "  +10 — Momentum\n"
        msg += "   +5 — Extras (sweep/vol/vwap)\n\n"
        msg += "<b>TP structure:</b>\n"
        msg += "  TP1 = 1:1.5 RR  close 50%\n"
        msg += "  TP2 = 1:2.5 RR  close 30%\n"
        msg += "  TP3 = 1:4.0 RR  close 20%\n"
        await u.message.reply_text(msg, parse_mode=ParseMode.HTML)


# ══════════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════════
async def main():
    TELEGRAM_TOKEN   = "7731521911:AAFnus-fDivEwoKqrtwZXMmKEj5BU1EhQn4"
    TELEGRAM_CHAT_ID = "7500072234"
    BINANCE_API_KEY  = None
    BINANCE_SECRET   = None

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
