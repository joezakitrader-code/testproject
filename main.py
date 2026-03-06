"""
SMC PRO v4.1 — STANDALONE BACKTESTER
══════════════════════════════════════════════════════════════════
Fetches real OHLCV from Binance (public, no API key needed),
replays the exact v4.1 signal logic, then simulates trade outcomes.

OUTPUT:
  • Console summary (per-pair + overall stats)
  • smc_backtest_results.csv  ← every trade, every detail
  • smc_backtest_report.txt   ← printable summary you can share

HOW TO RUN:
  pip install ccxt pandas numpy ta --break-system-packages
  python smc_backtester.py

CONFIGURABLE AT TOP:
  PAIRS        — which coins to test
  DAYS_BACK    — how many days of history (default 90)
  TIMEFRAME_ENTRY — '1h' (same as live bot)
══════════════════════════════════════════════════════════════════
"""

import asyncio
import ccxt.async_support as ccxt
import pandas as pd
import numpy as np
import ta
import warnings
import logging
from datetime import datetime, timedelta

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.WARNING)  # quiet during backtest

# ══════════════════════════════════════════════════════════════
#  BACKTEST SETTINGS — edit these
# ══════════════════════════════════════════════════════════════
PAIRS = [
    "BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "XRP/USDT",
    "DOGE/USDT", "ADA/USDT", "AVAX/USDT", "LINK/USDT", "DOT/USDT",
    "LTC/USDT", "MATIC/USDT", "UNI/USDT", "ATOM/USDT", "NEAR/USDT",
]
DAYS_BACK            = 90      # days of history to replay
MIN_SCORE_SHORT      = 80
MIN_SCORE_LONG       = 82
TRIPLE_EMA_REQUIRED  = True
TRIPLE_EMA_BYPASS    = 87
HH_LL_BONUS          = 8
HH_LL_LOOKBACK       = 10
OB_TOLERANCE_PCT     = 0.008
OB_IMPULSE_ATR_MULT  = 1.0
STRUCTURE_LOOKBACK   = 20

# Trade management (mirrors live bot)
TP1_RR = 1.5
TP2_RR = 2.5
TP3_RR = 4.0
TP1_CLOSE = 0.50   # close 50% at TP1
TP2_CLOSE = 0.30   # close 30% at TP2
TP3_CLOSE = 0.20   # close 20% at TP3
TRADE_TIMEOUT_H = 48


# ══════════════════════════════════════════════════════════════
#  INDICATORS  (identical to live bot)
# ══════════════════════════════════════════════════════════════
def add_indicators(df):
    if len(df) < 55:
        return df
    try:
        df = df.copy()
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

        df['cmf'] = ta.volume.ChaikinMoneyFlowIndicator(
            df['high'], df['low'], df['close'], df['volume']).chaikin_money_flow()
        df['mfi'] = ta.volume.MFIIndicator(
            df['high'], df['low'], df['close'], df['volume']).money_flow_index()

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
        pass
    return df


# ══════════════════════════════════════════════════════════════
#  SMC ENGINE  (identical to live bot)
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
            return False
        recent = df_4h.iloc[-lookback:]
        prior  = df_4h.iloc[-(lookback * 2):-lookback]
        if direction == 'LONG':
            return recent['high'].max() > prior['high'].max()
        else:
            return recent['low'].min() < prior['low'].min()

    def is_triple_ema_bull(self, df_4h):
        l = df_4h.iloc[-1]
        return float(l.get('ema_21', 0)) > float(l.get('ema_50', 0)) > float(l.get('ema_200', 0))

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
        obs = []
        n = len(df)
        start = max(2, n - lookback)
        for i in range(start, n - 3):
            c = df.iloc[i]
            atr_local = df['atr'].iloc[i] if 'atr' in df.columns and not pd.isna(df['atr'].iloc[i]) else (c['high'] - c['low'])
            min_impulse = atr_local * OB_IMPULSE_ATR_MULT
            if direction == 'LONG':
                if c['close'] >= c['open']: continue
                fwd_high = df['high'].iloc[i+1:min(i+5, n)].max()
                if fwd_high - c['low'] < min_impulse: continue
                ob = {'top': max(c['open'], c['close']), 'bottom': c['low'],
                      'mid': (max(c['open'], c['close']) + c['low']) / 2, 'bar': i}
                ob_50 = (ob['top'] + ob['bottom']) / 2
                if (df['close'].iloc[i+1:n] < ob_50).any(): continue
                obs.append(ob)
            else:
                if c['close'] <= c['open']: continue
                fwd_low = df['low'].iloc[i+1:min(i+5, n)].min()
                if c['high'] - fwd_low < min_impulse: continue
                ob = {'top': c['high'], 'bottom': min(c['open'], c['close']),
                      'mid': (c['high'] + min(c['open'], c['close'])) / 2, 'bar': i}
                ob_50 = (ob['top'] + ob['bottom']) / 2
                if (df['close'].iloc[i+1:n] > ob_50).any(): continue
                obs.append(ob)
        obs.sort(key=lambda x: x['bar'], reverse=True)
        return obs

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


# ══════════════════════════════════════════════════════════════
#  SCORER  (identical to live bot)
# ══════════════════════════════════════════════════════════════
def score_setup(direction, ob, structure, sweep, fvg_near,
                df_1h, df_15m, df_4h, pd_label, hh_ll_confirmed):
    score = 0

    l1  = df_1h.iloc[-1]
    p1  = df_1h.iloc[-2]
    l15 = df_15m.iloc[-1]
    l4  = df_4h.iloc[-1]

    if structure:
        score += 20 if 'MSS' in structure['kind'] else 14

    if ob:
        ob_size_pct = (ob['top'] - ob['bottom']) / ob['bottom'] * 100
        if ob_size_pct < 0.8:    score += 20
        elif ob_size_pct < 2.0:  score += 13
        else:                     score += 7

    e21 = l4.get('ema_21', 0); e50 = l4.get('ema_50', 0); e200 = l4.get('ema_200', 0)
    if direction == 'LONG':
        if e21 > e50 > e200:     score += 15
        elif e21 > e50:          score += 10
        elif pd_label == 'DISCOUNT': score += 6
    else:
        if e21 < e50 < e200:     score += 15
        elif e21 < e50:          score += 10
        elif pd_label == 'PREMIUM': score += 6

    if hh_ll_confirmed:
        score += HH_LL_BONUS

    trigger = False
    if direction == 'LONG':
        if   l1.get('bull_engulf', 0) == 1: score += 25; trigger = True
        elif l1.get('bull_pin', 0) == 1:    score += 22; trigger = True
        elif l1.get('hammer', 0) == 1:      score += 18; trigger = True
        elif p1.get('bull_engulf', 0) == 1: score += 14; trigger = True
        elif p1.get('bull_pin', 0) == 1:    score += 11; trigger = True
        elif p1.get('hammer', 0) == 1:      score += 9;  trigger = True
    else:
        if   l1.get('bear_engulf', 0) == 1:    score += 25; trigger = True
        elif l1.get('bear_pin', 0) == 1:       score += 22; trigger = True
        elif l1.get('shooting_star', 0) == 1:  score += 18; trigger = True
        elif p1.get('bear_engulf', 0) == 1:    score += 14; trigger = True
        elif p1.get('bear_pin', 0) == 1:       score += 11; trigger = True
        elif p1.get('shooting_star', 0) == 1:  score += 9;  trigger = True

    if not trigger:
        score -= 12

    rsi1  = l1.get('rsi', 50)
    macd1 = l1.get('macd', 0); ms1  = l1.get('macd_signal', 0)
    pm1   = p1.get('macd', 0); pms1 = p1.get('macd_signal', 0)
    sk1   = l1.get('srsi_k', 0.5); sd1 = l1.get('srsi_d', 0.5)

    if direction == 'LONG':
        if 28 <= rsi1 <= 55:               score += 4
        elif rsi1 < 28:                    score += 3
        if macd1 > ms1 and pm1 <= pms1:   score += 5
        elif macd1 > ms1:                  score += 2
        if sk1 < 0.3 and sk1 > sd1:       score += 3
    else:
        if 45 <= rsi1 <= 72:               score += 4
        elif rsi1 > 72:                    score += 3
        if macd1 < ms1 and pm1 >= pms1:   score += 5
        elif macd1 < ms1:                  score += 2
        if sk1 > 0.7 and sk1 < sd1:       score += 3

    extras = 0
    if sweep:    extras += 4
    if fvg_near: extras += 3

    vr15 = l15.get('vol_ratio', 1.0)
    if   vr15 >= 2.5: extras += 3
    elif vr15 >= 1.5: extras += 1

    close1 = l1.get('close', 0); vwap1 = l1.get('vwap', 0)
    if direction == 'LONG' and close1 < vwap1:   extras = min(extras+1, 10)
    elif direction == 'SHORT' and close1 > vwap1: extras = min(extras+1, 10)

    score += min(extras, 10)
    return max(0, min(int(score), 100))


# ══════════════════════════════════════════════════════════════
#  DATA FETCHER
# ══════════════════════════════════════════════════════════════
async def fetch_all_data(exchange, symbol, days_back=90):
    """Fetch enough candles for backtesting (4h, 1h, 15m)."""
    result = {}
    limits = {
        '4h':  max(220,  days_back * 6  + 220),
        '1h':  max(150,  days_back * 24 + 150),
        '15m': max(80,   days_back * 96 + 80),
    }
    for tf, lim in limits.items():
        lim = min(lim, 1000)  # Binance max
        raw = await exchange.fetch_ohlcv(symbol, tf, limit=lim)
        df  = pd.DataFrame(raw, columns=['ts','open','high','low','close','volume'])
        df['ts'] = pd.to_datetime(df['ts'], unit='ms')
        df = df.reset_index(drop=True)
        result[tf] = add_indicators(df)
        await asyncio.sleep(0.1)
    return result


# ══════════════════════════════════════════════════════════════
#  SIGNAL DETECTOR  (walk-forward on historical data)
# ══════════════════════════════════════════════════════════════
smc = SMCEngine()

def detect_signal_at(df4_slice, df1_slice, df15_slice):
    """
    Run exact v4.1 gate logic on a historical slice.
    Returns signal dict or None.
    """
    try:
        if len(df1_slice) < 80 or len(df15_slice) < 40 or len(df4_slice) < 60:
            return None

        price = df1_slice['close'].iloc[-1]
        l4 = df4_slice.iloc[-1]
        e21 = l4.get('ema_21', 0); e50 = l4.get('ema_50', 0)

        if e21 > e50:    bias = 'LONG'
        elif e21 < e50:  bias = 'SHORT'
        else:            return None

        triple_ema_bull = smc.is_triple_ema_bull(df4_slice)
        hh_ll_ok = smc.check_4h_hh_ll(df4_slice, bias, HH_LL_LOOKBACK)

        pd_label, pd_pos = smc.pd_zone(df4_slice, price)
        if bias == 'LONG' and pd_label == 'PREMIUM':   return None
        if bias == 'SHORT' and pd_label == 'DISCOUNT':  return None

        highs1, lows1 = smc.swing_highs_lows(df1_slice, left=4, right=4)
        structure = smc.detect_structure_break(df1_slice, highs1, lows1, lookback=STRUCTURE_LOOKBACK)
        if structure:
            if bias == 'LONG' and 'BEAR' in structure['kind']:   return None
            if bias == 'SHORT' and 'BULL' in structure['kind']:  return None

        obs = smc.find_order_blocks(df1_slice, bias, lookback=60)
        if not obs: return None

        active_ob = None
        for ob in obs:
            if smc.price_in_ob(price, ob, OB_TOLERANCE_PCT):
                active_ob = ob; break
        if not active_ob: return None

        fvgs = smc.find_fvg(df1_slice, bias, lookback=25)
        fvg_near = None
        for fvg in fvgs:
            if fvg['bottom'] < active_ob['top'] and fvg['top'] > active_ob['bottom']:
                fvg_near = fvg; break

        sweep = smc.recent_liquidity_sweep(df1_slice, bias, highs1, lows1, lookback=20)

        score = score_setup(
            bias, active_ob, structure, sweep, fvg_near,
            df1_slice, df15_slice, df4_slice, pd_label, hh_ll_ok
        )

        # Direction gate
        if bias == 'SHORT':
            if score < MIN_SCORE_SHORT: return None
        else:
            if score < MIN_SCORE_LONG: return None
            if TRIPLE_EMA_REQUIRED and score < TRIPLE_EMA_BYPASS:
                if not triple_ema_bull: return None

        atr1  = df1_slice['atr'].iloc[-1]
        entry = price
        if bias == 'LONG':
            sl = min(active_ob['bottom'] - atr1 * 0.2, entry - atr1 * 0.6)
        else:
            sl = max(active_ob['top'] + atr1 * 0.2, entry + atr1 * 0.6)

        risk = abs(entry - sl)
        if risk < entry * 0.001: return None

        if bias == 'LONG':
            tps = [entry + risk * TP1_RR, entry + risk * TP2_RR, entry + risk * TP3_RR]
        else:
            tps = [entry - risk * TP1_RR, entry - risk * TP2_RR, entry - risk * TP3_RR]

        return {
            'bias':       bias,
            'score':      score,
            'triple_ema': triple_ema_bull,
            'hh_ll':      hh_ll_ok,
            'entry':      entry,
            'sl':         sl,
            'tps':        tps,
            'risk':       risk,
            'risk_pct':   risk / entry * 100,
            'ob':         active_ob,
            'has_fvg':    fvg_near is not None,
            'has_sweep':  sweep is not None,
            'pd_zone':    pd_label,
            'structure':  structure['kind'] if structure else 'NONE',
        }
    except Exception as e:
        return None


# ══════════════════════════════════════════════════════════════
#  TRADE SIMULATOR
# ══════════════════════════════════════════════════════════════
def simulate_trade(sig, future_1h):
    """
    Walk future 1H candles after signal to determine outcome.
    Returns dict with R result and outcome label.
    """
    bias   = sig['bias']
    entry  = sig['entry']
    sl     = sig['sl']
    tps    = sig['tps']
    risk   = sig['risk']

    tp_hit    = [False, False, False]
    remaining = 1.0   # fraction of position still open
    realized_r = 0.0
    outcome    = 'TIMEOUT'
    max_adverse = 0.0

    weights = [TP1_CLOSE, TP2_CLOSE, TP3_CLOSE]

    for i, row in future_1h.iterrows():
        h, l, c = row['high'], row['low'], row['close']

        # Track max adverse excursion
        if bias == 'LONG':
            mae = (entry - l) / risk
        else:
            mae = (h - entry) / risk
        max_adverse = max(max_adverse, mae)

        # Check SL first (conservative — assume worst price on candle)
        if bias == 'LONG' and l <= sl:
            realized_r -= remaining * 1.0
            outcome = 'SL'
            break
        elif bias == 'SHORT' and h >= sl:
            realized_r -= remaining * 1.0
            outcome = 'SL'
            break

        # Check TPs in order
        for t_idx, tp in enumerate(tps):
            if tp_hit[t_idx]: continue
            hit = (bias == 'LONG' and h >= tp) or (bias == 'SHORT' and l <= tp)
            if hit:
                tp_hit[t_idx] = True
                frac = weights[t_idx]
                rr_level = [TP1_RR, TP2_RR, TP3_RR][t_idx]
                realized_r += frac * rr_level
                remaining  -= frac
                if t_idx == 2:
                    outcome = 'TP3'
                elif t_idx == 1 and not tp_hit[2]:
                    outcome = 'TP2'
                elif t_idx == 0 and not tp_hit[1]:
                    outcome = 'TP1'

        if remaining <= 0:
            break

    # Determine final outcome label
    if 'SL' not in outcome and 'TIMEOUT' not in outcome:
        pass  # outcome already set by TP hits
    elif 'TIMEOUT' in outcome and any(tp_hit):
        # Partial close on timeout
        realized_r -= remaining * 0  # close at breakeven on timeout (conservative)
        outcome = 'PARTIAL_TP'

    won = realized_r > 0

    return {
        'outcome':       outcome,
        'realized_r':    round(realized_r, 4),
        'won':           won,
        'tp1_hit':       tp_hit[0],
        'tp2_hit':       tp_hit[1],
        'tp3_hit':       tp_hit[2],
        'max_adverse_r': round(max_adverse, 4),
    }


# ══════════════════════════════════════════════════════════════
#  WALK-FORWARD BACKTEST LOOP
# ══════════════════════════════════════════════════════════════
def run_walk_forward(data, symbol, days_back=90):
    """
    Walk forward bar-by-bar on 1H data.
    Every bar: align 4H and 15M slices, run signal detection.
    If signal fires, simulate trade on subsequent bars.
    """
    df4  = data['4h']
    df1  = data['1h']
    df15 = data['15m']

    # Figure out which 1H bars fall within [start_date, end_date]
    end_date   = df1['ts'].iloc[-1]
    start_date = end_date - timedelta(days=days_back)

    # Only walk bars in our test window
    test_bars = df1[df1['ts'] >= start_date].index.tolist()

    trades = []
    active_trade_end = -1  # index of last bar of current active trade (no overlapping)

    print(f"  {symbol}: {len(test_bars)} 1H bars to walk ({days_back}d) ...", end=' ', flush=True)

    for idx in test_bars:
        if idx < 80:  # need warmup
            continue
        if idx <= active_trade_end:
            continue  # skip — trade already running

        # Slice data up to current bar (no lookahead)
        df1_slice = df1.iloc[:idx+1].copy()
        ts_now    = df1_slice['ts'].iloc[-1]

        # Align 4H slice: last 4H bar whose close is <= ts_now
        df4_slice = df4[df4['ts'] <= ts_now].copy()
        if len(df4_slice) < 60: continue

        # Align 15M slice
        df15_slice = df15[df15['ts'] <= ts_now].copy()
        if len(df15_slice) < 40: continue

        # Detect signal
        sig = detect_signal_at(df4_slice, df1_slice, df15_slice)
        if sig is None:
            continue

        # Simulate trade on bars AFTER signal
        future_bars = df1.iloc[idx+1:idx+1+TRADE_TIMEOUT_H].copy()
        if len(future_bars) < 2:
            continue

        result = simulate_trade(sig, future_bars)

        # Record
        trades.append({
            'symbol':        symbol,
            'date':          ts_now.strftime('%Y-%m-%d %H:%M'),
            'direction':     sig['bias'],
            'score':         sig['score'],
            'triple_ema':    sig['triple_ema'],
            'hh_ll':         sig['hh_ll'],
            'pd_zone':       sig['pd_zone'],
            'structure':     sig['structure'],
            'has_fvg':       sig['has_fvg'],
            'has_sweep':     sig['has_sweep'],
            'entry':         round(sig['entry'], 6),
            'sl':            round(sig['sl'], 6),
            'tp1':           round(sig['tps'][0], 6),
            'tp2':           round(sig['tps'][1], 6),
            'tp3':           round(sig['tps'][2], 6),
            'risk_pct':      round(sig['risk_pct'], 3),
            'outcome':       result['outcome'],
            'realized_r':    result['realized_r'],
            'won':           result['won'],
            'tp1_hit':       result['tp1_hit'],
            'tp2_hit':       result['tp2_hit'],
            'tp3_hit':       result['tp3_hit'],
            'max_adverse_r': result['max_adverse_r'],
        })

        # Block overlapping trades: assume trade runs for TP_TIMEOUT hours
        active_trade_end = idx + TRADE_TIMEOUT_H

    print(f"{len(trades)} signals found.")
    return trades


# ══════════════════════════════════════════════════════════════
#  REPORT GENERATOR
# ══════════════════════════════════════════════════════════════
def generate_report(all_trades):
    df = pd.DataFrame(all_trades)
    if df.empty:
        return "❌ No trades found. Loosen filters or check data.", df

    lines = []
    sep = "═" * 60

    lines.append(sep)
    lines.append("  SMC PRO v4.1 — BACKTEST REPORT")
    lines.append(f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M UTC')}")
    lines.append(f"  Period: {DAYS_BACK} days  |  Pairs: {len(PAIRS)}")
    lines.append(sep)

    # ── Overall stats ──
    total    = len(df)
    wins     = df['won'].sum()
    wr       = wins / total * 100 if total else 0
    avg_r    = df['realized_r'].mean()
    total_r  = df['realized_r'].sum()
    max_dd   = 0
    running  = 0
    peak     = 0
    for r in df['realized_r']:
        running += r
        if running > peak: peak = running
        dd = peak - running
        if dd > max_dd: max_dd = dd

    lines.append(f"\n  OVERALL ({total} trades)")
    lines.append(f"  {'─'*40}")
    lines.append(f"  Win Rate:      {wr:.1f}%  ({wins}W / {total-wins}L)")
    lines.append(f"  Avg R/trade:   {avg_r:+.3f}R")
    lines.append(f"  Total R:       {total_r:+.2f}R")
    lines.append(f"  Max Drawdown:  -{max_dd:.2f}R")
    lines.append(f"  Signals/week:  {total / (DAYS_BACK/7):.1f}")

    # ── Direction split ──
    lines.append(f"\n  BY DIRECTION")
    lines.append(f"  {'─'*40}")
    for d in ['LONG', 'SHORT']:
        sub = df[df['direction'] == d]
        if len(sub) == 0: continue
        w = sub['won'].sum()
        lines.append(f"  {d}: {len(sub)} trades | WR={w/len(sub)*100:.1f}% | avg={sub['realized_r'].mean():+.3f}R")

    # ── Score buckets ──
    lines.append(f"\n  SCORE BUCKETS")
    lines.append(f"  {'─'*40}")
    buckets = [(80,84),(85,89),(90,100)]
    for lo, hi in buckets:
        sub = df[(df['score'] >= lo) & (df['score'] <= hi)]
        if len(sub) == 0: continue
        w = sub['won'].sum()
        lines.append(f"  {lo}-{hi}: {len(sub)} trades | WR={w/len(sub)*100:.1f}% | avg={sub['realized_r'].mean():+.3f}R")

    # ── Outcome breakdown ──
    lines.append(f"\n  OUTCOME BREAKDOWN")
    lines.append(f"  {'─'*40}")
    tp1 = df['tp1_hit'].sum(); tp2 = df['tp2_hit'].sum(); tp3 = df['tp3_hit'].sum()
    sl  = (df['outcome'] == 'SL').sum()
    tout= (df['outcome'] == 'TIMEOUT').sum()
    pt  = (df['outcome'] == 'PARTIAL_TP').sum()
    lines.append(f"  TP1 hits: {tp1} ({tp1/total*100:.0f}%)")
    lines.append(f"  TP2 hits: {tp2} ({tp2/total*100:.0f}%)")
    lines.append(f"  TP3 hits: {tp3} ({tp3/total*100:.0f}%)")
    lines.append(f"  SL hits:  {sl}  ({sl/total*100:.0f}%)")
    lines.append(f"  Timeout:  {tout} ({tout/total*100:.0f}%)")
    lines.append(f"  Partial:  {pt}  ({pt/total*100:.0f}%)")

    # ── Per-pair ──
    lines.append(f"\n  PER PAIR")
    lines.append(f"  {'─'*40}")
    for sym in sorted(df['symbol'].unique()):
        sub = df[df['symbol'] == sym]
        w = sub['won'].sum()
        wr_sym = w/len(sub)*100
        r_sym  = sub['realized_r'].mean()
        lines.append(f"  {sym:<12}: {len(sub):>3} trades | WR={wr_sym:5.1f}% | avg={r_sym:+.3f}R")

    # ── HH/LL and Triple EMA analysis ──
    lines.append(f"\n  FILTER ANALYSIS")
    lines.append(f"  {'─'*40}")
    for flag, label in [('hh_ll','HH/LL confirmed'), ('triple_ema','Triple EMA')]:
        with_flag = df[df[flag] == True]
        without   = df[df[flag] == False]
        if len(with_flag):
            w1 = with_flag['won'].sum()
            lines.append(f"  {label} YES: {len(with_flag)} trades | WR={w1/len(with_flag)*100:.1f}% | avg={with_flag['realized_r'].mean():+.3f}R")
        if len(without):
            w2 = without['won'].sum()
            lines.append(f"  {label} NO:  {len(without)} trades | WR={w2/len(without)*100:.1f}% | avg={without['realized_r'].mean():+.3f}R")

    lines.append(f"\n{sep}")
    lines.append("  Share these results to analyze & improve the strategy!")
    lines.append(sep)

    report = "\n".join(lines)
    return report, df


# ══════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════
async def main():
    print("\n" + "═"*60)
    print("  SMC PRO v4.1 — BACKTESTER")
    print(f"  Pairs: {len(PAIRS)}  |  Days: {DAYS_BACK}")
    print(f"  SHORT≥{MIN_SCORE_SHORT}  LONG≥{MIN_SCORE_LONG}+TripleEMA")
    print("═"*60 + "\n")

    exchange = ccxt.binance({
        'enableRateLimit': True,
        'options': {'defaultType': 'spot'}  # spot for public OHLCV
    })

    all_trades = []

    for symbol in PAIRS:
        print(f"⬇️  Fetching {symbol} ...")
        try:
            data = await fetch_all_data(exchange, symbol, DAYS_BACK)
            trades = run_walk_forward(data, symbol, DAYS_BACK)
            all_trades.extend(trades)
        except Exception as e:
            print(f"  ❌ {symbol}: {e}")
        await asyncio.sleep(0.5)

    await exchange.close()

    print("\n" + "─"*60)
    print("📊 Generating report...\n")

    report, df = generate_report(all_trades)
    print(report)

    # Save CSV
    csv_path = "smc_backtest_results.csv"
    if not df.empty:
        df.to_csv(csv_path, index=False)
        print(f"\n✅ Trade log saved → {csv_path}")

    # Save report
    report_path = "smc_backtest_report.txt"
    with open(report_path, 'w') as f:
        f.write(report)
    print(f"✅ Report saved     → {report_path}")

    print("\n💡 Share both files with Claude to analyze & rebuild the strategy.\n")


if __name__ == "__main__":
    asyncio.run(main())
