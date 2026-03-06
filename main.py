"""
SMC PRO v4.1 — STANDALONE BACKTESTER  (fixed)
══════════════════════════════════════════════════════════════════
BUG FIX from v1: The OB violation check `(df['close'].iloc[i+1:n] < ob_50).any()`
was checking ALL remaining bars in the slice. In a 90-day walk-forward,
that means bars from 60 days ago are checked against 60 days of future data —
almost every OB gets "violated" and filtered out (0 signals).

FIX: VIOLATION_WINDOW = 40 bars (40 1H candles = ~40hrs forward).
This mirrors the live bot's actual behavior: it fetches 150 1H bars,
and OBs formed at bar 90 are only checked against bars 91-150 (~60 bars max).
Using 40 is conservative and avoids over-fitting.

HOW TO RUN:
  pip install ccxt pandas numpy ta
  python smc_backtester_v2.py

OUTPUTS:
  smc_backtest_results.csv   — every trade with full details
  smc_backtest_report.txt    — share this with Claude
  smc_near_misses.csv        — signals that almost fired (for analysis)
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
logging.basicConfig(level=logging.WARNING)

# ══════════════════════════════════════════════════════════════
#  CONFIG
# ══════════════════════════════════════════════════════════════
PAIRS = [
    "BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "XRP/USDT",
    "DOGE/USDT", "ADA/USDT", "AVAX/USDT", "LINK/USDT", "DOT/USDT",
    "LTC/USDT", "MATIC/USDT", "UNI/USDT", "ATOM/USDT", "NEAR/USDT",
]
DAYS_BACK              = 90
MIN_SCORE_SHORT        = 80
MIN_SCORE_LONG         = 82
TRIPLE_EMA_REQUIRED    = True
TRIPLE_EMA_BYPASS      = 87
HH_LL_BONUS            = 8
HH_LL_LOOKBACK         = 10
OB_TOLERANCE_PCT       = 0.008
OB_IMPULSE_ATR_MULT    = 1.0
OB_VIOLATION_WINDOW    = 40    # ← KEY FIX: was len(df) (entire history), now 40 bars
STRUCTURE_LOOKBACK     = 20
TP1_RR = 1.5;  TP2_RR = 2.5;  TP3_RR = 4.0
TP1_CLOSE = 0.50;  TP2_CLOSE = 0.30;  TP3_CLOSE = 0.20
TRADE_TIMEOUT_H = 48
MIN_BARS_BETWEEN_SIGNALS = 4   # avoid firing multiple signals on same setup


# ══════════════════════════════════════════════════════════════
#  INDICATORS
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

        stoch = ta.momentum.StochRSIIndicator(df['close'])
        df['srsi_k'] = stoch.stochrsi_k()
        df['srsi_d'] = stoch.stochrsi_d()

        df['atr'] = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close']).average_true_range()

        df['vol_sma']   = df['volume'].rolling(20).mean()
        df['vol_ratio'] = df['volume'] / df['vol_sma'].replace(0, np.nan)

        tp_col = (df['high'] + df['low'] + df['close']) / 3
        df['vwap'] = (tp_col * df['volume']).cumsum() / df['volume'].cumsum()

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
        df['hammer']       = ((lw > body*2.0) & (lw > uw*1.5)).astype(int)
        df['shooting_star']= ((uw > body*2.0) & (uw > lw*1.5)).astype(int)
    except Exception:
        pass
    return df


# ══════════════════════════════════════════════════════════════
#  SMC ENGINE
# ══════════════════════════════════════════════════════════════
class SMCEngine:

    def swing_highs_lows(self, df, left=4, right=4):
        highs, lows = [], []
        n = len(df)
        for i in range(left, n - right):
            hi = df['high'].iloc[i]; lo = df['low'].iloc[i]
            if all(hi >= df['high'].iloc[i-left:i]) and all(hi >= df['high'].iloc[i+1:i+right+1]):
                highs.append({'i': i, 'price': hi})
            if all(lo <= df['low'].iloc[i-left:i]) and all(lo <= df['low'].iloc[i+1:i+right+1]):
                lows.append({'i': i, 'price': lo})
        return highs, lows

    def check_4h_hh_ll(self, df_4h, direction, lookback=HH_LL_LOOKBACK):
        n = len(df_4h)
        if n < lookback * 2: return False
        recent = df_4h.iloc[-lookback:]; prior = df_4h.iloc[-(lookback*2):-lookback]
        if direction == 'LONG': return recent['high'].max() > prior['high'].max()
        else:                   return recent['low'].min()  < prior['low'].min()

    def is_triple_ema_bull(self, df_4h):
        l = df_4h.iloc[-1]
        return float(l.get('ema_21', 0)) > float(l.get('ema_50', 0)) > float(l.get('ema_200', 0))

    def detect_structure_break(self, df, highs, lows, lookback=STRUCTURE_LOOKBACK):
        events = []; close = df['close']; n = len(df); start = max(0, n-lookback-15)
        for k in range(1, len(highs)):
            ph = highs[k-1]; ch = highs[k]
            if ch['i'] < start: continue
            level = ph['price']
            for j in range(ch['i'], min(ch['i']+10, n)):
                if close.iloc[j] > level:
                    events.append({'kind': 'BOS_BULL' if ch['price']>ph['price'] else 'MSS_BULL', 'level': level, 'bar': j})
                    break
        for k in range(1, len(lows)):
            pl = lows[k-1]; cl = lows[k]
            if cl['i'] < start: continue
            level = pl['price']
            for j in range(cl['i'], min(cl['i']+10, n)):
                if close.iloc[j] < level:
                    events.append({'kind': 'BOS_BEAR' if cl['price']<pl['price'] else 'MSS_BEAR', 'level': level, 'bar': j})
                    break
        if not events: return None
        latest = sorted(events, key=lambda x: x['bar'])[-1]
        if latest['bar'] < n - lookback: return None
        return latest

    def find_order_blocks(self, df, direction, lookback=60):
        """
        FIXED: violation check uses OB_VIOLATION_WINDOW (40 bars) instead of
        entire remaining history. This matches the live bot's effective behavior
        where it only has ~150 bars total and OBs near bar 90 are checked
        against ~60 bars of history before signal bar.
        """
        obs = []; n = len(df); start = max(2, n - lookback)
        for i in range(start, n - 2):
            c = df.iloc[i]
            atr_local = float(df['atr'].iloc[i]) if 'atr' in df.columns and not pd.isna(df['atr'].iloc[i]) else (c['high'] - c['low'])
            if atr_local <= 0: atr_local = c['high'] - c['low']
            min_impulse = atr_local * OB_IMPULSE_ATR_MULT
            viol_end = min(i + 1 + OB_VIOLATION_WINDOW, n)

            if direction == 'LONG':
                if c['close'] >= c['open']: continue
                fwd = df['high'].iloc[i+1:min(i+6, n)]
                if len(fwd) == 0: continue
                if fwd.max() - c['low'] < min_impulse: continue
                ob = {'top': max(c['open'],c['close']), 'bottom': c['low'],
                      'mid': (max(c['open'],c['close'])+c['low'])/2, 'bar': i}
                if (df['close'].iloc[i+1:viol_end] < (ob['top']+ob['bottom'])/2).any(): continue
                obs.append(ob)
            else:
                if c['close'] <= c['open']: continue
                fwd = df['low'].iloc[i+1:min(i+6, n)]
                if len(fwd) == 0: continue
                if c['high'] - fwd.min() < min_impulse: continue
                ob = {'top': c['high'], 'bottom': min(c['open'],c['close']),
                      'mid': (c['high']+min(c['open'],c['close']))/2, 'bar': i}
                if (df['close'].iloc[i+1:viol_end] > (ob['top']+ob['bottom'])/2).any(): continue
                obs.append(ob)
        obs.sort(key=lambda x: x['bar'], reverse=True)
        return obs

    def price_in_ob(self, price, ob):
        tol = ob['top'] * OB_TOLERANCE_PCT
        return (ob['bottom'] - tol) <= price <= (ob['top'] + tol)

    def find_fvg(self, df, direction, lookback=25):
        fvgs = []; n = len(df)
        for i in range(max(1, n-lookback), n-1):
            prev = df.iloc[i-1]; nxt = df.iloc[i+1]
            if direction == 'LONG' and prev['high'] < nxt['low']:
                fvgs.append({'top': nxt['low'], 'bottom': prev['high'], 'bar': i})
            elif direction == 'SHORT' and prev['low'] > nxt['high']:
                fvgs.append({'top': prev['low'], 'bottom': nxt['high'], 'bar': i})
        return fvgs

    def recent_liquidity_sweep(self, df, direction, highs, lows, lookback=25):
        n = len(df); start = n - lookback
        if direction == 'LONG':
            for sl in reversed(lows):
                if sl['i'] < start: continue
                level = sl['price']
                for j in range(sl['i']+1, min(sl['i']+8, n)):
                    c = df.iloc[j]
                    if c['low'] < level and c['close'] > level:
                        return {'level': level, 'bar': j}
        else:
            for sh in reversed(highs):
                if sh['i'] < start: continue
                level = sh['price']
                for j in range(sh['i']+1, min(sh['i']+8, n)):
                    c = df.iloc[j]
                    if c['high'] > level and c['close'] < level:
                        return {'level': level, 'bar': j}
        return None

    def pd_zone(self, df_4h, price):
        hi = df_4h['high'].iloc[-50:].max(); lo = df_4h['low'].iloc[-50:].min()
        rang = hi - lo
        if rang == 0: return 'NEUTRAL', 0.5
        pos = (price - lo) / rang
        if pos < 0.40: return 'DISCOUNT', pos
        if pos > 0.60: return 'PREMIUM', pos
        return 'NEUTRAL', pos


# ══════════════════════════════════════════════════════════════
#  SCORER
# ══════════════════════════════════════════════════════════════
def score_setup(direction, ob, structure, sweep, fvg_near,
                df_1h, df_15m, df_4h, pd_label, hh_ll_ok):
    score = 0
    l1 = df_1h.iloc[-1]; p1 = df_1h.iloc[-2]
    l15 = df_15m.iloc[-1]; l4 = df_4h.iloc[-1]

    if structure:
        score += 20 if 'MSS' in structure['kind'] else 14
    if ob:
        pct = (ob['top']-ob['bottom'])/ob['bottom']*100
        score += 20 if pct < 0.8 else (13 if pct < 2.0 else 7)

    e21=l4.get('ema_21',0); e50=l4.get('ema_50',0); e200=l4.get('ema_200',0)
    if direction == 'LONG':
        if e21>e50>e200: score+=15
        elif e21>e50:    score+=10
        elif pd_label=='DISCOUNT': score+=6
    else:
        if e21<e50<e200: score+=15
        elif e21<e50:    score+=10
        elif pd_label=='PREMIUM': score+=6

    if hh_ll_ok: score += HH_LL_BONUS

    trigger = False
    if direction == 'LONG':
        if   l1.get('bull_engulf',0)==1: score+=25; trigger=True
        elif l1.get('bull_pin',0)==1:    score+=22; trigger=True
        elif l1.get('hammer',0)==1:      score+=18; trigger=True
        elif p1.get('bull_engulf',0)==1: score+=14; trigger=True
        elif p1.get('bull_pin',0)==1:    score+=11; trigger=True
        elif p1.get('hammer',0)==1:      score+=9;  trigger=True
    else:
        if   l1.get('bear_engulf',0)==1:   score+=25; trigger=True
        elif l1.get('bear_pin',0)==1:      score+=22; trigger=True
        elif l1.get('shooting_star',0)==1: score+=18; trigger=True
        elif p1.get('bear_engulf',0)==1:   score+=14; trigger=True
        elif p1.get('bear_pin',0)==1:      score+=11; trigger=True
        elif p1.get('shooting_star',0)==1: score+=9;  trigger=True
    if not trigger: score -= 12

    rsi1=l1.get('rsi',50); macd1=l1.get('macd',0); ms1=l1.get('macd_signal',0)
    pm1=p1.get('macd',0);  pms1=p1.get('macd_signal',0)
    sk1=l1.get('srsi_k',0.5); sd1=l1.get('srsi_d',0.5)

    if direction == 'LONG':
        if 28<=rsi1<=55:             score+=4
        elif rsi1<28:                score+=3
        if macd1>ms1 and pm1<=pms1: score+=5
        elif macd1>ms1:              score+=2
        if sk1<0.3 and sk1>sd1:     score+=3
    else:
        if 45<=rsi1<=72:             score+=4
        elif rsi1>72:                score+=3
        if macd1<ms1 and pm1>=pms1: score+=5
        elif macd1<ms1:              score+=2
        if sk1>0.7 and sk1<sd1:     score+=3

    extras = 0
    if sweep:    extras += 4
    if fvg_near: extras += 3
    vr = l15.get('vol_ratio', 1.0)
    if   vr >= 2.5: extras += 3
    elif vr >= 1.5: extras += 1
    close1=l1.get('close',0); vwap1=l1.get('vwap',0)
    if direction=='LONG'  and close1<vwap1: extras=min(extras+1,10)
    elif direction=='SHORT' and close1>vwap1: extras=min(extras+1,10)
    score += min(extras, 10)

    return max(0, min(int(score), 100)), trigger


# ══════════════════════════════════════════════════════════════
#  GATE LOGIC + SIGNAL DETECTION
# ══════════════════════════════════════════════════════════════
smc = SMCEngine()

def detect_signal_at(df4, df1, df15):
    """
    Returns (signal_dict or None, reject_reason_string)
    """
    try:
        if len(df1)<80 or len(df15)<40 or len(df4)<60:
            return None, 'not_enough_data'

        price = df1['close'].iloc[-1]
        l4 = df4.iloc[-1]
        e21=l4.get('ema_21',0); e50=l4.get('ema_50',0)

        if   e21 > e50: bias = 'LONG'
        elif e21 < e50: bias = 'SHORT'
        else:           return None, 'ema_flat'

        triple_ema = smc.is_triple_ema_bull(df4)
        hh_ll_ok   = smc.check_4h_hh_ll(df4, bias, HH_LL_LOOKBACK)

        pd_label, pd_pos = smc.pd_zone(df4, price)
        if bias=='LONG'  and pd_label=='PREMIUM':  return None, 'pd_zone_premium_long'
        if bias=='SHORT' and pd_label=='DISCOUNT': return None, 'pd_zone_discount_short'

        highs1, lows1 = smc.swing_highs_lows(df1, 4, 4)
        structure = smc.detect_structure_break(df1, highs1, lows1, STRUCTURE_LOOKBACK)
        if structure:
            if bias=='LONG'  and 'BEAR' in structure['kind']: return None, 'structure_opposes_long'
            if bias=='SHORT' and 'BULL' in structure['kind']: return None, 'structure_opposes_short'

        obs = smc.find_order_blocks(df1, bias, lookback=60)
        if not obs: return None, 'no_ob_found'

        active_ob = None
        for ob in obs:
            if smc.price_in_ob(price, ob):
                active_ob = ob; break
        if not active_ob:
            nearest = obs[0]
            dist = min(abs(price-nearest['top']),abs(price-nearest['bottom']))/price*100
            return None, f'price_not_at_ob_{dist:.1f}pct_away'

        fvgs = smc.find_fvg(df1, bias, 25)
        fvg_near = next((f for f in fvgs if f['bottom']<active_ob['top'] and f['top']>active_ob['bottom']), None)
        sweep = smc.recent_liquidity_sweep(df1, bias, highs1, lows1, 20)

        score, has_trigger = score_setup(
            bias, active_ob, structure, sweep, fvg_near,
            df1, df15, df4, pd_label, hh_ll_ok
        )

        if not has_trigger: return None, f'no_trigger_score_{score}'

        # Direction gate
        if bias == 'SHORT':
            if score < MIN_SCORE_SHORT: return None, f'score_{score}_below_short_min_{MIN_SCORE_SHORT}'
        else:
            if score < MIN_SCORE_LONG: return None, f'score_{score}_below_long_min_{MIN_SCORE_LONG}'
            if TRIPLE_EMA_REQUIRED and score < TRIPLE_EMA_BYPASS and not triple_ema:
                return None, f'score_{score}_no_triple_ema'

        atr1 = float(df1['atr'].iloc[-1])
        entry = price
        if bias == 'LONG':
            sl = min(active_ob['bottom'] - atr1*0.2, entry - atr1*0.6)
        else:
            sl = max(active_ob['top'] + atr1*0.2, entry + atr1*0.6)

        risk = abs(entry - sl)
        if risk < entry*0.001: return None, 'degenerate_sl'

        if bias == 'LONG':
            tps = [entry+risk*TP1_RR, entry+risk*TP2_RR, entry+risk*TP3_RR]
        else:
            tps = [entry-risk*TP1_RR, entry-risk*TP2_RR, entry-risk*TP3_RR]

        return {
            'bias': bias, 'score': score,
            'triple_ema': triple_ema, 'hh_ll': hh_ll_ok,
            'entry': entry, 'sl': sl, 'tps': tps, 'risk': risk,
            'risk_pct': risk/entry*100,
            'ob_size_pct': (active_ob['top']-active_ob['bottom'])/active_ob['bottom']*100,
            'has_fvg': fvg_near is not None,
            'has_sweep': sweep is not None,
            'pd_zone': pd_label, 'pd_pos': round(pd_pos,3),
            'structure': structure['kind'] if structure else 'NONE',
            'has_trigger': has_trigger,
        }, 'passed'

    except Exception as e:
        return None, f'exception_{str(e)[:40]}'


# ══════════════════════════════════════════════════════════════
#  TRADE SIMULATOR
# ══════════════════════════════════════════════════════════════
def simulate_trade(sig, future_1h):
    bias=sig['bias']; entry=sig['entry']; sl=sig['sl']; tps=sig['tps']; risk=sig['risk']
    tp_hit=[False,False,False]; remaining=1.0; realized_r=0.0; outcome='TIMEOUT'
    weights=[TP1_CLOSE, TP2_CLOSE, TP3_CLOSE]
    max_adverse=0.0; bars_to_outcome=len(future_1h)

    for bar_n, (i, row) in enumerate(future_1h.iterrows()):
        h,l = row['high'],row['low']
        mae = (entry-l)/risk if bias=='LONG' else (h-entry)/risk
        max_adverse = max(max_adverse, mae)

        # SL check first
        if bias=='LONG' and l<=sl:
            realized_r -= remaining; outcome='SL'; bars_to_outcome=bar_n+1; break
        elif bias=='SHORT' and h>=sl:
            realized_r -= remaining; outcome='SL'; bars_to_outcome=bar_n+1; break

        for t_idx, tp in enumerate(tps):
            if tp_hit[t_idx]: continue
            hit = (bias=='LONG' and h>=tp) or (bias=='SHORT' and l<=tp)
            if hit:
                tp_hit[t_idx]=True
                realized_r += weights[t_idx]*[TP1_RR,TP2_RR,TP3_RR][t_idx]
                remaining  -= weights[t_idx]
                if t_idx==2: outcome='TP3'; bars_to_outcome=bar_n+1
                elif t_idx==1 and not tp_hit[2]: outcome='TP2_open'
                elif t_idx==0 and not (tp_hit[1] or tp_hit[2]): outcome='TP1_open'
        if remaining <= 0.01: break

    if outcome=='TIMEOUT' and any(tp_hit):
        outcome='PARTIAL_TP'
    if 'TP3' in outcome: outcome='TP3'
    elif 'TP2' in outcome and tp_hit[1]: outcome='TP2' if not tp_hit[2] else 'TP3'
    elif 'TP1' in outcome and tp_hit[0]: outcome='TP1'

    return {
        'outcome': outcome, 'realized_r': round(realized_r,4),
        'won': realized_r>0,
        'tp1_hit':tp_hit[0], 'tp2_hit':tp_hit[1], 'tp3_hit':tp_hit[2],
        'max_adverse_r': round(max_adverse,4),
        'bars_to_outcome': bars_to_outcome,
    }


# ══════════════════════════════════════════════════════════════
#  DATA FETCH
# ══════════════════════════════════════════════════════════════
async def fetch_all(exchange, symbol, days_back):
    result = {}
    limits = {
        '4h':  min(1000, max(220, days_back*6+220)),
        '1h':  min(1000, max(150, days_back*24+150)),
        '15m': min(1000, max(80,  days_back*96+80)),
    }
    for tf, lim in limits.items():
        raw = await exchange.fetch_ohlcv(symbol, tf, limit=lim)
        df  = pd.DataFrame(raw, columns=['ts','open','high','low','close','volume'])
        df['ts'] = pd.to_datetime(df['ts'], unit='ms')
        result[tf] = add_indicators(df.reset_index(drop=True))
        await asyncio.sleep(0.12)
    return result


# ══════════════════════════════════════════════════════════════
#  WALK-FORWARD
# ══════════════════════════════════════════════════════════════
def run_walk_forward(data, symbol, days_back):
    df4=data['4h']; df1=data['1h']; df15=data['15m']
    end_dt  = df1['ts'].iloc[-1]
    start_dt= end_dt - timedelta(days=days_back)
    test_idx= df1[df1['ts']>=start_dt].index.tolist()

    trades=[]; near_misses=[]; reject_counts={}
    active_end=-1; last_signal_bar=-999

    print(f"  {symbol}: {len(test_idx)} bars  ...", end=' ', flush=True)

    for idx in test_idx:
        if idx < 100: continue
        if idx <= active_end: continue
        if idx - last_signal_bar < MIN_BARS_BETWEEN_SIGNALS: continue

        df1_s  = df1.iloc[:idx+1]
        ts_now = df1_s['ts'].iloc[-1]
        df4_s  = df4[df4['ts']<=ts_now]
        df15_s = df15[df15['ts']<=ts_now]
        if len(df4_s)<60 or len(df15_s)<40: continue

        sig, reason = detect_signal_at(df4_s, df1_s, df15_s)
        reject_counts[reason] = reject_counts.get(reason,0) + 1

        if sig is None:
            # Log near-misses that made it past OB gate
            if reason.startswith('score') or reason.startswith('no_trigger') or reason.startswith('no_triple'):
                near_misses.append({
                    'symbol': symbol, 'date': ts_now.strftime('%Y-%m-%d %H:%M'),
                    'reason': reason,
                })
            continue

        future = df1.iloc[idx+1:idx+1+TRADE_TIMEOUT_H].copy()
        if len(future) < 2: continue

        result = simulate_trade(sig, future)
        last_signal_bar = idx
        active_end = idx + result['bars_to_outcome'] + 1

        trades.append({
            'symbol':       symbol,
            'date':         ts_now.strftime('%Y-%m-%d %H:%M'),
            'direction':    sig['bias'],
            'score':        sig['score'],
            'triple_ema':   sig['triple_ema'],
            'hh_ll':        sig['hh_ll'],
            'pd_zone':      sig['pd_zone'],
            'structure':    sig['structure'],
            'ob_size_pct':  round(sig['ob_size_pct'],3),
            'has_fvg':      sig['has_fvg'],
            'has_sweep':    sig['has_sweep'],
            'entry':        round(sig['entry'],6),
            'sl':           round(sig['sl'],6),
            'tp1':          round(sig['tps'][0],6),
            'tp2':          round(sig['tps'][1],6),
            'tp3':          round(sig['tps'][2],6),
            'risk_pct':     round(sig['risk_pct'],3),
            'outcome':      result['outcome'],
            'realized_r':   result['realized_r'],
            'won':          result['won'],
            'tp1_hit':      result['tp1_hit'],
            'tp2_hit':      result['tp2_hit'],
            'tp3_hit':      result['tp3_hit'],
            'max_adverse_r':result['max_adverse_r'],
            'bars_held':    result['bars_to_outcome'],
        })

    top_rejects = sorted(reject_counts.items(), key=lambda x:-x[1])[:5]
    reject_str = ' | '.join(f'{k}={v}' for k,v in top_rejects if k != 'passed')
    print(f"{len(trades)} signals | top rejects: {reject_str}")
    return trades, near_misses


# ══════════════════════════════════════════════════════════════
#  REPORT
# ══════════════════════════════════════════════════════════════
def report(all_trades, all_near_misses):
    df = pd.DataFrame(all_trades)
    if df.empty:
        return "❌ 0 trades. Check reject reasons above.", df

    sep = "═"*62
    L = [sep, "  SMC PRO v4.1 — BACKTEST REPORT",
         f"  {datetime.now().strftime('%Y-%m-%d %H:%M UTC')}",
         f"  {DAYS_BACK}d | {len(PAIRS)} pairs | OB viol window={OB_VIOLATION_WINDOW}bars",
         sep]

    def stats_block(sub, label):
        if len(sub)==0: return
        w=sub['won'].sum(); t=len(sub)
        wr=w/t*100; avg_r=sub['realized_r'].mean(); total_r=sub['realized_r'].sum()
        run=0; peak=0; max_dd=0
        for r in sub['realized_r']:
            run+=r
            if run>peak: peak=run
            dd=peak-run
            if dd>max_dd: max_dd=dd
        L.append(f"\n  {label} ({t} trades)")
        L.append(f"  {'─'*44}")
        L.append(f"  WR:         {wr:.1f}%  ({w}W / {t-w}L)")
        L.append(f"  Avg R:      {avg_r:+.3f}R")
        L.append(f"  Total R:    {total_r:+.2f}R")
        L.append(f"  Max DD:     -{max_dd:.2f}R")
        L.append(f"  Signals/wk: {t/(DAYS_BACK/7):.1f}")

    stats_block(df, "OVERALL")

    L.append(f"\n  BY DIRECTION"); L.append(f"  {'─'*44}")
    for d in ['LONG','SHORT']:
        sub=df[df['direction']==d]
        if len(sub)==0: continue
        w=sub['won'].sum()
        L.append(f"  {d}: {len(sub):>3}t | WR={w/len(sub)*100:.1f}% | avg={sub['realized_r'].mean():+.3f}R | total={sub['realized_r'].sum():+.2f}R")

    L.append(f"\n  SCORE BUCKETS"); L.append(f"  {'─'*44}")
    for lo,hi in [(0,74),(75,79),(80,84),(85,89),(90,100)]:
        sub=df[(df['score']>=lo)&(df['score']<=hi)]
        if len(sub)==0: continue
        w=sub['won'].sum()
        L.append(f"  {lo:>2}-{hi}: {len(sub):>3}t | WR={w/len(sub)*100:.1f}% | avg={sub['realized_r'].mean():+.3f}R")

    L.append(f"\n  OUTCOME BREAKDOWN"); L.append(f"  {'─'*44}")
    for out in ['TP3','TP2','TP1','PARTIAL_TP','SL','TIMEOUT']:
        n=len(df[df['outcome']==out])
        if n: L.append(f"  {out:<12}: {n:>3} ({n/len(df)*100:.0f}%)")
    L.append(f"  TP1+ hit: {df['tp1_hit'].sum()} | TP2+ hit: {df['tp2_hit'].sum()} | TP3 hit: {df['tp3_hit'].sum()}")

    L.append(f"\n  FILTER ANALYSIS"); L.append(f"  {'─'*44}")
    for flag,label in [('hh_ll','HH/LL'),('triple_ema','TripleEMA'),('has_fvg','FVG'),('has_sweep','Sweep')]:
        y=df[df[flag]==True]; n2=df[df[flag]==False]
        if len(y): L.append(f"  {label} YES {len(y):>3}t | WR={y['won'].sum()/len(y)*100:.1f}% | avg={y['realized_r'].mean():+.3f}R")
        if len(n2): L.append(f"  {label} NO  {len(n2):>3}t | WR={n2['won'].sum()/len(n2)*100:.1f}% | avg={n2['realized_r'].mean():+.3f}R")

    L.append(f"\n  OB SIZE ANALYSIS"); L.append(f"  {'─'*44}")
    for lo,hi in [(0,0.8),(0.8,2.0),(2.0,99)]:
        sub=df[(df['ob_size_pct']>=lo)&(df['ob_size_pct']<hi)]
        if len(sub)==0: continue
        w=sub['won'].sum()
        L.append(f"  OB {lo:.1f}-{hi:.1f}%: {len(sub):>3}t | WR={w/len(sub)*100:.1f}% | avg={sub['realized_r'].mean():+.3f}R")

    L.append(f"\n  PER PAIR"); L.append(f"  {'─'*44}")
    for sym in sorted(df['symbol'].unique()):
        sub=df[df['symbol']==sym]; w=sub['won'].sum()
        L.append(f"  {sym:<12}: {len(sub):>3}t | WR={w/len(sub)*100:.1f}% | avg={sub['realized_r'].mean():+.3f}R")

    L.append(f"\n  NEAR MISSES SUMMARY ({len(all_near_misses)} bars reached OB gate)")
    L.append(f"  {'─'*44}")
    if all_near_misses:
        nm_df = pd.DataFrame(all_near_misses)
        by_reason = nm_df.groupby('reason').size().sort_values(ascending=False).head(8)
        for reason, cnt in by_reason.items():
            L.append(f"  {reason:<45}: {cnt}")

    L.append(f"\n{sep}")
    L.append("  Share smc_backtest_report.txt + smc_backtest_results.csv")
    L.append(sep)
    return "\n".join(L), df


# ══════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════
async def main():
    print(f"\n{'═'*62}")
    print(f"  SMC PRO v4.1 — BACKTESTER v2 (OB violation bug fixed)")
    print(f"  Pairs={len(PAIRS)} | Days={DAYS_BACK} | OB_viol_window={OB_VIOLATION_WINDOW}bars")
    print(f"  SHORT≥{MIN_SCORE_SHORT} | LONG≥{MIN_SCORE_LONG}+TripleEMA(bypass≥{TRIPLE_EMA_BYPASS})")
    print(f"{'═'*62}\n")

    exchange = ccxt.binance({'enableRateLimit': True, 'options':{'defaultType':'spot'}})
    all_trades=[]; all_near=[]; 

    for sym in PAIRS:
        print(f"⬇️  {sym} ...", end=' ', flush=True)
        try:
            data = await fetch_all(exchange, sym, DAYS_BACK)
            t, nm = run_walk_forward(data, sym, DAYS_BACK)
            all_trades.extend(t); all_near.extend(nm)
        except Exception as e:
            print(f"ERROR: {e}")
        await asyncio.sleep(0.5)

    await exchange.close()

    print(f"\n{'─'*62}\n📊 Generating report...\n")
    rpt, df = report(all_trades, all_near)
    print(rpt)

    df.to_csv("smc_backtest_results.csv", index=False)
    print(f"\n✅ smc_backtest_results.csv  ({len(df)} trades)")

    with open("smc_backtest_report.txt","w") as f:
        f.write(rpt)
    print(f"✅ smc_backtest_report.txt")

    if all_near:
        pd.DataFrame(all_near).to_csv("smc_near_misses.csv", index=False)
        print(f"✅ smc_near_misses.csv  ({len(all_near)} near-misses)")

    print("\n💡 Share all 3 files to rebuild the strategy.\n")


if __name__ == "__main__":
    asyncio.run(main())
