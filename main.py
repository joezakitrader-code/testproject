"""
SMC PRO v4.1 — BACKTESTER v3
══════════════════════════════════════════════════════════════════
FIXES from v2 (which still got 0 signals):

  ROOT CAUSE ANALYSIS of 0 signals:
  ─────────────────────────────────
  BTC:  pd_zone_premium_long=87   ← 87 LONGs killed because price
                                     was in "premium" per 50-bar range
  MATIC: pd_zone_discount_short=225 ← 225 SHORTs killed because price
                                       was in "discount" per 50-bar range

  The PD zone uses df_4h['high/low'].iloc[-50:] — in a 90-day downtrend,
  ALL 50-bar windows show price at the top of range = "premium" = no longs.
  In a bear market the filter correctly rejects longs... but this means the
  entire 90-day backtest period = 0 signals if the market is trending.

  FIX 1: PD zone uses 200-bar range (not 50), giving a more meaningful
          price context across the full backtest.

  FIX 2: Structure gate softened — "opposes" now only blocks if a RECENT
          (last 10 bars) opposing break exists. Older opposing structure
          is allowed (market may have reversed since).

  FIX 3: Score threshold temporarily lowered to 70/72 to diagnose what
          scores are actually achievable. We'll see the distribution and
          can raise it back.

  FIX 4: Added score-only mode that logs ALL setups that reach OB gate,
          regardless of score, so we can see the full distribution.

HOW TO RUN:
  pip install ccxt pandas numpy ta
  python smc_backtester_v3.py

OUTPUTS:
  smc_v3_results.csv       — trades that passed all filters
  smc_v3_all_setups.csv    — ALL setups reaching OB gate (score distribution)
  smc_v3_report.txt        — summary report
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
DAYS_BACK           = 90
HH_LL_BONUS         = 8
HH_LL_LOOKBACK      = 10
OB_TOLERANCE_PCT    = 0.010    # slightly wider (was 0.008)
OB_IMPULSE_ATR_MULT = 0.8     # slightly relaxed (was 1.0)
OB_VIOLATION_WINDOW = 40
STRUCTURE_LOOKBACK  = 20
PD_ZONE_BARS        = 200     # FIX: was 50, now 200 for meaningful context
STRUCTURE_OPPOSE_BARS = 10    # FIX: only block if opposing structure within last N bars

# Score thresholds — run at 70/72 to see distribution, raise later
MIN_SCORE_SHORT     = 70      # was 80
MIN_SCORE_LONG      = 72      # was 82
TRIPLE_EMA_REQUIRED = True
TRIPLE_EMA_BYPASS   = 80     # was 87

TP1_RR=1.5; TP2_RR=2.5; TP3_RR=4.0
TP1_CLOSE=0.50; TP2_CLOSE=0.30; TP3_CLOSE=0.20
TRADE_TIMEOUT_H=48
MIN_BARS_BETWEEN_SIGNALS=4


# ══════════════════════════════════════════════════════════════
#  INDICATORS
# ══════════════════════════════════════════════════════════════
def add_indicators(df):
    if len(df) < 55: return df
    try:
        df = df.copy()
        df['ema_21']  = ta.trend.EMAIndicator(df['close'], 21).ema_indicator()
        df['ema_50']  = ta.trend.EMAIndicator(df['close'], 50).ema_indicator()
        df['ema_200'] = ta.trend.EMAIndicator(df['close'], min(200, len(df)-1)).ema_indicator()
        df['rsi']     = ta.momentum.RSIIndicator(df['close'], 14).rsi()
        macd = ta.trend.MACD(df['close'])
        df['macd'] = macd.macd(); df['macd_signal'] = macd.macd_signal()
        stoch = ta.momentum.StochRSIIndicator(df['close'])
        df['srsi_k'] = stoch.stochrsi_k(); df['srsi_d'] = stoch.stochrsi_d()
        df['atr'] = ta.volatility.AverageTrueRange(df['high'],df['low'],df['close']).average_true_range()
        df['vol_sma']   = df['volume'].rolling(20).mean()
        df['vol_ratio'] = df['volume'] / df['vol_sma'].replace(0, np.nan)
        tp_col = (df['high']+df['low']+df['close'])/3
        df['vwap'] = (tp_col*df['volume']).cumsum() / df['volume'].cumsum()
        body=(df['close']-df['open']).abs()
        uw=df['high']-df[['open','close']].max(axis=1)
        lw=df[['open','close']].min(axis=1)-df['low']
        df['bull_engulf']=((df['close'].shift(1)<df['open'].shift(1))&(df['close']>df['open'])&(df['close']>df['open'].shift(1))&(df['open']<df['close'].shift(1))).astype(int)
        df['bear_engulf']=((df['close'].shift(1)>df['open'].shift(1))&(df['close']<df['open'])&(df['close']<df['open'].shift(1))&(df['open']>df['close'].shift(1))).astype(int)
        df['bull_pin']=((lw>body*2.5)&(lw>uw*2)&(df['close']>df['open'])).astype(int)
        df['bear_pin']=((uw>body*2.5)&(uw>lw*2)&(df['close']<df['open'])).astype(int)
        df['hammer']=((lw>body*2.0)&(lw>uw*1.5)).astype(int)
        df['shooting_star']=((uw>body*2.0)&(uw>lw*1.5)).astype(int)
    except Exception: pass
    return df


# ══════════════════════════════════════════════════════════════
#  SMC ENGINE
# ══════════════════════════════════════════════════════════════
class SMCEngine:

    def swing_highs_lows(self, df, left=4, right=4):
        highs=[]; lows=[]; n=len(df)
        for i in range(left, n-right):
            hi=df['high'].iloc[i]; lo=df['low'].iloc[i]
            if all(hi>=df['high'].iloc[i-left:i]) and all(hi>=df['high'].iloc[i+1:i+right+1]): highs.append({'i':i,'price':hi})
            if all(lo<=df['low'].iloc[i-left:i]) and all(lo<=df['low'].iloc[i+1:i+right+1]): lows.append({'i':i,'price':lo})
        return highs, lows

    def check_4h_hh_ll(self, df_4h, direction):
        n=len(df_4h)
        if n<HH_LL_LOOKBACK*2: return False
        recent=df_4h.iloc[-HH_LL_LOOKBACK:]; prior=df_4h.iloc[-HH_LL_LOOKBACK*2:-HH_LL_LOOKBACK]
        return recent['high'].max()>prior['high'].max() if direction=='LONG' else recent['low'].min()<prior['low'].min()

    def is_triple_ema_bull(self, df_4h):
        l=df_4h.iloc[-1]
        return float(l.get('ema_21',0))>float(l.get('ema_50',0))>float(l.get('ema_200',0))

    def detect_structure_break(self, df, highs, lows, lookback=STRUCTURE_LOOKBACK):
        events=[]; close=df['close']; n=len(df); start=max(0,n-lookback-15)
        for k in range(1,len(highs)):
            ph=highs[k-1]; ch=highs[k]
            if ch['i']<start: continue
            level=ph['price']
            for j in range(ch['i'],min(ch['i']+10,n)):
                if close.iloc[j]>level:
                    events.append({'kind':'BOS_BULL' if ch['price']>ph['price'] else 'MSS_BULL','level':level,'bar':j}); break
        for k in range(1,len(lows)):
            pl=lows[k-1]; cl=lows[k]
            if cl['i']<start: continue
            level=pl['price']
            for j in range(cl['i'],min(cl['i']+10,n)):
                if close.iloc[j]<level:
                    events.append({'kind':'BOS_BEAR' if cl['price']<pl['price'] else 'MSS_BEAR','level':level,'bar':j}); break
        if not events: return None
        latest=sorted(events,key=lambda x:x['bar'])[-1]
        if latest['bar']<n-lookback: return None
        return latest

    def find_order_blocks(self, df, direction, lookback=60):
        obs=[]; n=len(df); start=max(2,n-lookback)
        for i in range(start, n-2):
            c=df.iloc[i]
            atr_local=float(df['atr'].iloc[i]) if 'atr' in df.columns and not pd.isna(df['atr'].iloc[i]) else (c['high']-c['low'])
            if atr_local<=0: atr_local=c['high']-c['low']
            min_impulse=atr_local*OB_IMPULSE_ATR_MULT
            viol_end=min(i+1+OB_VIOLATION_WINDOW,n)
            if direction=='LONG':
                if c['close']>=c['open']: continue
                fwd=df['high'].iloc[i+1:min(i+6,n)]
                if len(fwd)==0 or fwd.max()-c['low']<min_impulse: continue
                ob={'top':max(c['open'],c['close']),'bottom':c['low'],'mid':(max(c['open'],c['close'])+c['low'])/2,'bar':i}
                if (df['close'].iloc[i+1:viol_end]<(ob['top']+ob['bottom'])/2).any(): continue
                obs.append(ob)
            else:
                if c['close']<=c['open']: continue
                fwd=df['low'].iloc[i+1:min(i+6,n)]
                if len(fwd)==0 or c['high']-fwd.min()<min_impulse: continue
                ob={'top':c['high'],'bottom':min(c['open'],c['close']),'mid':(c['high']+min(c['open'],c['close']))/2,'bar':i}
                if (df['close'].iloc[i+1:viol_end]>(ob['top']+ob['bottom'])/2).any(): continue
                obs.append(ob)
        obs.sort(key=lambda x:x['bar'],reverse=True)
        return obs

    def price_in_ob(self, price, ob):
        tol=ob['top']*OB_TOLERANCE_PCT
        return (ob['bottom']-tol)<=price<=(ob['top']+tol)

    def find_fvg(self, df, direction, lookback=25):
        fvgs=[]; n=len(df)
        for i in range(max(1,n-lookback),n-1):
            prev=df.iloc[i-1]; nxt=df.iloc[i+1]
            if direction=='LONG' and prev['high']<nxt['low']: fvgs.append({'top':nxt['low'],'bottom':prev['high'],'bar':i})
            elif direction=='SHORT' and prev['low']>nxt['high']: fvgs.append({'top':prev['low'],'bottom':nxt['high'],'bar':i})
        return fvgs

    def recent_liquidity_sweep(self, df, direction, highs, lows, lookback=25):
        n=len(df); start=n-lookback
        if direction=='LONG':
            for sl in reversed(lows):
                if sl['i']<start: continue
                level=sl['price']
                for j in range(sl['i']+1,min(sl['i']+8,n)):
                    c=df.iloc[j]
                    if c['low']<level and c['close']>level: return {'level':level,'bar':j}
        else:
            for sh in reversed(highs):
                if sh['i']<start: continue
                level=sh['price']
                for j in range(sh['i']+1,min(sh['i']+8,n)):
                    c=df.iloc[j]
                    if c['high']>level and c['close']<level: return {'level':level,'bar':j}
        return None

    def pd_zone(self, df_4h, price):
        """FIX: Use PD_ZONE_BARS (200) instead of 50 for more stable reference range."""
        n=len(df_4h)
        bars=min(PD_ZONE_BARS, n)
        hi=df_4h['high'].iloc[-bars:].max()
        lo=df_4h['low'].iloc[-bars:].min()
        rang=hi-lo
        if rang==0: return 'NEUTRAL',0.5
        pos=(price-lo)/rang
        if pos<0.35: return 'DISCOUNT',pos
        if pos>0.65: return 'PREMIUM',pos
        return 'NEUTRAL',pos


# ══════════════════════════════════════════════════════════════
#  SCORER
# ══════════════════════════════════════════════════════════════
def score_setup(direction, ob, structure, sweep, fvg_near,
                df_1h, df_15m, df_4h, pd_label, hh_ll_ok):
    score=0; reasons=[]
    l1=df_1h.iloc[-1]; p1=df_1h.iloc[-2]; l15=df_15m.iloc[-1]; l4=df_4h.iloc[-1]

    if structure:
        pts=20 if 'MSS' in structure['kind'] else 14
        score+=pts; reasons.append(f"structure_{structure['kind']}+{pts}")
    if ob:
        pct=(ob['top']-ob['bottom'])/ob['bottom']*100
        pts=20 if pct<0.8 else (13 if pct<2.0 else 7)
        score+=pts; reasons.append(f"ob_size_{pct:.1f}pct+{pts}")

    e21=l4.get('ema_21',0); e50=l4.get('ema_50',0); e200=l4.get('ema_200',0)
    if direction=='LONG':
        if e21>e50>e200:     score+=15; reasons.append("triple_ema_bull+15")
        elif e21>e50:        score+=10; reasons.append("ema21>50+10")
        elif pd_label=='DISCOUNT': score+=6; reasons.append("discount_zone+6")
    else:
        if e21<e50<e200:     score+=15; reasons.append("triple_ema_bear+15")
        elif e21<e50:        score+=10; reasons.append("ema21<50+10")
        elif pd_label=='PREMIUM': score+=6; reasons.append("premium_zone+6")

    if hh_ll_ok: score+=HH_LL_BONUS; reasons.append(f"hh_ll+{HH_LL_BONUS}")

    trigger=False; trigger_label=''
    if direction=='LONG':
        if   l1.get('bull_engulf',0)==1: score+=25; trigger=True; trigger_label='bull_engulf_cur'
        elif l1.get('bull_pin',0)==1:    score+=22; trigger=True; trigger_label='bull_pin_cur'
        elif l1.get('hammer',0)==1:      score+=18; trigger=True; trigger_label='hammer_cur'
        elif p1.get('bull_engulf',0)==1: score+=14; trigger=True; trigger_label='bull_engulf_prev'
        elif p1.get('bull_pin',0)==1:    score+=11; trigger=True; trigger_label='bull_pin_prev'
        elif p1.get('hammer',0)==1:      score+=9;  trigger=True; trigger_label='hammer_prev'
    else:
        if   l1.get('bear_engulf',0)==1:   score+=25; trigger=True; trigger_label='bear_engulf_cur'
        elif l1.get('bear_pin',0)==1:      score+=22; trigger=True; trigger_label='bear_pin_cur'
        elif l1.get('shooting_star',0)==1: score+=18; trigger=True; trigger_label='shooting_star_cur'
        elif p1.get('bear_engulf',0)==1:   score+=14; trigger=True; trigger_label='bear_engulf_prev'
        elif p1.get('bear_pin',0)==1:      score+=11; trigger=True; trigger_label='bear_pin_prev'
        elif p1.get('shooting_star',0)==1: score+=9;  trigger=True; trigger_label='shooting_star_prev'
    if not trigger: score-=12
    if trigger: reasons.append(f"{trigger_label}+pts")

    rsi1=l1.get('rsi',50); macd1=l1.get('macd',0); ms1=l1.get('macd_signal',0)
    pm1=p1.get('macd',0); pms1=p1.get('macd_signal',0)
    sk1=l1.get('srsi_k',0.5); sd1=l1.get('srsi_d',0.5)
    if direction=='LONG':
        if 28<=rsi1<=55:           score+=4
        elif rsi1<28:              score+=3
        if macd1>ms1 and pm1<=pms1: score+=5
        elif macd1>ms1:            score+=2
        if sk1<0.3 and sk1>sd1:   score+=3
    else:
        if 45<=rsi1<=72:           score+=4
        elif rsi1>72:              score+=3
        if macd1<ms1 and pm1>=pms1: score+=5
        elif macd1<ms1:            score+=2
        if sk1>0.7 and sk1<sd1:   score+=3

    extras=0
    if sweep:    extras+=4
    if fvg_near: extras+=3
    vr=l15.get('vol_ratio',1.0)
    if vr>=2.5: extras+=3
    elif vr>=1.5: extras+=1
    close1=l1.get('close',0); vwap1=l1.get('vwap',0)
    if direction=='LONG' and close1<vwap1: extras=min(extras+1,10)
    elif direction=='SHORT' and close1>vwap1: extras=min(extras+1,10)
    score+=min(extras,10)

    return max(0,min(int(score),100)), trigger, reasons


# ══════════════════════════════════════════════════════════════
#  SIGNAL DETECTION
# ══════════════════════════════════════════════════════════════
smc=SMCEngine()

def detect_signal_at(df4, df1, df15, log_all=False):
    """
    Returns (signal_dict or None, reject_reason, setup_info_for_logging)
    """
    setup_info = {}
    try:
        if len(df1)<80 or len(df15)<40 or len(df4)<60:
            return None,'not_enough_data',{}

        price=df1['close'].iloc[-1]
        l4=df4.iloc[-1]
        e21=l4.get('ema_21',0); e50=l4.get('ema_50',0)
        if   e21>e50: bias='LONG'
        elif e21<e50: bias='SHORT'
        else:         return None,'ema_flat',{}

        triple_ema=smc.is_triple_ema_bull(df4)
        hh_ll_ok=smc.check_4h_hh_ll(df4,bias)

        pd_label,pd_pos=smc.pd_zone(df4,price)
        if bias=='LONG'  and pd_label=='PREMIUM':  return None,'pd_zone_premium_long',{}
        if bias=='SHORT' and pd_label=='DISCOUNT': return None,'pd_zone_discount_short',{}

        highs1,lows1=smc.swing_highs_lows(df1,4,4)
        structure=smc.detect_structure_break(df1,highs1,lows1,STRUCTURE_LOOKBACK)

        # FIX: Only block if RECENT opposing structure (last N bars)
        if structure:
            n1=len(df1)
            recent_enough=structure['bar']>=(n1-STRUCTURE_OPPOSE_BARS)
            if recent_enough:
                if bias=='LONG'  and 'BEAR' in structure['kind']: return None,'structure_opposes_long_recent',{}
                if bias=='SHORT' and 'BULL' in structure['kind']: return None,'structure_opposes_short_recent',{}

        obs=smc.find_order_blocks(df1,bias,60)
        if not obs: return None,'no_ob_found',{}

        active_ob=None
        for ob in obs:
            if smc.price_in_ob(price,ob): active_ob=ob; break
        if not active_ob:
            nearest=obs[0]
            dist=min(abs(price-nearest['top']),abs(price-nearest['bottom']))/price*100
            return None,f'price_not_at_ob_{dist:.1f}pct',{}

        # ── AT THIS POINT: price is AT an OB — log everything ──
        fvgs=smc.find_fvg(df1,bias,25)
        fvg_near=next((f for f in fvgs if f['bottom']<active_ob['top'] and f['top']>active_ob['bottom']),None)
        sweep=smc.recent_liquidity_sweep(df1,bias,highs1,lows1,20)

        score,has_trigger,reasons=score_setup(
            bias,active_ob,structure,sweep,fvg_near,
            df1,df15,df4,pd_label,hh_ll_ok
        )

        setup_info={
            'bias':bias,'score':score,'trigger':has_trigger,
            'triple_ema':triple_ema,'hh_ll':hh_ll_ok,'pd_zone':pd_label,
            'structure':structure['kind'] if structure else 'NONE',
            'reasons':'|'.join(reasons[:5]),
        }

        if not has_trigger: return None,f'no_trigger_score_{score}',setup_info

        # Score gate
        min_sc=MIN_SCORE_SHORT if bias=='SHORT' else MIN_SCORE_LONG
        if score<min_sc: return None,f'score_{score}_below_{min_sc}',setup_info

        # Triple EMA gate for LONG
        if bias=='LONG' and TRIPLE_EMA_REQUIRED and score<TRIPLE_EMA_BYPASS and not triple_ema:
            return None,f'no_triple_ema_score_{score}',setup_info

        atr1=float(df1['atr'].iloc[-1])
        entry=price
        if bias=='LONG': sl=min(active_ob['bottom']-atr1*0.2, entry-atr1*0.6)
        else:            sl=max(active_ob['top']+atr1*0.2, entry+atr1*0.6)
        risk=abs(entry-sl)
        if risk<entry*0.001: return None,'degenerate_sl',setup_info

        tps=[entry+risk*TP1_RR,entry+risk*TP2_RR,entry+risk*TP3_RR] if bias=='LONG' \
        else [entry-risk*TP1_RR,entry-risk*TP2_RR,entry-risk*TP3_RR]

        return {
            'bias':bias,'score':score,'triple_ema':triple_ema,'hh_ll':hh_ll_ok,
            'entry':entry,'sl':sl,'tps':tps,'risk':risk,
            'risk_pct':risk/entry*100,
            'ob_size_pct':(active_ob['top']-active_ob['bottom'])/active_ob['bottom']*100,
            'has_fvg':fvg_near is not None,'has_sweep':sweep is not None,
            'pd_zone':pd_label,'pd_pos':round(pd_pos,3),
            'structure':structure['kind'] if structure else 'NONE',
            'trigger_label':reasons[5] if len(reasons)>5 else '',
        },'passed',setup_info

    except Exception as e:
        return None,f'exception_{str(e)[:50]}',{}


# ══════════════════════════════════════════════════════════════
#  TRADE SIMULATOR
# ══════════════════════════════════════════════════════════════
def simulate_trade(sig, future_1h):
    bias=sig['bias']; entry=sig['entry']; sl=sig['sl']; tps=sig['tps']; risk=sig['risk']
    tp_hit=[False,False,False]; remaining=1.0; realized_r=0.0; outcome='TIMEOUT'
    weights=[TP1_CLOSE,TP2_CLOSE,TP3_CLOSE]; max_adverse=0.0; bars_used=len(future_1h)

    for bar_n,(i,row) in enumerate(future_1h.iterrows()):
        h,l=row['high'],row['low']
        mae=(entry-l)/risk if bias=='LONG' else (h-entry)/risk
        max_adverse=max(max_adverse,mae)
        if bias=='LONG' and l<=sl:
            realized_r-=remaining; outcome='SL'; bars_used=bar_n+1; break
        elif bias=='SHORT' and h>=sl:
            realized_r-=remaining; outcome='SL'; bars_used=bar_n+1; break
        for t_idx,tp in enumerate(tps):
            if tp_hit[t_idx]: continue
            if (bias=='LONG' and h>=tp) or (bias=='SHORT' and l<=tp):
                tp_hit[t_idx]=True
                realized_r+=weights[t_idx]*[TP1_RR,TP2_RR,TP3_RR][t_idx]
                remaining-=weights[t_idx]
                if t_idx==2: outcome='TP3'; bars_used=bar_n+1
                elif t_idx==1: outcome='TP2'
                elif t_idx==0: outcome='TP1'
        if remaining<=0.01: break

    if outcome=='TIMEOUT' and any(tp_hit): outcome='PARTIAL_TP'
    return {
        'outcome':outcome,'realized_r':round(realized_r,4),'won':realized_r>0,
        'tp1_hit':tp_hit[0],'tp2_hit':tp_hit[1],'tp3_hit':tp_hit[2],
        'max_adverse_r':round(max_adverse,4),'bars_held':bars_used,
    }


# ══════════════════════════════════════════════════════════════
#  DATA FETCH
# ══════════════════════════════════════════════════════════════
async def fetch_all(exchange, symbol, days_back):
    result={}
    limits={'4h':min(1000,max(250,days_back*6+250)),
            '1h':min(1000,max(200,days_back*24+200)),
            '15m':min(1000,max(100,days_back*96+100))}
    for tf,lim in limits.items():
        raw=await exchange.fetch_ohlcv(symbol,tf,limit=lim)
        df=pd.DataFrame(raw,columns=['ts','open','high','low','close','volume'])
        df['ts']=pd.to_datetime(df['ts'],unit='ms')
        result[tf]=add_indicators(df.reset_index(drop=True))
        await asyncio.sleep(0.12)
    return result


# ══════════════════════════════════════════════════════════════
#  WALK-FORWARD
# ══════════════════════════════════════════════════════════════
def run_walk_forward(data, symbol, days_back):
    df4=data['4h']; df1=data['1h']; df15=data['15m']
    end_dt=df1['ts'].iloc[-1]; start_dt=end_dt-timedelta(days=days_back)
    test_idx=df1[df1['ts']>=start_dt].index.tolist()

    trades=[]; all_setups=[]; reject_counts={}
    active_end=-1; last_signal_bar=-999

    print(f"  {symbol}: {len(test_idx)} bars ...", end=' ', flush=True)

    for idx in test_idx:
        if idx<100 or idx<=active_end or idx-last_signal_bar<MIN_BARS_BETWEEN_SIGNALS: continue
        df1_s=df1.iloc[:idx+1]; ts_now=df1_s['ts'].iloc[-1]
        df4_s=df4[df4['ts']<=ts_now]; df15_s=df15[df15['ts']<=ts_now]
        if len(df4_s)<60 or len(df15_s)<40: continue

        sig,reason,setup_info=detect_signal_at(df4_s,df1_s,df15_s)
        reject_counts[reason]=reject_counts.get(reason,0)+1

        # Log all setups that reached OB gate
        if setup_info:
            all_setups.append({'symbol':symbol,'date':ts_now.strftime('%Y-%m-%d %H:%M'),
                               'reason':reason,**setup_info})

        if sig is None: continue

        future=df1.iloc[idx+1:idx+1+TRADE_TIMEOUT_H].copy()
        if len(future)<2: continue

        result=simulate_trade(sig,future)
        last_signal_bar=idx; active_end=idx+result['bars_held']+1

        trades.append({
            'symbol':symbol,'date':ts_now.strftime('%Y-%m-%d %H:%M'),
            'direction':sig['bias'],'score':sig['score'],
            'triple_ema':sig['triple_ema'],'hh_ll':sig['hh_ll'],
            'pd_zone':sig['pd_zone'],'structure':sig['structure'],
            'ob_size_pct':round(sig['ob_size_pct'],3),
            'has_fvg':sig['has_fvg'],'has_sweep':sig['has_sweep'],
            'entry':round(sig['entry'],6),'sl':round(sig['sl'],6),
            'tp1':round(sig['tps'][0],6),'tp2':round(sig['tps'][1],6),'tp3':round(sig['tps'][2],6),
            'risk_pct':round(sig['risk_pct'],3),
            'outcome':result['outcome'],'realized_r':result['realized_r'],
            'won':result['won'],'tp1_hit':result['tp1_hit'],
            'tp2_hit':result['tp2_hit'],'tp3_hit':result['tp3_hit'],
            'max_adverse_r':result['max_adverse_r'],'bars_held':result['bars_held'],
        })

    top_r=sorted(reject_counts.items(),key=lambda x:-x[1])[:4]
    rstr=' | '.join(f'{k}={v}' for k,v in top_r if k!='passed')
    print(f"{len(trades)} signals | {rstr}")
    return trades, all_setups


# ══════════════════════════════════════════════════════════════
#  REPORT
# ══════════════════════════════════════════════════════════════
def report(trades, all_setups):
    df=pd.DataFrame(trades)
    sep="═"*64

    lines=[sep,
           "  SMC PRO — BACKTEST REPORT v3",
           f"  {datetime.now().strftime('%Y-%m-%d %H:%M UTC')}",
           f"  {DAYS_BACK}d | {len(PAIRS)} pairs | Score SHORT≥{MIN_SCORE_SHORT} LONG≥{MIN_SCORE_LONG}",
           f"  PD zone bars={PD_ZONE_BARS} | OB viol window={OB_VIOLATION_WINDOW}",
           sep]

    if df.empty:
        lines.append("\n  ❌ 0 trades reached final gate.")
        lines.append(f"\n  All-setups (reached OB gate): {len(all_setups)}")
        if all_setups:
            as_df=pd.DataFrame(all_setups)
            lines.append(f"  Score distribution of OB-gate setups:")
            for lo,hi in [(0,49),(50,59),(60,69),(70,79),(80,89),(90,100)]:
                sub=as_df[(as_df['score']>=lo)&(as_df['score']<=hi)]
                if len(sub): lines.append(f"    {lo}-{hi}: {len(sub)} setups")
            lines.append(f"\n  Top reject reasons (at/past OB gate):")
            rc=as_df.groupby('reason').size().sort_values(ascending=False).head(10)
            for r,n in rc.items():
                lines.append(f"    {r:<50}: {n}")
        lines.append(sep)
        return "\n".join(lines), df

    total=len(df); wins=df['won'].sum(); wr=wins/total*100
    avg_r=df['realized_r'].mean(); total_r=df['realized_r'].sum()
    run=0; peak=0; max_dd=0
    for r in df['realized_r']:
        run+=r
        if run>peak: peak=run
        dd=peak-run
        if dd>max_dd: max_dd=dd

    lines+=[f"\n  OVERALL ({total} trades)",f"  {'─'*46}",
            f"  WR:          {wr:.1f}%  ({int(wins)}W / {total-int(wins)}L)",
            f"  Avg R:       {avg_r:+.3f}R",
            f"  Total R:     {total_r:+.2f}R",
            f"  Max DD:      -{max_dd:.2f}R",
            f"  Signals/wk:  {total/(DAYS_BACK/7):.1f}"]

    lines+=[f"\n  BY DIRECTION",f"  {'─'*46}"]
    for d in ['LONG','SHORT']:
        s=df[df['direction']==d]
        if len(s)==0: continue
        w=s['won'].sum()
        lines.append(f"  {d}: {len(s):>3}t | WR={w/len(s)*100:.1f}% | avg={s['realized_r'].mean():+.3f}R | total={s['realized_r'].sum():+.2f}R")

    lines+=[f"\n  SCORE BUCKETS",f"  {'─'*46}"]
    for lo,hi in [(60,69),(70,74),(75,79),(80,84),(85,89),(90,100)]:
        s=df[(df['score']>=lo)&(df['score']<=hi)]
        if len(s)==0: continue
        w=s['won'].sum()
        lines.append(f"  {lo}-{hi}: {len(s):>3}t | WR={w/len(s)*100:.1f}% | avg={s['realized_r'].mean():+.3f}R")

    lines+=[f"\n  OUTCOMES",f"  {'─'*46}"]
    for out in ['TP3','TP2','TP1','PARTIAL_TP','SL','TIMEOUT']:
        n=len(df[df['outcome']==out])
        if n: lines.append(f"  {out:<12}: {n:>3} ({n/total*100:.0f}%)")

    lines+=[f"\n  FILTER IMPACT",f"  {'─'*46}"]
    for flag,label in [('hh_ll','HH/LL'),('triple_ema','TripleEMA'),('has_fvg','FVG'),('has_sweep','Sweep')]:
        y=df[df[flag]==True]; n2=df[df[flag]==False]
        if len(y): lines.append(f"  {label} YES {len(y):>3}t | WR={y['won'].sum()/len(y)*100:.1f}% | avg={y['realized_r'].mean():+.3f}R")
        if len(n2): lines.append(f"  {label} NO  {len(n2):>3}t | WR={n2['won'].sum()/len(n2)*100:.1f}% | avg={n2['realized_r'].mean():+.3f}R")

    lines+=[f"\n  PER PAIR",f"  {'─'*46}"]
    for sym in sorted(df['symbol'].unique()):
        s=df[df['symbol']==sym]; w=s['won'].sum()
        lines.append(f"  {sym:<12}: {len(s):>3}t | WR={w/len(s)*100:.1f}% | avg={s['realized_r'].mean():+.3f}R")

    # Score distribution of all OB-gate setups
    if all_setups:
        as_df=pd.DataFrame(all_setups)
        lines+=[f"\n  ALL OB-GATE SETUPS — SCORE DISTRIBUTION ({len(all_setups)} total)",f"  {'─'*46}"]
        for lo,hi in [(0,49),(50,59),(60,69),(70,74),(75,79),(80,84),(85,89),(90,100)]:
            s=as_df[(as_df['score']>=lo)&(as_df['score']<=hi)]
            if len(s): lines.append(f"  {lo}-{hi}: {len(s):>4} setups")
        lines+=[f"\n  Top reject reasons at/past OB gate:",f"  {'─'*46}"]
        rc=as_df.groupby('reason').size().sort_values(ascending=False).head(10)
        for r,n in rc.items():
            lines.append(f"  {r:<48}: {n}")

    lines+=[f"\n{sep}",
            "  FILES: smc_v3_results.csv | smc_v3_all_setups.csv | smc_v3_report.txt",
            sep]
    return "\n".join(lines), df


# ══════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════
async def main():
    print(f"\n{'═'*64}")
    print(f"  SMC PRO — BACKTESTER v3")
    print(f"  Pairs={len(PAIRS)} | Days={DAYS_BACK} | PD_ZONE_BARS={PD_ZONE_BARS}")
    print(f"  Score thresholds: SHORT≥{MIN_SCORE_SHORT} | LONG≥{MIN_SCORE_LONG}")
    print(f"  OB violation window: {OB_VIOLATION_WINDOW} bars")
    print(f"{'═'*64}\n")

    exchange=ccxt.binance({'enableRateLimit':True,'options':{'defaultType':'spot'}})
    all_trades=[]; all_setups=[]

    for sym in PAIRS:
        print(f"⬇️  {sym} ...", end=' ', flush=True)
        try:
            data=await fetch_all(exchange,sym,DAYS_BACK)
            t,s=run_walk_forward(data,sym,DAYS_BACK)
            all_trades.extend(t); all_setups.extend(s)
        except Exception as e:
            print(f"ERROR: {e}")
        await asyncio.sleep(0.5)

    await exchange.close()
    print(f"\n{'─'*64}\n📊 Report...\n")

    rpt,df=report(all_trades,all_setups)
    print(rpt)

    df.to_csv("smc_v3_results.csv",index=False)
    print(f"\n✅ smc_v3_results.csv ({len(df)} trades)")
    with open("smc_v3_report.txt","w") as f: f.write(rpt)
    print(f"✅ smc_v3_report.txt")
    if all_setups:
        pd.DataFrame(all_setups).to_csv("smc_v3_all_setups.csv",index=False)
        print(f"✅ smc_v3_all_setups.csv ({len(all_setups)} OB-gate setups)")
    print("\n💡 Share all 3 files — we'll use the score distribution to rebuild.\n")

if __name__=="__main__":
    asyncio.run(main())
