"""
SMC PRO — BACKTESTER v5  (HONEST REBUILD)
══════════════════════════════════════════════════════════════════
FULL ANALYSIS FROM v1-v4 DATA:
────────────────────────────────
The scoring system has ZERO predictive power.
Winners and losers score identically across all components.
Higher scores actually correlate with WORSE outcomes (inverted).

This tells us: the current SMC signal approach identifies valid
setups, but cannot predict direction of the next move reliably.

WHAT ACTUALLY WORKS (from data):
  • SHORT bias in a downtrend: consistently better
  • Bull engulf trigger: 75% WR (4 trades, small sample)
  • Score 65-69 range: 50% WR, +0.52R avg
  • HH/LL filter HURTS: 25% WR with vs 71% without

WHAT DOESN'T WORK:
  • Any LONG in a downtrend (last 90d was bear market)
  • High scores (75+) actually perform worse than low scores
  • HH/LL bonus (anti-correlated with wins)
  • FVG overlap (anti-correlated with wins)

v5 APPROACH: Test two separate configs and compare
  CONFIG A: SHORT-ONLY (bear market aligned)
  CONFIG B: DIRECTION-AGNOSTIC with strict score 65-69 band only

Also added: 4H trend strength filter using ADX (only trade when
ADX > 20 confirming a real trend, not ranging chop).

Both configs logged separately so you can compare.
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
from collections import defaultdict

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.WARNING)

PAIRS = [
    "BTC/USDT","ETH/USDT","SOL/USDT","BNB/USDT","XRP/USDT",
    "DOGE/USDT","ADA/USDT","AVAX/USDT","LINK/USDT","DOT/USDT",
    "LTC/USDT","MATIC/USDT","UNI/USDT","ATOM/USDT","NEAR/USDT",
]
DAYS_BACK = 90

# Config A: SHORT-ONLY, strict
CFG_A = dict(
    name="CONFIG_A_SHORT_ONLY",
    allow_long=False, allow_short=True,
    min_score=62,
    ob_tol=0.012, ob_lookback=80, ob_viol=40, ob_atr=0.6,
    pd_bars=200, struct_oppose_bars=8,
    adx_min=20,        # require trending (not ranging)
    require_trigger=False,
    hh_ll_bonus=False,  # data showed HH/LL hurts
    fvg_bonus=False,    # data showed FVG hurts
)

# Config B: BOTH DIRECTIONS, only score 65-72 band (sweet spot from data)
CFG_B = dict(
    name="CONFIG_B_BOTH_DIRS_SWEET_SPOT",
    allow_long=True, allow_short=True,
    min_score=65, max_score=72,  # only the band that worked
    ob_tol=0.012, ob_lookback=80, ob_viol=40, ob_atr=0.6,
    pd_bars=200, struct_oppose_bars=8,
    adx_min=20,
    require_trigger=True,  # only triggered setups
    hh_ll_bonus=False,
    fvg_bonus=False,
)

# Config C: ENGULF-ONLY (high quality trigger, both directions)
CFG_C = dict(
    name="CONFIG_C_ENGULF_TRIGGER_ONLY",
    allow_long=True, allow_short=True,
    min_score=60,
    ob_tol=0.012, ob_lookback=80, ob_viol=40, ob_atr=0.6,
    pd_bars=200, struct_oppose_bars=8,
    adx_min=15,
    require_trigger=True,
    require_engulf=True,   # only engulf triggers (75% WR from data)
    hh_ll_bonus=False,
    fvg_bonus=False,
)

TP1_RR=1.5; TP2_RR=2.5; TP3_RR=4.0
TP1_CLOSE=0.50; TP2_CLOSE=0.30; TP3_CLOSE=0.20
TRADE_TIMEOUT_H=48
MIN_BARS_BETWEEN=6


# ══════════════════════════════════════════════════════════════
#  INDICATORS
# ══════════════════════════════════════════════════════════════
def add_indicators(df):
    if len(df)<55: return df
    try:
        df=df.copy()
        df['ema_21'] =ta.trend.EMAIndicator(df['close'],21).ema_indicator()
        df['ema_50'] =ta.trend.EMAIndicator(df['close'],50).ema_indicator()
        df['ema_200']=ta.trend.EMAIndicator(df['close'],min(200,len(df)-1)).ema_indicator()
        df['rsi']    =ta.momentum.RSIIndicator(df['close'],14).rsi()
        macd=ta.trend.MACD(df['close'])
        df['macd']=macd.macd(); df['macd_signal']=macd.macd_signal()
        stoch=ta.momentum.StochRSIIndicator(df['close'])
        df['srsi_k']=stoch.stochrsi_k(); df['srsi_d']=stoch.stochrsi_d()
        df['atr']=ta.volatility.AverageTrueRange(df['high'],df['low'],df['close']).average_true_range()
        adx=ta.trend.ADXIndicator(df['high'],df['low'],df['close'],14)
        df['adx']=adx.adx(); df['di_pos']=adx.adx_pos(); df['di_neg']=adx.adx_neg()
        df['vol_sma']=df['volume'].rolling(20).mean()
        df['vol_ratio']=df['volume']/df['vol_sma'].replace(0,np.nan)
        tp_col=(df['high']+df['low']+df['close'])/3
        df['vwap']=(tp_col*df['volume']).cumsum()/df['volume'].cumsum()
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
        for i in range(left,n-right):
            hi=df['high'].iloc[i]; lo=df['low'].iloc[i]
            if all(hi>=df['high'].iloc[i-left:i]) and all(hi>=df['high'].iloc[i+1:i+right+1]): highs.append({'i':i,'price':hi})
            if all(lo<=df['low'].iloc[i-left:i]) and all(lo<=df['low'].iloc[i+1:i+right+1]): lows.append({'i':i,'price':lo})
        return highs,lows

    def detect_structure(self, df, highs, lows, lookback=20):
        events=[]; close=df['close']; n=len(df); start=max(0,n-lookback-15)
        for k in range(1,len(highs)):
            ph=highs[k-1]; ch=highs[k]
            if ch['i']<start: continue
            for j in range(ch['i'],min(ch['i']+10,n)):
                if close.iloc[j]>ph['price']:
                    events.append({'kind':'BOS_BULL' if ch['price']>ph['price'] else 'MSS_BULL','bar':j}); break
        for k in range(1,len(lows)):
            pl=lows[k-1]; cl=lows[k]
            if cl['i']<start: continue
            for j in range(cl['i'],min(cl['i']+10,n)):
                if close.iloc[j]<pl['price']:
                    events.append({'kind':'BOS_BEAR' if cl['price']<pl['price'] else 'MSS_BEAR','bar':j}); break
        if not events: return None
        latest=sorted(events,key=lambda x:x['bar'])[-1]
        if latest['bar']<n-lookback: return None
        return latest

    def find_obs(self, df, direction, lookback, atr_mult, viol_window, tol_pct):
        obs=[]; n=len(df); start=max(2,n-lookback)
        for i in range(start,n-2):
            c=df.iloc[i]
            atr=float(df['atr'].iloc[i]) if 'atr' in df.columns and not pd.isna(df['atr'].iloc[i]) else (c['high']-c['low'])
            if atr<=0: atr=c['high']-c['low']
            viol_end=min(i+1+viol_window,n)
            if direction=='LONG':
                if c['close']>=c['open']: continue
                fwd=df['high'].iloc[i+1:min(i+6,n)]
                if len(fwd)==0 or fwd.max()-c['low']<atr*atr_mult: continue
                ob={'top':max(c['open'],c['close']),'bottom':c['low'],'bar':i,
                    'size_pct':(max(c['open'],c['close'])-c['low'])/max(c['low'],0.0001)*100}
                if (df['close'].iloc[i+1:viol_end]<(ob['top']+ob['bottom'])/2).any(): continue
                obs.append(ob)
            else:
                if c['close']<=c['open']: continue
                fwd=df['low'].iloc[i+1:min(i+6,n)]
                if len(fwd)==0 or c['high']-fwd.min()<atr*atr_mult: continue
                ob={'top':c['high'],'bottom':min(c['open'],c['close']),'bar':i,
                    'size_pct':(c['high']-min(c['open'],c['close']))/max(min(c['open'],c['close']),0.0001)*100}
                if (df['close'].iloc[i+1:viol_end]>(ob['top']+ob['bottom'])/2).any(): continue
                obs.append(ob)
        obs.sort(key=lambda x:x['bar'],reverse=True)
        return obs

    def in_ob(self, price, ob, tol_pct):
        tol=ob['top']*tol_pct
        return (ob['bottom']-tol)<=price<=(ob['top']+tol)

    def pd_zone(self, df4, price, bars):
        n=len(df4); b=min(bars,n)
        hi=df4['high'].iloc[-b:].max(); lo=df4['low'].iloc[-b:].min()
        rang=hi-lo
        if rang==0: return 'NEUTRAL',0.5
        pos=(price-lo)/rang
        if pos<0.35: return 'DISCOUNT',pos
        if pos>0.65: return 'PREMIUM',pos
        return 'NEUTRAL',pos

    def find_sweep(self, df, direction, highs, lows):
        n=len(df); start=n-25
        if direction=='LONG':
            for sl in reversed(lows):
                if sl['i']<start: continue
                for j in range(sl['i']+1,min(sl['i']+8,n)):
                    c=df.iloc[j]
                    if c['low']<sl['price'] and c['close']>sl['price']: return True
        else:
            for sh in reversed(highs):
                if sh['i']<start: continue
                for j in range(sh['i']+1,min(sh['i']+8,n)):
                    c=df.iloc[j]
                    if c['high']>sh['price'] and c['close']<sh['price']: return True
        return False

    def check_hh_ll(self, df4, direction, lookback=10):
        n=len(df4)
        if n<lookback*2: return False
        r=df4.iloc[-lookback:]; p=df4.iloc[-lookback*2:-lookback]
        return r['high'].max()>p['high'].max() if direction=='LONG' else r['low'].min()<p['low'].min()


smc=SMCEngine()


# ══════════════════════════════════════════════════════════════
#  SCORER (clean, minimal)
# ══════════════════════════════════════════════════════════════
def score_signal(direction, ob, structure, has_sweep, df1, df15, df4, pd_label, cfg):
    score=0
    l1=df1.iloc[-1]; p1=df1.iloc[-2]; l4=df4.iloc[-1]
    e21=l4.get('ema_21',0); e50=l4.get('ema_50',0); e200=l4.get('ema_200',0)

    # 4H Trend (30 pts)
    trend=0
    if direction=='LONG':
        if e21>e50>e200: trend=30
        elif e21>e50:    trend=20
    else:
        if e21<e50<e200: trend=30
        elif e21<e50:    trend=20
    score+=trend

    # OB Quality (25 pts)
    ob_q=0
    if ob:
        pct=ob.get('size_pct',2.0)
        if pct<0.8: ob_q=25
        elif pct<2.0: ob_q=17
        elif pct<4.0: ob_q=10
        else: ob_q=5
    score+=ob_q

    # Structure (20 pts)
    struct=0
    if structure:
        struct=20 if 'MSS' in structure['kind'] else 13
    score+=struct

    # Trigger (20 pts, no penalty)
    trig_pts=0; trig_label='none'; is_engulf=False
    if direction=='LONG':
        if   l1.get('bull_engulf',0)==1: trig_pts=20; trig_label='bull_engulf'; is_engulf=True
        elif l1.get('bull_pin',0)==1:    trig_pts=17; trig_label='bull_pin'
        elif l1.get('hammer',0)==1:      trig_pts=14; trig_label='hammer'
        elif p1.get('bull_engulf',0)==1: trig_pts=11; trig_label='bull_engulf_prev'; is_engulf=True
        elif p1.get('bull_pin',0)==1:    trig_pts=8;  trig_label='bull_pin_prev'
        elif p1.get('hammer',0)==1:      trig_pts=6;  trig_label='hammer_prev'
    else:
        if   l1.get('bear_engulf',0)==1:   trig_pts=20; trig_label='bear_engulf'; is_engulf=True
        elif l1.get('bear_pin',0)==1:      trig_pts=17; trig_label='bear_pin'
        elif l1.get('shooting_star',0)==1: trig_pts=14; trig_label='shooting_star'
        elif p1.get('bear_engulf',0)==1:   trig_pts=11; trig_label='bear_engulf_prev'; is_engulf=True
        elif p1.get('bear_pin',0)==1:      trig_pts=8;  trig_label='bear_pin_prev'
        elif p1.get('shooting_star',0)==1: trig_pts=6;  trig_label='shooting_star_prev'
    score+=trig_pts

    # Momentum (10 pts)
    mom=0
    rsi=l1.get('rsi',50); macd=l1.get('macd',0); msig=l1.get('macd_signal',0)
    pm=p1.get('macd',0); pms=p1.get('macd_signal',0)
    sk=l1.get('srsi_k',0.5); sd=l1.get('srsi_d',0.5)
    if direction=='LONG':
        if 28<=rsi<=55 or rsi<28: mom+=3
        if macd>msig and pm<=pms: mom+=4
        elif macd>msig: mom+=2
        if sk<0.3 and sk>sd: mom+=3
    else:
        if 45<=rsi<=72 or rsi>72: mom+=3
        if macd<msig and pm>=pms: mom+=4
        elif macd<msig: mom+=2
        if sk>0.7 and sk<sd: mom+=3
    score+=mom

    # Sweep bonus (3 pts)
    if has_sweep and cfg.get('fvg_bonus',True): score+=3
    elif has_sweep: score+=3  # sweep always counts

    return max(0,min(int(score),100)), trig_label, is_engulf, bool(trig_pts>0)


# ══════════════════════════════════════════════════════════════
#  SIGNAL DETECTION
# ══════════════════════════════════════════════════════════════
def detect(df4, df1, df15, cfg):
    try:
        if len(df1)<80 or len(df15)<40 or len(df4)<60: return None,'data'
        price=df1['close'].iloc[-1]
        l4=df4.iloc[-1]
        e21=l4.get('ema_21',0); e50=l4.get('ema_50',0)

        if   e21>e50: bias='LONG'
        elif e21<e50: bias='SHORT'
        else:         return None,'ema_flat'

        if bias=='LONG'  and not cfg.get('allow_long',True):  return None,'long_disabled'
        if bias=='SHORT' and not cfg.get('allow_short',True): return None,'short_disabled'

        # ADX filter: require trending market
        adx_val=float(l4.get('adx',0) or 0)
        if adx_val < cfg.get('adx_min',0): return None,f'adx_low_{adx_val:.0f}'

        pd_label,pd_pos=smc.pd_zone(df4,price,cfg['pd_bars'])
        if bias=='LONG'  and pd_label=='PREMIUM':  return None,'pd_premium'
        if bias=='SHORT' and pd_label=='DISCOUNT': return None,'pd_discount'

        highs,lows=smc.swing_highs_lows(df1,4,4)
        structure=smc.detect_structure(df1,highs,lows,20)
        if structure:
            n1=len(df1)
            if structure['bar']>=(n1-cfg['struct_oppose_bars']):
                if bias=='LONG'  and 'BEAR' in structure['kind']: return None,'struct_opp_long'
                if bias=='SHORT' and 'BULL' in structure['kind']: return None,'struct_opp_short'

        obs=smc.find_obs(df1,bias,cfg['ob_lookback'],cfg['ob_atr'],cfg['ob_viol'],cfg['ob_tol'])
        if not obs: return None,'no_ob'

        active_ob=None
        for ob in obs:
            if smc.in_ob(price,ob,cfg['ob_tol']): active_ob=ob; break
        if not active_ob:
            d=min(abs(price-obs[0]['top']),abs(price-obs[0]['bottom']))/price*100
            return None,f'not_at_ob_{d:.1f}pct'

        has_sweep=smc.find_sweep(df1,bias,highs,lows)
        hh_ll=smc.check_hh_ll(df4,bias)

        score,trig_label,is_engulf,has_trigger=score_signal(
            bias,active_ob,structure,has_sweep,df1,df15,df4,pd_label,cfg)

        if cfg.get('require_trigger',False) and not has_trigger:
            return None,f'no_trigger_sc{score}'

        if cfg.get('require_engulf',False) and not is_engulf:
            return None,f'no_engulf_trig_{trig_label}'

        min_sc=cfg['min_score']
        max_sc=cfg.get('max_score',100)
        if score<min_sc: return None,f'score_{score}<{min_sc}'
        if score>max_sc: return None,f'score_{score}>{max_sc}'

        atr1=float(df1['atr'].iloc[-1])
        entry=price
        if bias=='LONG': sl=min(active_ob['bottom']-atr1*0.2,entry-atr1*0.6)
        else:            sl=max(active_ob['top']+atr1*0.2,entry+atr1*0.6)
        risk=abs(entry-sl)
        if risk<entry*0.0008: return None,'bad_sl'

        tps=[entry+risk*TP1_RR,entry+risk*TP2_RR,entry+risk*TP3_RR] if bias=='LONG' \
        else [entry-risk*TP1_RR,entry-risk*TP2_RR,entry-risk*TP3_RR]

        return {
            'bias':bias,'score':score,'trigger':trig_label,'is_engulf':is_engulf,
            'has_trigger':has_trigger,'hh_ll':hh_ll,'has_sweep':has_sweep,
            'entry':entry,'sl':sl,'tps':tps,'risk':risk,
            'risk_pct':risk/entry*100,
            'ob_size_pct':active_ob.get('size_pct',2.0),
            'pd_zone':pd_label,'pd_pos':round(pd_pos,3),
            'structure':structure['kind'] if structure else 'NONE',
            'adx':round(adx_val,1),
        },'passed'

    except Exception as e:
        return None,f'exc_{str(e)[:40]}'


# ══════════════════════════════════════════════════════════════
#  SIMULATOR
# ══════════════════════════════════════════════════════════════
def simulate(sig, future):
    bias=sig['bias']; entry=sig['entry']; sl=sig['sl']; tps=sig['tps']; risk=sig['risk']
    tp_hit=[False,False,False]; remaining=1.0; realized_r=0.0; outcome='TIMEOUT'
    weights=[TP1_CLOSE,TP2_CLOSE,TP3_CLOSE]; max_mae=0.0; bars_used=len(future)

    for bn,(_,row) in enumerate(future.iterrows()):
        h,l=row['high'],row['low']
        mae=(entry-l)/risk if bias=='LONG' else (h-entry)/risk
        max_mae=max(max_mae,mae)
        if bias=='LONG' and l<=sl:
            realized_r-=remaining; outcome='SL'; bars_used=bn+1; break
        elif bias=='SHORT' and h>=sl:
            realized_r-=remaining; outcome='SL'; bars_used=bn+1; break
        for ti,tp in enumerate(tps):
            if tp_hit[ti]: continue
            if (bias=='LONG' and h>=tp) or (bias=='SHORT' and l<=tp):
                tp_hit[ti]=True
                realized_r+=weights[ti]*[TP1_RR,TP2_RR,TP3_RR][ti]
                remaining-=weights[ti]
                if ti==2: outcome='TP3'; bars_used=bn+1
                elif ti==1: outcome='TP2'
                elif ti==0: outcome='TP1'
        if remaining<=0.01: break

    if outcome=='TIMEOUT' and any(tp_hit): outcome='PARTIAL_TP'
    return {'outcome':outcome,'realized_r':round(realized_r,4),'won':realized_r>0,
            'tp1_hit':tp_hit[0],'tp2_hit':tp_hit[1],'tp3_hit':tp_hit[2],
            'max_adverse_r':round(max_mae,4),'bars_held':bars_used}


# ══════════════════════════════════════════════════════════════
#  DATA + WALK-FORWARD
# ══════════════════════════════════════════════════════════════
async def fetch(exchange, symbol, days_back):
    result={}
    limits={'4h':min(1000,days_back*6+250),'1h':min(1000,days_back*24+200),'15m':min(1000,days_back*96+100)}
    for tf,lim in limits.items():
        raw=await exchange.fetch_ohlcv(symbol,tf,limit=lim)
        df=pd.DataFrame(raw,columns=['ts','open','high','low','close','volume'])
        df['ts']=pd.to_datetime(df['ts'],unit='ms')
        result[tf]=add_indicators(df.reset_index(drop=True))
        await asyncio.sleep(0.12)
    return result

def walk_forward(data, symbol, days_back, cfg):
    df4=data['4h']; df1=data['1h']; df15=data['15m']
    end_dt=df1['ts'].iloc[-1]; start_dt=end_dt-timedelta(days=days_back)
    test_idx=df1[df1['ts']>=start_dt].index.tolist()
    trades=[]; rejects=defaultdict(int); last_sig=-999; active_end=-1

    for idx in test_idx:
        if idx<100 or idx<=active_end or idx-last_sig<MIN_BARS_BETWEEN: continue
        df1_s=df1.iloc[:idx+1]; ts=df1_s['ts'].iloc[-1]
        df4_s=df4[df4['ts']<=ts]; df15_s=df15[df15['ts']<=ts]
        if len(df4_s)<60 or len(df15_s)<40: continue

        sig,reason=detect(df4_s,df1_s,df15_s,cfg)
        rejects[reason]+=1
        if sig is None: continue

        future=df1.iloc[idx+1:idx+1+TRADE_TIMEOUT_H].copy()
        if len(future)<2: continue

        res=simulate(sig,future)
        last_sig=idx; active_end=idx+res['bars_held']+1

        trades.append({
            'symbol':symbol,'date':ts.strftime('%Y-%m-%d %H:%M'),
            'config':cfg['name'],
            'direction':sig['bias'],'score':sig['score'],
            'trigger':sig['trigger'],'is_engulf':sig['is_engulf'],
            'hh_ll':sig['hh_ll'],'has_sweep':sig['has_sweep'],
            'pd_zone':sig['pd_zone'],'structure':sig['structure'],
            'ob_size_pct':round(sig['ob_size_pct'],3),
            'adx':sig['adx'],
            'entry':round(sig['entry'],6),'sl':round(sig['sl'],6),
            'tp1':round(sig['tps'][0],6),'tp2':round(sig['tps'][1],6),'tp3':round(sig['tps'][2],6),
            'risk_pct':round(sig['risk_pct'],3),
            'outcome':res['outcome'],'realized_r':res['realized_r'],
            'won':res['won'],'tp1_hit':res['tp1_hit'],'tp2_hit':res['tp2_hit'],'tp3_hit':res['tp3_hit'],
            'max_adverse_r':res['max_adverse_r'],'bars_held':res['bars_held'],
        })

    return trades, dict(rejects)


# ══════════════════════════════════════════════════════════════
#  REPORT
# ══════════════════════════════════════════════════════════════
def section(df, label, lines, total_days=90, n_pairs=15):
    if len(df)==0:
        lines.append(f"\n  {label}: 0 trades"); return
    total=len(df); wins=df['won'].sum(); wr=wins/total*100
    avg_r=df['realized_r'].mean(); total_r=df['realized_r'].sum()
    run=0; peak=0; max_dd=0
    for r in df['realized_r']:
        run+=r
        if run>peak: peak=run
        dd=peak-run
        if dd>max_dd: max_dd=dd

    lines+=[f"\n  ┌─ {label} ({total} trades) ─────────────────────────",
            f"  │  WR:         {wr:.1f}%  ({int(wins)}W / {total-int(wins)}L)",
            f"  │  Avg R:      {avg_r:+.3f}R",
            f"  │  Total R:    {total_r:+.2f}R",
            f"  │  Max DD:     -{max_dd:.2f}R",
            f"  │  Per week:   {total/(total_days/7):.1f}"]

    lines.append(f"  │  Scores:  ",)
    for lo,hi in [(55,64),(65,69),(70,74),(75,79),(80,100)]:
        s=df[(df['score']>=lo)&(df['score']<=hi)]
        if len(s): w2=s['won'].sum(); lines.append(f"  │    {lo}-{hi}: {len(s):>2}t WR={w2/len(s)*100:.0f}% avg={s['realized_r'].mean():+.2f}R")

    lines.append(f"  │  Triggers:")
    for trig in sorted(df['trigger'].unique()):
        s=df[df['trigger']==trig]; w2=s['won'].sum()
        lines.append(f"  │    {trig:<22}: {len(s):>2}t WR={w2/len(s)*100:.0f}% avg={s['realized_r'].mean():+.2f}R")

    lines.append(f"  │  Outcomes:")
    for out in ['TP3','TP2','TP1','PARTIAL_TP','SL','TIMEOUT']:
        n=len(df[df['outcome']==out])
        if n: lines.append(f"  │    {out:<12}: {n:>2} ({n/total*100:.0f}%)")

    lines.append(f"  │  ADX analysis:")
    for lo,hi in [(0,19),(20,24),(25,29),(30,49),(50,100)]:
        s=df[(df['adx']>=lo)&(df['adx']<=hi)]
        if len(s): w2=s['won'].sum(); lines.append(f"  │    ADX {lo}-{hi}: {len(s):>2}t WR={w2/len(s)*100:.0f}% avg={s['realized_r'].mean():+.2f}R")

    lines.append(f"  └────────────────────────────────────────────────────")

def make_report(all_trades):
    df=pd.DataFrame(all_trades) if all_trades else pd.DataFrame()
    sep="═"*66
    L=[sep, "  SMC PRO v5 — MULTI-CONFIG BACKTEST REPORT",
       f"  {datetime.now().strftime('%Y-%m-%d %H:%M UTC')}",
       f"  {DAYS_BACK}d | {len(PAIRS)} pairs", sep]

    if df.empty:
        L.append("\n  ❌ 0 trades across all configs."); L.append(sep)
        return "\n".join(L), df

    for cfg in [CFG_A,CFG_B,CFG_C]:
        sub=df[df['config']==cfg['name']]
        section(sub, cfg['name'], L)

    # Direction breakdown
    L+=[f"\n  DIRECTION SUMMARY (all configs)",f"  {'─'*50}"]
    for d in ['LONG','SHORT']:
        s=df[df['direction']==d]
        if len(s)==0: continue
        w=s['won'].sum()
        L.append(f"  {d}: {len(s):>3}t | WR={w/len(s)*100:.1f}% | avg={s['realized_r'].mean():+.3f}R | total={s['realized_r'].sum():+.2f}R")

    # Best trigger across all
    L+=[f"\n  TRIGGER PERFORMANCE (all configs)",f"  {'─'*50}"]
    for trig in sorted(df['trigger'].unique()):
        s=df[df['trigger']==trig]; w=s['won'].sum()
        L.append(f"  {trig:<22}: {len(s):>3}t | WR={w/len(s)*100:.1f}% | avg={s['realized_r'].mean():+.3f}R")

    # Per pair
    L+=[f"\n  PER PAIR (all configs combined)",f"  {'─'*50}"]
    for sym in sorted(df['symbol'].unique()):
        s=df[df['symbol']==sym]; w=s['won'].sum()
        L.append(f"  {sym:<12}: {len(s):>3}t | WR={w/len(s)*100:.1f}% | avg={s['realized_r'].mean():+.3f}R")

    L+=[f"\n{sep}","  smc_v5_results.csv | smc_v5_report.txt",sep]
    return "\n".join(L), df


# ══════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════
async def main():
    print(f"\n{'═'*66}")
    print(f"  SMC PRO — BACKTESTER v5 (multi-config comparison)")
    print(f"  Testing 3 configs: SHORT-ONLY | SWEET-SPOT | ENGULF-ONLY")
    print(f"  {len(PAIRS)} pairs | {DAYS_BACK} days")
    print(f"{'═'*66}\n")

    exchange=ccxt.binance({'enableRateLimit':True,'options':{'defaultType':'spot'}})
    all_trades=[]

    for sym in PAIRS:
        print(f"⬇️  {sym} ...", end=' ', flush=True)
        try:
            data=await fetch(exchange,sym,DAYS_BACK)
            for cfg in [CFG_A,CFG_B,CFG_C]:
                t,_=walk_forward(data,sym,DAYS_BACK,cfg)
                all_trades.extend(t)
            df_sym=pd.DataFrame(all_trades) if all_trades else pd.DataFrame()
            cnts={}
            for cfg in [CFG_A,CFG_B,CFG_C]:
                c=df_sym[df_sym['config']==cfg['name']] if not df_sym.empty else pd.DataFrame()
                sym_c=c[c['symbol']==sym] if not c.empty else pd.DataFrame()
                cnts[cfg['name'].split('_')[1]]=len(sym_c)
            print(f"A={cnts.get('A',0)} B={cnts.get('B',0)} C={cnts.get('C',0)}")
        except Exception as e:
            print(f"ERROR: {e}")
        await asyncio.sleep(0.5)

    await exchange.close()
    print(f"\n{'─'*66}\n📊 Report...\n")

    rpt,df=make_report(all_trades)
    print(rpt)

    df.to_csv("smc_v5_results.csv",index=False)
    print(f"\n✅ smc_v5_results.csv ({len(df)} trades)")
    with open("smc_v5_report.txt","w") as f: f.write(rpt)
    print(f"✅ smc_v5_report.txt")
    print("\n💡 This is the final diagnostic. Share to get the rebuilt live bot.\n")

if __name__=="__main__":
    asyncio.run(main())
