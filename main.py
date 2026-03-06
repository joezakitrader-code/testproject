"""
SMC PRO — BACKTESTER v4  (DATA-DRIVEN REBUILD)
══════════════════════════════════════════════════════════════════
WHAT THE v3 DATA TOLD US:
─────────────────────────
  663 OB-gate setups, only 15 passed filters, and here's why:

  1. TRIGGER BOTTLENECK: 462 of 663 setups (70%) had score 0-49.
     These ALL show "no_trigger_score_33-49". With no trigger, score
     starts at ≈37 and can never reach 70+. The trigger requirement
     is effectively a HARD GATE that blocks 70% of setups.
     → FIX: Remove trigger as a hard gate. Make it a bonus only.
             Score starts at 0, trigger adds +15-25, but no -12 penalty.

  2. SHORTS WORK, LONGS DON'T:
     SHORT: WR=67%, avg=+1.07R
     LONG:  WR=22%, avg=-0.58R
     The last 90d was a bear market. Longs into a downtrend = losses.
     → FIX: Stronger 4H trend alignment requirement.
             LONG requires full triple EMA (21>50>200) OR HH confirmed.
             SHORT requires EMA21 < EMA50 only (already works).

  3. FVG HURTS (-0.53R with FVG vs +0.61R without):
     FVG overlap with OB may indicate the zone is "used up" / messy.
     → FIX: FVG overlap gives 0 bonus (was +3). Track it for analysis.

  4. SCORE DISTRIBUTION: Only 11 setups scored 75-89 out of 663.
     The scoring is too binary — most weight goes to rare events.
     → FIX: Redistribute points more evenly. More setups should score
             60-80. Use dynamic thresholds based on what's achievable.

  5. TRADE FREQUENCY: 1.2/week across 15 pairs is too low for a bot.
     Target: 3-6/week = 0.2-0.4/pair/week.
     → FIX: Loosen OB tolerance to 1.2%, lookback to 80 bars.

NEW SCORING SYSTEM:
────────────────────
  Max 100 pts, threshold SHORT≥65, LONG≥68

  4H Trend (30pts max):
    Triple EMA stack (21>50>200)    = 30
    EMA 21>50 (or 21<50)            = 20
    HH/LL confirmed                 = +8 bonus

  OB Quality (25pts max):
    Tight OB (<0.8%)                = 25
    Medium OB (0.8-2%)              = 17
    Wide OB (2-4%)                  = 10
    Very wide (>4%)                 = 5

  Structure (20pts max):
    MSS (reversal)                  = 20
    BOS (continuation)              = 13
    No structure                    = 0

  Entry Trigger (20pts max):
    Engulf current bar              = 20
    Pin bar current bar             = 17
    Hammer/SS current bar           = 14
    Engulf prev bar                 = 11
    Pin prev bar                    = 8
    Hammer/SS prev bar              = 6
    No trigger                      = 0  (was -12!)

  Momentum (10pts max):
    RSI in ideal zone               = 3
    MACD cross                      = 4
    StochRSI cross                  = 3

  Extras (5pts max):
    Liquidity sweep                 = 3
    Volume spike 2x+                = 2
    VWAP alignment                  = 1

HOW TO RUN:
  pip install ccxt pandas numpy ta
  python smc_backtester_v4.py
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
DAYS_BACK             = 90
MIN_SCORE_SHORT       = 65
MIN_SCORE_LONG        = 68
# LONG hard gate: requires EITHER triple EMA OR confirmed HH
LONG_REQUIRES_TREND   = True
OB_TOLERANCE_PCT      = 0.012   # 1.2% (was 0.8-1.0%)
OB_LOOKBACK           = 80      # was 60
OB_IMPULSE_ATR_MULT   = 0.6     # was 0.8-1.0
OB_VIOLATION_WINDOW   = 40
PD_ZONE_BARS          = 200
HH_LL_LOOKBACK        = 10
STRUCTURE_LOOKBACK    = 20
STRUCTURE_OPPOSE_BARS = 8
TP1_RR=1.5; TP2_RR=2.5; TP3_RR=4.0
TP1_CLOSE=0.50; TP2_CLOSE=0.30; TP3_CLOSE=0.20
TRADE_TIMEOUT_H=48
MIN_BARS_BETWEEN_SIGNALS=6


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
            if all(hi>=df['high'].iloc[i-left:i]) and all(hi>=df['high'].iloc[i+1:i+right+1]):
                highs.append({'i':i,'price':hi})
            if all(lo<=df['low'].iloc[i-left:i]) and all(lo<=df['low'].iloc[i+1:i+right+1]):
                lows.append({'i':i,'price':lo})
        return highs,lows

    def check_hh_ll(self, df_4h, direction):
        n=len(df_4h)
        if n<HH_LL_LOOKBACK*2: return False
        r=df_4h.iloc[-HH_LL_LOOKBACK:]; p=df_4h.iloc[-HH_LL_LOOKBACK*2:-HH_LL_LOOKBACK]
        return r['high'].max()>p['high'].max() if direction=='LONG' else r['low'].min()<p['low'].min()

    def triple_ema_bull(self, df4):
        l=df4.iloc[-1]
        return float(l.get('ema_21',0))>float(l.get('ema_50',0))>float(l.get('ema_200',0))

    def triple_ema_bear(self, df4):
        l=df4.iloc[-1]
        return float(l.get('ema_21',0))<float(l.get('ema_50',0))<float(l.get('ema_200',0))

    def detect_structure(self, df, highs, lows):
        events=[]; close=df['close']; n=len(df); start=max(0,n-STRUCTURE_LOOKBACK-15)
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
        if latest['bar']<n-STRUCTURE_LOOKBACK: return None
        return latest

    def find_obs(self, df, direction):
        obs=[]; n=len(df); start=max(2,n-OB_LOOKBACK)
        for i in range(start,n-2):
            c=df.iloc[i]
            atr=float(df['atr'].iloc[i]) if 'atr' in df.columns and not pd.isna(df['atr'].iloc[i]) else (c['high']-c['low'])
            if atr<=0: atr=c['high']-c['low']
            viol_end=min(i+1+OB_VIOLATION_WINDOW,n)
            if direction=='LONG':
                if c['close']>=c['open']: continue
                fwd=df['high'].iloc[i+1:min(i+6,n)]
                if len(fwd)==0 or fwd.max()-c['low']<atr*OB_IMPULSE_ATR_MULT: continue
                ob={'top':max(c['open'],c['close']),'bottom':c['low'],'bar':i,
                    'size_pct':(max(c['open'],c['close'])-c['low'])/c['low']*100}
                if (df['close'].iloc[i+1:viol_end]<(ob['top']+ob['bottom'])/2).any(): continue
                obs.append(ob)
            else:
                if c['close']<=c['open']: continue
                fwd=df['low'].iloc[i+1:min(i+6,n)]
                if len(fwd)==0 or c['high']-fwd.min()<atr*OB_IMPULSE_ATR_MULT: continue
                ob={'top':c['high'],'bottom':min(c['open'],c['close']),'bar':i,
                    'size_pct':(c['high']-min(c['open'],c['close']))/min(c['open'],c['close'])*100}
                if (df['close'].iloc[i+1:viol_end]>(ob['top']+ob['bottom'])/2).any(): continue
                obs.append(ob)
        obs.sort(key=lambda x:x['bar'],reverse=True)
        return obs

    def in_ob(self, price, ob):
        tol=ob['top']*OB_TOLERANCE_PCT
        return (ob['bottom']-tol)<=price<=(ob['top']+tol)

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

    def pd_zone(self, df4, price):
        n=len(df4); bars=min(PD_ZONE_BARS,n)
        hi=df4['high'].iloc[-bars:].max(); lo=df4['low'].iloc[-bars:].min()
        rang=hi-lo
        if rang==0: return 'NEUTRAL',0.5
        pos=(price-lo)/rang
        if pos<0.35: return 'DISCOUNT',pos
        if pos>0.65: return 'PREMIUM',pos
        return 'NEUTRAL',pos


# ══════════════════════════════════════════════════════════════
#  NEW SCORER
# ══════════════════════════════════════════════════════════════
def score_v4(direction, ob, structure, has_sweep, df1, df15, df4, pd_label, hh_ll, t_ema_bull, t_ema_bear):
    score=0; breakdown={}

    l1=df1.iloc[-1]; p1=df1.iloc[-2]; l15=df15.iloc[-1]; l4=df4.iloc[-1]
    e21=l4.get('ema_21',0); e50=l4.get('ema_50',0)

    # ── 1. 4H TREND (30 pts max) ──────────────────────────
    trend_pts=0
    if direction=='LONG':
        if t_ema_bull:   trend_pts=30
        elif e21>e50:    trend_pts=20
    else:
        if t_ema_bear:   trend_pts=30
        elif e21<e50:    trend_pts=20
    if hh_ll: trend_pts=min(trend_pts+8,30)
    score+=trend_pts; breakdown['trend']=trend_pts

    # ── 2. OB QUALITY (25 pts max) ────────────────────────
    ob_pts=0
    if ob:
        pct=ob.get('size_pct',2.0)
        if pct<0.8:    ob_pts=25
        elif pct<2.0:  ob_pts=17
        elif pct<4.0:  ob_pts=10
        else:          ob_pts=5
    score+=ob_pts; breakdown['ob_quality']=ob_pts

    # ── 3. STRUCTURE (20 pts max) ─────────────────────────
    struct_pts=0
    if structure:
        struct_pts=20 if 'MSS' in structure['kind'] else 13
    score+=struct_pts; breakdown['structure']=struct_pts

    # ── 4. ENTRY TRIGGER (20 pts max) — NO PENALTY ────────
    trigger_pts=0; trigger_label='none'
    if direction=='LONG':
        if   l1.get('bull_engulf',0)==1: trigger_pts=20; trigger_label='bull_engulf'
        elif l1.get('bull_pin',0)==1:    trigger_pts=17; trigger_label='bull_pin'
        elif l1.get('hammer',0)==1:      trigger_pts=14; trigger_label='hammer'
        elif p1.get('bull_engulf',0)==1: trigger_pts=11; trigger_label='bull_engulf_prev'
        elif p1.get('bull_pin',0)==1:    trigger_pts=8;  trigger_label='bull_pin_prev'
        elif p1.get('hammer',0)==1:      trigger_pts=6;  trigger_label='hammer_prev'
    else:
        if   l1.get('bear_engulf',0)==1:   trigger_pts=20; trigger_label='bear_engulf'
        elif l1.get('bear_pin',0)==1:      trigger_pts=17; trigger_label='bear_pin'
        elif l1.get('shooting_star',0)==1: trigger_pts=14; trigger_label='shooting_star'
        elif p1.get('bear_engulf',0)==1:   trigger_pts=11; trigger_label='bear_engulf_prev'
        elif p1.get('bear_pin',0)==1:      trigger_pts=8;  trigger_label='bear_pin_prev'
        elif p1.get('shooting_star',0)==1: trigger_pts=6;  trigger_label='shooting_star_prev'
    score+=trigger_pts; breakdown['trigger']=trigger_pts; breakdown['trigger_label']=trigger_label

    # ── 5. MOMENTUM (10 pts max) ──────────────────────────
    mom=0
    rsi=l1.get('rsi',50)
    macd=l1.get('macd',0); msig=l1.get('macd_signal',0)
    pm=p1.get('macd',0); pms=p1.get('macd_signal',0)
    sk=l1.get('srsi_k',0.5); sd=l1.get('srsi_d',0.5)
    if direction=='LONG':
        if 28<=rsi<=55: mom+=3
        elif rsi<28:    mom+=3
        if macd>msig and pm<=pms: mom+=4
        elif macd>msig:           mom+=2
        if sk<0.3 and sk>sd:      mom+=3
    else:
        if 45<=rsi<=72: mom+=3
        elif rsi>72:    mom+=3
        if macd<msig and pm>=pms: mom+=4
        elif macd<msig:           mom+=2
        if sk>0.7 and sk<sd:      mom+=3
    score+=mom; breakdown['momentum']=mom

    # ── 6. EXTRAS (5 pts max) ─────────────────────────────
    ext=0
    if has_sweep: ext+=3
    vr=l15.get('vol_ratio',1.0)
    if vr>=2.0: ext+=2
    elif vr>=1.5: ext+=1
    close1=l1.get('close',0); vwap1=l1.get('vwap',0)
    if direction=='LONG' and close1<vwap1: ext+=1
    elif direction=='SHORT' and close1>vwap1: ext+=1
    score+=min(ext,5); breakdown['extras']=min(ext,5)

    return max(0,min(int(score),100)), breakdown, trigger_label


# ══════════════════════════════════════════════════════════════
#  SIGNAL DETECTION
# ══════════════════════════════════════════════════════════════
smc=SMCEngine()

def detect_signal(df4, df1, df15):
    try:
        if len(df1)<80 or len(df15)<40 or len(df4)<60: return None,'data'

        price=df1['close'].iloc[-1]
        l4=df4.iloc[-1]
        e21=l4.get('ema_21',0); e50=l4.get('ema_50',0)
        if   e21>e50: bias='LONG'
        elif e21<e50: bias='SHORT'
        else:         return None,'ema_flat'

        t_bull=smc.triple_ema_bull(df4); t_bear=smc.triple_ema_bear(df4)
        hh_ll=smc.check_hh_ll(df4,bias)

        # LONG trend gate: need triple EMA OR HH confirmed
        if bias=='LONG' and LONG_REQUIRES_TREND:
            if not t_bull and not hh_ll:
                return None,'long_no_trend'

        pd_label,pd_pos=smc.pd_zone(df4,price)
        if bias=='LONG'  and pd_label=='PREMIUM':  return None,'pd_premium_long'
        if bias=='SHORT' and pd_label=='DISCOUNT': return None,'pd_discount_short'

        highs,lows=smc.swing_highs_lows(df1,4,4)
        structure=smc.detect_structure(df1,highs,lows)
        if structure:
            n1=len(df1)
            if structure['bar']>=(n1-STRUCTURE_OPPOSE_BARS):
                if bias=='LONG'  and 'BEAR' in structure['kind']: return None,'structure_opposes_long'
                if bias=='SHORT' and 'BULL' in structure['kind']: return None,'structure_opposes_short'

        obs=smc.find_obs(df1,bias)
        if not obs: return None,'no_ob'

        active_ob=None
        for ob in obs:
            if smc.in_ob(price,ob): active_ob=ob; break
        if not active_ob:
            nearest=obs[0]
            dist=min(abs(price-nearest['top']),abs(price-nearest['bottom']))/price*100
            return None,f'not_at_ob_{dist:.1f}pct'

        has_sweep=smc.find_sweep(df1,bias,highs,lows)

        score,breakdown,trigger=score_v4(
            bias,active_ob,structure,has_sweep,
            df1,df15,df4,pd_label,hh_ll,t_bull,t_bear
        )

        min_sc=MIN_SCORE_SHORT if bias=='SHORT' else MIN_SCORE_LONG
        if score<min_sc: return None,f'score_{score}_below_{min_sc}'

        atr1=float(df1['atr'].iloc[-1])
        entry=price
        if bias=='LONG': sl=min(active_ob['bottom']-atr1*0.2,entry-atr1*0.6)
        else:            sl=max(active_ob['top']+atr1*0.2,entry+atr1*0.6)
        risk=abs(entry-sl)
        if risk<entry*0.0008: return None,'bad_sl'

        tps=[entry+risk*TP1_RR,entry+risk*TP2_RR,entry+risk*TP3_RR] if bias=='LONG' \
        else [entry-risk*TP1_RR,entry-risk*TP2_RR,entry-risk*TP3_RR]

        return {
            'bias':bias,'score':score,'breakdown':breakdown,
            'trigger':trigger,'t_ema_bull':t_bull,'t_ema_bear':t_bear,
            'hh_ll':hh_ll,'entry':entry,'sl':sl,'tps':tps,'risk':risk,
            'risk_pct':risk/entry*100,
            'ob_size_pct':active_ob.get('size_pct',2.0),
            'has_sweep':has_sweep,'pd_zone':pd_label,'pd_pos':round(pd_pos,3),
            'structure':structure['kind'] if structure else 'NONE',
            'trend_pts':breakdown['trend'],'ob_pts':breakdown['ob_quality'],
            'struct_pts':breakdown['structure'],'trigger_pts':breakdown['trigger'],
            'mom_pts':breakdown['momentum'],'extra_pts':breakdown['extras'],
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

    for bn,(i,row) in enumerate(future.iterrows()):
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

def walk_forward(data, symbol, days_back):
    df4=data['4h']; df1=data['1h']; df15=data['15m']
    end_dt=df1['ts'].iloc[-1]; start_dt=end_dt-timedelta(days=days_back)
    test_idx=df1[df1['ts']>=start_dt].index.tolist()

    trades=[]; rejects={}; last_sig=-999; active_end=-1

    print(f"  {symbol}: {len(test_idx)} bars ...", end=' ', flush=True)

    for idx in test_idx:
        if idx<100 or idx<=active_end or idx-last_sig<MIN_BARS_BETWEEN_SIGNALS: continue
        df1_s=df1.iloc[:idx+1]; ts=df1_s['ts'].iloc[-1]
        df4_s=df4[df4['ts']<=ts]; df15_s=df15[df15['ts']<=ts]
        if len(df4_s)<60 or len(df15_s)<40: continue

        sig,reason=detect_signal(df4_s,df1_s,df15_s)
        rejects[reason]=rejects.get(reason,0)+1
        if sig is None: continue

        future=df1.iloc[idx+1:idx+1+TRADE_TIMEOUT_H].copy()
        if len(future)<2: continue

        res=simulate(sig,future)
        last_sig=idx; active_end=idx+res['bars_held']+1

        trades.append({
            'symbol':symbol,'date':ts.strftime('%Y-%m-%d %H:%M'),
            'direction':sig['bias'],'score':sig['score'],
            'trigger':sig['trigger'],'hh_ll':sig['hh_ll'],
            't_ema_bull':sig['t_ema_bull'],'t_ema_bear':sig['t_ema_bear'],
            'pd_zone':sig['pd_zone'],'structure':sig['structure'],
            'ob_size_pct':round(sig['ob_size_pct'],3),
            'has_sweep':sig['has_sweep'],
            'trend_pts':sig['trend_pts'],'ob_pts':sig['ob_pts'],
            'struct_pts':sig['struct_pts'],'trigger_pts':sig['trigger_pts'],
            'mom_pts':sig['mom_pts'],'extra_pts':sig['extra_pts'],
            'entry':round(sig['entry'],6),'sl':round(sig['sl'],6),
            'tp1':round(sig['tps'][0],6),'tp2':round(sig['tps'][1],6),'tp3':round(sig['tps'][2],6),
            'risk_pct':round(sig['risk_pct'],3),
            'outcome':res['outcome'],'realized_r':res['realized_r'],
            'won':res['won'],'tp1_hit':res['tp1_hit'],'tp2_hit':res['tp2_hit'],'tp3_hit':res['tp3_hit'],
            'max_adverse_r':res['max_adverse_r'],'bars_held':res['bars_held'],
        })

    top=sorted(rejects.items(),key=lambda x:-x[1])[:5]
    rstr=' | '.join(f'{k}={v}' for k,v in top if k!='passed')
    print(f"{len(trades)} signals | {rstr}")
    return trades


# ══════════════════════════════════════════════════════════════
#  REPORT
# ══════════════════════════════════════════════════════════════
def make_report(trades):
    df=pd.DataFrame(trades)
    sep="═"*64
    L=[sep,"  SMC PRO v4 — DATA-DRIVEN BACKTEST REPORT",
       f"  {datetime.now().strftime('%Y-%m-%d %H:%M UTC')}",
       f"  {DAYS_BACK}d | {len(PAIRS)} pairs | SHORT≥{MIN_SCORE_SHORT} LONG≥{MIN_SCORE_LONG}",
       f"  OB tol={OB_TOLERANCE_PCT*100:.1f}% lookback={OB_LOOKBACK} viol_window={OB_VIOLATION_WINDOW}",
       sep]

    if df.empty:
        L.append("\n  ❌ 0 trades. Filters still too strict or market had no setups.")
        L.append(sep); return "\n".join(L),df

    total=len(df); wins=df['won'].sum(); wr=wins/total*100
    avg_r=df['realized_r'].mean(); total_r=df['realized_r'].sum()
    run=0; peak=0; max_dd=0
    for r in df['realized_r']:
        run+=r
        if run>peak: peak=run
        dd=peak-run
        if dd>max_dd: max_dd=dd

    L+=[f"\n  OVERALL ({total} trades)",f"  {'─'*46}",
        f"  WR:          {wr:.1f}%  ({int(wins)}W / {total-int(wins)}L)",
        f"  Avg R:       {avg_r:+.3f}R",
        f"  Total R:     {total_r:+.2f}R",
        f"  Max DD:      -{max_dd:.2f}R",
        f"  Signals/wk:  {total/(DAYS_BACK/7):.1f}"]

    L+=[f"\n  BY DIRECTION",f"  {'─'*46}"]
    for d in ['LONG','SHORT']:
        s=df[df['direction']==d]
        if len(s)==0: continue
        w=s['won'].sum()
        L.append(f"  {d}: {len(s):>3}t | WR={w/len(s)*100:.1f}% | avg={s['realized_r'].mean():+.3f}R | total={s['realized_r'].sum():+.2f}R")

    L+=[f"\n  SCORE BUCKETS",f"  {'─'*46}"]
    for lo,hi in [(0,54),(55,59),(60,64),(65,69),(70,74),(75,79),(80,84),(85,100)]:
        s=df[(df['score']>=lo)&(df['score']<=hi)]
        if len(s)==0: continue
        w=s['won'].sum()
        L.append(f"  {lo:>2}-{hi}: {len(s):>3}t | WR={w/len(s)*100:.1f}% | avg={s['realized_r'].mean():+.3f}R")

    L+=[f"\n  TRIGGER TYPE ANALYSIS",f"  {'─'*46}"]
    for trig in df['trigger'].unique():
        s=df[df['trigger']==trig]
        w=s['won'].sum()
        L.append(f"  {trig:<22}: {len(s):>3}t | WR={w/len(s)*100:.1f}% | avg={s['realized_r'].mean():+.3f}R")

    L+=[f"\n  SCORE COMPONENT vs OUTCOME",f"  {'─'*46}"]
    winners=df[df['won']==True]; losers=df[df['won']==False]
    for col in ['trend_pts','ob_pts','struct_pts','trigger_pts','mom_pts','extra_pts']:
        wm=winners[col].mean() if len(winners) else 0
        lm=losers[col].mean() if len(losers) else 0
        L.append(f"  {col:<15}: winners={wm:.1f}  losers={lm:.1f}  diff={wm-lm:+.1f}")

    L+=[f"\n  OUTCOMES",f"  {'─'*46}"]
    for out in ['TP3','TP2','TP1','PARTIAL_TP','SL','TIMEOUT']:
        n=len(df[df['outcome']==out])
        if n: L.append(f"  {out:<12}: {n:>3} ({n/total*100:.0f}%)")

    L+=[f"\n  FILTER ANALYSIS",f"  {'─'*46}"]
    for flag,label in [('hh_ll','HH/LL'),('t_ema_bull','TripleEMA_bull'),('has_sweep','Sweep')]:
        y=df[df[flag]==True]; n2=df[df[flag]==False]
        if len(y): L.append(f"  {label} YES {len(y):>3}t | WR={y['won'].sum()/len(y)*100:.1f}% | avg={y['realized_r'].mean():+.3f}R")
        if len(n2): L.append(f"  {label} NO  {len(n2):>3}t | WR={n2['won'].sum()/len(n2)*100:.1f}% | avg={n2['realized_r'].mean():+.3f}R")

    L+=[f"\n  PER PAIR",f"  {'─'*46}"]
    for sym in sorted(df['symbol'].unique()):
        s=df[df['symbol']==sym]; w=s['won'].sum()
        L.append(f"  {sym:<12}: {len(s):>3}t | WR={w/len(s)*100:.1f}% | avg={s['realized_r'].mean():+.3f}R")

    L+=[f"\n{sep}",
        "  FILES: smc_v4_results.csv | smc_v4_report.txt",
        "  Share these to finalize the new strategy config.",sep]
    return "\n".join(L),df


# ══════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════
async def main():
    print(f"\n{'═'*64}")
    print(f"  SMC PRO — BACKTESTER v4 (data-driven rebuild)")
    print(f"  Pairs={len(PAIRS)} | Days={DAYS_BACK}")
    print(f"  Score: SHORT≥{MIN_SCORE_SHORT} | LONG≥{MIN_SCORE_LONG} + trend gate")
    print(f"  OB: tol={OB_TOLERANCE_PCT*100:.1f}% lookback={OB_LOOKBACK} viol_win={OB_VIOLATION_WINDOW}")
    print(f"{'═'*64}\n")

    exchange=ccxt.binance({'enableRateLimit':True,'options':{'defaultType':'spot'}})
    all_trades=[]

    for sym in PAIRS:
        print(f"⬇️  {sym} ...", end=' ', flush=True)
        try:
            data=await fetch(exchange,sym,DAYS_BACK)
            t=walk_forward(data,sym,DAYS_BACK)
            all_trades.extend(t)
        except Exception as e:
            print(f"ERROR: {e}")
        await asyncio.sleep(0.5)

    await exchange.close()
    print(f"\n{'─'*64}\n📊 Report...\n")

    rpt,df=make_report(all_trades)
    print(rpt)

    df.to_csv("smc_v4_results.csv",index=False)
    print(f"\n✅ smc_v4_results.csv ({len(df)} trades)")
    with open("smc_v4_report.txt","w") as f: f.write(rpt)
    print(f"✅ smc_v4_report.txt")
    print("\n💡 Share both files — final threshold tuning next.\n")

if __name__=="__main__":
    asyncio.run(main())
