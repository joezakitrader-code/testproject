"""
SMC PRO — LIVE SCANNER + TELEGRAM ALERTS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Scans all USDT perpetuals on Binance every closed 1H candle.
Sends Telegram alerts for:
  🟢/🔴 Entry signal detected
  🎯 TP1 hit  →  SL moved to BE+0.25R (no further SL alerts)
  🎯🎯 TP2 hit →  SL moved to TP1
  🏆 TP3 hit  →  trade closed
  🛡️ SL hit   →  only if TP1 was NOT hit
  ⏱️ TIMEOUT  →  48H max hold, closed at market

STRATEGY MODES (set STRATEGY below):
  'B_PLUS'   — LONG≥82+TripleEMA+BTC | SHORT≥83
  'C_SNIPER' — LONG≥87+TripleEMA+BTC | SHORT≥85
  'BOTH'     — run both, label each alert

SETUP:
  pip install ccxt ta pandas numpy python-telegram-bot aiohttp
  Set TELEGRAM_TOKEN and TELEGRAM_CHAT_ID below.

DEPLOY:
  python smc_live_scanner.py
  (or run in tmux / systemd / docker for 24/7)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import asyncio
import json
import logging
import os
import time
from datetime import datetime, timezone
from typing import Optional

import aiohttp
import ccxt.async_support as ccxt
import numpy as np
import pandas as pd
import ta

# ════════════════════════════════════════════════
#  ⚙️  CONFIG — EDIT THESE
# ════════════════════════════════════════════════
TELEGRAM_TOKEN   = "7732870721:AAEHG3QJdo31S9sA8xjJzf-cXj6Tn4mo2uo"       # from @BotFather
TELEGRAM_CHAT_ID = "7500072234"         # from @userinfobot

STRATEGY         = 'B_PLUS'    # 'B_PLUS' | 'C_SNIPER' | 'BOTH'

MIN_VOLUME_USDT  = 10_000_000  # 10M — filter illiquid pairs
SCAN_INTERVAL_S  = 60          # seconds between completion checks (fine polling)
TRAIL_BE_R       = 0.25        # SL trail after TP1: BE + 0.25R
MAX_HOLD_HOURS   = 48          # TIMEOUT after this many hours
DEDUPE_HOURS     = 2           # min gap between signals on same pair

EXCLUDE_SYMBOLS = {
    'USDC/USDT:USDT','BUSD/USDT:USDT','TUSD/USDT:USDT',
    'USDP/USDT:USDT','DAI/USDT:USDT','FDUSD/USDT:USDT',
}

STATE_FILE = 'live_scanner_state.json'  # persists open trades across restarts

# ════════════════════════════════════════════════
#  STRATEGY CONFIGS
# ════════════════════════════════════════════════
STRATEGY_CONFIGS = {
    'B_PLUS': {
        'label': 'B+',
        'min_score_long':  82,
        'min_score_short': 83,
        'triple_ema_long': True,
        'btc_filter':      True,
    },
    'C_SNIPER': {
        'label': 'Sniper',
        'min_score_long':  87,
        'min_score_short': 85,
        'triple_ema_long': True,
        'btc_filter':      True,
    },
}

# ════════════════════════════════════════════════
#  LOGGING
# ════════════════════════════════════════════════
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('smc_live.log'),
    ]
)
log = logging.getLogger(__name__)

# ════════════════════════════════════════════════
#  CONSTANTS (same as backtester)
# ════════════════════════════════════════════════
STRUCTURE_LOOKBACK  = 30
HH_LL_LOOKBACK      = 15
HH_LL_BONUS         = 8
OB_IMPULSE_ATR_MULT = 0.5
OB_TOLERANCE_PCT    = 0.003
WARM_UP_BARS        = 100     # 1H bars needed before scanning
CANDLES_4H          = 250
CANDLES_1H          = 500
CANDLES_15M         = 300


# ════════════════════════════════════════════════
#  TELEGRAM
# ════════════════════════════════════════════════
async def tg_send(session: aiohttp.ClientSession, text: str):
    """Send a Telegram message. Never raises — logs failure instead."""
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {
        'chat_id':    TELEGRAM_CHAT_ID,
        'text':       text,
        'parse_mode': 'HTML',
        'disable_web_page_preview': True,
    }
    try:
        async with session.post(url, json=payload, timeout=aiohttp.ClientTimeout(total=10)) as r:
            if r.status != 200:
                body = await r.text()
                log.warning(f"Telegram error {r.status}: {body[:200]}")
    except Exception as e:
        log.warning(f"Telegram send failed: {e}")


def fmt_price(p: float, ref: float) -> str:
    """Format price with adaptive decimal places based on magnitude."""
    if ref < 0.001:   return f"{p:.8f}"
    if ref < 0.1:     return f"{p:.5f}"
    if ref < 1:       return f"{p:.4f}"
    if ref < 100:     return f"{p:.3f}"
    if ref < 10000:   return f"{p:.2f}"
    return f"{p:.1f}"


def entry_alert(sig: dict, cfg_label: str) -> str:
    s   = sig
    ref = s['entry']
    fp  = lambda p: fmt_price(p, ref)
    sym = s['symbol'].replace('/USDT:USDT', '')
    bias_emoji = '🟢 LONG' if s['bias'] == 'LONG' else '🔴 SHORT'
    quality_emoji = {'ELITE': '⚡', 'PREMIUM': '🔥', 'HIGH': '✅'}.get(s['quality'], '✅')

    lines = [
        f"{quality_emoji} <b>SMC SIGNAL [{cfg_label}]</b>",
        f"",
        f"<b>{bias_emoji}  {sym}/USDT  •  Score {s['score']}</b>",
        f"",
        f"📍 Entry  <code>{fp(s['entry'])}</code>",
        f"🛡️ SL     <code>{fp(s['sl'])}</code>  ({s['risk_pct']:.2f}% risk)",
        f"",
        f"🎯 TP1   <code>{fp(s['tp1'])}</code>  (+1.5R)",
        f"🎯 TP2   <code>{fp(s['tp2'])}</code>  (+2.5R)",
        f"🏆 TP3   <code>{fp(s['tp3'])}</code>  (+4.0R)",
        f"",
        f"📊 {s['structure']}  •  {s['pd_zone']}  •  {'HH/LL ✓' if s['hh_ll'] else ''}",
        f"🔍 {s['reasons'][:80]}",
        f"",
        f"🕐 {s['entry_time']} UTC",
    ]
    return '\n'.join(lines)


def tp1_alert(trade: dict) -> str:
    s   = trade
    ref = s['entry']
    fp  = lambda p: fmt_price(p, ref)
    sym = s['symbol'].replace('/USDT:USDT', '')
    new_sl = s['current_sl']
    return (
        f"🎯 <b>TP1 HIT — {sym}</b>  [{s['cfg_label']}]\n"
        f"\n"
        f"✅ Half position closed at <code>{fp(s['tp1'])}</code>  (+1.5R)\n"
        f"🛡️ SL moved to <code>{fp(new_sl)}</code>  (BE+{TRAIL_BE_R}R)\n"
        f"🎯 Targeting TP2 <code>{fp(s['tp2'])}</code>  (+2.5R)\n"
        f"🏆 Targeting TP3 <code>{fp(s['tp3'])}</code>  (+4.0R)\n"
        f"\n"
        f"⚡ <i>SL alerts suppressed — trade protected</i>"
    )


def tp2_alert(trade: dict) -> str:
    s   = trade
    ref = s['entry']
    fp  = lambda p: fmt_price(p, ref)
    sym = s['symbol'].replace('/USDT:USDT', '')
    return (
        f"🎯🎯 <b>TP2 HIT — {sym}</b>  [{s['cfg_label']}]\n"
        f"\n"
        f"✅ Partial closed at <code>{fp(s['tp2'])}</code>  (+2.5R)\n"
        f"🛡️ SL trailed to <code>{fp(s['current_sl'])}</code>  (TP1 level)\n"
        f"🏆 Riding to TP3 <code>{fp(s['tp3'])}</code>  (+4.0R)\n"
    )


def tp3_alert(trade: dict) -> str:
    s   = trade
    ref = s['entry']
    fp  = lambda p: fmt_price(p, ref)
    sym = s['symbol'].replace('/USDT:USDT', '')
    return (
        f"🏆 <b>TP3 HIT — {sym}</b>  [{s['cfg_label']}]\n"
        f"\n"
        f"💰 Full position closed at <code>{fp(s['tp3'])}</code>\n"
        f"📈 Avg return: <b>+2.67R</b>  🔥\n"
        f"⏱️ Held: {trade.get('bars_held', '?')} bars\n"
    )


def sl_alert(trade: dict, hit_price: float) -> str:
    s   = trade
    ref = s['entry']
    fp  = lambda p: fmt_price(p, ref)
    sym = s['symbol'].replace('/USDT:USDT', '')
    return (
        f"🛡️ <b>SL HIT — {sym}</b>  [{s['cfg_label']}]\n"
        f"\n"
        f"❌ Stopped out at <code>{fp(hit_price)}</code>  (-1.0R)\n"
        f"📉 No TPs were hit\n"
    )


def timeout_alert(trade: dict, exit_price: float) -> str:
    s   = trade
    ref = s['entry']
    fp  = lambda p: fmt_price(p, ref)
    sym = s['symbol'].replace('/USDT:USDT', '')
    pnl = (exit_price - s['entry']) / abs(s['entry'] - s['sl'])
    if s['bias'] == 'SHORT':
        pnl = -pnl
    return (
        f"⏱️ <b>TIMEOUT — {sym}</b>  [{s['cfg_label']}]\n"
        f"\n"
        f"Trade closed after {MAX_HOLD_HOURS}H at <code>{fp(exit_price)}</code>\n"
        f"P&L: <b>{pnl:+.2f}R</b>\n"
    )


# ════════════════════════════════════════════════
#  BTC REGIME
# ════════════════════════════════════════════════
class BTCRegime:
    def __init__(self):
        self._above_200 = None

    def update(self, df4: pd.DataFrame):
        if df4 is None or len(df4) < 5:
            return
        close = float(df4['close'].iloc[-1])
        ema200 = float(df4['ema_200'].iloc[-1]) if 'ema_200' in df4.columns else 0
        self._above_200 = close > ema200
        log.info(f"  BTC regime: {'BULL ▲' if self._above_200 else 'BEAR ▼'}  close={close:.0f}  ema200={ema200:.0f}")

    def longs_allowed(self) -> bool:
        if self._above_200 is None:
            return True
        return self._above_200

btc_regime = BTCRegime()


# ════════════════════════════════════════════════
#  INDICATORS
# ════════════════════════════════════════════════
def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
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
        df['cmf'] = ta.volume.ChaikinMoneyFlowIndicator(df['high'], df['low'], df['close'], df['volume']).chaikin_money_flow()
        df['mfi'] = ta.volume.MFIIndicator(df['high'], df['low'], df['close'], df['volume']).money_flow_index()
        df['vol_sma']   = df['volume'].rolling(20).mean()
        df['vol_ratio'] = df['volume'] / df['vol_sma'].replace(0, np.nan)
        tp_col = (df['high'] + df['low'] + df['close']) / 3
        df['vwap'] = (tp_col * df['volume']).cumsum() / df['volume'].cumsum()
        body = (df['close'] - df['open']).abs()
        uw   = df['high'] - df[['open','close']].max(axis=1)
        lw   = df[['open','close']].min(axis=1) - df['low']
        df['bull_engulf']  = ((df['close'].shift(1) < df['open'].shift(1)) & (df['close'] > df['open']) & (df['close'] > df['open'].shift(1)) & (df['open'] < df['close'].shift(1))).astype(int)
        df['bear_engulf']  = ((df['close'].shift(1) > df['open'].shift(1)) & (df['close'] < df['open']) & (df['close'] < df['open'].shift(1)) & (df['open'] > df['close'].shift(1))).astype(int)
        df['bull_pin']     = ((lw > body * 2.5) & (lw > uw * 2) & (df['close'] > df['open'])).astype(int)
        df['bear_pin']     = ((uw > body * 2.5) & (uw > lw * 2) & (df['close'] < df['open'])).astype(int)
        df['hammer']       = ((lw > body * 2.0) & (lw > uw * 1.5)).astype(int)
        df['shooting_star'] = ((uw > body * 2.0) & (uw > lw * 1.5)).astype(int)
    except Exception as e:
        log.debug(f"Indicator error: {e}")
    return df


# ════════════════════════════════════════════════
#  SMC ENGINE
# ════════════════════════════════════════════════
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
            return False, "Not enough data"
        recent = df_4h.iloc[-lookback:]
        prior  = df_4h.iloc[-(lookback*2):-lookback]
        if direction == 'LONG':
            rh, ph = recent['high'].max(), prior['high'].max()
            return (rh > ph), f"4H HH {ph:.5f}→{rh:.5f}"
        else:
            rl, pl = recent['low'].min(), prior['low'].min()
            return (rl < pl), f"4H LL {pl:.5f}→{rl:.5f}"

    def detect_structure_break(self, df, highs, lows, lookback=STRUCTURE_LOOKBACK):
        events = []
        close  = df['close']
        n      = len(df)
        start  = max(0, n - lookback - 15)
        for k in range(1, len(highs)):
            ph = highs[k-1]; ch = highs[k]
            if ch['i'] < start: continue
            level = ph['price']
            for j in range(ch['i'], min(ch['i']+10, n)):
                if close.iloc[j] > level:
                    events.append({'kind': 'BOS_BULL' if ch['price'] > ph['price'] else 'MSS_BULL', 'level': level, 'bar': j})
                    break
        for k in range(1, len(lows)):
            pl = lows[k-1]; cl = lows[k]
            if cl['i'] < start: continue
            level = pl['price']
            for j in range(cl['i'], min(cl['i']+10, n)):
                if close.iloc[j] < level:
                    events.append({'kind': 'BOS_BEAR' if cl['price'] < pl['price'] else 'MSS_BEAR', 'level': level, 'bar': j})
                    break
        if not events:
            return None
        latest = sorted(events, key=lambda x: x['bar'])[-1]
        return latest if latest['bar'] >= n - lookback else None

    def find_order_blocks(self, df, direction, lookback=60):
        obs = []
        n   = len(df)
        start = max(2, n - lookback)
        for i in range(start, n - 3):
            c = df.iloc[i]
            atr_local = df['atr'].iloc[i] if ('atr' in df.columns and not pd.isna(df['atr'].iloc[i])) else (c['high'] - c['low'])
            min_impulse = atr_local * OB_IMPULSE_ATR_MULT
            if direction == 'LONG':
                if c['close'] >= c['open']: continue
                if df['high'].iloc[i+1:min(i+5, n)].max() - c['low'] < min_impulse: continue
                ob = {'top': max(c['open'], c['close']), 'bottom': c['low'], 'mid': (max(c['open'], c['close']) + c['low']) / 2, 'bar': i}
                if (df['close'].iloc[i+1:n] < (ob['top'] + ob['bottom']) / 2).any(): continue
                obs.append(ob)
            else:
                if c['close'] <= c['open']: continue
                if c['high'] - df['low'].iloc[i+1:min(i+5, n)].min() < min_impulse: continue
                ob = {'top': c['high'], 'bottom': min(c['open'], c['close']), 'mid': (c['high'] + min(c['open'], c['close'])) / 2, 'bar': i}
                if (df['close'].iloc[i+1:n] > (ob['top'] + ob['bottom']) / 2).any(): continue
                obs.append(ob)
        obs.sort(key=lambda x: x['bar'], reverse=True)
        return obs

    def price_in_ob(self, price, ob, tolerance_pct=OB_TOLERANCE_PCT):
        tol = ob['top'] * tolerance_pct
        return (ob['bottom'] - tol) <= price <= (ob['top'] + tol)

    def find_fvg(self, df, direction, lookback=25):
        fvgs = []
        n    = len(df)
        for i in range(max(1, n - lookback), n - 1):
            prev = df.iloc[i-1]; nxt = df.iloc[i+1]
            if direction == 'LONG' and prev['high'] < nxt['low']:
                fvgs.append({'top': nxt['low'], 'bottom': prev['high'], 'mid': (nxt['low'] + prev['high']) / 2, 'bar': i})
            elif direction == 'SHORT' and prev['low'] > nxt['high']:
                fvgs.append({'top': prev['low'], 'bottom': nxt['high'], 'mid': (prev['low'] + nxt['high']) / 2, 'bar': i})
        return fvgs

    def recent_liquidity_sweep(self, df, direction, highs, lows, lookback=25):
        n     = len(df)
        start = n - lookback
        if direction == 'LONG':
            for sl in reversed(lows):
                if sl['i'] < start: continue
                level = sl['price']
                for j in range(sl['i']+1, min(sl['i']+8, n)):
                    c = df.iloc[j]
                    if c['low'] < level and c['close'] > level:
                        return {'level': level, 'bar': j, 'type': 'SWEEP_LOW'}
        else:
            for sh in reversed(highs):
                if sh['i'] < start: continue
                level = sh['price']
                for j in range(sh['i']+1, min(sh['i']+8, n)):
                    c = df.iloc[j]
                    if c['high'] > level and c['close'] < level:
                        return {'level': level, 'bar': j, 'type': 'SWEEP_HIGH'}
        return None

    def pd_zone(self, df_4h, price):
        hi  = df_4h['high'].iloc[-50:].max()
        lo  = df_4h['low'].iloc[-50:].min()
        rng = hi - lo
        if rng == 0:
            return 'NEUTRAL', 0.5
        pos = (price - lo) / rng
        if pos < 0.40:   return 'DISCOUNT', pos
        elif pos > 0.60: return 'PREMIUM',  pos
        return 'NEUTRAL', pos

    def is_triple_ema_bull(self, df_4h):
        l = df_4h.iloc[-1]
        return float(l.get('ema_21', 0)) > float(l.get('ema_50', 0)) > float(l.get('ema_200', 0))

smc = SMCEngine()


# ════════════════════════════════════════════════
#  SCORER
# ════════════════════════════════════════════════
def score_setup(direction, ob, structure, sweep, fvg_near,
                df_1h, df_15m, df_4h, pd_label, hh_ll_confirmed):
    score   = 0
    reasons = []
    l1  = df_1h.iloc[-1]; p1 = df_1h.iloc[-2]
    l15 = df_15m.iloc[-1]; l4 = df_4h.iloc[-1]

    if structure:
        if 'MSS' in structure['kind']: score += 20; reasons.append(f"MSS({structure['kind']})")
        else:                           score += 14; reasons.append(f"BOS({structure['kind']})")

    if ob:
        ob_sz = (ob['top'] - ob['bottom']) / ob['bottom'] * 100
        if ob_sz < 0.8:   score += 20; reasons.append(f"TightOB({ob_sz:.2f}%)")
        elif ob_sz < 2.0: score += 13; reasons.append(f"OB({ob_sz:.2f}%)")
        else:             score += 7;  reasons.append(f"WideOB({ob_sz:.2f}%)")

    e21 = float(l4.get('ema_21', 0)); e50 = float(l4.get('ema_50', 0)); e200 = float(l4.get('ema_200', 0))
    if direction == 'LONG':
        if e21 > e50 > e200:           score += 15; reasons.append("4H_TripleEMA_Bull")
        elif e21 > e50:                score += 10; reasons.append("4H_EMA_Bull")
        elif pd_label == 'DISCOUNT':   score += 6;  reasons.append("Discount")
    else:
        if e21 < e50 < e200:           score += 15; reasons.append("4H_TripleEMA_Bear")
        elif e21 < e50:                score += 10; reasons.append("4H_EMA_Bear")
        elif pd_label == 'PREMIUM':    score += 6;  reasons.append("Premium")

    if hh_ll_confirmed: score += HH_LL_BONUS; reasons.append(f"HH/LL+{HH_LL_BONUS}")

    trigger = False
    if direction == 'LONG':
        if l1.get('bull_engulf', 0):   score += 25; trigger = True; reasons.append("1H_BullEngulf")
        elif l1.get('bull_pin', 0):    score += 22; trigger = True; reasons.append("1H_BullPin")
        elif l1.get('hammer', 0):      score += 18; trigger = True; reasons.append("1H_Hammer")
        elif p1.get('bull_engulf', 0): score += 14; trigger = True; reasons.append("1H_BullEngulf_prev")
        elif p1.get('bull_pin', 0):    score += 11; trigger = True; reasons.append("1H_BullPin_prev")
        elif p1.get('hammer', 0):      score += 9;  trigger = True; reasons.append("1H_Hammer_prev")
    else:
        if l1.get('bear_engulf', 0):      score += 25; trigger = True; reasons.append("1H_BearEngulf")
        elif l1.get('bear_pin', 0):       score += 22; trigger = True; reasons.append("1H_BearPin")
        elif l1.get('shooting_star', 0):  score += 18; trigger = True; reasons.append("1H_ShootStar")
        elif p1.get('bear_engulf', 0):    score += 14; trigger = True; reasons.append("1H_BearEngulf_prev")
        elif p1.get('bear_pin', 0):       score += 11; trigger = True; reasons.append("1H_BearPin_prev")
        elif p1.get('shooting_star', 0):  score += 9;  trigger = True; reasons.append("1H_SS_prev")
    if not trigger: score -= 12

    rsi1 = float(l1.get('rsi', 50)); macd1 = float(l1.get('macd', 0)); ms1 = float(l1.get('macd_signal', 0))
    pm1  = float(p1.get('macd', 0));  pms1 = float(p1.get('macd_signal', 0))
    sk1  = float(l1.get('srsi_k', 0.5))
    sd1  = float(l1.get('srsi_d', 0.5))
    if direction == 'LONG':
        if 28 <= rsi1 <= 55:             score += 4; reasons.append(f"RSI_reset({rsi1:.0f})")
        elif rsi1 < 28:                  score += 3; reasons.append(f"RSI_OS({rsi1:.0f})")
        if macd1 > ms1 and pm1 <= pms1:  score += 5; reasons.append("MACD_BullX")
        elif macd1 > ms1:                score += 2; reasons.append("MACD_bull")
        if sk1 < 0.3 and sk1 > sd1:      score += 3; reasons.append("Stoch_BullX")
    else:
        if 45 <= rsi1 <= 72:             score += 4; reasons.append(f"RSI_OBzone({rsi1:.0f})")
        elif rsi1 > 72:                  score += 3; reasons.append(f"RSI_OB({rsi1:.0f})")
        if macd1 < ms1 and pm1 >= pms1:  score += 5; reasons.append("MACD_BearX")
        elif macd1 < ms1:                score += 2; reasons.append("MACD_bear")
        if sk1 > 0.7 and sk1 < sd1:      score += 3; reasons.append("Stoch_BearX")

    extras = 0
    if sweep:    extras += 4; reasons.append("LiqSweep")
    if fvg_near: extras += 3; reasons.append("FVG+OB")
    vr15 = float(l15.get('vol_ratio', 1.0)) if not pd.isna(l15.get('vol_ratio', 1.0)) else 1.0
    if   vr15 >= 2.5: extras += 3; reasons.append(f"15M_vol{vr15:.1f}x")
    elif vr15 >= 1.5: extras += 1; reasons.append(f"15M_vol{vr15:.1f}x")
    close1 = float(l1.get('close', 0)); vwap1 = float(l1.get('vwap', 0))
    if direction == 'LONG' and close1 < vwap1:    extras += 1; reasons.append("BelowVWAP")
    elif direction == 'SHORT' and close1 > vwap1: extras += 1; reasons.append("AboveVWAP")
    score += min(extras, 10)
    return max(0, min(int(score), 100)), reasons


# ════════════════════════════════════════════════
#  SIGNAL GENERATOR
# ════════════════════════════════════════════════
def analyse_symbol(df4: pd.DataFrame, df1: pd.DataFrame, df15: pd.DataFrame, symbol: str) -> Optional[dict]:
    try:
        if len(df1) < 80 or len(df15) < 40:
            return None
        price = float(df1['close'].iloc[-1])
        l4 = df4.iloc[-1]
        e21 = float(l4.get('ema_21', 0)); e50 = float(l4.get('ema_50', 0))
        if e21 > e50:   bias = 'LONG'
        elif e21 < e50: bias = 'SHORT'
        else: return None

        triple_ema_bull = smc.is_triple_ema_bull(df4)
        btc_long_ok     = btc_regime.longs_allowed()
        hh_ll_ok, _    = smc.check_4h_hh_ll(df4, bias, HH_LL_LOOKBACK)
        pd_label, _    = smc.pd_zone(df4, price)

        if bias == 'LONG'  and pd_label == 'PREMIUM':  return None
        if bias == 'SHORT' and pd_label == 'DISCOUNT': return None

        highs1, lows1 = smc.swing_highs_lows(df1, left=4, right=4)
        structure = smc.detect_structure_break(df1, highs1, lows1, lookback=STRUCTURE_LOOKBACK)
        if structure:
            if bias == 'LONG'  and 'BEAR' in structure['kind']: return None
            if bias == 'SHORT' and 'BULL' in structure['kind']: return None

        obs = smc.find_order_blocks(df1, bias, lookback=60)
        if not obs: return None
        active_ob = next((ob for ob in obs if smc.price_in_ob(price, ob, OB_TOLERANCE_PCT)), None)
        if not active_ob: return None

        fvgs     = smc.find_fvg(df1, bias, lookback=25)
        fvg_near = next((f for f in fvgs if f['bottom'] < active_ob['top'] and f['top'] > active_ob['bottom']), None)
        sweep    = smc.recent_liquidity_sweep(df1, bias, highs1, lows1, lookback=20)

        score, reasons = score_setup(bias, active_ob, structure, sweep, fvg_near,
                                     df1, df15, df4, pd_label, hh_ll_ok)

        atr1  = float(df1['atr'].iloc[-1])
        entry = price
        if bias == 'LONG':
            sl = min(active_ob['bottom'] - atr1 * 0.2, entry - atr1 * 0.6)
        else:
            sl = max(active_ob['top'] + atr1 * 0.2, entry + atr1 * 0.6)

        risk = abs(entry - sl)
        if risk < entry * 0.001: return None

        tps = ([entry + risk*1.5, entry + risk*2.5, entry + risk*4.0] if bias == 'LONG'
               else [entry - risk*1.5, entry - risk*2.5, entry - risk*4.0])

        quality = 'ELITE' if score >= 92 else ('PREMIUM' if score >= 85 else 'HIGH')

        return {
            'symbol':       symbol,
            'bias':         bias,
            'quality':      quality,
            'score':        score,
            'entry':        entry,
            'sl':           sl,
            'tp1':          tps[0],
            'tp2':          tps[1],
            'tp3':          tps[2],
            'risk_pct':     risk / entry * 100,
            'pd_zone':      pd_label,
            'hh_ll':        hh_ll_ok,
            'triple_ema':   triple_ema_bull,
            'btc_long_ok':  btc_long_ok,
            'structure':    structure['kind'] if structure else 'NONE',
            'reasons':      ' | '.join(reasons[:8]),
            'entry_time':   datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M'),
        }
    except Exception as e:
        log.debug(f"analyse_symbol {symbol}: {e}")
        return None


def passes_config(sig: dict, cfg: dict) -> bool:
    bias = sig['bias']
    if bias == 'LONG':
        if sig['score'] < cfg['min_score_long']:    return False
        if cfg['btc_filter'] and not sig['btc_long_ok']: return False
        if cfg['triple_ema_long'] and sig['score'] < 87 and not sig['triple_ema']: return False
    else:
        if sig['score'] < cfg['min_score_short']:   return False
    return True


# ════════════════════════════════════════════════
#  CANDLE FETCHER
# ════════════════════════════════════════════════
async def fetch_ohlcv(exchange, symbol: str, tf: str, limit: int) -> Optional[pd.DataFrame]:
    try:
        raw = await exchange.fetch_ohlcv(symbol, tf, limit=limit)
        if not raw or len(raw) < 10:
            return None
        df = pd.DataFrame(raw, columns=['ts', 'open', 'high', 'low', 'close', 'volume'])
        df['ts'] = pd.to_datetime(df['ts'], unit='ms', utc=True)
        # Drop the current (incomplete) candle — use only closed bars
        df = df.iloc[:-1].reset_index(drop=True)
        for c in ['open', 'high', 'low', 'close', 'volume']:
            df[c] = df[c].astype(float)
        return df
    except Exception as e:
        log.debug(f"fetch_ohlcv {symbol} {tf}: {e}")
        return None


# ════════════════════════════════════════════════
#  STATE PERSISTENCE
# ════════════════════════════════════════════════
def load_state() -> dict:
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE) as f:
                return json.load(f)
        except Exception:
            pass
    return {'open_trades': {}, 'last_signal': {}}


def save_state(state: dict):
    try:
        with open(STATE_FILE, 'w') as f:
            json.dump(state, f, indent=2, default=str)
    except Exception as e:
        log.warning(f"save_state: {e}")


# ════════════════════════════════════════════════
#  LIVE TRADE TRACKER
#  Checks open trades on every new 1H candle.
#  TP1 hit → suppress future SL alerts for this trade.
# ════════════════════════════════════════════════
async def check_open_trades(exchange, session: aiohttp.ClientSession, state: dict):
    trades = state.get('open_trades', {})
    if not trades:
        return

    for trade_id, trade in list(trades.items()):
        symbol = trade['symbol']
        try:
            df1 = await fetch_ohlcv(exchange, symbol, '1h', 3)
            if df1 is None or len(df1) < 1:
                continue

            bar = df1.iloc[-1]
            hi  = float(bar['high'])
            lo  = float(bar['low'])
            now_utc = datetime.now(timezone.utc)

            entry      = trade['entry']
            sl_current = trade['current_sl']
            tp1        = trade['tp1']
            tp2        = trade['tp2']
            tp3        = trade['tp3']
            bias       = trade['bias']
            tp1_hit    = trade.get('tp1_hit', False)
            tp2_hit    = trade.get('tp2_hit', False)
            risk       = abs(entry - trade['sl'])

            be_plus = (entry + risk * TRAIL_BE_R) if bias == 'LONG' else (entry - risk * TRAIL_BE_R)

            # ── LONG checks ──────────────────────────────
            if bias == 'LONG':
                # TP3 full close
                if hi >= tp3:
                    trade['bars_held'] = trade.get('bars_held', 0) + 1
                    await tg_send(session, tp3_alert(trade))
                    log.info(f"TP3 hit: {symbol}")
                    del trades[trade_id]
                    continue
                # TP2
                if hi >= tp2 and not tp2_hit:
                    trade['tp2_hit']    = True
                    trade['current_sl'] = tp1           # trail to TP1
                    await tg_send(session, tp2_alert(trade))
                    log.info(f"TP2 hit: {symbol}")
                # TP1
                if hi >= tp1 and not tp1_hit:
                    trade['tp1_hit']    = True
                    trade['current_sl'] = be_plus       # trail to BE+0.25R
                    await tg_send(session, tp1_alert(trade))
                    log.info(f"TP1 hit: {symbol}")
                # SL — only alert if TP1 was never hit
                if lo <= sl_current:
                    if not trade.get('tp1_hit', False):
                        await tg_send(session, sl_alert(trade, sl_current))
                        log.info(f"SL hit: {symbol} (no TP1)")
                    else:
                        log.info(f"SL hit: {symbol} (trailing, TP1 was hit — no alert)")
                    del trades[trade_id]
                    continue

            # ── SHORT checks ─────────────────────────────
            else:
                # TP3 full close
                if lo <= tp3:
                    trade['bars_held'] = trade.get('bars_held', 0) + 1
                    await tg_send(session, tp3_alert(trade))
                    log.info(f"TP3 hit: {symbol}")
                    del trades[trade_id]
                    continue
                # TP2
                if lo <= tp2 and not tp2_hit:
                    trade['tp2_hit']    = True
                    trade['current_sl'] = tp1
                    await tg_send(session, tp2_alert(trade))
                    log.info(f"TP2 hit: {symbol}")
                # TP1
                if lo <= tp1 and not tp1_hit:
                    trade['tp1_hit']    = True
                    trade['current_sl'] = be_plus
                    await tg_send(session, tp1_alert(trade))
                    log.info(f"TP1 hit: {symbol}")
                # SL — only alert if TP1 was never hit
                if hi >= sl_current:
                    if not trade.get('tp1_hit', False):
                        await tg_send(session, sl_alert(trade, sl_current))
                        log.info(f"SL hit: {symbol} (no TP1)")
                    else:
                        log.info(f"SL hit: {symbol} (trailing, TP1 was hit — no alert)")
                    del trades[trade_id]
                    continue

            # ── TIMEOUT check ────────────────────────────
            entry_dt = datetime.fromisoformat(trade['entry_time'].replace(' ', 'T') + '+00:00')
            hours_held = (now_utc - entry_dt).total_seconds() / 3600
            if hours_held >= MAX_HOLD_HOURS:
                exit_price = float(df1.iloc[-1]['close'])
                await tg_send(session, timeout_alert(trade, exit_price))
                log.info(f"TIMEOUT: {symbol} after {hours_held:.1f}H")
                del trades[trade_id]
                continue

            trade['bars_held'] = trade.get('bars_held', 0) + 1

        except Exception as e:
            log.warning(f"check_open_trades {symbol}: {e}")

    save_state(state)


# ════════════════════════════════════════════════
#  PAIR SCANNER — runs on each 1H close
# ════════════════════════════════════════════════
async def scan_symbols(exchange, session: aiohttp.ClientSession,
                       symbols: list, state: dict):
    active_configs = {}
    if STRATEGY in ('B_PLUS', 'BOTH'):
        active_configs['B_PLUS'] = STRATEGY_CONFIGS['B_PLUS']
    if STRATEGY in ('C_SNIPER', 'BOTH'):
        active_configs['C_SNIPER'] = STRATEGY_CONFIGS['C_SNIPER']

    new_signals = 0

    for symbol in symbols:
        try:
            df4  = await fetch_ohlcv(exchange, symbol, '4h',  CANDLES_4H)
            df1  = await fetch_ohlcv(exchange, symbol, '1h',  CANDLES_1H)
            df15 = await fetch_ohlcv(exchange, symbol, '15m', CANDLES_15M)

            if df4 is None or df1 is None or df15 is None: continue
            if len(df1) < WARM_UP_BARS: continue

            df4  = add_indicators(df4)
            df1  = add_indicators(df1)
            df15 = add_indicators(df15)

            sig = analyse_symbol(df4, df1, df15, symbol)
            if sig is None: continue

            for cfg_key, cfg in active_configs.items():
                if not passes_config(sig, cfg): continue

                # Dedupe: skip if we already have a recent signal for this pair+config
                dedupe_key = f"{symbol}_{cfg_key}"
                last_ts    = state['last_signal'].get(dedupe_key)
                now_ts     = time.time()
                if last_ts and (now_ts - last_ts) < DEDUPE_HOURS * 3600:
                    continue

                state['last_signal'][dedupe_key] = now_ts

                # Build trade record for tracking
                risk = abs(sig['entry'] - sig['sl'])
                be_plus = (sig['entry'] + risk * TRAIL_BE_R) if sig['bias'] == 'LONG' else (sig['entry'] - risk * TRAIL_BE_R)

                trade_id = f"{symbol}_{cfg_key}_{int(now_ts)}"
                trade = {
                    **sig,
                    'cfg_key':     cfg_key,
                    'cfg_label':   cfg['label'],
                    'current_sl':  sig['sl'],
                    'be_plus':     be_plus,
                    'tp1_hit':     False,
                    'tp2_hit':     False,
                    'bars_held':   0,
                    'entry_time':  sig['entry_time'],
                }
                state['open_trades'][trade_id] = trade

                # Send Telegram entry alert
                await tg_send(session, entry_alert(sig, cfg['label']))
                log.info(f"🚨 SIGNAL [{cfg['label']}] {symbol} {sig['bias']} sc={sig['score']}")
                new_signals += 1

            await asyncio.sleep(0.3)   # gentle rate limiting per symbol

        except Exception as e:
            log.debug(f"scan {symbol}: {e}")

    save_state(state)
    return new_signals


# ════════════════════════════════════════════════
#  CANDLE CLOSE DETECTOR
#  Returns True on the first poll after a new 1H candle closes.
# ════════════════════════════════════════════════
class CandleCloseDetector:
    def __init__(self, tf_seconds: int = 3600):
        self.tf_seconds   = tf_seconds
        self.last_candle  = -1

    def is_new_candle(self) -> bool:
        now    = int(time.time())
        bucket = now - (now % self.tf_seconds)
        if bucket != self.last_candle:
            self.last_candle = bucket
            return True
        return False


# ════════════════════════════════════════════════
#  SYMBOL FETCHER
# ════════════════════════════════════════════════
async def fetch_symbols(exchange) -> list:
    await exchange.load_markets()
    tickers = await exchange.fetch_tickers()
    pairs   = []
    for sym, tick in tickers.items():
        if not sym.endswith('/USDT:USDT'): continue
        if sym in EXCLUDE_SYMBOLS:        continue
        vol = tick.get('quoteVolume') or 0
        if vol >= MIN_VOLUME_USDT:
            pairs.append(sym)
    pairs.sort()
    return pairs


# ════════════════════════════════════════════════
#  MAIN LOOP
# ════════════════════════════════════════════════
async def main():
    log.info("═" * 55)
    log.info("  SMC PRO — LIVE SCANNER")
    log.info(f"  Strategy : {STRATEGY}")
    log.info(f"  Min vol  : {MIN_VOLUME_USDT/1e6:.0f}M USDT")
    log.info(f"  Dedupe   : {DEDUPE_HOURS}H  |  Max hold: {MAX_HOLD_HOURS}H")
    log.info("═" * 55)

    if TELEGRAM_TOKEN == "YOUR_BOT_TOKEN_HERE":
        log.warning("⚠️  TELEGRAM_TOKEN not set — alerts disabled")
    if TELEGRAM_CHAT_ID == "YOUR_CHAT_ID_HERE":
        log.warning("⚠️  TELEGRAM_CHAT_ID not set — alerts disabled")

    exchange = ccxt.binance({
        'enableRateLimit': True,
        'options': {'defaultType': 'future'},
    })

    state = load_state()
    log.info(f"  Loaded {len(state.get('open_trades', {}))} open trades from state file")

    # Fetch symbol list once; refresh every 24H
    log.info("  Fetching active USDT perp pairs...")
    symbols = await fetch_symbols(exchange)
    log.info(f"  Found {len(symbols)} pairs ≥{MIN_VOLUME_USDT/1e6:.0f}M USDT")
    symbols_fetched_at = time.time()

    detector = CandleCloseDetector(tf_seconds=3600)

    # Send startup message
    async with aiohttp.ClientSession() as session:
        await tg_send(session, (
            f"🤖 <b>SMC PRO Scanner started</b>\n"
            f"Strategy: <b>{STRATEGY}</b>\n"
            f"Watching <b>{len(symbols)}</b> USDT perp pairs\n"
            f"Volume filter: ≥{MIN_VOLUME_USDT/1e6:.0f}M USDT\n"
            f"Trail: BE+{TRAIL_BE_R}R after TP1"
        ))

        while True:
            try:
                # Refresh symbol list every 24H
                if time.time() - symbols_fetched_at > 86400:
                    symbols = await fetch_symbols(exchange)
                    symbols_fetched_at = time.time()
                    log.info(f"  Symbol refresh: {len(symbols)} pairs")

                # Always check open trade TP/SL on every poll
                await check_open_trades(exchange, session, state)

                # Only scan for new signals on 1H candle close
                if detector.is_new_candle():
                    # Refresh BTC regime
                    df4_btc = await fetch_ohlcv(exchange, 'BTC/USDT:USDT', '4h', 220)
                    if df4_btc is not None:
                        df4_btc = add_indicators(df4_btc)
                        btc_regime.update(df4_btc)

                    log.info(f"  🔍 1H candle closed — scanning {len(symbols)} pairs...")
                    t0 = time.time()
                    n  = await scan_symbols(exchange, session, symbols, state)
                    elapsed = time.time() - t0
                    log.info(f"  Scan done in {elapsed:.1f}s — {n} new signal(s)  |  {len(state.get('open_trades', {}))} open trades")

            except Exception as e:
                log.error(f"Main loop error: {e}", exc_info=True)

            await asyncio.sleep(SCAN_INTERVAL_S)


if __name__ == '__main__':
    asyncio.run(main())
