"""
SWING ICT OB LIVE SCANNER v1.0
================================
Railway deployment — 4H swing trade scanner.

BACKTEST RESULTS (v4.0 validation):
  Config  : V1 — S2 SHORT-only in BEAR, LONG-only in BULL
  WR      : 59.3% TP1
  Avg TP1 : +9.32% per hit (50% position)
  Ladder  : +2.49% avg (50/30/20 partials)
  RR      : 2.23:1
  Hold    : 2.3d avg (max 14d)
  Signals : ~0.15/day (4/month) in current BEAR market

KEY DIFFERENCE from scalping scanner:
  4H timeframe — not 1H
  TP1 = 2x ATR (~9% avg) not 1x
  Max hold = 14 days not 24h
  SHORT only in BEAR | LONG only in BULL (regime-aligned)
  Discount zone = strict 50% (only deep premium/discount)
  BTC regime via EMA50 (more reactive than EMA21 for swing)

Set in Railway > Variables:
  TELEGRAM_TOKEN    = your bot token
  TELEGRAM_CHAT_ID  = your chat id
  BINANCE_API_KEY   = optional
  BINANCE_SECRET    = optional
"""

import asyncio
import os
import sys
import logging
import ccxt.async_support as ccxt
from telegram import Bot, Update
from telegram.ext import Application, CommandHandler, ContextTypes
from telegram.constants import ParseMode
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
import ta
from collections import deque
import warnings

warnings.filterwarnings('ignore')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
# CREDENTIALS
# ═══════════════════════════════════════════════════════════════

TELEGRAM_TOKEN   = "7957028587:AAE7aSYtE4hCxxTIPkAs_1ULJ9e8alkY6Ic"
TELEGRAM_CHAT_ID = "-1003659830260"
BINANCE_API_KEY  = None
BINANCE_SECRET   = None


# ═══════════════════════════════════════════════════════════════
# SWING SETTINGS — V1 FINAL CONFIG
# ═══════════════════════════════════════════════════════════════

# OB structure
DISPLACEMENT_MULT = 1.5
SWEEP_REQUIRED    = True
FVG_REQUIRED      = False
BOS_REQUIRED      = True
DISCOUNT_PCT      = 0.50     # SHORT only above 50% of range (premium)
                             # LONG  only below 50% of range (discount)
SWING_LOOKBACK    = 15
RANGE_LOOKBACK    = 80

# Confirmation
ADX_MIN           = 20
RSI_THRESH        = 55       # not extreme OS — swing uses broader RSI
VOL_MIN           = 1.3      # 4H vol threshold
EMA_REQUIRED      = False    # EMA_ALIGN adds quality but not required

# Targets — swing sized
ATR_TP1_MULT = 2.0           # avg +9.32% per TP1 hit
ATR_TP2_MULT = 4.0
ATR_TP3_MULT = 8.0
OB_SL_BUFFER = 0.002         # 0.2% wider for swing

# Close plan (regime-aligned: SHORT 50/30/20, LONG 40/35/25)
SHORT_TP1_PCT = 0.50
SHORT_TP2_PCT = 0.30
SHORT_TP3_PCT = 0.20
LONG_TP1_PCT  = 0.40
LONG_TP2_PCT  = 0.35
LONG_TP3_PCT  = 0.25

# Scanner
TIMEFRAME         = '4h'
CANDLE_LIMIT      = 500      # need more history for 4H swing OBs
SCAN_INTERVAL_MIN = 240      # scan every 4H (on candle close)
MAX_PAIRS         = 300
MIN_VOLUME_USDT   = 3_000_000
COOLDOWN_HOURS    = 24
MAX_TRADE_BARS    = 336      # 14 days on 4H (336 bars)
MIN_RETEST_BARS   = 3


# ═══════════════════════════════════════════════════════════════
# SCANNER
# ═══════════════════════════════════════════════════════════════

class SwingScanner:

    def __init__(self):
        self.exchange = ccxt.binance({
            'apiKey':          BINANCE_API_KEY,
            'secret':          BINANCE_SECRET,
            'enableRateLimit': True,
            'options':         {'defaultType': 'future'},
        })
        self.bot            = Bot(token=TELEGRAM_TOKEN)
        self.chat_id        = TELEGRAM_CHAT_ID
        self.btc_regime     = 'UNKNOWN'
        self.btc_price      = None
        self.btc_ema50      = None
        self.signal_history = deque(maxlen=200)
        self.active_trades  = {}
        self.cooldown       = {}
        self.fired          = set()
        self.is_scanning    = False
        self.stats = {
            'total_signals':  0,
            'long_signals':   0,
            'short_signals':  0,
            'tp1_hits':       0,
            'tp2_hits':       0,
            'tp3_hits':       0,
            'sl_hits':        0,
            'timeouts':       0,
            'be_saves':       0,
            'regime_blocked': 0,
            'last_scan':      None,
            'session_start':  datetime.now(),
        }

    # ── BTC Regime (EMA50 for swing) ─────────────────────────

    async def update_btc_regime(self):
        try:
            ohlcv = await self.exchange.fetch_ohlcv('BTC/USDT:USDT', '4h', limit=100)
            df = pd.DataFrame(ohlcv, columns=['timestamp','open','high','low','close','volume'])
            df['ema50']     = ta.trend.EMAIndicator(df['close'], window=50).ema_indicator()
            last            = df.iloc[-1]
            prev            = self.btc_regime
            self.btc_price  = last['close']
            self.btc_ema50  = last['ema50']
            self.btc_regime = 'BULL' if self.btc_price > self.btc_ema50 else 'BEAR'
            if prev != self.btc_regime:
                logger.info(f'Regime flip: {prev} -> {self.btc_regime}')
                await self.send_msg(
                    f"🔄 <b>BTC Regime Flip!</b>\n"
                    f"{prev} → <b>{self.btc_regime}</b>\n"
                    f"Price: ${self.btc_price:,.0f} | EMA50: ${self.btc_ema50:,.0f}\n\n"
                    f"{'Now scanning LONGs' if self.btc_regime=='BULL' else 'Now scanning SHORTs'}"
                )
            logger.info(f'BTC: {self.btc_regime} (${self.btc_price:,.0f} vs EMA50 ${self.btc_ema50:,.0f})')
        except Exception as e:
            logger.error(f'BTC regime error: {e}')

    # ── Pairs ────────────────────────────────────────────────

    async def get_pairs(self):
        try:
            await self.exchange.load_markets()
            tickers = await self.exchange.fetch_tickers()
            pairs = [
                s for s in self.exchange.symbols
                if s.endswith('/USDT:USDT')
                and 'PERP' not in s
                and tickers.get(s, {}).get('quoteVolume', 0) > MIN_VOLUME_USDT
            ]
            pairs.sort(key=lambda x: tickers.get(x, {}).get('quoteVolume', 0), reverse=True)
            logger.info(f'{len(pairs)} pairs loaded')
            return pairs[:MAX_PAIRS]
        except Exception as e:
            logger.error(f'Pairs error: {e}')
            return []

    # ── Candles + Indicators ─────────────────────────────────

    async def fetch_candles(self, symbol):
        try:
            ohlcv = await self.exchange.fetch_ohlcv(symbol, TIMEFRAME, limit=CANDLE_LIMIT)
            df = pd.DataFrame(ohlcv, columns=['timestamp','open','high','low','close','volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
            return df if len(df) >= 80 else None
        except:
            return None

    def add_indicators(self, df):
        if len(df) < 50: return df
        try:
            df['ema_21']   = ta.trend.EMAIndicator(df['close'], window=21).ema_indicator()
            df['ema_50']   = ta.trend.EMAIndicator(df['close'], window=min(50, len(df)-1)).ema_indicator()
            df['atr']      = ta.volatility.AverageTrueRange(df['high'],df['low'],df['close'],window=14).average_true_range()
            df['vol_sma']  = df['volume'].rolling(20).mean()
            df['vol_ratio']= df['volume'] / df['vol_sma'].replace(0, np.nan)
            df['rsi']      = ta.momentum.RSIIndicator(df['close'], window=14).rsi()
            adx            = ta.trend.ADXIndicator(df['high'],df['low'],df['close'],window=14)
            df['adx']      = adx.adx()
            df['di_plus']  = adx.adx_pos()
            df['di_minus'] = adx.adx_neg()
        except Exception as e:
            logger.debug(f'Indicators: {e}')
        return df

    # ── ICT Helpers ──────────────────────────────────────────

    def has_liquidity_sweep(self, df, ob_idx, direction):
        pre = df.iloc[max(0, ob_idx-30):ob_idx]
        if len(pre) < 6: return False
        highs = pre['high'].values; lows = pre['low'].values
        if direction == 'LONG':
            for j in range(5, len(lows)-1):
                if lows[j] < min(lows[:j]): return True
        else:
            for j in range(5, len(highs)-1):
                if highs[j] > max(highs[:j]): return True
        return False

    def check_bos(self, df, ob_idx, direction):
        if ob_idx < 8: return False, None
        pre = df.iloc[max(0, ob_idx-30):ob_idx]
        end = min(len(df), ob_idx+40)
        if direction == 'LONG':
            ref = pre['high'].max()
            for j in range(ob_idx+1, end):
                if df.iloc[j]['close'] > ref: return True, j
        else:
            ref = pre['low'].min()
            for j in range(ob_idx+1, end):
                if df.iloc[j]['close'] < ref: return True, j
        return False, None

    def in_discount_premium(self, df, ob_idx, direction):
        w = df.iloc[max(0, ob_idx-RANGE_LOOKBACK):ob_idx]
        if len(w) < 15: return True
        hi = w['high'].max(); lo = w['low'].min()
        if hi == lo: return True
        pos = (df.iloc[ob_idx]['close'] - lo) / (hi - lo)
        if direction == 'LONG':  return pos <= DISCOUNT_PCT
        else:                    return pos >= (1.0 - DISCOUNT_PCT)

    # ── OB Detection ─────────────────────────────────────────

    def find_valid_obs(self, df):
        obs = []
        highs  = df['high'].values;  lows   = df['low'].values
        opens  = df['open'].values;  closes = df['close'].values
        atrs   = df['atr'].values

        for i in range(SWING_LOOKBACK+3, len(df)-5):
            atr = atrs[i]
            if pd.isna(atr) or atr == 0: continue

            # Bullish OB
            if closes[i] < opens[i] and i+1 < len(df):
                nc = df.iloc[i+1]
                if nc['close'] - nc['open'] > DISPLACEMENT_MULT * atr:
                    d = 'LONG'
                    sweep   = self.has_liquidity_sweep(df, i, d)
                    bos, bi = self.check_bos(df, i, d)
                    disc    = self.in_discount_premium(df, i, d)
                    if SWEEP_REQUIRED and not sweep: continue
                    if not bos:                      continue
                    if not disc:                     continue
                    obs.append({
                        'type': 'BULL', 'idx': i,
                        'top': highs[i], 'btm': lows[i],
                        'sl_level': closes[i],
                        'bos_idx': bi or i+3,
                        'touched': False, 'mitigated': False,
                        'ts': str(df.iloc[i]['timestamp']),
                        'sweep': sweep, 'bos': bos, 'discount': disc,
                    })

            # Bearish OB
            if closes[i] > opens[i] and i+1 < len(df):
                nc = df.iloc[i+1]
                if nc['open'] - nc['close'] > DISPLACEMENT_MULT * atr:
                    d = 'SHORT'
                    sweep   = self.has_liquidity_sweep(df, i, d)
                    bos, bi = self.check_bos(df, i, d)
                    prem    = self.in_discount_premium(df, i, d)
                    if SWEEP_REQUIRED and not sweep: continue
                    if not bos:                      continue
                    if not prem:                     continue
                    obs.append({
                        'type': 'BEAR', 'idx': i,
                        'top': highs[i], 'btm': lows[i],
                        'sl_level': closes[i],
                        'bos_idx': bi or i+3,
                        'touched': False, 'mitigated': False,
                        'ts': str(df.iloc[i]['timestamp']),
                        'sweep': sweep, 'bos': bos, 'discount': prem,
                    })
        return obs

    # ── Confirmation ─────────────────────────────────────────

    def check_confirm(self, df, idx, direction):
        if idx >= len(df) or idx < 1: return False, []
        row  = df.iloc[idx]
        tags = []; score = 0.0

        # ADX > 20
        adx = row.get('adx', np.nan)
        if not pd.isna(adx) and adx >= ADX_MIN:
            tags.append(f'ADX{adx:.0f}'); score += 1.0

        # EMA alignment (EMA21 vs EMA50)
        e21 = row.get('ema_21', np.nan); e50 = row.get('ema_50', np.nan)
        if not any(pd.isna(x) for x in [e21, e50]):
            if direction == 'LONG'  and e21 > e50: tags.append('EMA_ALIGN'); score += 1.0
            if direction == 'SHORT' and e21 < e50: tags.append('EMA_ALIGN'); score += 1.0
        elif EMA_REQUIRED:
            return False, []

        # RSI < 55
        rsi = row.get('rsi', np.nan)
        if not pd.isna(rsi) and rsi < RSI_THRESH:
            tags.append(f'RSI{rsi:.0f}'); score += 0.5

        # VOL spike >= 1.3x
        vr = row.get('vol_ratio', np.nan)
        if not pd.isna(vr) and vr >= VOL_MIN:
            tags.append(f'VOL_{vr:.1f}x'); score += 0.5

        # DI direction
        dip = row.get('di_plus', np.nan); dim = row.get('di_minus', np.nan)
        if not any(pd.isna(x) for x in [dip, dim]):
            if direction == 'LONG'  and dip > dim: tags.append('DI+'); score += 0.5
            if direction == 'SHORT' and dim > dip: tags.append('DI-'); score += 0.5

        return score >= 2.0, tags

    # ── Scan One Symbol ──────────────────────────────────────

    async def scan_symbol(self, symbol):
        df = await self.fetch_candles(symbol)
        if df is None: return None

        df  = self.add_indicators(df)
        obs = self.find_valid_obs(df)
        if not obs: return None

        highs  = df['high'].values
        lows   = df['low'].values
        closes = df['close'].values
        i      = len(df) - 2   # last closed 4H candle
        now    = df.iloc[i]['timestamp']

        active = [ob for ob in obs
                  if ob['bos_idx'] is not None and ob['bos_idx'] <= i
                  and not ob['touched'] and not ob['mitigated']]

        for ob in active:
            if ob['type'] == 'BULL' and lows[i]  < ob['btm']:
                ob['mitigated'] = True; continue
            if ob['type'] == 'BEAR' and highs[i] > ob['top']:
                ob['mitigated'] = True; continue

            in_bull = ob['type'] == 'BULL' and lows[i]  <= ob['top'] and highs[i] >= ob['btm']
            in_bear = ob['type'] == 'BEAR' and highs[i] >= ob['btm'] and lows[i]  <= ob['top']
            if not (in_bull or in_bear): continue

            direction = 'LONG' if in_bull else 'SHORT'

            # Strict regime alignment (SHORT in BEAR, LONG in BULL)
            if direction == 'LONG'  and self.btc_regime == 'BEAR':
                self.stats['regime_blocked'] += 1; continue
            if direction == 'SHORT' and self.btc_regime == 'BULL':
                self.stats['regime_blocked'] += 1; continue

            if (i - ob['bos_idx']) < MIN_RETEST_BARS: continue

            ck   = (symbol, direction)
            last = self.cooldown.get(ck)
            if last and (now - last).total_seconds() < COOLDOWN_HOURS * 3600: continue

            fk = (symbol, direction, ob['ts'])
            if fk in self.fired: continue

            confirmed, tags = self.check_confirm(df, i, direction)
            if not confirmed: continue

            atr   = df.iloc[i].get('atr', np.nan)
            entry = closes[i]
            if pd.isna(atr) or atr == 0 or pd.isna(entry) or entry == 0: continue

            sl_level = ob.get('sl_level', ob['btm'] if direction == 'LONG' else ob['top'])
            if direction == 'LONG':
                sl  = sl_level * (1 - OB_SL_BUFFER)
                tp1 = entry + atr * ATR_TP1_MULT
                tp2 = entry + atr * ATR_TP2_MULT
                tp3 = entry + atr * ATR_TP3_MULT
            else:
                sl  = sl_level * (1 + OB_SL_BUFFER)
                tp1 = entry - atr * ATR_TP1_MULT
                tp2 = entry - atr * ATR_TP2_MULT
                tp3 = entry - atr * ATR_TP3_MULT

            if direction == 'LONG'  and sl >= entry: continue
            if direction == 'SHORT' and sl <= entry: continue

            risk_pct = abs((sl  - entry) / entry * 100)
            tp1_pct  = abs((tp1 - entry) / entry * 100)
            tp2_pct  = abs((tp2 - entry) / entry * 100)
            tp3_pct  = abs((tp3 - entry) / entry * 100)
            rr       = abs((tp1 - entry) / abs(sl - entry)) if sl != entry else 0

            ob['touched'] = True
            self.cooldown[ck] = now
            self.fired.add(fk)

            tid = f"{symbol.replace('/USDT:USDT','')}_{now.strftime('%Y%m%d%H%M')}"

            ict_tags = []
            if ob['sweep']:   ict_tags.append('Sweep')
            if ob['bos']:     ict_tags.append('BOS')
            if ob['discount']:ict_tags.append('Premium' if direction=='SHORT' else 'Discount')

            if direction == 'LONG':
                close_plan = (
                    f"📋 <b>LONG Close Plan:</b>\n"
                    f"  TP1 → close <b>{int(LONG_TP1_PCT*100)}%</b> → move SL to BE\n"
                    f"  TP2 → close <b>{int(LONG_TP2_PCT*100)}%</b>\n"
                    f"  TP3 → close remaining <b>{int(LONG_TP3_PCT*100)}%</b> (runner)"
                )
            else:
                close_plan = (
                    f"📋 <b>SHORT Close Plan:</b>\n"
                    f"  TP1 → close <b>{int(SHORT_TP1_PCT*100)}%</b> → move SL to BE\n"
                    f"  TP2 → close <b>{int(SHORT_TP2_PCT*100)}%</b>\n"
                    f"  TP3 → close remaining <b>{int(SHORT_TP3_PCT*100)}%</b> (runner)"
                )

            return {
                'trade_id':    tid,
                'symbol':      symbol.replace('/USDT:USDT', ''),
                'full_symbol': symbol,
                'signal':      direction,
                'timestamp':   now,
                'btc_regime':  self.btc_regime,
                'entry':       entry,
                'stop_loss':   sl,
                'tp1': tp1, 'tp1_pct': tp1_pct,
                'tp2': tp2, 'tp2_pct': tp2_pct,
                'tp3': tp3, 'tp3_pct': tp3_pct,
                'risk_pct':    round(risk_pct, 3),
                'rr':          round(rr, 2),
                'confirms':    '|'.join(tags),
                'ict_tags':    ' | '.join(ict_tags),
                'close_plan':  close_plan,
                'tp1_hit': False, 'tp2_hit': False,
                'tp3_hit': False, 'sl_hit':  False,
                'be_active': False,
            }

        return None

    # ── Signal Message ───────────────────────────────────────

    def fmt_signal(self, sig):
        arrow = '📉' if sig['signal'] == 'SHORT' else '📈'
        re    = '🐂' if sig['btc_regime'] == 'BULL' else '🐻'
        m  = f"{'─'*44}\n"
        m += f"{arrow} <b>SWING ICT OB — {sig['signal']}</b>\n"
        m += f"{'─'*44}\n\n"
        m += f"<b>Pair:</b>    #{sig['symbol']}  {re} {sig['btc_regime']}\n"
        m += f"<b>ICT:</b>     {sig['ict_tags']}\n"
        m += f"<b>Confirm:</b> {sig['confirms']}\n"
        m += f"<b>TF:</b>      4H | Max hold: 14 days\n\n"
        m += f"<b>Entry:</b>   <code>${sig['entry']:.6g}</code>\n"
        m += f"<b>TP1:</b>     <code>${sig['tp1']:.6g}</code>  +{sig['tp1_pct']:.2f}%  (avg +9%)\n"
        m += f"<b>TP2:</b>     <code>${sig['tp2']:.6g}</code>  +{sig['tp2_pct']:.2f}%\n"
        m += f"<b>TP3:</b>     <code>${sig['tp3']:.6g}</code>  +{sig['tp3_pct']:.2f}%\n"
        m += f"<b>SL:</b>      <code>${sig['stop_loss']:.6g}</code>  -{sig['risk_pct']:.2f}%\n"
        m += f"<b>RR (TP1):</b> {sig['rr']:.2f}:1\n\n"
        m += f"{sig['close_plan']}\n\n"
        m += f"<i>ID: {sig['trade_id']}</i>\n"
        m += f"<i>{sig['timestamp'].strftime('%d %b %H:%M UTC')} | Swing v1.0</i>"
        return m

    # ── Telegram ─────────────────────────────────────────────

    async def send_msg(self, text):
        try:
            await self.bot.send_message(
                chat_id=self.chat_id, text=text, parse_mode=ParseMode.HTML)
        except Exception as e:
            logger.error(f'Telegram error: {e}')

    # ── TP/SL Alerts ─────────────────────────────────────────

    async def _tp1_alert(self, trade, price):
        gain      = abs((price - trade['entry']) / trade['entry'] * 100)
        close_pct = int(SHORT_TP1_PCT*100) if trade['signal']=='SHORT' else int(LONG_TP1_PCT*100)
        m  = f"✅ <b>TP1 HIT — SWING</b>\n\n<b>{trade['symbol']}</b>  {trade['signal']}\n"
        m += f"Entry: <code>${trade['entry']:.6g}</code>\n"
        m += f"TP1:   <code>${price:.6g}</code>  <b>+{gain:.2f}%</b>\n\n"
        m += f"Close <b>{close_pct}%</b> of position\n"
        m += f"Move SL to breakeven: <code>${trade['entry']:.6g}</code>\n"
        m += f"Let rest run to TP2: <code>${trade['tp2']:.6g}</code>  (+{trade['tp2_pct']:.2f}%)\n"
        m += f"\n<i>{trade['trade_id']}</i>"
        await self.send_msg(m)
        trade['tp1_hit'] = True; trade['be_active'] = True
        self.stats['tp1_hits'] += 1

    async def _tp2_alert(self, trade, price):
        gain      = abs((price - trade['entry']) / trade['entry'] * 100)
        close_pct = int(SHORT_TP2_PCT*100) if trade['signal']=='SHORT' else int(LONG_TP2_PCT*100)
        runner_pct= int(SHORT_TP3_PCT*100) if trade['signal']=='SHORT' else int(LONG_TP3_PCT*100)
        m  = f"💰 <b>TP2 HIT — SWING</b>\n\n<b>{trade['symbol']}</b>  {trade['signal']}\n"
        m += f"TP2: <code>${price:.6g}</code>  <b>+{gain:.2f}%</b>\n\n"
        m += f"Close <b>{close_pct}%</b> more\n"
        m += f"Runner ({runner_pct}%) open → TP3: <code>${trade['tp3']:.6g}</code>  (+{trade['tp3_pct']:.2f}%)\n"
        m += f"\n<i>{trade['trade_id']}</i>"
        await self.send_msg(m)
        trade['tp2_hit'] = True
        self.stats['tp2_hits'] += 1

    async def _tp3_alert(self, trade, price):
        gain = abs((price - trade['entry']) / trade['entry'] * 100)
        m  = f"🚀 <b>TP3 HIT — FULL SWING RUNNER!</b>\n\n<b>{trade['symbol']}</b>  {trade['signal']}\n"
        m += f"Entry: <code>${trade['entry']:.6g}</code>\n"
        m += f"TP3:   <code>${price:.6g}</code>  <b>+{gain:.2f}%</b>\n\n"
        m += f"Close remaining position — swing complete\n"
        m += f"\n<i>{trade['trade_id']}</i>"
        await self.send_msg(m)
        trade['tp3_hit'] = True
        self.stats['tp3_hits'] += 1

    async def _sl_alert(self, trade, price, be_save=False):
        if be_save:
            m  = f"🔒 <b>BREAKEVEN CLOSE</b>\n\n<b>{trade['symbol']}</b>  {trade['signal']}\n"
            m += f"TP1 was hit — closed remainder at entry — no loss\n"
            m += f"\n<i>{trade['trade_id']}</i>"
            self.stats['be_saves'] += 1
        else:
            loss = abs((price - trade['entry']) / trade['entry'] * 100)
            m  = f"⛔ <b>STOP LOSS</b>\n\n<b>{trade['symbol']}</b>  {trade['signal']}\n"
            m += f"Entry: <code>${trade['entry']:.6g}</code>\n"
            m += f"SL:    <code>${price:.6g}</code>  <b>-{loss:.2f}%</b>\n"
            m += f"\n<i>{trade['trade_id']}</i>"
            self.stats['sl_hits'] += 1
        await self.send_msg(m)

    async def _timeout_alert(self, trade):
        m  = f"⏰ <b>SWING TIMEOUT — 14 DAYS</b>\n\n<b>{trade['symbol']}</b>  {trade['signal']}\n"
        m += f"Entry: <code>${trade['entry']:.6g}</code>\n"
        m += f"No target hit in 14 days — close position manually\n"
        m += f"\n<i>{trade['trade_id']}</i>"
        await self.send_msg(m)
        self.stats['timeouts'] += 1

    # ── Trade Tracker (polls every 5 min for 4H swing) ───────

    async def track_trades(self):
        logger.info('Swing trade tracker started')
        while True:
            try:
                if not self.active_trades:
                    await asyncio.sleep(300); continue

                done = []
                for tid, t in list(self.active_trades.items()):
                    try:
                        age_hours = (datetime.now(timezone.utc) - t['timestamp']).total_seconds() / 3600
                        if age_hours > MAX_TRADE_BARS * 4:  # 14 days
                            await self._timeout_alert(t)
                            done.append(tid); continue

                        ticker    = await self.exchange.fetch_ticker(t['full_symbol'])
                        price     = ticker['last']
                        direction = t['signal']
                        active_sl = t['entry'] if t['be_active'] else t['stop_loss']

                        if direction == 'LONG':
                            if not t['tp3_hit'] and t['tp2_hit'] and price >= t['tp3']:
                                await self._tp3_alert(t, price); done.append(tid); continue
                            if not t['tp2_hit'] and t['tp1_hit'] and price >= t['tp2']:
                                await self._tp2_alert(t, price)
                            if not t['tp1_hit'] and price >= t['tp1']:
                                await self._tp1_alert(t, price)
                            if price <= active_sl:
                                await self._sl_alert(t, price, t['be_active'] and active_sl==t['entry'])
                                done.append(tid)
                        else:
                            if not t['tp3_hit'] and t['tp2_hit'] and price <= t['tp3']:
                                await self._tp3_alert(t, price); done.append(tid); continue
                            if not t['tp2_hit'] and t['tp1_hit'] and price <= t['tp2']:
                                await self._tp2_alert(t, price)
                            if not t['tp1_hit'] and price <= t['tp1']:
                                await self._tp1_alert(t, price)
                            if price >= active_sl:
                                await self._sl_alert(t, price, t['be_active'] and active_sl==t['entry'])
                                done.append(tid)

                        await asyncio.sleep(0.2)
                    except Exception as e:
                        logger.error(f'Track {tid}: {e}')

                for tid in done:
                    self.active_trades.pop(tid, None)

                await asyncio.sleep(300)   # check every 5 min

            except Exception as e:
                logger.error(f'Tracker: {e}'); await asyncio.sleep(60)

    # ── Main Scan ─────────────────────────────────────────────

    async def scan_all(self):
        if self.is_scanning: return []
        self.is_scanning = True
        signals = []

        await self.update_btc_regime()
        pairs = await self.get_pairs()
        logger.info(f'Swing scan | {len(pairs)} pairs | BTC: {self.btc_regime} | Scanning {"SHORTs" if self.btc_regime=="BEAR" else "LONGs"}')

        for idx, pair in enumerate(pairs):
            try:
                sig = await self.scan_symbol(pair)
                if sig:
                    self.active_trades[sig['trade_id']] = sig
                    self.signal_history.append(sig)
                    signals.append(sig)
                    self.stats['total_signals'] += 1
                    if sig['signal'] == 'LONG': self.stats['long_signals'] += 1
                    else:                       self.stats['short_signals'] += 1
                    await self.send_msg(self.fmt_signal(sig))
                    logger.info(f'SIGNAL: {sig["symbol"]} {sig["signal"]} | {sig["confirms"]} | TP1 +{sig["tp1_pct"]:.1f}%')
                    await asyncio.sleep(1)
                await asyncio.sleep(0.15)
                if (idx+1) % 75 == 0:
                    await self.update_btc_regime()
                    logger.info(f'{idx+1}/{len(pairs)} scanned | {len(signals)} signals')
            except Exception as e:
                logger.error(f'Scan {pair}: {e}')

        self.stats['last_scan'] = datetime.now()
        logger.info(f'Scan done | {len(signals)} swing signals | tracking: {len(self.active_trades)}')
        self.is_scanning = False
        return signals

    # ── Daily Report ──────────────────────────────────────────

    async def send_daily_report(self):
        while True:
            await asyncio.sleep(24 * 3600)
            try:
                s   = self.stats
                tp1 = s['tp1_hits']; tp2 = s['tp2_hits']; tp3 = s['tp3_hits']
                sl  = s['sl_hits'];  be  = s['be_saves']
                tot = tp1 + sl
                wr  = round(tp1 / tot * 100, 1) if tot > 0 else 0
                hrs = round((datetime.now() - s['session_start']).total_seconds() / 3600, 1)
                re  = '🐂' if self.btc_regime == 'BULL' else '🐻'
                bar = '▰' * int(wr/10) + '▱' * (10 - int(wr/10))
                if wr >= 60:   status = '✅ On track (target 59%+)'
                elif wr >= 50: status = '⚠️ Watch closely'
                else:          status = '🚨 Below target'

                m  = f"{'─'*42}\n📅 <b>24H REPORT — SWING ICT OB v1.0</b>\n{'─'*42}\n\n"
                m += f"{re} BTC: <b>{self.btc_regime}</b>  |  Session: {hrs}h\n"
                m += f"Price: ${self.btc_price:,.0f}  EMA50: ${self.btc_ema50:,.0f}\n\n"
                m += f"<b>Signals (all-time):</b> {s['total_signals']}\n"
                m += f"  📈 Long: {s['long_signals']}  |  📉 Short: {s['short_signals']}\n"
                m += f"  Regime blocked: {s['regime_blocked']}\n\n"
                m += f"<b>TP Performance:</b>\n"
                m += f"  ✅ TP1: <b>{tp1}</b>\n"
                m += f"  💰 TP2: <b>{tp2}</b>  ({round(tp2/max(tp1,1)*100)}% of TP1s)\n"
                m += f"  🚀 TP3: <b>{tp3}</b>  ({round(tp3/max(tp1,1)*100)}% of TP1s)\n"
                m += f"  🔒 BE saves: <b>{be}</b>\n"
                m += f"  ❌ SL: <b>{sl}</b>\n\n"
                m += f"<b>TP1 Win Rate: {wr}%</b>\n{bar}\n\n"
                m += f"{status}\n"
                m += f"Active trades: {len(self.active_trades)}\n"
                m += f"<i>{datetime.now().strftime('%d %b %Y %H:%M UTC')}</i>"
                await self.send_msg(m)
            except Exception as e:
                logger.error(f'Daily report: {e}')

    async def run(self):
        logger.info('Swing ICT OB Scanner v1.0 started')
        asyncio.create_task(self.track_trades())
        asyncio.create_task(self.send_daily_report())
        while True:
            try:
                await self.scan_all()
                await asyncio.sleep(SCAN_INTERVAL_MIN * 60)
            except Exception as e:
                logger.error(f'Run error: {e}'); await asyncio.sleep(60)

    async def close(self):
        await self.exchange.close()


# ═══════════════════════════════════════════════════════════════
# TELEGRAM COMMANDS
# ═══════════════════════════════════════════════════════════════

class BotCommands:

    def __init__(self, scanner: SwingScanner):
        self.s = scanner

    async def cmd_start(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        re = '🐂' if self.s.btc_regime == 'BULL' else '🐻'
        m  = "📊 <b>Swing ICT OB Scanner v1.0</b>\n\n"
        m += f"Current regime: {re} <b>{self.s.btc_regime}</b>\n"
        m += f"Currently scanning: {'📈 LONGs' if self.s.btc_regime=='BULL' else '📉 SHORTs'}\n\n"
        m += "<b>Config (v4.0 validated):</b>\n"
        m += f"  4H timeframe | Disp={DISPLACEMENT_MULT}x ATR\n"
        m += f"  Sweep + BOS + Premium/Discount zone\n"
        m += f"  ADX>{ADX_MIN} | RSI<{RSI_THRESH} | VOL>{VOL_MIN}x\n"
        m += f"  TP1={ATR_TP1_MULT}x | TP2={ATR_TP2_MULT}x | TP3={ATR_TP3_MULT}x ATR\n"
        m += f"  Max hold: 14 days | Cooldown: 24h\n\n"
        m += "<b>Backtest (180d, 300 pairs):</b>\n"
        m += "  59.3% TP1 | avg +9.32% per TP1 | 2.23 RR\n\n"
        m += "/scan /stats /trades /regime /help"
        await update.message.reply_text(m, parse_mode=ParseMode.HTML)

    async def cmd_scan(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        if self.s.is_scanning:
            await update.message.reply_text('Scan already running!'); return
        await update.message.reply_text('🔍 Running swing scan...')
        asyncio.create_task(self.s.scan_all())

    async def cmd_stats(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        s   = self.s.stats
        tp1 = s['tp1_hits']; sl = s['sl_hits']
        wr  = round(tp1 / max(tp1+sl, 1) * 100, 1)
        hrs = round((datetime.now() - s['session_start']).total_seconds() / 3600, 1)
        re  = '🐂' if self.s.btc_regime == 'BULL' else '🐻'

        m  = f"📊 <b>STATS — Swing ICT OB v1.0</b>\n\nSession: {hrs}h\n"
        m += f"BTC: {re} <b>{self.s.btc_regime}</b>"
        if self.s.btc_price:
            m += f"  (${self.s.btc_price:,.0f})"
        m += f"\n\n<b>Signals:</b> {s['total_signals']}\n"
        m += f"  📈 Long: {s['long_signals']}  |  📉 Short: {s['short_signals']}\n"
        m += f"  Regime blocked: {s['regime_blocked']}\n\n"
        m += f"<b>Performance:</b>\n"
        m += f"  ✅ TP1: {tp1}  ({wr}% WR)\n"
        m += f"  💰 TP2: {s['tp2_hits']}  |  🚀 TP3: {s['tp3_hits']}\n"
        m += f"  🔒 BE: {s['be_saves']}  |  ❌ SL: {sl}\n\n"
        m += f"Active trades: {len(self.s.active_trades)}\n"
        if s['last_scan']:
            m += f"Last scan: {s['last_scan'].strftime('%H:%M UTC')}"
        await update.message.reply_text(m, parse_mode=ParseMode.HTML)

    async def cmd_trades(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        trades = self.s.active_trades
        if not trades:
            await update.message.reply_text('No active swing trades.'); return
        m = f"📡 <b>ACTIVE SWING TRADES ({len(trades)})</b>\n\n"
        for tid, t in list(trades.items())[:8]:
            age_d = round((datetime.now(timezone.utc) - t['timestamp']).total_seconds() / 86400, 1)
            be_s  = ' 🔒BE' if t['be_active'] else ''
            m += f"<b>{t['symbol']}</b>  {t['signal']}{be_s}  |  {age_d}d ago\n"
            m += f"  Entry: <code>${t['entry']:.6g}</code>  TP1: <code>${t['tp1']:.6g}</code>\n"
            m += f"  TP1:{'✅' if t['tp1_hit'] else '⏳'} TP2:{'✅' if t['tp2_hit'] else '⏳'} TP3:{'✅' if t['tp3_hit'] else '⏳'}\n"
            m += f"  {t['confirms']}\n\n"
        await update.message.reply_text(m, parse_mode=ParseMode.HTML)

    async def cmd_regime(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        r  = self.s.btc_regime
        re = '🐂' if r == 'BULL' else '🐻'
        m  = f"{re} <b>BTC 4H Regime: {r}</b>\n\n"
        if self.s.btc_price and self.s.btc_ema50:
            m += f"Price: ${self.s.btc_price:,.2f}\n"
            m += f"EMA50 (4H): ${self.s.btc_ema50:,.2f}\n\n"
        m += ("📈 Scanning LONGs | SHORTs off\n" if r=='BULL'
              else "📉 Scanning SHORTs | LONGs off\n")
        m += f"\n<i>Backtest: 59.3% TP1 | avg +9.32% per trade</i>"
        await update.message.reply_text(m, parse_mode=ParseMode.HTML)

    async def cmd_help(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        m  = "📚 <b>Swing ICT OB Scanner v1.0</b>\n\n"
        m += "<b>Strategy:</b>\n"
        m += "  4H OB retest at premium/discount zone\n"
        m += "  Confirmed: Sweep + BOS + ADX/EMA/RSI/VOL\n"
        m += "  SHORT in BEAR | LONG in BULL (regime-aligned)\n\n"
        m += "<b>Targets:</b>\n"
        m += f"  TP1 = {ATR_TP1_MULT}x ATR  (~9% avg move)\n"
        m += f"  TP2 = {ATR_TP2_MULT}x ATR  (~18% avg)\n"
        m += f"  TP3 = {ATR_TP3_MULT}x ATR  (~36% avg)\n\n"
        m += "<b>Close plan:</b>\n"
        m += f"  SHORT: {int(SHORT_TP1_PCT*100)}/{int(SHORT_TP2_PCT*100)}/{int(SHORT_TP3_PCT*100)} at TP1/TP2/TP3\n"
        m += f"  LONG:  {int(LONG_TP1_PCT*100)}/{int(LONG_TP2_PCT*100)}/{int(LONG_TP3_PCT*100)} at TP1/TP2/TP3\n"
        m += "  After TP1: always move SL to BE\n\n"
        m += "/scan /stats /trades /regime /help"
        await update.message.reply_text(m, parse_mode=ParseMode.HTML)


# ═══════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════

async def main():
    scanner = SwingScanner()
    app     = Application.builder().token(TELEGRAM_TOKEN).build()
    cmds    = BotCommands(scanner)

    for cmd, fn in [
        ('start',  cmds.cmd_start),
        ('scan',   cmds.cmd_scan),
        ('stats',  cmds.cmd_stats),
        ('trades', cmds.cmd_trades),
        ('regime', cmds.cmd_regime),
        ('help',   cmds.cmd_help),
    ]:
        app.add_handler(CommandHandler(cmd, fn))

    await app.initialize()
    await app.start()

    logger.info('=== SWING ICT OB LIVE SCANNER v1.0 ===')
    logger.info(f'4H | 59.3% TP1 | avg +9.32% | 2.23 RR | SHORT in BEAR / LONG in BULL')
    logger.info(f'Scan every {SCAN_INTERVAL_MIN}min | Max hold 14d | {MAX_PAIRS} pairs')

    try:
        await asyncio.gather(
            scanner.run(),
            app.updater.start_polling(),
        )
    except (KeyboardInterrupt, SystemExit):
        logger.info('Shutting down...')
    finally:
        await scanner.close()
        await app.stop()
        await app.shutdown()


if __name__ == '__main__':
    asyncio.run(main())
