"""
ICT OB LIVE SCANNER v10.6-M — BEST CONFIG
==========================================
Railway deployment — credentials via environment variables.

FINAL BACKTEST RESULTS (7 rounds of optimization):
  Config  : v10.6-M — Full baseline OB + No ADX + RSI<35
  WR      : 88.9% TP1
  RR      : 2.21:1
  Avg PnL : +2.209%
  Max DD  : -0.56%
  Signals : ~0.21/day (BEAR market — will increase in BULL)

KEY CHANGES from v10.3:
  ADX filter DISABLED — was killing 40%+ of valid signals
  RSI threshold 35    — widened from 30 (+62% more signals)
  Everything else unchanged (sweep/FVG/BOS/discount all ON)

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

TELEGRAM_TOKEN   = os.environ['TELEGRAM_TOKEN']
TELEGRAM_CHAT_ID = os.environ['TELEGRAM_CHAT_ID']
BINANCE_API_KEY  = os.environ.get('BINANCE_API_KEY', None)
BINANCE_SECRET   = os.environ.get('BINANCE_SECRET',  None)


# ═══════════════════════════════════════════════════════════════
# ICT v10.6-M SETTINGS — FINAL OPTIMIZED
# ═══════════════════════════════════════════════════════════════

REGIME_MODE       = 'HARD'
DISPLACEMENT_MULT = 2.0
SWING_LOOKBACK    = 10

SWEEP_REQUIRED    = True
FVG_REQUIRED      = True
BOS_REQUIRED      = True
DISCOUNT_REQUIRED = True

RANGE_LOOKBACK  = 50
VOL_SPIKE_MIN   = 2.5       # hard floor — never lower
VOL_SPIKE_MAX   = 3.5       # trap cap — never raise
ADX_MIN         = None      # DISABLED — was the bottleneck (v10.6 finding)
RSI_THRESH      = 35        # widened from 30
MIN_RETEST_BARS = 3
OB_SL_BUFFER    = 0.001

ATR_TP1_MULT = 1.0
ATR_TP2_MULT = 1.8
ATR_TP3_MULT = 3.0

LONG_TP1_PCT  = 0.33
LONG_TP2_PCT  = 0.33
LONG_TP3_PCT  = 0.34
SHORT_TP1_PCT = 0.70
SHORT_TP2_PCT = 0.30


# ═══════════════════════════════════════════════════════════════
# SCANNER SETTINGS
# ═══════════════════════════════════════════════════════════════

TIMEFRAME         = '1h'
CANDLE_LIMIT      = 300
SCAN_INTERVAL_MIN = 60
MAX_PAIRS         = 300
MIN_VOLUME_USDT   = 1_000_000
COOLDOWN_HOURS    = 6
MAX_TRADE_HOURS   = 24


# ═══════════════════════════════════════════════════════════════
# SCANNER
# ═══════════════════════════════════════════════════════════════

class ICTScanner:

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
        self.btc_ema        = None
        self.signal_history = deque(maxlen=300)
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
            'pairs_scanned':  0,
            'session_start':  datetime.now(),
        }

    # ── BTC Regime ───────────────────────────────────────────

    async def update_btc_regime(self):
        try:
            ohlcv = await self.exchange.fetch_ohlcv('BTC/USDT:USDT', '4h', limit=50)
            df = pd.DataFrame(ohlcv, columns=['timestamp','open','high','low','close','volume'])
            df['ema21']     = ta.trend.EMAIndicator(df['close'], window=21).ema_indicator()
            last            = df.iloc[-1]
            prev            = self.btc_regime
            self.btc_price  = last['close']
            self.btc_ema    = last['ema21']
            self.btc_regime = 'BULL' if self.btc_price > self.btc_ema else 'BEAR'
            if prev != self.btc_regime:
                logger.info(f'Regime flip: {prev} -> {self.btc_regime}')
            logger.info(f'BTC: {self.btc_regime} (${self.btc_price:,.0f})')
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
            return df if len(df) >= 100 else None
        except:
            return None

    def add_indicators(self, df):
        if len(df) < 30: return df
        try:
            df['ema_9']    = ta.trend.EMAIndicator(df['close'], window=9).ema_indicator()
            df['ema_21']   = ta.trend.EMAIndicator(df['close'], window=21).ema_indicator()
            df['atr']      = ta.volatility.AverageTrueRange(df['high'],df['low'],df['close']).average_true_range()
            df['vol_sma']  = df['volume'].rolling(20).mean()
            df['vol_ratio']= df['volume'] / df['vol_sma'].replace(0, np.nan)
            df['rsi']      = ta.momentum.RSIIndicator(df['close'], window=14).rsi()
        except Exception as e:
            logger.debug(f'Indicators: {e}')
        return df

    # ── ICT Helpers ──────────────────────────────────────────

    def has_liquidity_sweep(self, df, ob_idx, direction):
        pre = df.iloc[max(0, ob_idx-30):ob_idx]
        if len(pre) < 6: return False
        highs = pre['high'].values; lows = pre['low'].values
        if direction == 'LONG':
            for j in range(4, len(lows)-1):
                if lows[j] < min(lows[:j]): return True
        else:
            for j in range(4, len(highs)-1):
                if highs[j] > max(highs[:j]): return True
        return False

    def check_fvg(self, df, ob_idx, direction):
        if ob_idx + 2 >= len(df): return False
        c0 = df.iloc[ob_idx]; c2 = df.iloc[ob_idx+2]
        if direction == 'LONG'  and c2['low']  > c0['high']: return True
        if direction == 'SHORT' and c2['high'] < c0['low']:  return True
        return False

    def check_bos(self, df, ob_idx, direction):
        if ob_idx < 5: return False, None
        pre = df.iloc[max(0, ob_idx-20):ob_idx]
        end = min(len(df), ob_idx+25)
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
        if len(w) < 10: return True
        hi = w['high'].max(); lo = w['low'].min()
        if hi == lo: return True
        pos = (df.iloc[ob_idx]['close'] - lo) / (hi - lo)
        return pos <= 0.5 if direction == 'LONG' else pos >= 0.5

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
                    fvg     = self.check_fvg(df, i, d)
                    bos, bi = self.check_bos(df, i, d)
                    disc    = self.in_discount_premium(df, i, d)
                    if SWEEP_REQUIRED    and not sweep: continue
                    if FVG_REQUIRED      and not fvg:   continue
                    if BOS_REQUIRED      and not bos:   continue
                    if DISCOUNT_REQUIRED and not disc:  continue
                    obs.append({
                        'type': 'BULL', 'idx': i,
                        'top': highs[i], 'btm': lows[i],
                        'sl_level': closes[i],
                        'bos_idx': bi or i+5,
                        'touched': False, 'mitigated': False,
                        'ts': str(df.iloc[i]['timestamp']),
                        'sweep': sweep, 'fvg': fvg, 'bos': bos, 'discount': disc,
                    })

            # Bearish OB
            if closes[i] > opens[i] and i+1 < len(df):
                nc = df.iloc[i+1]
                if nc['open'] - nc['close'] > DISPLACEMENT_MULT * atr:
                    d = 'SHORT'
                    sweep   = self.has_liquidity_sweep(df, i, d)
                    fvg     = self.check_fvg(df, i, d)
                    bos, bi = self.check_bos(df, i, d)
                    prem    = self.in_discount_premium(df, i, d)
                    if SWEEP_REQUIRED    and not sweep: continue
                    if FVG_REQUIRED      and not fvg:   continue
                    if BOS_REQUIRED      and not bos:   continue
                    if DISCOUNT_REQUIRED and not prem:  continue
                    obs.append({
                        'type': 'BEAR', 'idx': i,
                        'top': highs[i], 'btm': lows[i],
                        'sl_level': closes[i],
                        'bos_idx': bi or i+5,
                        'touched': False, 'mitigated': False,
                        'ts': str(df.iloc[i]['timestamp']),
                        'sweep': sweep, 'fvg': fvg, 'bos': bos, 'discount': prem,
                    })
        return obs

    # ── Retest Confirmation (v10.6-M) ────────────────────────

    def check_retest_confirm(self, df, idx, direction):
        if idx >= len(df) or idx < 1: return False, []
        row  = df.iloc[idx]
        prev = df.iloc[idx-1]
        tags = []

        # VOL spike 2.5x-3.5x
        vr = row.get('vol_ratio', np.nan)
        if not pd.isna(vr) and VOL_SPIKE_MIN <= vr <= VOL_SPIKE_MAX:
            tags.append(f'VOL_{vr:.1f}x')

        # RSI < 35
        rsi = row.get('rsi', np.nan)
        if not pd.isna(rsi) and rsi < RSI_THRESH:
            tags.append(f'RSI<{RSI_THRESH}')

        # EMA_BEAR for SHORTs only
        if direction == 'SHORT':
            e9  = row.get('ema_9',  np.nan); e21 = row.get('ema_21', np.nan)
            p9  = prev.get('ema_9', np.nan); p21 = prev.get('ema_21', np.nan)
            if not any(pd.isna(x) for x in [e9, e21, p9, p21]):
                if p9 >= p21 and e9 < e21:
                    tags.append('EMA_BEAR')

        return len(tags) > 0, tags

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
        i      = len(df) - 2
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

            if REGIME_MODE == 'HARD':
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

            confirmed, tags = self.check_retest_confirm(df, i, direction)
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

            tid = f"{symbol.replace('/USDT:USDT','')}_{now.strftime('%Y%m%d%H%M%S')}"

            ict_tags = []
            if ob['sweep']:    ict_tags.append('Sweep')
            if ob['fvg']:      ict_tags.append('FVG')
            if ob['bos']:      ict_tags.append('BOS')
            if ob['discount']: ict_tags.append('Discount' if direction == 'LONG' else 'Premium')

            if direction == 'LONG':
                close_plan = (
                    f"📋 <b>Close plan (LONG):</b>\n"
                    f"  • TP1 → close <b>{int(LONG_TP1_PCT*100)}%</b> → move SL to BE\n"
                    f"  • TP2 → close <b>{int(LONG_TP2_PCT*100)}%</b>\n"
                    f"  • TP3 → close remaining <b>{int(LONG_TP3_PCT*100)}%</b> (runner)"
                )
            else:
                close_plan = (
                    f"📋 <b>Close plan (SHORT):</b>\n"
                    f"  • TP1 → close <b>{int(SHORT_TP1_PCT*100)}%</b> → move SL to BE\n"
                    f"  • TP2 → close remaining <b>{int(SHORT_TP2_PCT*100)}%</b>"
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
        arrow = '🚀' if sig['signal'] == 'LONG' else '🔻'
        re    = '🐂' if sig['btc_regime'] == 'BULL' else '🐻'
        m  = f"{'─'*42}\n"
        m += f"{arrow} <b>ICT OB v10.6-M — {sig['signal']}</b>\n"
        m += f"{'─'*42}\n\n"
        m += f"<b>Pair:</b>    #{sig['symbol']}  {re} {sig['btc_regime']}\n"
        m += f"<b>ICT:</b>     {sig['ict_tags']}\n"
        m += f"<b>Confirm:</b> {sig['confirms']}\n\n"
        m += f"<b>Entry:</b>   <code>${sig['entry']:.6g}</code>\n"
        m += f"<b>TP1:</b>     <code>${sig['tp1']:.6g}</code>  +{sig['tp1_pct']:.2f}%\n"
        m += f"<b>TP2:</b>     <code>${sig['tp2']:.6g}</code>  +{sig['tp2_pct']:.2f}%\n"
        m += f"<b>TP3:</b>     <code>${sig['tp3']:.6g}</code>  +{sig['tp3_pct']:.2f}%\n"
        m += f"<b>SL:</b>      <code>${sig['stop_loss']:.6g}</code>  -{sig['risk_pct']:.2f}%\n"
        m += f"<b>RR (TP1):</b> {sig['rr']:.2f}:1\n\n"
        m += f"{sig['close_plan']}\n\n"
        m += f"<i>ID: {sig['trade_id']}</i>\n"
        m += f"<i>{sig['timestamp'].strftime('%H:%M UTC')} | v10.6-M</i>"
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
        close_pct = int(LONG_TP1_PCT*100) if trade['signal']=='LONG' else int(SHORT_TP1_PCT*100)
        m  = f"✅ <b>TP1 HIT</b>\n\n<b>{trade['symbol']}</b>  {trade['signal']}\n"
        m += f"Entry: <code>${trade['entry']:.6g}</code>\n"
        m += f"TP1:   <code>${price:.6g}</code>  <b>+{gain:.2f}%</b>\n\n"
        m += f"Close <b>{close_pct}%</b>  |  Move SL to BE: <code>${trade['entry']:.6g}</code>\n"
        m += f"Next: <code>${trade['tp2']:.6g}</code>  (+{trade['tp2_pct']:.2f}%)\n"
        m += f"\n<i>{trade['trade_id']}</i>"
        await self.send_msg(m)
        trade['tp1_hit'] = True; trade['be_active'] = True
        self.stats['tp1_hits'] += 1

    async def _tp2_alert(self, trade, price):
        gain       = abs((price - trade['entry']) / trade['entry'] * 100)
        has_runner = (trade['signal'] == 'LONG')
        close_pct  = int(LONG_TP2_PCT*100) if has_runner else int(SHORT_TP2_PCT*100)
        m  = f"💰 <b>TP2 HIT</b>\n\n<b>{trade['symbol']}</b>  {trade['signal']}\n"
        m += f"TP2: <code>${price:.6g}</code>  <b>+{gain:.2f}%</b>\n\n"
        m += f"Close <b>{close_pct}%</b>\n"
        if has_runner:
            m += f"Runner open → TP3: <code>${trade['tp3']:.6g}</code>\n"
        else:
            m += f"<b>Trade complete</b>\n"
        m += f"\n<i>{trade['trade_id']}</i>"
        await self.send_msg(m)
        trade['tp2_hit'] = True
        self.stats['tp2_hits'] += 1

    async def _tp3_alert(self, trade, price):
        gain = abs((price - trade['entry']) / trade['entry'] * 100)
        m  = f"🔥 <b>TP3 HIT — FULL RUNNER!</b>\n\n<b>{trade['symbol']}</b>  {trade['signal']}\n"
        m += f"TP3: <code>${price:.6g}</code>  <b>+{gain:.2f}%</b>\n"
        m += f"Close remaining {int(LONG_TP3_PCT*100)}% — trade complete\n"
        m += f"\n<i>{trade['trade_id']}</i>"
        await self.send_msg(m)
        trade['tp3_hit'] = True
        self.stats['tp3_hits'] += 1

    async def _sl_alert(self, trade, price, be_save=False):
        if be_save:
            m  = f"🔒 <b>BREAKEVEN CLOSE</b>\n\n<b>{trade['symbol']}</b>  {trade['signal']}\n"
            m += f"TP1 hit — closed remainder at entry — no loss\n"
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

    # ── Trade Tracker ─────────────────────────────────────────

    async def track_trades(self):
        logger.info('Trade tracker started')
        while True:
            try:
                if not self.active_trades:
                    await asyncio.sleep(30); continue
                done = []
                for tid, t in list(self.active_trades.items()):
                    try:
                        age = (datetime.now(timezone.utc) - t['timestamp']).total_seconds()
                        if age > MAX_TRADE_HOURS * 3600:
                            self.stats['timeouts'] += 1; done.append(tid); continue
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
                            if not t['tp2_hit'] and t['tp1_hit'] and price <= t['tp2']:
                                await self._tp2_alert(t, price); done.append(tid); continue
                            if not t['tp1_hit'] and price <= t['tp1']:
                                await self._tp1_alert(t, price)
                            if price >= active_sl:
                                await self._sl_alert(t, price, t['be_active'] and active_sl==t['entry'])
                                done.append(tid)
                        await asyncio.sleep(0.1)
                    except Exception as e:
                        logger.error(f'Track {tid}: {e}')
                for tid in done:
                    self.active_trades.pop(tid, None)
                await asyncio.sleep(30)
            except Exception as e:
                logger.error(f'Tracker: {e}'); await asyncio.sleep(60)

    # ── Main Scan ─────────────────────────────────────────────

    async def scan_all(self):
        if self.is_scanning: return []
        self.is_scanning = True
        signals = []
        await self.update_btc_regime()
        pairs = await self.get_pairs()
        logger.info(f'Scan started | {len(pairs)} pairs | BTC: {self.btc_regime}')
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
                    logger.info(f'SIGNAL: {sig["symbol"]} {sig["signal"]} | {sig["confirms"]}')
                    await asyncio.sleep(1)
                await asyncio.sleep(0.1)
                if (idx+1) % 50 == 0:
                    await self.update_btc_regime()
                    logger.info(f'{idx+1}/{len(pairs)} scanned | {len(signals)} signals')
            except Exception as e:
                logger.error(f'Scan {pair}: {e}')
        self.stats['last_scan']     = datetime.now()
        self.stats['pairs_scanned'] = len(pairs)
        logger.info(f'Scan done | {len(signals)} signals | tracking: {len(self.active_trades)}')
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
                cutoff   = datetime.now() - timedelta(hours=24)
                day_sigs = [t for t in self.signal_history
                            if t['timestamp'].replace(tzinfo=None) >= cutoff]
                re  = '🐂' if self.btc_regime == 'BULL' else '🐻'
                bar = '▰' * int(wr/10) + '▱' * (10 - int(wr/10))
                if wr >= 88:   status = '🔥 Excellent'
                elif wr >= 80: status = '✅ Good'
                elif wr >= 70: status = '⚠️ Watch closely'
                else:          status = '🚨 Below target'
                m  = f"{'─'*40}\n📅 <b>24H REPORT — ICT OB v10.6-M</b>\n{'─'*40}\n\n"
                m += f"{re} BTC: <b>{self.btc_regime}</b>  |  Session: {hrs}h\n\n"
                m += f"<b>Today:</b> {len(day_sigs)} signals\n\n"
                m += f"<b>All-time:</b>\n"
                m += f"  ✅ TP1: <b>{tp1}</b>\n  💰 TP2: <b>{tp2}</b>\n  🔥 TP3: <b>{tp3}</b>\n"
                m += f"  🔒 BE: <b>{be}</b>  |  ❌ SL: <b>{sl}</b>\n\n"
                m += f"<b>Win Rate: {wr}%</b>  {bar}\n{status}\n"
                m += f"Tracking: {len(self.active_trades)} trades\n"
                m += f"<i>{datetime.now().strftime('%d %b %Y %H:%M UTC')}</i>"
                await self.send_msg(m)
            except Exception as e:
                logger.error(f'Daily report: {e}')

    async def run(self):
        logger.info('ICT OB v10.6-M started')
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

    def __init__(self, scanner: ICTScanner):
        self.s = scanner

    async def cmd_start(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        m  = "🚀 <b>ICT OB Live Scanner v10.6-M</b>\n\n"
        m += "<b>Final optimized config (7 backtest rounds):</b>\n"
        m += f"  Sweep={SWEEP_REQUIRED} | FVG={FVG_REQUIRED} | BOS={BOS_REQUIRED} | Discount={DISCOUNT_REQUIRED}\n"
        m += f"  Disp={DISPLACEMENT_MULT}x | VOL {VOL_SPIKE_MIN}x-{VOL_SPIKE_MAX}x | ADX=OFF | RSI<{RSI_THRESH}\n"
        m += f"  TP1={ATR_TP1_MULT}x | TP2={ATR_TP2_MULT}x | TP3={ATR_TP3_MULT}x\n\n"
        m += "<b>Backtest (90d, 600 pairs):</b>\n"
        m += "  88.9% TP1 | 2.21 RR | -0.56% MaxDD | +2.21% avg\n\n"
        m += "/scan /stats /trades /regime /help"
        await update.message.reply_text(m, parse_mode=ParseMode.HTML)

    async def cmd_scan(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        if self.s.is_scanning:
            await update.message.reply_text('Scan already running!'); return
        await update.message.reply_text('🔍 Scanning all pairs...')
        asyncio.create_task(self.s.scan_all())

    async def cmd_stats(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        s   = self.s.stats
        tp1 = s['tp1_hits']; sl = s['sl_hits']
        wr  = round(tp1 / max(tp1+sl, 1) * 100, 1)
        hrs = round((datetime.now() - s['session_start']).total_seconds() / 3600, 1)
        re  = '🐂' if self.s.btc_regime == 'BULL' else '🐻'
        m  = f"📊 <b>STATS — v10.6-M</b>\n\nSession: {hrs}h\n"
        m += f"BTC: {re} <b>{self.s.btc_regime}</b>"
        if self.s.btc_price: m += f"  (${self.s.btc_price:,.0f})"
        m += f"\n\n<b>Signals:</b> {s['total_signals']}\n"
        m += f"  🟢 Long: {s['long_signals']}  |  🔴 Short: {s['short_signals']}\n"
        m += f"  Regime blocked: {s['regime_blocked']}\n\n"
        m += f"<b>TP Performance:</b>\n"
        m += f"  ✅ TP1: {tp1}  ({wr}% WR)\n"
        m += f"  💰 TP2: {s['tp2_hits']}  |  🔥 TP3: {s['tp3_hits']}\n"
        m += f"  🔒 BE: {s['be_saves']}  |  ❌ SL: {sl}\n\n"
        m += f"Tracking: {len(self.s.active_trades)} trades\n"
        if s['last_scan']: m += f"Last scan: {s['last_scan'].strftime('%H:%M UTC')}"
        await update.message.reply_text(m, parse_mode=ParseMode.HTML)

    async def cmd_trades(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        trades = self.s.active_trades
        if not trades:
            await update.message.reply_text('No active trades.'); return
        m = f"📡 <b>ACTIVE ({len(trades)})</b>\n\n"
        for tid, t in list(trades.items())[:10]:
            age  = int((datetime.now(timezone.utc) - t['timestamp']).total_seconds() / 3600)
            be_s = ' 🔒BE' if t['be_active'] else ''
            m += f"<b>{t['symbol']}</b>  {t['signal']}{be_s}  |  {age}h ago\n"
            m += f"  Entry: <code>${t['entry']:.6g}</code>  RR:{t['rr']}\n"
            m += f"  TP1:{'✅' if t['tp1_hit'] else '⏳'}  TP2:{'✅' if t['tp2_hit'] else '⏳'}  TP3:{'✅' if t['tp3_hit'] else '⏳'}\n"
            m += f"  {t['confirms']}\n\n"
        await update.message.reply_text(m, parse_mode=ParseMode.HTML)

    async def cmd_regime(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        r  = self.s.btc_regime
        re = '🐂' if r == 'BULL' else '🐻'
        m  = f"{re} <b>BTC 4H Regime: {r}</b>\n\n"
        if self.s.btc_price and self.s.btc_ema:
            m += f"Price:  ${self.s.btc_price:,.2f}\n"
            m += f"EMA21:  ${self.s.btc_ema:,.2f}\n\n"
        m += ("✅ LONGs active\n🚫 SHORTs BLOCKED\n" if r == 'BULL'
              else "✅ SHORTs active\n🚫 LONGs BLOCKED\n")
        m += f"\n<i>Target: 88.9% TP1 | 2.21 RR</i>"
        await update.message.reply_text(m, parse_mode=ParseMode.HTML)

    async def cmd_help(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        m  = "📚 <b>ICT OB v10.6-M Help</b>\n\n"
        m += "<b>Signal confirms:</b>\n"
        m += f"  VOL_Xx  = volume spike {VOL_SPIKE_MIN}x-{VOL_SPIKE_MAX}x\n"
        m += f"  RSI<{RSI_THRESH}  = oversold/weak momentum\n"
        m += f"  EMA_BEAR = bearish cross (SHORTs only)\n\n"
        m += "<b>Close plan:</b>\n"
        m += f"  LONG  — {int(LONG_TP1_PCT*100)}/{int(LONG_TP2_PCT*100)}/{int(LONG_TP3_PCT*100)} → runner to TP3\n"
        m += f"  SHORT — {int(SHORT_TP1_PCT*100)}/{int(SHORT_TP2_PCT*100)}\n\n"
        m += "<b>After TP1:</b> always move SL to breakeven\n\n"
        m += "/scan /stats /trades /regime /help"
        await update.message.reply_text(m, parse_mode=ParseMode.HTML)


# ═══════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════

async def main():
    scanner = ICTScanner()
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

    logger.info('=== ICT OB LIVE SCANNER v10.6-M ===')
    logger.info(f'88.9% TP1 | 2.21 RR | -0.56% MaxDD | ADX=OFF | RSI<{RSI_THRESH}')
    logger.info(f'TF={TIMEFRAME} | {SCAN_INTERVAL_MIN}min scan | HARD regime | {MAX_PAIRS} pairs')

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
