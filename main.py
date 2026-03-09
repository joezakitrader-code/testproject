"""
ADVANCED DAY TRADING SCANNER v4.0 — FINAL OPTIMIZED
======================================================
Changes from v3 based on backtest results:

  REGIME (biggest fix):
  ✅ HARD_BLOCK_REGIME = True  (53.7% → 96.4% WR by blocking counter-trend)

  TPs:
  ✅ TP2: 0.8x ATR  (was 1.0 — still never hit)
  ✅ TP3: 1.2x ATR  (was 1.5 — more realistic)
  ✅ TP1: 0.6x ATR  (kept — near-certain hit confirmed)

  SCORING — removed weak indicators:
  ✅ Removed: uo_oversold / uo_overbought (25-50% WR)
  ✅ Removed: williams_oversold / williams_overbought (47-50% WR)
  ✅ Removed: below_vwap for LONG (57% WR — too noisy)
  ✅ Removed: rsi_overbought standalone (42.9% WR)
  ✅ Boosted: bullish_divergence +0.5pts (100% WR)
  ✅ Boosted: vol_spike_bull +0.5pts (100% WR)
  ✅ Boosted: macd_cross_bull +0.5pts (100% WR)

  QUALITY:
  ✅ Removed HIGH tier — was 61.9% WR (worse than GOOD at 82.7%)
  ✅ Only PREMIUM (65%+) and GOOD (51%+) remain
  ✅ MIN_SCORE_PCT raised to 0.53 to cut noisy signals

  SIGNAL FREQUENCY:
  ✅ Removed 6h cooldown — was blocking valid signals
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

# ─────────────────────────────────────────────────────────────
# PARAMETERS — tuned from 3 rounds of backtesting
# ─────────────────────────────────────────────────────────────
ATR_SL_MULT  = 1.5
ATR_TP1_MULT = 0.6   # Confirmed ~78% hit rate
ATR_TP2_MULT = 0.8   # Was 1.0 — still never hit, tightened again
ATR_TP3_MULT = 1.2   # Was 1.5 — more realistic

MIN_SCORE_PCT       = 0.53   # Raised from 0.51 — cuts noisy signals
QUALITY_PREMIUM_PCT = 0.65   # 65%+ = PREMIUM

HARD_BLOCK_REGIME     = True   # KEY FIX: 53.7% → 96.4% WR
USE_LONG_TREND_FILTER = True   # Keep — was working

MAX_TRADE_HOURS    = 24
SCAN_INTERVAL_MIN  = 15
MIN_VOLUME_USDT    = 1_000_000

# ─────────────────────────────────────────────────────────────

class AdvancedDayTradingScanner:
    def __init__(self, telegram_token, telegram_chat_id, binance_api_key=None, binance_secret=None):
        self.telegram_token = telegram_token
        self.telegram_bot   = Bot(token=telegram_token)
        self.chat_id        = telegram_chat_id
        self.exchange = ccxt.binance({
            'apiKey':          binance_api_key,
            'secret':          binance_secret,
            'enableRateLimit': True,
            'options':         {'defaultType': 'future'}
        })
        self.signal_history = deque(maxlen=200)
        self.active_trades  = {}
        self.btc_regime     = None
        self.stats = {
            'total_signals': 0, 'long_signals': 0, 'short_signals': 0,
            'premium_signals': 0,
            'tp1_hits': 0, 'tp2_hits': 0, 'tp3_hits': 0, 'sl_hits': 0,
            'regime_blocked': 0, 'filtered_long': 0,
            'last_scan_time': None, 'pairs_scanned': 0,
        }
        self.is_scanning = False

    # ── BTC Regime ────────────────────────────────────────────

    async def update_btc_regime(self):
        try:
            ohlcv = await self.exchange.fetch_ohlcv('BTC/USDT:USDT', '4h', limit=30)
            df = pd.DataFrame(ohlcv, columns=['timestamp','open','high','low','close','volume'])
            df['ema21'] = ta.trend.EMAIndicator(df['close'], window=21).ema_indicator()
            last_close = df['close'].iloc[-1]
            last_ema   = df['ema21'].iloc[-1]
            prev       = self.btc_regime
            self.btc_regime = 'BULL' if last_close > last_ema else 'BEAR'
            if prev != self.btc_regime:
                emoji = '🐂' if self.btc_regime == 'BULL' else '🐻'
                await self.send_msg(
                    f"{emoji} <b>BTC Regime Flip!</b>\n"
                    f"{prev} → <b>{self.btc_regime}</b>\n"
                    f"BTC: ${last_close:,.2f} | EMA21: ${last_ema:,.2f}\n\n"
                    f"{'✅ LONGs now active, SHORTs blocked' if self.btc_regime == 'BULL' else '✅ SHORTs now active, LONGs blocked'}"
                )
            logger.info(f"📡 BTC: {self.btc_regime} (${last_close:,.0f} vs EMA ${last_ema:,.0f})")
        except Exception as e:
            logger.error(f"BTC regime error: {e}")

    # ── All USDT Pairs ────────────────────────────────────────

    async def get_all_usdt_pairs(self):
        try:
            await self.exchange.load_markets()
            tickers = await self.exchange.fetch_tickers()
            pairs = [
                s for s in self.exchange.symbols
                if s.endswith('/USDT:USDT') and 'PERP' not in s
                and tickers.get(s, {}).get('quoteVolume', 0) > MIN_VOLUME_USDT
            ]
            pairs.sort(key=lambda x: tickers.get(x, {}).get('quoteVolume', 0), reverse=True)
            logger.info(f"✅ {len(pairs)} pairs to scan")
            return pairs
        except Exception as e:
            logger.error(f"Pairs error: {e}")
            return []

    # ── Data Fetch ────────────────────────────────────────────

    async def fetch_data(self, symbol):
        data = {}
        try:
            for tf, limit in [('1h', 100), ('4h', 100), ('15m', 50)]:
                ohlcv = await self.exchange.fetch_ohlcv(symbol, tf, limit=limit)
                df = pd.DataFrame(ohlcv, columns=['timestamp','open','high','low','close','volume'])
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                data[tf] = df
                await asyncio.sleep(0.05)
            return data
        except Exception as e:
            logger.error(f"Fetch error {symbol}: {e}")
            return None

    # ── Indicators ────────────────────────────────────────────

    def calculate_supertrend(self, df, period=10, multiplier=3):
        try:
            hl2 = (df['high'] + df['low']) / 2
            atr = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close'], window=period).average_true_range()
            upper = hl2 + (multiplier * atr)
            lower = hl2 - (multiplier * atr)
            st = [0] * len(df)
            for i in range(1, len(df)):
                if df['close'].iloc[i] > upper.iloc[i-1]:
                    st[i] = lower.iloc[i]
                elif df['close'].iloc[i] < lower.iloc[i-1]:
                    st[i] = upper.iloc[i]
                else:
                    st[i] = st[i-1]
            return pd.Series(st, index=df.index)
        except:
            return pd.Series([0] * len(df), index=df.index)

    def add_indicators(self, df):
        try:
            if len(df) < 30:
                return df

            # TREND
            df['ema_9']  = ta.trend.EMAIndicator(df['close'], window=9).ema_indicator()
            df['ema_21'] = ta.trend.EMAIndicator(df['close'], window=21).ema_indicator()
            df['ema_50'] = ta.trend.EMAIndicator(df['close'], window=min(50, len(df)-1)).ema_indicator()
            df['supertrend'] = self.calculate_supertrend(df)

            # MOMENTUM
            df['rsi']         = ta.momentum.RSIIndicator(df['close'], window=14).rsi()
            srsi = ta.momentum.StochRSIIndicator(df['close'])
            df['stoch_rsi_k'] = srsi.stochrsi_k()
            df['stoch_rsi_d'] = srsi.stochrsi_d()
            macd = ta.trend.MACD(df['close'])
            df['macd']        = macd.macd()
            df['macd_signal'] = macd.macd_signal()
            df['roc']         = ta.momentum.ROCIndicator(df['close'], window=12).roc()

            # VOLATILITY
            bb = ta.volatility.BollingerBands(df['close'], window=20, window_dev=2)
            df['bb_upper']  = bb.bollinger_hband()
            df['bb_lower']  = bb.bollinger_lband()
            df['bb_pband']  = bb.bollinger_pband()
            df['atr']       = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close']).average_true_range()

            # VOLUME
            df['volume_sma']   = df['volume'].rolling(20).mean()
            df['volume_ratio'] = df['volume'] / df['volume_sma'].replace(0, np.nan)
            df['obv']     = ta.volume.OnBalanceVolumeIndicator(df['close'], df['volume']).on_balance_volume()
            df['obv_ema'] = df['obv'].ewm(span=20).mean()
            df['mfi']     = ta.volume.MFIIndicator(df['high'], df['low'], df['close'], df['volume']).money_flow_index()
            df['cmf']     = ta.volume.ChaikinMoneyFlowIndicator(df['high'], df['low'], df['close'], df['volume']).chaikin_money_flow()

            # TREND STRENGTH
            adx = ta.trend.ADXIndicator(df['high'], df['low'], df['close'])
            df['adx']      = adx.adx()
            df['di_plus']  = adx.adx_pos()
            df['di_minus'] = adx.adx_neg()
            df['cci']      = ta.trend.CCIIndicator(df['high'], df['low'], df['close']).cci()
            aroon = ta.trend.AroonIndicator(df['high'], df['low'])
            df['aroon_ind'] = aroon.aroon_up() - aroon.aroon_down()

            # VWAP
            tp = (df['high'] + df['low'] + df['close']) / 3
            df['vwap'] = (tp * df['volume']).cumsum() / df['volume'].cumsum()
            df['vwap'].fillna(df['close'], inplace=True)

            # PATTERNS
            df['bullish_engulfing'] = (
                (df['close'].shift(1) < df['open'].shift(1)) &
                (df['close'] > df['open']) &
                (df['open'] <= df['close'].shift(1)) &
                (df['close'] >= df['open'].shift(1))
            ).astype(int)
            df['bearish_engulfing'] = (
                (df['close'].shift(1) > df['open'].shift(1)) &
                (df['close'] < df['open']) &
                (df['open'] >= df['close'].shift(1)) &
                (df['close'] <= df['open'].shift(1))
            ).astype(int)
            df['bullish_divergence'] = (
                (df['low'] < df['low'].shift(1)) & (df['rsi'] > df['rsi'].shift(1))
            ).astype(int)
            df['bearish_divergence'] = (
                (df['high'] > df['high'].shift(1)) & (df['rsi'] < df['rsi'].shift(1))
            ).astype(int)

        except Exception as e:
            logger.error(f"Indicator error: {e}")
        return df

    def detect_volume_spike(self, df):
        if len(df) < 20:
            return False, 1.0
        recent = df['volume'].iloc[-1]
        avg    = df['volume'].iloc[-20:].mean()
        if avg == 0 or pd.isna(avg):
            return False, 1.0
        ratio = recent / avg
        return ratio > 2.5, ratio

    # ── Signal Detection v4 ───────────────────────────────────

    def detect_signal(self, data, symbol):
        try:
            if not data or '1h' not in data:
                return None

            for tf in data:
                data[tf] = self.add_indicators(data[tf])

            df_1h  = data['1h']
            df_4h  = data['4h']
            df_15m = data['15m']

            if len(df_1h) < 50:
                return None

            r1h  = df_1h.iloc[-1]
            p1h  = df_1h.iloc[-2]
            r4h  = df_4h.iloc[-1]
            r15m = df_15m.iloc[-1]

            for col in ['ema_9','ema_21','rsi','macd','vwap','bb_pband','atr']:
                if col not in r1h.index or pd.isna(r1h[col]):
                    return None

            volume_spike, vol_ratio = self.detect_volume_spike(df_1h)
            macd_cross_bull = r1h['macd'] > r1h['macd_signal'] and p1h['macd'] <= p1h['macd_signal']
            macd_cross_bear = r1h['macd'] < r1h['macd_signal'] and p1h['macd'] >= p1h['macd_signal']

            ls = ss = 0
            lr = []; sr = []

            # ── TREND (6 pts) ──────────────────────────────────
            if r4h['ema_9'] > r4h['ema_21'] > r4h['ema_50']:
                ls += 3; lr.append('🔥 4H Uptrend')
            elif r4h['ema_9'] < r4h['ema_21'] < r4h['ema_50']:
                ss += 3; sr.append('🔥 4H Downtrend')

            if r1h['ema_9'] > r1h['ema_21']:
                ls += 2; lr.append('1H EMA Bullish')
            elif r1h['ema_9'] < r1h['ema_21']:
                ss += 2; sr.append('1H EMA Bearish')

            if r1h['close'] > r1h['supertrend']:
                ls += 1; lr.append('SuperTrend ↑')
            elif r1h['close'] < r1h['supertrend']:
                ss += 1; sr.append('SuperTrend ↓')

            # ── MOMENTUM (8.5 pts — removed weak UO/Williams) ──
            rsi = r1h['rsi']
            if rsi < 30:
                ls += 3.5; lr.append(f'💎 RSI Oversold ({rsi:.0f})')
            elif rsi < 40:
                ls += 2;   lr.append(f'RSI Low ({rsi:.0f})')
            elif 40 <= rsi <= 50:
                ls += 1;   lr.append(f'RSI Buy Zone ({rsi:.0f})')

            if rsi > 70:
                ss += 3.5; sr.append(f'💎 RSI Overbought ({rsi:.0f})')
            elif rsi > 60:
                ss += 2;   sr.append(f'RSI High ({rsi:.0f})')
            elif 50 <= rsi <= 60:
                ss += 1;   sr.append(f'RSI Sell Zone ({rsi:.0f})')

            sk = r1h['stoch_rsi_k']; sd = r1h['stoch_rsi_d']
            if sk < 0.2 and sk > sd:
                ls += 2; lr.append('⚡ StochRSI Cross ↑')
            elif sk > 0.8 and sk < sd:
                ss += 2; sr.append('⚡ StochRSI Cross ↓')

            # MACD — boosted +0.5 (100% WR confirmed)
            if macd_cross_bull:
                ls += 3; lr.append('🎯 MACD Cross ↑')
            elif macd_cross_bear:
                ss += 3; sr.append('🎯 MACD Cross ↓')

            # ── VOLUME (5.5 pts) ──────────────────────────────
            # vol_spike_bull boosted +0.5 (100% WR confirmed)
            if volume_spike:
                if r1h['close'] > p1h['close']:
                    ls += 3.5; lr.append(f'🚀 Vol Spike ({vol_ratio:.1f}x)')
                else:
                    ss += 3;   sr.append(f'💥 Vol Dump ({vol_ratio:.1f}x)')

            if r1h['mfi'] < 20:
                ls += 1.5; lr.append(f'MFI Oversold ({r1h["mfi"]:.0f})')
            elif r1h['mfi'] > 80:
                ss += 1.5; sr.append(f'MFI Overbought ({r1h["mfi"]:.0f})')

            if r1h['cmf'] > 0.15:
                ls += 1; lr.append('CMF Buying')
            elif r1h['cmf'] < -0.15:
                ss += 1; sr.append('CMF Selling')

            if r1h['obv'] > r1h['obv_ema']:
                ls += 0.5; lr.append('OBV Accumulation')
            else:
                ss += 0.5; sr.append('OBV Distribution')

            # ── VOLATILITY (4 pts — removed weak below_vwap for longs) ──
            bbp = r1h['bb_pband']
            if bbp < 0.1:
                ls += 2.5; lr.append('💎 Lower BB')
            elif bbp > 0.9:
                ss += 2.5; sr.append('💎 Upper BB')

            if r1h['cci'] < -150:
                ls += 1.5; lr.append('CCI Oversold')
            elif r1h['cci'] > 150:
                ss += 1.5; sr.append('CCI Overbought')

            # above_vwap for SHORT (100% WR), below_vwap removed for LONG (57% WR)
            if r1h['close'] > r1h['vwap'] * 1.02:
                ss += 1; sr.append('Above VWAP')

            # ── TREND STRENGTH (4 pts) ────────────────────────
            if r1h['adx'] > 30:
                if r1h['di_plus'] > r1h['di_minus']:
                    ls += 2; lr.append(f'ADX Strong Up ({r1h["adx"]:.0f})')
                else:
                    ss += 2; sr.append(f'ADX Strong Down ({r1h["adx"]:.0f})')
            elif r1h['adx'] > 25:
                if r1h['di_plus'] > r1h['di_minus']:
                    ls += 1
                else:
                    ss += 1

            if r1h['aroon_ind'] > 50:
                ls += 1; lr.append('Aroon Bullish')
            elif r1h['aroon_ind'] < -50:
                ss += 1; sr.append('Aroon Bearish')

            if r1h['roc'] > 3:
                ls += 1; lr.append('ROC Positive')
            elif r1h['roc'] < -3:
                ss += 1; sr.append('ROC Negative')

            # ── PATTERNS & DIVERGENCE (3.5 pts) ──────────────
            # bullish_divergence boosted +0.5 (100% WR confirmed)
            if r1h['bullish_divergence']:
                ls += 2.5; lr.append('🎯 Bullish Divergence')
            elif r1h['bearish_divergence']:
                ss += 2;   sr.append('🎯 Bearish Divergence')

            if r15m['bullish_engulfing']:
                ls += 1.5; lr.append('📊 Bullish Engulfing')
            elif r15m['bearish_engulfing']:
                ss += 1.5; sr.append('📊 Bearish Engulfing')

            # ── HTF (2 pts) ───────────────────────────────────
            if r4h['close'] > r4h['vwap']:
                ls += 1; lr.append('4H Above VWAP')
            else:
                ss += 1; sr.append('4H Below VWAP')

            if r4h['rsi'] < 50:
                ls += 1
            elif r4h['rsi'] > 50:
                ss += 1

            # ── DETERMINE SIGNAL ──────────────────────────────
            max_score     = 35
            min_threshold = max_score * MIN_SCORE_PCT
            signal        = None

            if ls > ss and ls >= min_threshold:
                signal = 'LONG';  score = ls; reasons = lr
            elif ss > ls and ss >= min_threshold:
                signal = 'SHORT'; score = ss; reasons = sr
            if not signal:
                return None

            # ── HARD REGIME BLOCK (key v4 fix) ───────────────
            if HARD_BLOCK_REGIME and self.btc_regime:
                if signal == 'LONG' and self.btc_regime == 'BEAR':
                    self.stats['regime_blocked'] += 1
                    logger.info(f"  🐻 {symbol} LONG blocked — BTC BEAR regime")
                    return None
                if signal == 'SHORT' and self.btc_regime == 'BULL':
                    self.stats['regime_blocked'] += 1
                    logger.info(f"  🐂 {symbol} SHORT blocked — BTC BULL regime")
                    return None

            # ── LONG trend filter (any one of 5) ─────────────
            if signal == 'LONG' and USE_LONG_TREND_FILTER:
                confirms = [
                    r4h['ema_9'] > r4h['ema_21'],
                    r1h['ema_9'] > r1h['ema_21'],
                    macd_cross_bull,
                    volume_spike and r1h['close'] > p1h['close'],
                    rsi < 35,
                ]
                if not any(confirms):
                    self.stats['filtered_long'] += 1
                    return None

            # ── Quality (2 tiers only — HIGH removed) ────────
            pct = score / max_score
            quality = 'PREMIUM 💎' if pct >= QUALITY_PREMIUM_PCT else 'GOOD ✅'

            entry = r15m['close']
            atr   = r1h['atr']
            if pd.isna(atr) or atr == 0 or pd.isna(entry) or entry == 0:
                return None

            if signal == 'LONG':
                sl      = entry - atr * ATR_SL_MULT
                targets = [entry + atr*ATR_TP1_MULT, entry + atr*ATR_TP2_MULT, entry + atr*ATR_TP3_MULT]
            else:
                sl      = entry + atr * ATR_SL_MULT
                targets = [entry - atr*ATR_TP1_MULT, entry - atr*ATR_TP2_MULT, entry - atr*ATR_TP3_MULT]

            risk_pct = abs((sl - entry) / entry * 100)
            rr       = [abs(tp - entry) / abs(sl - entry) for tp in targets]
            trade_id = f"{symbol.replace('/USDT:USDT','')}_{datetime.now().strftime('%Y%m%d%H%M%S')}"

            return {
                'trade_id':      trade_id,
                'symbol':        symbol.replace('/USDT:USDT', ''),
                'full_symbol':   symbol,
                'signal':        signal,
                'quality':       quality,
                'score':         score,
                'max_score':     max_score,
                'score_percent': pct * 100,
                'entry':         entry,
                'stop_loss':     sl,
                'targets':       targets,
                'reward_ratios': rr,
                'risk_percent':  risk_pct,
                'reasons':       reasons[:10],
                'tp_hit':        [False, False, False],
                'sl_hit':        False,
                'timestamp':     datetime.now(),
                'btc_regime':    self.btc_regime or 'N/A',
            }

        except Exception as e:
            logger.error(f"Signal error {symbol}: {e}")
            return None

    # ── Format Signal ─────────────────────────────────────────

    def format_signal(self, sig):
        emoji    = '🚀' if sig['signal'] == 'LONG' else '🔻'
        regime   = sig.get('btc_regime', '?')
        r_emoji  = '🐂' if regime == 'BULL' else '🐻'

        msg  = f"{'='*42}\n"
        msg += f"{emoji} <b>DAY TRADE — {sig['quality']}</b> {emoji}\n"
        msg += f"{'='*42}\n\n"
        msg += f"<b>🆔</b> <code>{sig['trade_id']}</code>\n"
        msg += f"<b>📊 PAIR:</b> #{sig['symbol']}  {r_emoji} BTC:{regime}\n"
        msg += f"<b>📍 DIR:</b>  <b>{sig['signal']}</b>  (with trend ✅)\n"
        msg += f"<b>⭐ SCORE:</b> {sig['score']:.1f}/{sig['max_score']} ({sig['score_percent']:.0f}%)\n"
        filled = int(sig['score_percent'] / 10)
        msg += f"{'▰'*filled}{'▱'*(10-filled)}\n\n"

        msg += f"<b>💰 ENTRY:</b> ${sig['entry']:.6f}\n\n"
        msg += f"<b>🎯 TARGETS:</b>\n"
        for i, (tp, rr) in enumerate(zip(sig['targets'], sig['reward_ratios']), 1):
            pct_g = abs((tp - sig['entry']) / sig['entry'] * 100)
            sizes = ['40%', '40%', '20%']
            msg  += f"  TP{i}: ${tp:.6f}  +{pct_g:.2f}%  [RR {rr:.1f}:1]  → close {sizes[i-1]}\n"

        msg += f"\n<b>🛑 SL:</b> ${sig['stop_loss']:.6f}  (-{sig['risk_percent']:.2f}%)\n"
        msg += f"<i>💡 TP1 hit → move SL to breakeven immediately</i>\n\n"

        msg += f"<b>📋 REASONS:</b>\n"
        for r in sig['reasons']:
            msg += f"  • {r}\n"

        msg += f"\n📡 <b>TRACKING LIVE</b> | v4.0\n"
        msg += f"<i>⏰ {sig['timestamp'].strftime('%H:%M:%S UTC')}</i>\n"
        msg += f"{'='*42}"
        return msg

    # ── Telegram ──────────────────────────────────────────────

    async def send_msg(self, msg):
        try:
            await self.telegram_bot.send_message(
                chat_id=self.chat_id, text=msg, parse_mode=ParseMode.HTML
            )
        except Exception as e:
            logger.error(f"Send error: {e}")

    async def send_tp_alert(self, trade, tp_num, price):
        emoji = '🎉' if trade['signal'] == 'LONG' else '💰'
        tp    = trade['targets'][tp_num - 1]
        pct   = abs((tp - trade['entry']) / trade['entry'] * 100)
        sizes = ['40%', '40%', '20%']
        actions = [
            "Take 40% profit\n🔒 Move SL to <b>ENTRY (breakeven)</b>",
            "Take 40% more profit\nLet last 20% run to TP3",
            "Close remaining 20%\n🎊 <b>TRADE COMPLETE!</b>",
        ]
        msg  = f"{emoji} <b>TP{tp_num} HIT!</b> {emoji}\n\n"
        msg += f"<code>{trade['trade_id']}</code>\n"
        msg += f"<b>{trade['symbol']}</b> {trade['signal']}\n\n"
        msg += f"Target:  ${tp:.6f}\n"
        msg += f"Profit:  <b>+{pct:.2f}%</b>  ({sizes[tp_num-1]})\n\n"
        msg += f"📋 {actions[tp_num-1]}"
        await self.send_msg(msg)
        self.stats[f'tp{tp_num}_hits'] += 1

    async def send_sl_alert(self, trade, price):
        loss = abs((price - trade['entry']) / trade['entry'] * 100)
        msg  = f"⛔ <b>STOP LOSS HIT</b> ⛔\n\n"
        msg += f"<code>{trade['trade_id']}</code>\n"
        msg += f"{trade['symbol']} {trade['signal']}\n\n"
        msg += f"Entry: ${trade['entry']:.6f}\n"
        msg += f"SL:    ${trade['stop_loss']:.6f}\n"
        msg += f"Price: ${price:.6f}\n"
        msg += f"Loss:  <b>-{loss:.2f}%</b>\n\n"
        msg += f"<i>Next signal incoming — stay patient 🎯</i>"
        await self.send_msg(msg)
        self.stats['sl_hits'] += 1

    # ── Trade Tracker ─────────────────────────────────────────

    async def track_trades(self):
        logger.info("📡 Tracker started")
        while True:
            try:
                if not self.active_trades:
                    await asyncio.sleep(30)
                    continue

                to_remove = []
                for tid, trade in list(self.active_trades.items()):
                    try:
                        if datetime.now() - trade['timestamp'] > timedelta(hours=MAX_TRADE_HOURS):
                            await self.send_msg(
                                f"⏰ <b>24H TIMEOUT</b>\n<code>{tid}</code>\n"
                                f"{trade['symbol']} — close manually at market price!"
                            )
                            to_remove.append(tid)
                            continue

                        ticker = await self.exchange.fetch_ticker(trade['full_symbol'])
                        price  = ticker['last']

                        if trade['signal'] == 'LONG':
                            if not trade['sl_hit'] and price <= trade['stop_loss']:
                                await self.send_sl_alert(trade, price)
                                trade['sl_hit'] = True
                                to_remove.append(tid)
                                continue
                            for i, tp in enumerate(trade['targets']):
                                if not trade['tp_hit'][i] and price >= tp:
                                    await self.send_tp_alert(trade, i+1, price)
                                    trade['tp_hit'][i] = True
                                    if i == 2:
                                        to_remove.append(tid)
                        else:
                            if not trade['sl_hit'] and price >= trade['stop_loss']:
                                await self.send_sl_alert(trade, price)
                                trade['sl_hit'] = True
                                to_remove.append(tid)
                                continue
                            for i, tp in enumerate(trade['targets']):
                                if not trade['tp_hit'][i] and price <= tp:
                                    await self.send_tp_alert(trade, i+1, price)
                                    trade['tp_hit'][i] = True
                                    if i == 2:
                                        to_remove.append(tid)

                    except Exception as e:
                        logger.error(f"Track error {tid}: {e}")

                for tid in to_remove:
                    if tid in self.active_trades:
                        del self.active_trades[tid]

                await asyncio.sleep(30)

            except Exception as e:
                logger.error(f"Tracker error: {e}")
                await asyncio.sleep(60)

    # ── Main Scanner ──────────────────────────────────────────

    async def scan_all(self):
        if self.is_scanning:
            return []
        self.is_scanning = True
        logger.info("🔍 Scan started")

        await self.update_btc_regime()
        pairs   = await self.get_all_usdt_pairs()
        signals = []
        scanned = 0

        for pair in pairs:
            try:
                logger.info(f"  📊 {pair}")
                data = await self.fetch_data(pair)
                if data:
                    sig = self.detect_signal(data, pair)
                    if sig:
                        signals.append(sig)
                        self.signal_history.append(sig)
                        self.stats['total_signals'] += 1
                        if sig['signal'] == 'LONG':  self.stats['long_signals'] += 1
                        else:                         self.stats['short_signals'] += 1
                        if 'PREMIUM' in sig['quality']:
                            self.stats['premium_signals'] += 1
                        self.active_trades[sig['trade_id']] = sig
                        await self.send_msg(self.format_signal(sig))
                        await asyncio.sleep(1.5)

                scanned += 1
                if scanned % 30 == 0:
                    logger.info(f"  📈 {scanned}/{len(pairs)} scanned")
                await asyncio.sleep(0.5)

            except Exception as e:
                logger.error(f"❌ {pair}: {e}")

        self.stats['last_scan_time'] = datetime.now()
        self.stats['pairs_scanned']  = scanned

        longs   = sum(1 for s in signals if s['signal'] == 'LONG')
        shorts  = len(signals) - longs
        premium = sum(1 for s in signals if 'PREMIUM' in s['quality'])

        direction_tag = '🟢 BULL mode — LONG only' if self.btc_regime == 'BULL' else '🔴 BEAR mode — SHORT only' if self.btc_regime == 'BEAR' else '❓ Regime unknown'

        summary  = f"✅ <b>SCAN DONE</b> — v4.0\n\n"
        summary += f"📡 {direction_tag}\n"
        summary += f"📊 Scanned: {scanned} pairs\n\n"
        summary += f"🎯 Signals: <b>{len(signals)}</b>\n"
        summary += f"  🟢 Long:    {longs}\n"
        summary += f"  🔴 Short:   {shorts}\n"
        summary += f"  💎 Premium: {premium}\n\n"
        summary += f"🚫 Regime blocked: {self.stats['regime_blocked']}\n"
        summary += f"⚡ Long filtered:  {self.stats['filtered_long']}\n"
        summary += f"📡 Tracking:       {len(self.active_trades)}\n"
        summary += f"⏰ {datetime.now().strftime('%H:%M:%S')}"
        await self.send_msg(summary)

        logger.info(f"✅ Scan done: {len(signals)} signals")
        self.is_scanning = False
        return signals

    async def run(self, interval=SCAN_INTERVAL_MIN):
        logger.info("🚀 v4.0 STARTING")

        welcome  = "🔥 <b>ADVANCED DAY TRADING SCANNER v4.0</b> 🔥\n"
        welcome += "<i>Final optimized — 3 rounds of backtesting</i>\n\n"
        welcome += "<b>Key improvements over v3:</b>\n"
        welcome += "🐻/🐂 Hard BTC regime filter (96% WR aligned)\n"
        welcome += "🎯 Tighter TPs: 0.6 / 0.8 / 1.2x ATR\n"
        welcome += "🗑️  Removed 4 weak indicators (UO, Williams)\n"
        welcome += "⬆️  Boosted MACD, Vol Spike, Divergence signals\n"
        welcome += "✅ 2 quality tiers: PREMIUM 💎 / GOOD ✅\n\n"
        welcome += f"⏱ Scanning every <b>{interval} min</b>\n\n"
        welcome += "<b>Commands:</b>\n"
        welcome += "/scan /stats /trades /regime /help"
        await self.send_msg(welcome)

        asyncio.create_task(self.track_trades())

        while True:
            try:
                await self.scan_all()
                await asyncio.sleep(interval * 60)
            except Exception as e:
                logger.error(f"Run error: {e}")
                await asyncio.sleep(60)

    async def close(self):
        await self.exchange.close()


# ─────────────────────────────────────────────────────────────
# COMMANDS
# ─────────────────────────────────────────────────────────────

class BotCommands:
    def __init__(self, scanner: AdvancedDayTradingScanner):
        self.scanner = scanner

    async def cmd_start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        msg  = "🚀 <b>Advanced Day Trading Scanner v4.0</b>\n\n"
        msg += "3x backtested & optimized.\n"
        msg += "Only trades WITH the BTC market trend.\n\n"
        msg += "/scan /stats /trades /regime /help"
        await update.message.reply_text(msg, parse_mode=ParseMode.HTML)

    async def cmd_scan(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if self.scanner.is_scanning:
            await update.message.reply_text("⚠️ Scan already running!")
            return
        await update.message.reply_text("🔍 Scanning all USDT pairs...")
        asyncio.create_task(self.scanner.scan_all())

    async def cmd_stats(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        s   = self.scanner.stats
        tp1 = s['tp1_hits']; tp2 = s['tp2_hits']; tp3 = s['tp3_hits']; sl = s['sl_hits']
        completed = tp1 + sl
        wr = round(tp1 / completed * 100, 1) if completed > 0 else 0

        msg  = f"📊 <b>LIVE STATS — v4.0</b>\n\n"
        msg += f"<b>Signals:</b>\n"
        msg += f"  Total:   {s['total_signals']}\n"
        msg += f"  Long:    {s['long_signals']} 🟢\n"
        msg += f"  Short:   {s['short_signals']} 🔴\n"
        msg += f"  Premium: {s['premium_signals']} 💎\n\n"
        msg += f"<b>Filters:</b>\n"
        msg += f"  Regime blocked: {s['regime_blocked']} 🚫\n"
        msg += f"  Long filtered:  {s['filtered_long']} ⚡\n\n"
        msg += f"<b>Results:</b>\n"
        msg += f"  TP1: {tp1} 🎯  TP2: {tp2} 🎯  TP3: {tp3} 🎯\n"
        msg += f"  SL:  {sl} ❌\n"
        msg += f"  Win Rate: <b>{wr}%</b>\n\n"
        msg += f"<b>Market:</b>  {self.scanner.btc_regime or 'Unknown'}\n"
        msg += f"<b>Tracking:</b> {len(self.scanner.active_trades)} trades"
        if s['last_scan_time']:
            msg += f"\n\nLast scan: {s['last_scan_time'].strftime('%H:%M:%S')}"
        await update.message.reply_text(msg, parse_mode=ParseMode.HTML)

    async def cmd_trades(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        trades = self.scanner.active_trades
        if not trades:
            await update.message.reply_text("📭 No active trades.")
            return
        msg = f"📡 <b>ACTIVE TRADES ({len(trades)})</b>\n\n"
        for tid, t in list(trades.items())[:10]:
            age = int((datetime.now() - t['timestamp']).total_seconds() / 3600)
            tps = ''.join(['✅' if h else '⏳' for h in t['tp_hit']])
            msg += f"<b>{t['symbol']}</b> {t['signal']} {t['quality']}\n"
            msg += f"  {tps} | {age}h | Score: {t['score_percent']:.0f}%\n"
            msg += f"  Entry: ${t['entry']:.6f}\n\n"
        await update.message.reply_text(msg, parse_mode=ParseMode.HTML)

    async def cmd_regime(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        regime = self.scanner.btc_regime or 'Unknown'
        emoji  = '🐂' if regime == 'BULL' else '🐻' if regime == 'BEAR' else '❓'
        msg  = f"{emoji} <b>BTC Regime: {regime}</b>\n\n"
        if regime == 'BULL':
            msg += "BTC above 4H EMA21\n\n"
            msg += "✅ <b>LONG signals ACTIVE</b>\n"
            msg += "🚫 SHORT signals BLOCKED\n\n"
            msg += "<i>Backtest: 96.4% WR when trading with regime</i>"
        elif regime == 'BEAR':
            msg += "BTC below 4H EMA21\n\n"
            msg += "✅ <b>SHORT signals ACTIVE</b>\n"
            msg += "🚫 LONG signals BLOCKED\n\n"
            msg += "<i>Backtest: 96.4% WR when trading with regime</i>"
        else:
            msg += "Run /scan to determine regime"
        await update.message.reply_text(msg, parse_mode=ParseMode.HTML)

    async def cmd_help(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        msg  = "📚 <b>SCANNER v4.0 — COMPLETE GUIDE</b>\n\n"
        msg += "<b>TP Strategy (backtested):</b>\n"
        msg += "  TP1: 0.6x ATR → Take 40% + move SL to entry\n"
        msg += "  TP2: 0.8x ATR → Take 40%\n"
        msg += "  TP3: 1.2x ATR → Take final 20%\n"
        msg += "  SL:  1.5x ATR\n\n"
        msg += "<b>Why regime filtering matters:</b>\n"
        msg += "  With regime: 96.4% WR ✅\n"
        msg += "  Against regime: 53.7% WR ❌\n\n"
        msg += "<b>Quality Tiers:</b>\n"
        msg += "  💎 PREMIUM (65%+ score) — highest conviction\n"
        msg += "  ✅ GOOD (53%+ score) — solid setup\n\n"
        msg += "<b>Indicators (optimized from backtests):</b>\n"
        msg += "  Trend:    EMA 9/21/50, SuperTrend, Aroon\n"
        msg += "  Momentum: RSI, MACD (+boosted), StochRSI\n"
        msg += "  Volume:   OBV, MFI, CMF, Vol Spike (+boosted)\n"
        msg += "  Pattern:  Divergence (+boosted), Engulfing\n"
        msg += "  Removed:  UO, Williams R (proven weak)\n\n"
        msg += "/scan /stats /trades /regime /help"
        await update.message.reply_text(msg, parse_mode=ParseMode.HTML)


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

async def main():
    # ════════════════════════════
    TELEGRAM_TOKEN   = "7957028587:AAE7aSYtE4hCxxTIPkAs_1ULJ9e8alkY6Ic"
    TELEGRAM_CHAT_ID = "-1002442074724"
    BINANCE_API_KEY  = None
    BINANCE_SECRET   = None
    # ════════════════════════════

    scanner = AdvancedDayTradingScanner(
        telegram_token=TELEGRAM_TOKEN,
        telegram_chat_id=TELEGRAM_CHAT_ID,
        binance_api_key=BINANCE_API_KEY,
        binance_secret=BINANCE_SECRET,
    )

    app = Application.builder().token(TELEGRAM_TOKEN).build()
    cmds = BotCommands(scanner)

    app.add_handler(CommandHandler("start",  cmds.cmd_start))
    app.add_handler(CommandHandler("scan",   cmds.cmd_scan))
    app.add_handler(CommandHandler("stats",  cmds.cmd_stats))
    app.add_handler(CommandHandler("trades", cmds.cmd_trades))
    app.add_handler(CommandHandler("regime", cmds.cmd_regime))
    app.add_handler(CommandHandler("help",   cmds.cmd_help))

    await app.initialize()
    await app.start()
    logger.info("🤖 Bot v4.0 ready!")

    try:
        await scanner.run(interval=SCAN_INTERVAL_MIN)
    except KeyboardInterrupt:
        logger.info("Shutting down...")
    finally:
        await scanner.close()
        await app.stop()
        await app.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
