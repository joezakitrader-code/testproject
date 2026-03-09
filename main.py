"""
ADVANCED DAY TRADING SCANNER v3.0
===================================
CHANGES FROM v2 (based on backtest v2 results):
  ✅ Scans ALL USDT pairs (no whitelist default)
  ✅ TP multipliers ultra-tight: 0.6 / 1.0 / 1.5x ATR  (TP2/TP3 were never hit at 1.3/1.8)
  ✅ BTC regime filter SOFTENED: doesn't block signals, just warns + reduces score threshold
  ✅ LONG filter kept but relaxed: requires ANY ONE of 5 confirmations (was 3)
  ✅ Trailing SL suggestion after TP1 hit
  ✅ Signal frequency improved — more signals from more pairs
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
# TUNABLE PARAMETERS
# ─────────────────────────────────────────────────────────────
ATR_SL_MULT   = 1.5
ATR_TP1_MULT  = 0.6   # Very tight — near-guaranteed hit
ATR_TP2_MULT  = 1.0   # Same as v1 TP1 — should now hit
ATR_TP3_MULT  = 1.5   # Realistic extended target

MIN_SCORE_PCT           = 0.51
QUALITY_PREMIUM_PCT     = 0.65
QUALITY_HIGH_PCT        = 0.55

# Regime: softened — warns instead of blocks
# Set to True to hard-block counter-regime trades
HARD_BLOCK_REGIME       = False

USE_LONG_TREND_FILTER   = True   # Still filter weak longs
MAX_TRADE_HOURS         = 24
SCAN_INTERVAL_MIN       = 15
MIN_VOLUME_USDT         = 1_000_000

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
            'premium_signals': 0, 'high_signals': 0,
            'tp1_hits': 0, 'tp2_hits': 0, 'tp3_hits': 0, 'sl_hits': 0,
            'last_scan_time': None, 'pairs_scanned': 0, 'filtered_long': 0,
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
            prev_regime = self.btc_regime
            self.btc_regime = 'BULL' if last_close > last_ema else 'BEAR'
            if prev_regime != self.btc_regime:
                await self.send_msg(
                    f"📡 <b>BTC Regime Changed!</b>\n"
                    f"{prev_regime} → <b>{self.btc_regime}</b>\n"
                    f"BTC: ${last_close:,.0f} | EMA21: ${last_ema:,.0f}"
                )
            logger.info(f"📡 BTC Regime: {self.btc_regime}")
        except Exception as e:
            logger.error(f"BTC regime error: {e}")
            self.btc_regime = None

    # ── All USDT pairs ────────────────────────────────────────

    async def get_all_usdt_pairs(self):
        try:
            await self.exchange.load_markets()
            tickers = await self.exchange.fetch_tickers()
            pairs = []
            for symbol in self.exchange.symbols:
                if symbol.endswith('/USDT:USDT') and 'PERP' not in symbol:
                    vol = tickers.get(symbol, {}).get('quoteVolume', 0)
                    if vol > MIN_VOLUME_USDT:
                        pairs.append(symbol)
            pairs.sort(key=lambda x: tickers.get(x, {}).get('quoteVolume', 0), reverse=True)
            logger.info(f"✅ {len(pairs)} pairs found")
            return pairs
        except Exception as e:
            logger.error(f"Pairs error: {e}")
            return []

    # ── Data fetch ────────────────────────────────────────────

    async def fetch_day_trading_data(self, symbol):
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
            upper_band = hl2 + (multiplier * atr)
            lower_band = hl2 - (multiplier * atr)
            supertrend = [0] * len(df)
            for i in range(1, len(df)):
                if df['close'].iloc[i] > upper_band.iloc[i-1]:
                    supertrend[i] = lower_band.iloc[i]
                elif df['close'].iloc[i] < lower_band.iloc[i-1]:
                    supertrend[i] = upper_band.iloc[i]
                else:
                    supertrend[i] = supertrend[i-1]
            return pd.Series(supertrend, index=df.index)
        except:
            return pd.Series([0] * len(df), index=df.index)

    def calculate_advanced_indicators(self, df):
        try:
            if len(df) < 30:
                return df
            df['ema_9']  = ta.trend.EMAIndicator(df['close'], window=9).ema_indicator()
            df['ema_21'] = ta.trend.EMAIndicator(df['close'], window=21).ema_indicator()
            df['ema_50'] = ta.trend.EMAIndicator(df['close'], window=min(50, len(df)-1)).ema_indicator()
            df['supertrend'] = self.calculate_supertrend(df)
            psar = ta.trend.PSARIndicator(df['high'], df['low'], df['close'])
            df['psar'] = psar.psar()

            df['rsi']   = ta.momentum.RSIIndicator(df['close'], window=14).rsi()
            df['rsi_6'] = ta.momentum.RSIIndicator(df['close'], window=6).rsi()
            srsi = ta.momentum.StochRSIIndicator(df['close'])
            df['stoch_rsi_k'] = srsi.stochrsi_k()
            df['stoch_rsi_d'] = srsi.stochrsi_d()
            macd = ta.trend.MACD(df['close'])
            df['macd']        = macd.macd()
            df['macd_signal'] = macd.macd_signal()
            df['macd_hist']   = macd.macd_diff()
            stoch = ta.momentum.StochasticOscillator(df['high'], df['low'], df['close'])
            df['stoch_k']    = stoch.stoch()
            df['stoch_d']    = stoch.stoch_signal()
            df['williams_r'] = ta.momentum.WilliamsRIndicator(df['high'], df['low'], df['close']).williams_r()
            df['roc']        = ta.momentum.ROCIndicator(df['close'], window=12).roc()
            df['uo']         = ta.momentum.UltimateOscillator(df['high'], df['low'], df['close']).ultimate_oscillator()

            bb = ta.volatility.BollingerBands(df['close'], window=20, window_dev=2)
            df['bb_upper']  = bb.bollinger_hband()
            df['bb_middle'] = bb.bollinger_mavg()
            df['bb_lower']  = bb.bollinger_lband()
            df['bb_width']  = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
            df['bb_pband']  = bb.bollinger_pband()
            kc = ta.volatility.KeltnerChannel(df['high'], df['low'], df['close'])
            df['kc_upper'] = kc.keltner_channel_hband()
            df['kc_lower'] = kc.keltner_channel_lband()
            df['atr'] = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close']).average_true_range()
            dc = ta.volatility.DonchianChannel(df['high'], df['low'], df['close'])
            df['dc_upper'] = dc.donchian_channel_hband()
            df['dc_lower'] = dc.donchian_channel_lband()

            df['volume_sma']   = df['volume'].rolling(window=20).mean()
            df['volume_ratio'] = df['volume'] / df['volume_sma'].replace(0, np.nan)
            df['obv']     = ta.volume.OnBalanceVolumeIndicator(df['close'], df['volume']).on_balance_volume()
            df['obv_ema'] = df['obv'].ewm(span=20).mean()
            df['mfi']     = ta.volume.MFIIndicator(df['high'], df['low'], df['close'], df['volume']).money_flow_index()
            df['ad']      = ta.volume.AccDistIndexIndicator(df['high'], df['low'], df['close'], df['volume']).acc_dist_index()
            df['cmf']     = ta.volume.ChaikinMoneyFlowIndicator(df['high'], df['low'], df['close'], df['volume']).chaikin_money_flow()

            adx = ta.trend.ADXIndicator(df['high'], df['low'], df['close'])
            df['adx']      = adx.adx()
            df['di_plus']  = adx.adx_pos()
            df['di_minus'] = adx.adx_neg()
            df['cci'] = ta.trend.CCIIndicator(df['high'], df['low'], df['close']).cci()
            aroon = ta.trend.AroonIndicator(df['high'], df['low'])
            df['aroon_up']   = aroon.aroon_up()
            df['aroon_down'] = aroon.aroon_down()
            df['aroon_ind']  = df['aroon_up'] - df['aroon_down']

            tp_price = (df['high'] + df['low'] + df['close']) / 3
            df['vwap'] = (tp_price * df['volume']).cumsum() / df['volume'].cumsum()
            df['vwap'].fillna(df['close'], inplace=True)

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

    # ── Signal detection ──────────────────────────────────────

    def detect_signal(self, data, symbol):
        try:
            if not data or '1h' not in data:
                return None
            for tf in data:
                data[tf] = self.calculate_advanced_indicators(data[tf])

            df_1h  = data['1h']
            df_4h  = data['4h']
            df_15m = data['15m']

            if len(df_1h) < 50:
                return None

            latest_1h  = df_1h.iloc[-1]
            prev_1h    = df_1h.iloc[-2]
            latest_4h  = df_4h.iloc[-1]
            latest_15m = df_15m.iloc[-1]

            for col in ['ema_9','ema_21','rsi','macd','vwap','bb_pband']:
                if col not in latest_1h.index or pd.isna(latest_1h[col]):
                    return None

            volume_spike, vol_ratio = self.detect_volume_spike(df_1h)

            long_score  = 0
            short_score = 0
            max_score   = 35
            long_reasons  = []
            short_reasons = []

            # ── TREND (6 pts) ──
            if latest_4h['ema_9'] > latest_4h['ema_21'] > latest_4h['ema_50']:
                long_score += 3;  long_reasons.append('🔥 4H Uptrend')
            elif latest_4h['ema_9'] < latest_4h['ema_21'] < latest_4h['ema_50']:
                short_score += 3; short_reasons.append('🔥 4H Downtrend')

            if latest_1h['ema_9'] > latest_1h['ema_21']:
                long_score += 2;  long_reasons.append('1H Bullish EMA')
            elif latest_1h['ema_9'] < latest_1h['ema_21']:
                short_score += 2; short_reasons.append('1H Bearish EMA')

            if latest_1h['close'] > latest_1h['supertrend']:
                long_score += 1;  long_reasons.append('SuperTrend Bull')
            elif latest_1h['close'] < latest_1h['supertrend']:
                short_score += 1; short_reasons.append('SuperTrend Bear')

            # ── MOMENTUM (9 pts) ──
            rsi = latest_1h['rsi']
            if rsi < 30:
                long_score += 3.5; long_reasons.append(f'💎 RSI Oversold ({rsi:.0f})')
            elif rsi < 40:
                long_score += 2;   long_reasons.append(f'RSI Weak ({rsi:.0f})')
            elif 40 <= rsi <= 50:
                long_score += 1;   long_reasons.append(f'RSI Buy Zone ({rsi:.0f})')

            if rsi > 70:
                short_score += 3.5; short_reasons.append(f'💎 RSI Overbought ({rsi:.0f})')
            elif rsi > 60:
                short_score += 2;   short_reasons.append(f'RSI High ({rsi:.0f})')
            elif 50 <= rsi <= 60:
                short_score += 1;   short_reasons.append(f'RSI Sell Zone ({rsi:.0f})')

            sk = latest_1h['stoch_rsi_k']; sd = latest_1h['stoch_rsi_d']
            if sk < 0.2 and sk > sd:
                long_score += 2;  long_reasons.append('⚡ StochRSI Cross Up')
            elif sk > 0.8 and sk < sd:
                short_score += 2; short_reasons.append('⚡ StochRSI Cross Down')

            macd_cross_bull = latest_1h['macd'] > latest_1h['macd_signal'] and prev_1h['macd'] <= prev_1h['macd_signal']
            macd_cross_bear = latest_1h['macd'] < latest_1h['macd_signal'] and prev_1h['macd'] >= prev_1h['macd_signal']
            if macd_cross_bull:
                long_score += 2.5;  long_reasons.append('🎯 MACD Cross ↑')
            elif macd_cross_bear:
                short_score += 2.5; short_reasons.append('🎯 MACD Cross ↓')

            if latest_1h['uo'] < 30:
                long_score += 1.5;  long_reasons.append('UO Oversold')
            elif latest_1h['uo'] > 70:
                short_score += 1.5; short_reasons.append('UO Overbought')

            # ── VOLUME (5 pts) ──
            if volume_spike:
                if latest_1h['close'] > prev_1h['close']:
                    long_score += 3;  long_reasons.append(f'🚀 Vol Spike ({vol_ratio:.1f}x)')
                else:
                    short_score += 3; short_reasons.append(f'💥 Vol Dump ({vol_ratio:.1f}x)')

            if latest_1h['mfi'] < 20:
                long_score += 1.5;  long_reasons.append(f'MFI Oversold ({latest_1h["mfi"]:.0f})')
            elif latest_1h['mfi'] > 80:
                short_score += 1.5; short_reasons.append(f'MFI Overbought ({latest_1h["mfi"]:.0f})')

            if latest_1h['cmf'] > 0.15:
                long_score += 1;  long_reasons.append('CMF Buying')
            elif latest_1h['cmf'] < -0.15:
                short_score += 1; short_reasons.append('CMF Selling')

            obv_trend = df_1h['obv'].iloc[-5:].diff().mean()
            if obv_trend > 0 and latest_1h['obv'] > latest_1h['obv_ema']:
                long_score += 0.5;  long_reasons.append('OBV Accumulation')
            elif obv_trend < 0 and latest_1h['obv'] < latest_1h['obv_ema']:
                short_score += 0.5; short_reasons.append('OBV Distribution')

            # ── VOLATILITY (6 pts) ──
            if latest_1h['bb_pband'] < 0.1:
                long_score += 2.5;  long_reasons.append('💎 Lower BB Touch')
            elif latest_1h['bb_pband'] > 0.9:
                short_score += 2.5; short_reasons.append('💎 Upper BB Touch')

            if latest_1h['cci'] < -150:
                long_score += 1.5;  long_reasons.append('CCI Oversold')
            elif latest_1h['cci'] > 150:
                short_score += 1.5; short_reasons.append('CCI Overbought')

            if latest_1h['williams_r'] < -85:
                long_score += 1;  long_reasons.append('Williams Oversold')
            elif latest_1h['williams_r'] > -15:
                short_score += 1; short_reasons.append('Williams Overbought')

            if latest_1h['close'] < latest_1h['vwap'] * 0.98:
                long_score += 1;  long_reasons.append('Below VWAP')
            elif latest_1h['close'] > latest_1h['vwap'] * 1.02:
                short_score += 1; short_reasons.append('Above VWAP')

            # ── TREND STRENGTH (4 pts) ──
            if latest_1h['adx'] > 30:
                if latest_1h['di_plus'] > latest_1h['di_minus']:
                    long_score += 2;  long_reasons.append(f'ADX Strong Up ({latest_1h["adx"]:.0f})')
                else:
                    short_score += 2; short_reasons.append(f'ADX Strong Down ({latest_1h["adx"]:.0f})')
            elif latest_1h['adx'] > 25:
                if latest_1h['di_plus'] > latest_1h['di_minus']:
                    long_score += 1
                else:
                    short_score += 1

            if latest_1h['aroon_ind'] > 50:
                long_score += 1;  long_reasons.append('Aroon Bullish')
            elif latest_1h['aroon_ind'] < -50:
                short_score += 1; short_reasons.append('Aroon Bearish')

            if latest_1h['roc'] > 3:
                long_score += 1;  long_reasons.append('ROC Positive')
            elif latest_1h['roc'] < -3:
                short_score += 1; short_reasons.append('ROC Negative')

            # ── PATTERNS & DIVERGENCE (3 pts) ──
            if latest_1h['bullish_divergence'] == 1:
                long_score += 2;  long_reasons.append('🎯 Bullish Divergence')
            elif latest_1h['bearish_divergence'] == 1:
                short_score += 2; short_reasons.append('🎯 Bearish Divergence')

            if latest_15m['bullish_engulfing'] == 1:
                long_score += 1.5;  long_reasons.append('📊 Bullish Engulfing (15m)')
            elif latest_15m['bearish_engulfing'] == 1:
                short_score += 1.5; short_reasons.append('📊 Bearish Engulfing (15m)')

            # ── HTF (2 pts) ──
            if latest_4h['close'] > latest_4h['vwap']:
                long_score += 1
            else:
                short_score += 1

            if latest_4h['rsi'] < 50:
                long_score += 1
            elif latest_4h['rsi'] > 50:
                short_score += 1

            # ── DETERMINE SIGNAL ──
            min_threshold = max_score * MIN_SCORE_PCT
            signal = None
            if long_score > short_score and long_score >= min_threshold:
                signal = 'LONG';  score = long_score;  reasons = long_reasons
            elif short_score > long_score and short_score >= min_threshold:
                signal = 'SHORT'; score = short_score; reasons = short_reasons

            if not signal:
                return None

            # ── LONG trend filter (relaxed — any ONE of 5 checks) ──
            if signal == 'LONG' and USE_LONG_TREND_FILTER:
                trend_confirms = [
                    latest_4h['ema_9'] > latest_4h['ema_21'],              # 4H trend up
                    latest_1h['ema_9'] > latest_1h['ema_21'],              # 1H trend up
                    macd_cross_bull,                                        # MACD just crossed up
                    volume_spike and latest_1h['close'] > prev_1h['close'],# Volume spike up
                    latest_1h['rsi'] < 35,                                 # Deep oversold bounce
                ]
                if not any(trend_confirms):
                    self.stats['filtered_long'] += 1
                    return None

            # ── BTC regime (soft warning, hard block optional) ──
            regime_warning = ''
            if self.btc_regime:
                if signal == 'LONG' and self.btc_regime == 'BEAR':
                    if HARD_BLOCK_REGIME:
                        return None
                    regime_warning = ' ⚠️ Counter-regime'
                elif signal == 'SHORT' and self.btc_regime == 'BULL':
                    if HARD_BLOCK_REGIME:
                        return None
                    regime_warning = ' ⚠️ Counter-regime'

            # ── Quality ──
            pct = score / max_score
            if pct >= QUALITY_PREMIUM_PCT:
                quality = 'PREMIUM 💎'
            elif pct >= QUALITY_HIGH_PCT:
                quality = 'HIGH 🔥'
            else:
                quality = 'GOOD ✅'

            entry = latest_15m['close']
            atr   = latest_1h['atr']
            if pd.isna(atr) or atr == 0 or pd.isna(entry) or entry == 0:
                return None

            if signal == 'LONG':
                sl      = entry - (atr * ATR_SL_MULT)
                targets = [entry + atr*ATR_TP1_MULT, entry + atr*ATR_TP2_MULT, entry + atr*ATR_TP3_MULT]
            else:
                sl      = entry + (atr * ATR_SL_MULT)
                targets = [entry - atr*ATR_TP1_MULT, entry - atr*ATR_TP2_MULT, entry - atr*ATR_TP3_MULT]

            risk_pct = abs((sl - entry) / entry * 100)
            rr = [(abs(tp - entry) / abs(sl - entry)) for tp in targets]

            trade_id = f"{symbol.replace('/USDT:USDT','')}_{datetime.now().strftime('%Y%m%d%H%M%S')}"

            return {
                'trade_id':       trade_id,
                'symbol':         symbol.replace('/USDT:USDT', ''),
                'full_symbol':    symbol,
                'signal':         signal,
                'quality':        quality,
                'score':          score,
                'max_score':      max_score,
                'score_percent':  pct * 100,
                'entry':          entry,
                'stop_loss':      sl,
                'targets':        targets,
                'reward_ratios':  rr,
                'risk_percent':   risk_pct,
                'reasons':        reasons[:10],
                'regime_warning': regime_warning,
                'tp_hit':         [False, False, False],
                'sl_hit':         False,
                'timestamp':      datetime.now(),
                'status':         'ACTIVE',
                'btc_regime':     self.btc_regime or 'N/A',
            }

        except Exception as e:
            logger.error(f"Signal error {symbol}: {e}")
            return None

    # ── Format signal ─────────────────────────────────────────

    def format_signal(self, sig):
        emoji = '🚀' if sig['signal'] == 'LONG' else '🔻'
        regime_tag = f"[BTC:{sig['btc_regime']}]{sig.get('regime_warning','')}"

        msg  = f"{'='*42}\n"
        msg += f"{emoji} <b>DAY TRADE — {sig['quality']}</b> {emoji}\n"
        msg += f"{'='*42}\n\n"
        msg += f"<b>🆔</b> <code>{sig['trade_id']}</code>\n"
        msg += f"<b>📊 PAIR:</b> #{sig['symbol']}  <i>{regime_tag}</i>\n"
        msg += f"<b>📍 DIR:</b> <b>{sig['signal']}</b>\n"
        msg += f"<b>⭐ SCORE:</b> {sig['score']:.1f}/{sig['max_score']} ({sig['score_percent']:.0f}%)\n"
        filled = int(sig['score_percent'] / 10)
        msg += f"{'▰'*filled}{'▱'*(10-filled)}\n\n"

        msg += f"<b>💰 ENTRY:</b> ${sig['entry']:.6f}\n\n"
        msg += f"<b>🎯 TARGETS:</b>\n"
        durations = ['1-2h', '2-6h', '6-12h']
        for i, (tp, rr, dur) in enumerate(zip(sig['targets'], sig['reward_ratios'], durations), 1):
            pct_gain = abs((tp - sig['entry']) / sig['entry'] * 100)
            msg += f"  TP{i} ({dur}): ${tp:.6f}  +{pct_gain:.2f}%  [RR {rr:.1f}:1]\n"

        msg += f"\n<b>🛑 SL:</b> ${sig['stop_loss']:.6f}  (-{sig['risk_percent']:.2f}%)\n"
        msg += f"<i>💡 After TP1: move SL to entry (breakeven)</i>\n\n"

        msg += f"<b>✅ REASONS:</b>\n"
        for r in sig['reasons']:
            msg += f"  • {r}\n"

        msg += f"\n<b>📡 LIVE TRACKING ON</b> | v3.0\n"
        msg += f"<i>⏰ {sig['timestamp'].strftime('%H:%M:%S')}</i>\n"
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
        tp  = trade['targets'][tp_num - 1]
        pct = abs((tp - trade['entry']) / trade['entry'] * 100)

        msg  = f"{emoji} <b>TP{tp_num} HIT!</b> {emoji}\n\n"
        msg += f"<code>{trade['trade_id']}</code>\n"
        msg += f"<b>{trade['symbol']}</b> {trade['signal']}\n\n"
        msg += f"Target: ${tp:.6f}\n"
        msg += f"Profit: <b>+{pct:.2f}%</b>\n\n"

        if tp_num == 1:
            msg += "📋 <b>Action:</b> Take 40% profit\n"
            msg += "🔒 Move SL to <b>breakeven</b> (entry price)\n"
            msg += "Let rest ride to TP2 ✨"
        elif tp_num == 2:
            msg += "📋 <b>Action:</b> Take another 40% profit\n"
            msg += "Let last 20% ride to TP3 🎯"
        else:
            msg += "📋 <b>Action:</b> Close remaining 20%\n"
            msg += "🎊 <b>FULL TRADE COMPLETE!</b>"

        await self.send_msg(msg)
        self.stats[f'tp{tp_num}_hits'] += 1

    async def send_sl_alert(self, trade, price):
        loss = abs((price - trade['entry']) / trade['entry'] * 100)
        msg  = f"⚠️ <b>STOP LOSS HIT</b> ⚠️\n\n"
        msg += f"<code>{trade['trade_id']}</code>\n"
        msg += f"{trade['symbol']} {trade['signal']}\n\n"
        msg += f"Entry:  ${trade['entry']:.6f}\n"
        msg += f"SL:     ${trade['stop_loss']:.6f}\n"
        msg += f"Price:  ${price:.6f}\n"
        msg += f"Loss:   <b>-{loss:.2f}%</b>"
        await self.send_msg(msg)
        self.stats['sl_hits'] += 1

    # ── Trade tracker ─────────────────────────────────────────

    async def track_trades(self):
        logger.info("📡 Trade tracker started")
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
                                f"{trade['symbol']} — please close manually!"
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
                logger.error(f"Tracker loop error: {e}")
                await asyncio.sleep(60)

    # ── Main scanner ──────────────────────────────────────────

    async def scan_all(self):
        if self.is_scanning:
            logger.info("⚠️ Scan in progress, skipping...")
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
                data = await self.fetch_day_trading_data(pair)
                if data:
                    sig = self.detect_signal(data, pair)
                    if sig:
                        signals.append(sig)
                        self.signal_history.append(sig)
                        self.stats['total_signals'] += 1
                        if sig['signal'] == 'LONG':
                            self.stats['long_signals'] += 1
                        else:
                            self.stats['short_signals'] += 1
                        if 'PREMIUM' in sig['quality']:
                            self.stats['premium_signals'] += 1
                        elif 'HIGH' in sig['quality']:
                            self.stats['high_signals'] += 1
                        self.active_trades[sig['trade_id']] = sig
                        await self.send_msg(self.format_signal(sig))
                        await asyncio.sleep(1.5)

                scanned += 1
                if scanned % 30 == 0:
                    logger.info(f"  📈 Progress: {scanned}/{len(pairs)}")
                await asyncio.sleep(0.5)

            except Exception as e:
                logger.error(f"❌ {pair}: {e}")

        self.stats['last_scan_time'] = datetime.now()
        self.stats['pairs_scanned']  = scanned

        longs   = sum(1 for s in signals if s['signal'] == 'LONG')
        shorts  = len(signals) - longs
        premium = sum(1 for s in signals if 'PREMIUM' in s['quality'])
        high    = sum(1 for s in signals if 'HIGH' in s['quality'])
        counter = sum(1 for s in signals if s.get('regime_warning'))

        summary  = f"✅ <b>SCAN COMPLETE</b> — v3.0\n\n"
        summary += f"📊 Scanned: {scanned} pairs\n"
        summary += f"🌍 BTC Regime: <b>{self.btc_regime or '?'}</b>\n\n"
        summary += f"🎯 Signals: <b>{len(signals)}</b>\n"
        summary += f"  🟢 Long:  {longs}\n"
        summary += f"  🔴 Short: {shorts}\n"
        summary += f"  💎 Premium: {premium}\n"
        summary += f"  🔥 High:    {high}\n"
        if counter:
            summary += f"  ⚠️ Counter-regime: {counter}\n"
        summary += f"\n⚡ Filtered weak longs: {self.stats['filtered_long']}\n"
        summary += f"📡 Active trades: {len(self.active_trades)}\n"
        summary += f"⏰ {datetime.now().strftime('%H:%M:%S')}"
        await self.send_msg(summary)

        logger.info(f"✅ Scan done: {len(signals)} signals / {scanned} pairs")
        self.is_scanning = False
        return signals

    async def run(self, interval=SCAN_INTERVAL_MIN):
        logger.info("🚀 ADVANCED DAY TRADING SCANNER v3.0 STARTING")

        welcome  = "🔥 <b>ADVANCED DAY TRADING SCANNER v3.0</b> 🔥\n\n"
        welcome += "<b>What's new in v3:</b>\n"
        welcome += "✅ ALL USDT pairs scanned\n"
        welcome += "✅ Tighter TPs: 0.6/1.0/1.5x ATR\n"
        welcome += "✅ Smart LONG filter (relaxed)\n"
        welcome += "✅ BTC regime warning (not block)\n"
        welcome += "✅ Breakeven SL tip after TP1\n"
        welcome += "✅ 25+ indicators | 35pt scoring\n\n"
        welcome += f"⏱ Scanning every <b>{interval} min</b>\n\n"
        welcome += "<b>Commands:</b>\n"
        welcome += "/scan — Force scan now\n"
        welcome += "/stats — Live statistics\n"
        welcome += "/trades — Active trades\n"
        welcome += "/regime — BTC market regime\n"
        welcome += "/help — Full help\n\n"
        welcome += "🎯 <i>Built with backtest data. Let's get it.</i>"
        await self.send_msg(welcome)

        asyncio.create_task(self.track_trades())

        while True:
            try:
                await self.scan_all()
                logger.info(f"💤 Sleeping {interval} min")
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
        msg  = "🚀 <b>Advanced Day Trading Scanner v3.0</b>\n\n"
        msg += "Backtest-optimized. Scans ALL USDT pairs.\n\n"
        msg += "/scan — Force scan\n/stats — Stats\n/trades — Active\n/regime — BTC regime\n/help — Help"
        await update.message.reply_text(msg, parse_mode=ParseMode.HTML)

    async def cmd_scan(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if self.scanner.is_scanning:
            await update.message.reply_text("⚠️ Scan already running! Wait for it to finish.")
            return
        await update.message.reply_text("🔍 Starting full scan of ALL USDT pairs...")
        asyncio.create_task(self.scanner.scan_all())

    async def cmd_stats(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        s   = self.scanner.stats
        tot = s['total_signals']
        tp1 = s['tp1_hits']; tp2 = s['tp2_hits']; tp3 = s['tp3_hits']; sl = s['sl_hits']
        completed = tp1 + sl
        wr = round(tp1 / completed * 100, 1) if completed > 0 else 0

        msg  = f"📊 <b>LIVE STATS — v3.0</b>\n\n"
        msg += f"<b>Signals Generated:</b>\n"
        msg += f"  Total:   {tot}\n"
        msg += f"  Long:    {s['long_signals']} 🟢\n"
        msg += f"  Short:   {s['short_signals']} 🔴\n"
        msg += f"  Premium: {s['premium_signals']} 💎\n"
        msg += f"  High:    {s['high_signals']} 🔥\n"
        msg += f"  Filtered longs: {s['filtered_long']} ⚡\n\n"
        msg += f"<b>Outcomes:</b>\n"
        msg += f"  TP1: {tp1} 🎯  TP2: {tp2} 🎯  TP3: {tp3} 🎯\n"
        msg += f"  SL:  {sl} ❌\n"
        msg += f"  Live Win Rate: <b>{wr}%</b>\n\n"
        msg += f"<b>Market:</b>\n"
        msg += f"  BTC Regime: {self.scanner.btc_regime or 'Unknown'}\n"
        msg += f"  Tracking:   {len(self.scanner.active_trades)} trades\n"
        if s['last_scan_time']:
            msg += f"\nLast scan: {s['last_scan_time'].strftime('%H:%M:%S')}\n"
            msg += f"Pairs scanned: {s['pairs_scanned']}"
        await update.message.reply_text(msg, parse_mode=ParseMode.HTML)

    async def cmd_trades(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        trades = self.scanner.active_trades
        if not trades:
            await update.message.reply_text("📭 No active trades right now.")
            return
        msg = f"📡 <b>ACTIVE TRADES ({len(trades)})</b>\n\n"
        for tid, t in list(trades.items())[:10]:
            age = int((datetime.now() - t['timestamp']).total_seconds() / 3600)
            tps = ''.join(['✅' if h else '⏳' for h in t['tp_hit']])
            warn = t.get('regime_warning', '')
            msg += f"<b>{t['symbol']}</b> {t['signal']} {t['quality']}{warn}\n"
            msg += f"  {tps} | {age}h | {t['score_percent']:.0f}%\n"
            msg += f"  Entry: ${t['entry']:.6f}\n\n"
        await update.message.reply_text(msg, parse_mode=ParseMode.HTML)

    async def cmd_regime(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        regime = self.scanner.btc_regime or 'Unknown'
        emoji  = '🐂' if regime == 'BULL' else '🐻' if regime == 'BEAR' else '❓'
        msg  = f"{emoji} <b>BTC Market Regime: {regime}</b>\n\n"
        if regime == 'BULL':
            msg += "BTC is <b>above</b> 4H EMA21\n"
            msg += "✅ LONG signals are with the trend\n"
            msg += "⚠️ SHORT signals flagged as counter-regime"
        elif regime == 'BEAR':
            msg += "BTC is <b>below</b> 4H EMA21\n"
            msg += "✅ SHORT signals are with the trend\n"
            msg += "⚠️ LONG signals flagged as counter-regime"
        else:
            msg += "Regime not yet determined — run /scan first"
        await update.message.reply_text(msg, parse_mode=ParseMode.HTML)

    async def cmd_help(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        msg  = "📚 <b>SCANNER v3.0 — FULL GUIDE</b>\n\n"
        msg += "<b>TP Strategy (v3):</b>\n"
        msg += "  TP1: 0.6x ATR — fast, near-certain\n"
        msg += "  TP2: 1.0x ATR — realistic target\n"
        msg += "  TP3: 1.5x ATR — extended run\n"
        msg += "  SL:  1.5x ATR — tight protection\n\n"
        msg += "<b>Position Management:</b>\n"
        msg += "  TP1 hit → Take 40%, move SL to entry\n"
        msg += "  TP2 hit → Take 40%\n"
        msg += "  TP3 hit → Take final 20%\n\n"
        msg += "<b>Signal Quality:</b>\n"
        msg += "  💎 PREMIUM (65%+) — max conviction\n"
        msg += "  🔥 HIGH (55%+) — strong setup\n"
        msg += "  ✅ GOOD (51%+) — valid signal\n\n"
        msg += "<b>Regime Warning ⚠️:</b>\n"
        msg += "  Signal goes against BTC trend\n"
        msg += "  Still valid — just trade smaller size\n\n"
        msg += "<b>Indicators used (35pt system):</b>\n"
        msg += "  Trend: EMA9/21/50, SuperTrend, PSAR, Aroon\n"
        msg += "  Momentum: RSI, MACD, StochRSI, UO, Williams\n"
        msg += "  Volume: OBV, MFI, CMF, A/D Line\n"
        msg += "  Volatility: BB, Keltner, Donchian, ATR\n"
        msg += "  Patterns: Engulfing, Divergence, VWAP\n\n"
        msg += "<b>Commands:</b>\n"
        msg += "/scan /stats /trades /regime /help"
        await update.message.reply_text(msg, parse_mode=ParseMode.HTML)


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

async def main():
    # ════════════════════════════════════
    #   YOUR CREDENTIALS
    # ════════════════════════════════════
    TELEGRAM_TOKEN   = "7957028587:AAE7aSYtE4hCxxTIPkAs_1ULJ9e8alkY6Ic"
    TELEGRAM_CHAT_ID = "-1002442074724"
    BINANCE_API_KEY  = None   # Not needed for public data
    BINANCE_SECRET   = None

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
    logger.info("🤖 Bot v3.0 ready!")

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
