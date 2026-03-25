"""
HYBRID FVG PULLBACK SCANNER v1.0 - Futures + Volume Filter
"""

import asyncio
import ccxt.async_support as ccxt
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dataclasses import dataclass
from telegram import Bot, Update
from telegram.ext import Application, CommandHandler, ContextTypes
from telegram.constants import ParseMode
import logging
from collections import deque

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ========================= YOUR CREDENTIALS =========================
TELEGRAM_TOKEN = "7957028587:AAE7aSYtE4hCxxTIPkAs_1ULJ9e8alkY6Ic"
TELEGRAM_CHAT_ID = "-1003659830260"
# =================================================================

PAIRS = []

SCAN_INTERVAL_MIN = 15
MAX_TRADE_HOURS = 72

EMA_PERIOD = 20
FVG_THRESHOLD_ATR = 0.28
MIN_WICK_ATR = 0.30
OB_BUF_ATR = 0.55
MIN_VOLUME_MULT = 1.15

ATR_SL_BUF = 0.35
TP1_R = 1.0
TP2_R = 2.0
TP1_PCT = 0.50
ATR_PERIOD = 14

bot = Bot(token=TELEGRAM_TOKEN)
exchange = None
active_trades = {}
signal_history = deque(maxlen=500)
stats = {
    'total_signals': 0, 'long_signals': 0, 'short_signals': 0,
    'tp1_hits': 0, 'tp2_hits': 0, 'sl_hits': 0, 'be_saves': 0,
    'session_start': datetime.now()
}

@dataclass
class LiveTrade:
    symbol: str
    direction: str
    entry: float
    sl: float
    tp1: float
    tp2: float
    entry_time: datetime
    tp1_hit: bool = False
    be_active: bool = False

# ===================== LOAD FUTURES PAIRS =====================

async def load_usdt_pairs():
    global exchange, PAIRS

    if exchange is None:
        exchange = ccxt.binance({
            'enableRateLimit': True,
            'options': {'defaultType': 'future'}
        })

    markets = await exchange.load_markets()

    PAIRS = []
    
    for symbol, data in markets.items():
        try:
            if (
                symbol.endswith('/USDT') and
                data['active'] and
                data.get('linear', False)
            ):
                ticker = await exchange.fetch_ticker(symbol)
                vol = ticker.get('quoteVolume', 0)

                if vol and vol >= 500000:
                    PAIRS.append(symbol)

                await asyncio.sleep(0.1)

        except:
            continue

    logger.info(f"Loaded {len(PAIRS)} FUTURES USDT pairs (vol ≥ 500k)")

# ===================== INDICATORS =====================

def calc_atr(H, L, C, period=14):
    n = len(C)
    tr = np.zeros(n)
    tr[0] = H[0] - L[0]
    for i in range(1, n):
        tr[i] = max(H[i]-L[i], abs(H[i]-C[i-1]), abs(L[i]-C[i-1]))
    atr = np.zeros(n)
    if n >= period:
        atr[period-1] = np.mean(tr[:period])
        for i in range(period, n):
            atr[i] = (atr[i-1]*(period-1) + tr[i]) / period
        for i in range(period-1):
            atr[i] = atr[period-1]
    return atr

def calc_ema(C, period=20):
    return pd.Series(C).ewm(span=period, adjust=False).mean().values

def detect_fvg(H, L, C, i):
    if i < 2: return None, 0.0
    if L[i] > H[i-2]: return 'bull', L[i] - H[i-2]
    if H[i] < L[i-2]: return 'bear', L[i-2] - H[i]
    return None, 0.0

# ===================== DATA =====================

async def fetch_ohlcv(symbol, tf, limit):
    global exchange
    if exchange is None:
        exchange = ccxt.binance({'enableRateLimit': True, 'options': {'defaultType': 'future'}})
    try:
        ohlcv = await exchange.fetch_ohlcv(symbol, tf, limit=limit)
        df = pd.DataFrame(ohlcv, columns=['timestamp','open','high','low','close','volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        return df
    except Exception as e:
        logger.error(f"Fetch {symbol} {tf}: {e}")
        return None

# ===================== SIGNAL =====================

def build_signal(symbol, direction, entry, sl, tp1, tp2, trend4, fvg_size, entry_type):
    e = "🚀" if direction == "LONG" else "🔻"
    m = f"{'─'*38}\n{e} <b>{direction} — FVG Pullback</b>\n{'─'*38}\n\n"
    m += f"<b>Pair:</b> #{symbol}\n"
    m += f"<b>4H Trend:</b> {trend4.upper()}\n"
    m += f"<b>FVG:</b> {fvg_size:.4f} ATR\n"
    m += f"<b>Entry:</b> {entry_type}\n\n"
    m += f"<b>Entry:</b> <code>{entry:.6f}</code>\n"
    m += f"<b>TP1:</b> <code>{tp1:.6f}</code>\n"
    m += f"<b>TP2:</b> <code>{tp2:.6f}</code>\n"
    m += f"<b>SL:</b> <code>{sl:.6f}</code>\n\n"
    m += f"📋 Close 50% at TP1 → SL to BE\n"
    m += f"🎯 Runner remaining to TP2\n\n"
    m += f"<i>⏰ {datetime.now().strftime('%H:%M UTC')} | Hybrid Scanner</i>"
    return m

# ===================== SCANNER =====================

async def scan_all():
    global exchange

    for symbol in PAIRS:
        try:
            df4h = await fetch_ohlcv(symbol, '4h', 300)
            df15m = await fetch_ohlcv(symbol, '15m', 400)
            if df4h is None or df15m is None: continue

            H4 = df4h['high'].values.astype(float)
            C4 = df4h['close'].values.astype(float)
            EMA4 = calc_ema(C4, EMA_PERIOD)

            H15 = df15m['high'].values.astype(float)
            L15 = df15m['low'].values.astype(float)
            C15 = df15m['close'].values.astype(float)
            O15 = df15m['open'].values.astype(float)
            V15 = df15m['volume'].values.astype(float)
            ATR15 = calc_atr(H15, L15, C15, ATR_PERIOD)

            b15 = len(H15) - 1
            b4 = len(H4) - 1

            trend4 = 'uptrend' if C4[b4] > EMA4[b4] else 'downtrend' if C4[b4] < EMA4[b4] else 'range'
            if trend4 == 'range': continue

            fvg_dir, fvg_size = detect_fvg(H15, L15, C15, b15)
            if not fvg_dir or fvg_size < FVG_THRESHOLD_ATR * ATR15[b15]: continue

            avg_vol = np.mean(V15[max(0, b15-20):b15+1])
            if V15[b15] < avg_vol * MIN_VOLUME_MULT: continue

            ema_dist = abs(C15[b15] - EMA4[b4]) / ATR15[b15] if ATR15[b15] > 0 else 999
            if ema_dist > 2.0: continue

            if (trend4 == 'uptrend' and fvg_dir != 'bull') or (trend4 == 'downtrend' and fvg_dir != 'bear'):
                continue

            i = b15
            atr = ATR15[i]
            o, h, l, c = O15[i], H15[i], L15[i], C15[i]
            rng = h - l
            valid = False
            entry_type = ""

            if trend4 == 'uptrend' and c > o and (c - l) / max(rng, 1e-9) > 0.55:
                valid = True
                entry_type = "Bullish Rejection"
            elif trend4 == 'downtrend' and c < o and (h - c) / max(rng, 1e-9) > 0.55:
                valid = True
                entry_type = "Bearish Rejection"

            if not valid: continue

            entry = c
            if trend4 == 'uptrend':
                sl = l - atr * ATR_SL_BUF
                direction = 'LONG'
            else:
                sl = h + atr * ATR_SL_BUF
                direction = 'SHORT'

            risk = abs(entry - sl)
            if risk < 1e-9: continue

            tp1 = entry + risk * TP1_R if direction == 'LONG' else entry - risk * TP1_R
            tp2 = entry + risk * TP2_R if direction == 'LONG' else entry - risk * TP2_R

            signal_text = build_signal(symbol.replace('/USDT',''), direction, entry, sl, tp1, tp2, trend4, fvg_size, entry_type)

            await bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=signal_text, parse_mode=ParseMode.HTML)

            tid = f"{symbol.replace('/USDT','')}_{direction[0]}_{datetime.now().strftime('%H%M%S')}"
            active_trades[tid] = LiveTrade(symbol=symbol.replace('/USDT',''), direction=direction,
                                           entry=entry, sl=sl, tp1=tp1, tp2=tp2, entry_time=datetime.now())

            stats['total_signals'] += 1
            if direction == 'LONG': stats['long_signals'] += 1
            else: stats['short_signals'] += 1

            await asyncio.sleep(1.2)

        except Exception as e:
            logger.error(f"Scan error {symbol}: {e}")

    stats['last_scan'] = datetime.now()

# ===================== MAIN =====================

async def main():
    logger.info("🚀 Hybrid FVG Scanner Started")

    app = Application.builder().token(TELEGRAM_TOKEN).build()

    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("stats", cmd_stats))
    app.add_handler(CommandHandler("trades", cmd_trades))
    app.add_handler(CommandHandler("scan", cmd_scan))

    await app.initialize()
    await app.start()

    await load_usdt_pairs()  # 👈 IMPORTANT

    asyncio.create_task(track_trades())
    asyncio.create_task(main_loop())

    await asyncio.Event().wait()

async def main_loop():
    while True:
        await scan_all()
        await asyncio.sleep(SCAN_INTERVAL_MIN * 60)

if __name__ == '__main__':
    asyncio.run(main())
