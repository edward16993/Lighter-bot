import os, asyncio, logging, json
import numpy as np
from datetime import datetime
import pandas as pd
import httpx
from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes
import lighter

logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

TELEGRAM_BOT_TOKEN  = os.environ["TELEGRAM_BOT_TOKEN"]
TELEGRAM_CHAT_ID    = os.environ["TELEGRAM_CHAT_ID"]
ACCOUNT_INDEX       = int(os.environ["ACCOUNT_INDEX"])
API_KEY_INDEX       = int(os.environ["API_KEY_INDEX"])
LIGHTER_PRIVATE_KEY = os.environ["LIGHTER_PRIVATE_KEY"]

BASE_URL       = "https://mainnet.zklighter.elliot.ai"
LEVERAGE       = 5
CHECK_INTERVAL = 300
SL_ATR_MULT    = float(os.environ.get("SL_MULT", "2.0"))
TP_ATR_MULT    = float(os.environ.get("TP_MULT", "3.0"))
ADX_THRESHOLD  = 20.0   # ADX>20 = Trending = Trade!
MIN_MARGIN     = 0.50
MARKET_INDEX   = 0
SYMBOL         = "ETH-USDT-SWAP"
STATS_FILE     = "eth_stats.json"
START_MARGIN   = float(os.environ.get("ETH_MARGIN", "50"))
DECIMALS       = 3
MIN_SIZE       = 0.002

position      = None  # None / "LONG" / "SHORT"
entry_price   = 0.0
entry_size    = 0.0
entry_margin  = 0.0
sl_price      = 0.0
tp_price      = 0.0
stats         = {}
signer_client = None
tg_app        = None

def load_stats():
    try:
        if os.path.exists(STATS_FILE):
            with open(STATS_FILE) as f:
                return json.load(f)
    except Exception:
        pass
    return {
        "current_margin": START_MARGIN, "total_trades": 0,
        "wins": 0, "losses": 0, "total_pnl": 0.0,
        "peak_margin": START_MARGIN, "long_trades": 0, "short_trades": 0,
        "entry_price": 0.0, "entry_size": 0.0, "entry_margin": 0.0,
        "sl_price": 0.0, "tp_price": 0.0, "history": []
    }

def save_stats():
    with open(STATS_FILE, "w") as f:
        json.dump(stats, f, indent=2)

def calc_alma(series, period, sigma=0.85, offset=0.85):
    m = offset * (period - 1)
    s = period / sigma
    weights = np.array([np.exp(-((i - m)**2) / (2 * s**2)) for i in range(period)])
    weights /= weights.sum()
    result = pd.Series(np.nan, index=series.index)
    for i in range(period-1, len(series)):
        result.iloc[i] = np.dot(weights, series.iloc[i-period+1:i+1].values)
    return result

def calc_atr(df, period=10):
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - df["close"].shift()).abs(),
        (df["low"]  - df["close"].shift()).abs()
    ], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()

def calc_adx(df, period=14):
    tr    = pd.concat([
        df["high"] - df["low"],
        (df["high"] - df["close"].shift()).abs(),
        (df["low"]  - df["close"].shift()).abs()
    ], axis=1).max(axis=1).ewm(span=period, adjust=False).mean()
    dm_p  = (df["high"].diff()).clip(lower=0)
    dm_m  = (-df["low"].diff()).clip(lower=0)
    dm_p  = dm_p.where(dm_p > dm_m, 0)
    dm_m  = dm_m.where(dm_m > dm_p, 0)
    di_p  = 100 * dm_p.ewm(span=period, adjust=False).mean() / tr.replace(0, np.nan)
    di_m  = 100 * dm_m.ewm(span=period, adjust=False).mean() / tr.replace(0, np.nan)
    dx    = 100 * (di_p - di_m).abs() / (di_p + di_m).replace(0, np.nan)
    return dx.ewm(span=period, adjust=False).mean()

def calc_rsi(series, period=14):
    delta = series.diff()
    gain  = delta.clip(lower=0).ewm(span=period, adjust=False).mean()
    loss  = (-delta.clip(upper=0)).ewm(span=period, adjust=False).mean()
    rs    = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def calc_indicators(df):
    df = df.copy()
    df["alma_fast"] = calc_alma(df["close"], 13)
    df["alma_slow"] = calc_alma(df["close"], 21)
    df["ema200"]    = df["close"].ewm(span=200, adjust=False).mean()
    df["atr"]       = calc_atr(df, 10)
    df["rsi"]       = calc_rsi(df["close"], 14)
    df["adx"]       = calc_adx(df, 14)
    return df

async def fetch_candles(limit=250):
    async with httpx.AsyncClient(timeout=15) as c:
        r = await c.get(
            "https://www.okx.com/api/v5/market/candles",
            params={"instId": SYMBOL, "bar": "5m", "limit": str(limit)}
        )
        data = r.json()
        if data.get("code") != "0" or not data.get("data"):
            raise Exception(f"OKX error: {data}")
        rows = list(reversed(data["data"]))
        df = pd.DataFrame(rows, columns=["time","open","high","low","close","vol","volCcy","volCcyQuote","confirm"])
        for col in ["high", "low", "close"]:
            df[col] = df[col].astype(float)
        return df

async def place_order(side, size, price, reduce_only=False):
    order_price = int(price * (1.005 if side == "BUY" else 0.995) * 100)
    base_amt    = int(size * 10000)
    logger.info(f"Order: {side} {size} ETH @ ~{price:.2f}")
    tx, tx_hash, err = await signer_client.create_order(
        market_index=MARKET_INDEX,
        client_order_index=int(datetime.now().timestamp()),
        base_amount=base_amt,
        price=order_price,
        is_ask=(side == "SELL"),
        order_type=signer_client.ORDER_TYPE_MARKET,
        time_in_force=signer_client.ORDER_TIME_IN_FORCE_IMMEDIATE_OR_CANCEL,
        reduce_only=reduce_only,
        order_expiry=signer_client.DEFAULT_IOC_EXPIRY,
    )
    if err:
        raise Exception(str(err))
    return tx_hash

def calc_size(margin, price):
    size = (margin * LEVERAGE) / price
    size = max(size, MIN_SIZE)
    return round(size, DECIMALS)

def record_close(exit_price, reason, side):
    global stats
    if side == "LONG":
        pnl = round((exit_price - stats["entry_price"]) * stats["entry_size"], 4)
    else:
        pnl = round((stats["entry_price"] - exit_price) * stats["entry_size"], 4)
    new_m = round(max(stats["entry_margin"] + pnl, MIN_MARGIN), 4)
    stats["total_trades"] += 1
    stats["total_pnl"]    += round(pnl, 4)
    stats["current_margin"] = new_m
    if new_m > stats["peak_margin"]:
        stats["peak_margin"] = new_m
    result = "✅ WIN" if pnl >= 0 else "❌ LOSS"
    if pnl >= 0: stats["wins"] += 1
    else:        stats["losses"] += 1
    stats["history"].append({
        "no": stats["total_trades"], "side": side,
        "entry": stats["entry_price"], "exit": exit_price,
        "pnl": pnl, "reason": reason,
        "old_margin": stats["entry_margin"], "new_margin": new_m,
        "result": result, "time": datetime.now().strftime("%Y-%m-%d %H:%M")
    })
    stats["history"] = stats["history"][-20:]
    save_stats()
    return pnl, new_m, result

async def send_tg(msg):
    try:
        await tg_app.bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=msg, parse_mode="Markdown")
    except Exception as e:
        logger.error(f"TG: {e}")

async def strategy_loop():
    global position, entry_price, entry_size, entry_margin, sl_price, tp_price, stats

    try:
        api = lighter.ApiClient(lighter.Configuration(host=BASE_URL))
        acc = lighter.AccountApi(api)
        r   = await acc.account(str(ACCOUNT_INDEX))
        await api.close()
        for pos in (getattr(r, "positions", []) or []):
            if getattr(pos, "market_index", -1) == MARKET_INDEX:
                size = float(getattr(pos, "base_amount", 0) or 0)
                if size != 0:
                    position     = "LONG" if size > 0 else "SHORT"
                    entry_price  = float(stats.get("entry_price", 0))
                    entry_size   = abs(size)
                    entry_margin = float(stats.get("entry_margin", 0))
                    sl_price     = float(stats.get("sl_price", 0))
                    tp_price     = float(stats.get("tp_price", 0))
                    logger.info(f"ETH {position} position restored!")
    except Exception as e:
        logger.error(f"Startup check: {e}")

    await send_tg(
        f"🤖 *ALMA Long+Short Bot*\n"
        f"💰 Margin:`${stats['current_margin']}` | {position or '⚪ Wait'}\n"
        f"📈 Long: ALMA(13/21) cross + EMA200 + ADX>20\n"
        f"📉 Short: ALMA(13/21) cross + RSI<50 + ADX>20\n"
        f"⚡ {LEVERAGE}x | 5min | SL:{SL_ATR_MULT}x TP:{TP_ATR_MULT}x"
    )

    while True:
        try:
            df   = await fetch_candles(300)
            df   = calc_indicators(df)
            df   = df.dropna()
            curr = df.iloc[-1]
            prev = df.iloc[-2]
            price = curr["close"]
            atr   = curr["atr"]
            rsi   = curr["rsi"]
            adx   = curr["adx"]

            bull_cross   = (curr["alma_fast"] > curr["alma_slow"]) and (prev["alma_fast"] <= prev["alma_slow"])
            bear_cross   = (curr["alma_fast"] < curr["alma_slow"]) and (prev["alma_fast"] >= prev["alma_slow"])
            bull_trend   = price > curr["ema200"]
            adx_trending = adx > ADX_THRESHOLD
            long_signal  = bull_cross and bull_trend and adx_trending
            short_signal = bear_cross and (rsi < 50) and adx_trending

            logger.info(
                f"ETH=${price:.2f} RSI={rsi:.1f} ADX={adx:.1f} "
                f"Bull={bull_trend} Trending={adx_trending} Pos={position}"
            )

            # ── NO POSITION ──
            if position is None:
                if long_signal and stats["current_margin"] >= MIN_MARGIN:
                    margin = stats["current_margin"]
                    size   = calc_size(margin, price)
                    new_sl = round(price - (SL_ATR_MULT * atr), 2)
                    new_tp = round(price + (TP_ATR_MULT * atr), 2)
                    sl_pct = round((price - new_sl) / price * 100, 2)
                    tp_pct = round((new_tp - price) / price * 100, 2)
                    await send_tg(
                        f"📈 *ETH LONG Signal!*\n"
                        f"ALMA cross + EMA200 + ADX>{ADX_THRESHOLD:.0f} 🟢\n"
                        f"Price:`${price:.2f}` RSI:`{rsi:.1f}` ADX:`{adx:.1f}`\n"
                        f"🛡️ SL:`${new_sl}` (-{sl_pct}%)\n"
                        f"🎯 TP:`${new_tp}` (+{tp_pct}%)\n"
                        f"Size:`{size} ETH` | Margin:`${margin}` x {LEVERAGE}x"
                    )
                    try:
                        await place_order("BUY", size, price)
                        position     = "LONG"
                        entry_price  = price
                        entry_size   = size
                        entry_margin = margin
                        sl_price     = new_sl
                        tp_price     = new_tp
                        stats["entry_price"]   = price
                        stats["entry_size"]    = size
                        stats["entry_margin"]  = margin
                        stats["sl_price"]      = new_sl
                        stats["tp_price"]      = new_tp
                        stats["long_trades"]   = stats.get("long_trades", 0) + 1
                        save_stats()
                        await send_tg(
                            f"✅ *ETH LONG Opened!*\n"
                            f"Entry:`${price:.2f}` | Size:`{size} ETH`\n"
                            f"🛡️ SL:`${new_sl}` | 🎯 TP:`${new_tp}`\n"
                            f"💰 Margin:`${margin}`"
                        )
                    except Exception as e:
                        position = None
                        await send_tg(f"❌ ETH LONG Failed: `{e}`")

                elif short_signal and stats["current_margin"] >= MIN_MARGIN:
                    margin = stats["current_margin"]
                    size   = calc_size(margin, price)
                    new_sl = round(price + (SL_ATR_MULT * atr), 2)
                    new_tp = round(price - (TP_ATR_MULT * atr), 2)
                    sl_pct = round((new_sl - price) / price * 100, 2)
                    tp_pct = round((price - new_tp) / price * 100, 2)
                    await send_tg(
                        f"📉 *ETH SHORT Signal!*\n"
                        f"ALMA cross + RSI<50 + ADX>{ADX_THRESHOLD:.0f} 🔴\n"
                        f"Price:`${price:.2f}` RSI:`{rsi:.1f}` ADX:`{adx:.1f}`\n"
                        f"🛡️ SL:`${new_sl}` (+{sl_pct}%)\n"
                        f"🎯 TP:`${new_tp}` (-{tp_pct}%)\n"
                        f"Size:`{size} ETH` | Margin:`${margin}` x {LEVERAGE}x"
                    )
                    try:
                        await place_order("SELL", size, price)
                        position     = "SHORT"
                        entry_price  = price
                        entry_size   = size
                        entry_margin = margin
                        sl_price     = new_sl
                        tp_price     = new_tp
                        stats["entry_price"]   = price
                        stats["entry_size"]    = size
                        stats["entry_margin"]  = margin
                        stats["sl_price"]      = new_sl
                        stats["tp_price"]      = new_tp
                        stats["short_trades"]  = stats.get("short_trades", 0) + 1
                        save_stats()
                        await send_tg(
                            f"✅ *ETH SHORT Opened!*\n"
                            f"Entry:`${price:.2f}` | Size:`{size} ETH`\n"
                            f"🛡️ SL:`${new_sl}` | 🎯 TP:`${new_tp}`\n"
                            f"💰 Margin:`${margin}`"
                        )
                    except Exception as e:
                        position = None
                        await send_tg(f"❌ ETH SHORT Failed: `{e}`")

            # ── LONG POSITION ──
            elif position == "LONG":
                unrealized = round((price - entry_price) * entry_size, 4)
                reason = None
                if price >= tp_price:   reason = "TP"
                elif price <= sl_price: reason = "SL"
                elif bear_cross:        reason = "Cross"
                if reason:
                    if reason in ["TP","SL"]:
                        await send_tg(
                            f"{'🎯' if reason=='TP' else '🛑'} *ETH LONG {reason}!*\n"
                            f"Price:`${price:.2f}` | Unrealized:`${unrealized:+.4f}`"
                        )
                    try:
                        await place_order("SELL", entry_size, price, reduce_only=True)
                        position = None
                        pnl, new_m, outcome = record_close(price, reason, "LONG")
                        wr = round(stats["wins"] / max(stats["total_trades"], 1) * 100, 1)
                        await send_tg(
                            f"{outcome} *ETH LONG #{stats['total_trades']}* [{reason}]\n"
                            f"Entry:`${entry_price:.2f}` -> Exit:`${price:.2f}`\n"
                            f"PnL:`${pnl:+.4f}` | WR:`{wr}%`\n"
                            f"💰 Margin:`${new_m}`"
                        )
                    except Exception as e:
                        await send_tg(f"❌ LONG Close Failed: `{e}`")

            # ── SHORT POSITION ──
            elif position == "SHORT":
                unrealized = round((entry_price - price) * entry_size, 4)
                reason = None
                if price <= tp_price:   reason = "TP"
                elif price >= sl_price: reason = "SL"
                elif bull_cross:        reason = "Cross"
                if reason:
                    if reason in ["TP","SL"]:
                        await send_tg(
                            f"{'🎯' if reason=='TP' else '🛑'} *ETH SHORT {reason}!*\n"
                            f"Price:`${price:.2f}` | Unrealized:`${unrealized:+.4f}`"
                        )
                    try:
                        await place_order("BUY", entry_size, price, reduce_only=True)
                        position = None
                        pnl, new_m, outcome = record_close(price, reason, "SHORT")
                        wr = round(stats["wins"] / max(stats["total_trades"], 1) * 100, 1)
                        await send_tg(
                            f"{outcome} *ETH SHORT #{stats['total_trades']}* [{reason}]\n"
                            f"Entry:`${entry_price:.2f}` -> Exit:`${price:.2f}`\n"
                            f"PnL:`${pnl:+.4f}` | WR:`{wr}%`\n"
                            f"💰 Margin:`${new_m}`"
                        )
                    except Exception as e:
                        await send_tg(f"❌ SHORT Close Failed: `{e}`")

        except Exception as e:
            logger.error(f"Loop error: {e}")

        await asyncio.sleep(CHECK_INTERVAL)

async def cmd_start(u: Update, c: ContextTypes.DEFAULT_TYPE):
    await u.message.reply_text(
        f"🤖 *ALMA Long+Short Bot*\n"
        f"💰 Margin:`${stats['current_margin']}` | {position or '⚪ Wait'}\n"
        f"📈 Long: ALMA cross + EMA200\n"
        f"📉 Short: ALMA cross + RSI<50\n"
        f"⚡ {LEVERAGE}x | 5min\n"
        f"📊 3M:+289% | 1Y:+111%\n"
        "/status /signal /stats /history /balance /backtest",
        parse_mode="Markdown")

async def cmd_status(u: Update, c: ContextTypes.DEFAULT_TYPE):
    try:
        df   = await fetch_candles(300)
        df   = calc_indicators(df)
        df   = df.dropna()
        curr = df.iloc[-1]
        price  = curr["close"]
        ema200 = round(curr["ema200"], 2)
        atr    = round(curr["atr"], 2)
        rsi    = round(curr["rsi"], 1)
        adx    = round(curr["adx"], 1)
        trend  = "🟢 Bull" if price > ema200 else "🔴 Bear"
        adx_s  = f"✅ Trending" if adx > ADX_THRESHOLD else f"❌ Sideways"
        pos    = position or "⚪ Waiting"
        extra  = ""
        if position and entry_price > 0:
            if position == "LONG":
                upnl = round((price - entry_price) * entry_size, 4)
            else:
                upnl = round((entry_price - price) * entry_size, 4)
            extra = (
                f"\nUnrealized:`${upnl:+.4f}`\n"
                f"SL:`${sl_price}` | TP:`${tp_price}`"
            )
        await u.message.reply_text(
            f"📊 *ETH Status*\n"
            f"Price:`${price:.2f}` | {trend}\n"
            f"RSI:`{rsi}` | ATR:`${atr}`\n"
            f"ADX:`{adx}` {adx_s}\n"
            f"EMA200:`${ema200}`\n"
            f"Position: {pos}{extra}\n"
            f"💰 Margin:`${stats['current_margin']}`",
            parse_mode="Markdown")
    except Exception as e:
        await u.message.reply_text(f"❌ {e}")

async def cmd_signal(u: Update, c: ContextTypes.DEFAULT_TYPE):
    try:
        df   = await fetch_candles(300)
        df   = calc_indicators(df)
        df   = df.dropna()
        curr = df.iloc[-1]
        prev = df.iloc[-2]
        price = curr["close"]
        rsi   = round(curr["rsi"], 1)
        adx   = round(curr["adx"], 1)
        atr   = curr["atr"]
        bull_cross   = (curr["alma_fast"] > curr["alma_slow"]) and (prev["alma_fast"] <= prev["alma_slow"])
        bear_cross   = (curr["alma_fast"] < curr["alma_slow"]) and (prev["alma_fast"] >= prev["alma_slow"])
        bull_trend   = price > curr["ema200"]
        adx_trending = adx > ADX_THRESHOLD
        if bull_cross and bull_trend and adx_trending:
            sig = "🚨 LONG NOW!"
            extra = f"\nSL:`${round(price-(SL_ATR_MULT*atr),2)}` TP:`${round(price+(TP_ATR_MULT*atr),2)}`"
        elif bear_cross and rsi < 50 and adx_trending:
            sig = "🚨 SHORT NOW!"
            extra = f"\nSL:`${round(price+(SL_ATR_MULT*atr),2)}` TP:`${round(price-(TP_ATR_MULT*atr),2)}`"
        elif bull_cross and bull_trend and not adx_trending:
            sig = "⛔ LONG blocked (ADX too low)"
            extra = ""
        elif bear_cross and rsi < 50 and not adx_trending:
            sig = "⛔ SHORT blocked (ADX too low)"
            extra = ""
        elif bull_trend and curr["alma_fast"] > curr["alma_slow"]:
            sig = "🟡 Bullish - Wait cross"
            extra = ""
        elif rsi < 50 and curr["alma_fast"] < curr["alma_slow"]:
            sig = "🟡 Bearish RSI<50 - Wait cross"
            extra = ""
        else:
            sig = "⚪ No signal"
            extra = ""
        adx_icon 
