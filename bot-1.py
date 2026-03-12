# -*- coding: utf-8 -*-
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
ADX_THRESHOLD  = 20.0
MIN_MARGIN     = 0.50
MARKET_INDEX   = 0
SYMBOL         = "ETH-USDT-SWAP"
STATS_FILE     = "eth_stats.json"
START_MARGIN   = float(os.environ.get("ETH_MARGIN", "50"))
DECIMALS       = 3
MIN_SIZE       = 0.002

position      = None
entry_price   = 0.0
entry_size    = 0.0
entry_margin  = 0.0
sl_price      = 0.0
tp_price      = 0.0
tg_app        = None
signer_client = None
stats         = {}

def load_stats():
    try:
        with open(STATS_FILE) as f:
            return json.load(f)
    except Exception:
        return {
            "total_trades": 0, "wins": 0, "losses": 0,
            "total_pnl": 0.0, "current_margin": START_MARGIN,
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
    tr   = calc_atr(df, period)
    dm_p = (df["high"].diff()).clip(lower=0)
    dm_m = (-df["low"].diff()).clip(lower=0)
    dm_p = dm_p.where(dm_p > dm_m, 0)
    dm_m = dm_m.where(dm_m > dm_p, 0)
    di_p = 100 * dm_p.ewm(span=period, adjust=False).mean() / tr.replace(0, np.nan)
    di_m = 100 * dm_m.ewm(span=period, adjust=False).mean() / tr.replace(0, np.nan)
    dx   = 100 * (di_p - di_m).abs() / (di_p + di_m).replace(0, np.nan)
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

async def fetch_candles(limit=300):
    async with httpx.AsyncClient(timeout=15) as c:
        r = await c.get(
            "https://www.okx.com/api/v5/market/candles",
            params={"instId": SYMBOL, "bar": "5m", "limit": str(limit)}
        )
        data = r.json().get("data", [])
    df = pd.DataFrame(data)[[0, 2, 3, 4]].copy()
    df.columns = ["time", "high", "low", "close"]
    for col in ["high", "low", "close"]:
        df[col] = df[col].astype(float)
    df["time"] = pd.to_datetime(df["time"].astype(int), unit="ms")
    return df.sort_values("time").reset_index(drop=True)

async def place_order(side, size, price, reduce_only=False):
    slippage  = 0.002
    if side == "BUY":
        order_price = round(price * (1 + slippage), 2)
    else:
        order_price = round(price * (1 - slippage), 2)
    base_amt = int(size * 10000)
    logger.info("Order: %s %s ETH @ ~%.2f", side, size, price)
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
    result = "WIN" if pnl >= 0 else "LOSS"
    if pnl >= 0:
        stats["wins"] += 1
    else:
        stats["losses"] += 1
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
        logger.error("TG: %s", e)

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
                    logger.info("ETH %s position restored!", position)
    except Exception as e:
        logger.error("Startup check: %s", e)

    pos_str = position if position else "Wait"
    await send_tg(
        "*ALMA Long+Short Bot*\n"
        "Margin: $" + str(stats["current_margin"]) + " | " + pos_str + "\n"
        "Long:  ALMA(13/21) cross + EMA200 + ADX>20\n"
        "Short: ALMA(13/21) cross + RSI<50  + ADX>20\n"
        "Leverage:" + str(LEVERAGE) + "x | 5min | SL:" + str(SL_ATR_MULT) + "x TP:" + str(TP_ATR_MULT) + "x"
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

            logger.info("ETH=$%.2f RSI=%.1f ADX=%.1f Bull=%s Trending=%s Pos=%s",
                        price, rsi, adx, bull_trend, adx_trending, position)

            if position is None:
                if long_signal and stats["current_margin"] >= MIN_MARGIN:
                    margin = stats["current_margin"]
                    size   = calc_size(margin, price)
                    new_sl = round(price - (SL_ATR_MULT * atr), 2)
                    new_tp = round(price + (TP_ATR_MULT * atr), 2)
                    sl_pct = round((price - new_sl) / price * 100, 2)
                    tp_pct = round((new_tp - price) / price * 100, 2)
                    await send_tg(
                        "*ETH LONG Signal!*\n"
                        "ALMA(13/21) cross + EMA200 + ADX>" + str(ADX_THRESHOLD) + "\n"
                        "Price: $" + str(round(price, 2)) + " RSI: " + str(round(rsi, 1)) + " ADX: " + str(round(adx, 1)) + "\n"
                        "SL: $" + str(new_sl) + " (-" + str(sl_pct) + "%)\n"
                        "TP: $" + str(new_tp) + " (+" + str(tp_pct) + "%)\n"
                        "Size: " + str(size) + " ETH | Margin: $" + str(margin) + " x" + str(LEVERAGE)
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
                            "*ETH LONG Opened!*\n"
                            "Entry: $" + str(round(price, 2)) + " | Size: " + str(size) + " ETH\n"
                            "SL: $" + str(new_sl) + " | TP: $" + str(new_tp) + "\n"
                            "Margin: $" + str(margin)
                        )
                    except Exception as e:
                        position = None
                        await send_tg("ETH LONG Failed: " + str(e))

                elif short_signal and stats["current_margin"] >= MIN_MARGIN:
                    margin = stats["current_margin"]
                    size   = calc_size(margin, price)
                    new_sl = round(price + (SL_ATR_MULT * atr), 2)
                    new_tp = round(price - (TP_ATR_MULT * atr), 2)
                    sl_pct = round((new_sl - price) / price * 100, 2)
                    tp_pct = round((price - new_tp) / price * 100, 2)
                    await send_tg(
                        "*ETH SHORT Signal!*\n"
                        "ALMA(13/21) cross + RSI<50 + ADX>" + str(ADX_THRESHOLD) + "\n"
                        "Price: $" + str(round(price, 2)) + " RSI: " + str(round(rsi, 1)) + " ADX: " + str(round(adx, 1)) + "\n"
                        "SL: $" + str(new_sl) + " (+" + str(sl_pct) + "%)\n"
                        "TP: $" + str(new_tp) + " (-" + str(tp_pct) + "%)\n"
                        "Size: " + str(size) + " ETH | Margin: $" + str(margin) + " x" + str(LEVERAGE)
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
                            "*ETH SHORT Opened!*\n"
                            "Entry: $" + str(round(price, 2)) + " | Size: " + str(size) + " ETH\n"
                            "SL: $" + str(new_sl) + " | TP: $" + str(new_tp) + "\n"
                            "Margin: $" + str(margin)
                        )
                    except Exception as e:
                        position = None
                        await send_tg("ETH SHORT Failed: " + str(e))

            elif position == "LONG":
                unrealized = round((price - entry_price) * entry_size, 4)
                reason = None
                if price >= tp_price:
                    reason = "TP"
                elif price <= sl_price:
                    reason = "SL"
                elif bear_cross:
                    reason = "Cross"
                if reason:
                    lbl = "TP HIT" if reason == "TP" else "SL HIT" if reason == "SL" else "Cross Exit"
                    await send_tg(
                        "*ETH LONG " + lbl + "*\n"
                        "Price: $" + str(round(price, 2)) + " | Unrealized: $" + str(unrealized)
                    )
                    try:
                        await place_order("SELL", entry_size, price, reduce_only=True)
                        pnl, new_m, outcome = record_close(price, reason, "LONG")
                        wr = round(stats["wins"] / max(stats["total_trades"], 1) * 100, 1)
                        await send_tg(
                            outcome + " *ETH LONG #" + str(stats["total_trades"]) + "* [" + reason + "]\n"
                            "Entry: $" + str(entry_price) + " -> Exit: $" + str(round(price, 2)) + "\n"
                            "PnL: $" + str(pnl) + " | WR: " + str(wr) + "%\n"
                            "Margin: $" + str(new_m)
                        )
                        position = None
                    except Exception as e:
                        await send_tg("ETH LONG Close Failed: " + str(e))

            elif position == "SHORT":
                unrealized = round((entry_price - price) * entry_size, 4)
                reason = None
                if price <= tp_price:
                    reason = "TP"
                elif price >= sl_price:
                    reason = "SL"
                elif bull_cross:
                    reason = "Cross"
                if reason:
                    lbl = "TP HIT" if reason == "TP" else "SL HIT" if reason == "SL" else "Cross Exit"
                    await send_tg(
                        "*ETH SHORT " + lbl + "*\n"
                        "Price: $" + str(round(price, 2)) + " | Unrealized: $" + str(unrealized)
                    )
                    try:
                        await place_order("BUY", entry_size, price, reduce_only=True)
                        pnl, new_m, outcome = record_close(price, reason, "SHORT")
                        wr = round(stats["wins"] / max(stats["total_trades"], 1) * 100, 1)
                        await send_tg(
                            outcome + " *ETH SHORT #" + str(stats["total_trades"]) + "* [" + reason + "]\n"
                            "Entry: $" + str(entry_price) + " -> Exit: $" + str(round(price, 2)) + "\n"
                            "PnL: $" + str(pnl) + " | WR: " + str(wr) + "%\n"
                            "Margin: $" + str(new_m)
                        )
                        position = None
                    except Exception as e:
                        await send_tg("ETH SHORT Close Failed: " + str(e))

        except Exception as e:
            logger.error("Loop error: %s", e)
            await send_tg("Loop error: " + str(e))

        await asyncio.sleep(CHECK_INTERVAL)

async def cmd_start(u: Update, c: ContextTypes.DEFAULT_TYPE):
    await u.message.reply_text(
        "*ALMA Long+Short Bot*\n"
        "Commands:\n"
        "/status  - Price, ADX, position\n"
        "/signal  - Current signal\n"
        "/stats   - Trade stats\n"
        "/history - Last 5 trades\n"
        "/balance - DEX balance\n"
        "/backtest - Run backtest"
    )

async def cmd_status(u: Update, c: ContextTypes.DEFAULT_TYPE):
    try:
        df     = await fetch_candles(300)
        df     = calc_indicators(df)
        df     = df.dropna()
        curr   = df.iloc[-1]
        price  = curr["close"]
        ema200 = round(curr["ema200"], 2)
        atr    = round(curr["atr"], 2)
        rsi    = round(curr["rsi"], 1)
        adx    = round(curr["adx"], 1)
        trend  = "Bull" if price > ema200 else "Bear"
        adx_s  = "Trending" if adx > ADX_THRESHOLD else "Sideways-SKIP"
        pos    = position if position else "Waiting"
        extra  = ""
        if position and entry_price > 0:
            if position == "LONG":
                upnl = round((price - entry_price) * entry_size, 4)
            else:
                upnl = round((entry_price - price) * entry_size, 4)
            extra = "\nUnrealized: $" + str(upnl) + "\nSL: $" + str(sl_price) + " | TP: $" + str(tp_price)
        await u.message.reply_text(
            "*ETH Status*\n"
            "Price: $" + str(round(price, 2)) + " | " + trend + "\n"
            "RSI: " + str(rsi) + " | ATR: $" + str(atr) + "\n"
            "ADX: " + str(adx) + " [" + adx_s + "]\n"
            "EMA200: $" + str(ema200) + "\n"
            "Position: " + pos + extra + "\n"
            "Margin: $" + str(stats["current_margin"]),
            parse_mode="Markdown"
        )
    except Exception as e:
        await u.message.reply_text("Error: " + str(e))

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
        fast  = curr["alma_fast"]
        slow  = curr["alma_slow"]
        bull_cross   = (fast > slow) and (prev["alma_fast"] <= prev["alma_slow"])
        bear_cross   = (fast < slow) and (prev["alma_fast"] >= prev["alma_slow"])
        bull_trend   = price > curr["ema200"]
        adx_trending = adx > ADX_THRESHOLD
        alma_dir     = "Fast>Slow" if fast > slow else "Fast<Slow"
        trend_str    = "Bull" if bull_trend else "Bear"
        adx_str      = "Trending" if adx_trending else "Sideways-SKIP"
        if bull_cross and bull_trend and adx_trending:
            sig   = "LONG NOW!"
            extra = "\nSL: $" + str(round(price - SL_ATR_MULT * atr, 2)) + " TP: $" + str(round(price + TP_ATR_MULT * atr, 2))
        elif bear_cross and rsi < 50 and adx_trending:
            sig   = "SHORT NOW!"
            extra = "\nSL: $" + str(round(price + SL_ATR_MULT * atr, 2)) + " TP: $" + str(round(price - TP_ATR_MULT * atr, 2))
        elif bull_cross and bull_trend and not adx_trending:
            sig   = "LONG blocked - ADX low"
            extra = ""
        elif bear_cross and rsi < 50 and not adx_trending:
            sig   = "SHORT blocked - ADX low"
            extra = ""
        elif bull_trend and fast > slow:
            sig   = "Bullish - Wait cross"
            extra = ""
        elif rsi < 50 and fast < slow:
            sig   = "Bearish RSI<50 - Wait cross"
            extra = ""
        else:
            sig   = "No signal"
            extra = ""
        await u.message.reply_text(
            "*Signal Check*\n"
            "Price: $" + str(round(price, 2)) + " RSI: " + str(rsi) + "\n"
            "ADX: " + str(adx) + " [" + adx_str + "]\n"
            "Trend: " + trend_str + " | ALMA: " + alma_dir + "\n"
            "Signal: " + sig + extra,
            parse_mode="Markdown"
        )
    except Exception as e:
        await u.message.reply_text("Error: " + str(e))

async def cmd_stats(u: Update, c: ContextTypes.DEFAULT_TYPE):
    t  = stats["total_trades"]
    wr = round(stats["wins"] / max(t, 1) * 100, 1)
    g  = round((stats["current_margin"] - START_MARGIN) / START_MARGIN * 100, 1)
    lt = stats.get("long_trades", 0)
    st = stats.get("short_trades", 0)
    sign = "+" if g >= 0 else ""
    await u.message.reply_text(
        "*ALMA L+S Stats*\n"
        "Trades: " + str(t) + " (L:" + str(lt) + " S:" + str(st) + ")\n"
        "WR: " + str(wr) + "% | " + sign + str(g) + "%\n"
        "Total PnL: $" + str(round(stats["total_pnl"], 4)) + "\n"
        "Margin: $" + str(stats["current_margin"]) + "\n"
        "Peak: $" + str(stats["peak_margin"]),
        parse_mode="Markdown"
    )

async def cmd_history(u: Update, c: ContextTypes.DEFAULT_TYPE):
    h = stats.get("history", [])
    if not h:
        await u.message.reply_text("No trades yet!")
        return
    lines = ["*Last trades:*"]
    for t in h[-5:]:
        sign = "+" if t["pnl"] >= 0 else ""
        lines.append(
            "#" + str(t["no"]) + " " + t["side"] + " [" + t["reason"] + "] " +
            sign + "$" + str(t["pnl"]) + " -> $" + str(t["new_margin"])
        )
    await u.message.reply_text("\n".join(lines), parse_mode="Markdown")

async def cmd_balance(u: Update, c: ContextTypes.DEFAULT_TYPE):
    try:
        api = lighter.ApiClient(lighter.Configuration(host=BASE_URL))
        acc = lighter.AccountApi(api)
        r   = await acc.account(str(ACCOUNT_INDEX))
        await api.close()
        col  = getattr(r, "collateral", "?")
        upnl = getattr(r, "unrealized_pnl", "?")
        await u.message.reply_text(
            "*Balance*\nCollateral: $" + str(col) + "\nUnrealized: $" + str(upnl),
            parse_mode="Markdown"
        )
    except Exception as e:
        await u.message.reply_text("Error: " + str(e))

async def run_quick_backtest(days):
    import time as time_mod
    all_candles = []
    current    = int(datetime.now().timestamp() * 1000)
    start_time = int((datetime.now().timestamp() - days * 86400) * 1000)
    async with httpx.AsyncClient(timeout=30) as c:
        while current > start_time:
            r = await c.get(
                "https://www.okx.com/api/v5/market/history-candles",
                params={"instId": SYMBOL, "bar": "5m", "after": str(current), "limit": "300"}
            )
            candles = r.json().get("data", [])
            if not candles:
                break
            all_candles.extend(candles)
            current = int(candles[-1][0]) - 1
            if len(candles) < 300:
                break
            await asyncio.sleep(0.1)
    df = pd.DataFrame(all_candles)[[0, 2, 3, 4]].copy()
    df.columns = ["time", "high", "low", "close"]
    for col in ["high", "low", "close"]:
        df[col] = df[col].astype(float)
    df["time"] = pd.to_datetime(df["time"].astype(int), unit="ms")
    df = df.drop_duplicates().sort_values("time").reset_index(drop=True)
    df = calc_indicators(df)
    df = df.dropna().reset_index(drop=True)

    m = 8.0
    pos_bt = None
    ep = em = sl_p = tp_p = 0.0
    trades = []
    long_t = short_t = wins = 0

    for i in range(1, len(df)):
        curr = df.iloc[i]
        prev = df.iloc[i-1]
        price = curr["close"]
        atr   = curr["atr"]
        rsi   = curr["rsi"]
        adx   = curr["adx"]
        if any(pd.isna(curr[x]) for x in ["alma_fast", "alma_slow", "atr", "ema200", "rsi", "adx"]):
            continue

        bull_cross   = (curr["alma_fast"] > curr["alma_slow"]) and (prev["alma_fast"] <= prev["alma_slow"])
        bear_cross   = (curr["alma_fast"] < curr["alma_slow"]) and (prev["alma_fast"] >= prev["alma_slow"])
        bull_trend   = price > curr["ema200"]
        adx_ok       = adx > ADX_THRESHOLD
        long_signal  = bull_cross and bull_trend and adx_ok
        short_signal = bear_cross and (rsi < 50) and adx_ok

        if pos_bt == "LONG":
            reason = None
            if price >= tp_p:
                reason = "TP"
            elif price <= sl_p:
                reason = "SL"
            elif bear_cross:
                reason = "Cross"
            if reason:
                pnl = (price - ep) / ep * em * LEVERAGE
                m   = max(em + pnl, 0.5)
                trades.append(pnl)
                if pnl > 0:
                    wins += 1
                pos_bt = None

        elif pos_bt == "SHORT":
            reason = None
            if price <= tp_p:
                reason = "TP"
            elif price >= sl_p:
                reason = "SL"
            elif bull_cross:
                reason = "Cross"
            if reason:
                pnl = (ep - price) / ep * em * LEVERAGE
                m   = max(em + pnl, 0.5)
                trades.append(pnl)
                if pnl > 0:
                    wins += 1
                pos_bt = None

        if pos_bt is None and m >= 0.5:
            if long_signal:
                pos_bt = "LONG"
                ep = price; em = m
                sl_p = price - (SL_ATR_MULT * atr)
                tp_p = price + (TP_ATR_MULT * atr)
                long_t += 1
            elif short_signal:
                pos_bt = "SHORT"
                ep = price; em = m
                sl_p = price + (SL_ATR_MULT * atr)
                tp_p = price - (TP_ATR_MULT * atr)
                short_t += 1

    total  = len(trades)
    wr     = round(wins / max(total, 1) * 100, 1)
    growth = round((m - 8.0) / 8.0 * 100, 1)
    return total, long_t, short_t, wr, growth, round(m, 2)

async def cmd_backtest(u: Update, c: ContextTypes.DEFAULT_TYPE):
    await u.message.reply_text("Backtest running... (30-60 sec)")
    try:
        results = []
        for days, label in [(90, "3M"), (365, "1Y")]:
            total, lt, st, wr, growth, final = await run_quick_backtest(days)
            sign = "+" if growth > 0 else ""
            results.append(
                "*" + label + "*\n"
                "Trades: " + str(total) + " (L:" + str(lt) + " S:" + str(st) + ")\n"
                "WR: " + str(wr) + "% | " + sign + str(growth) + "%\n"
                "$8 -> $" + str(final) + "\n"
            )
        msg = (
            "*Backtest Results*\n"
            "ALMA(13/21) + ADX>20\n"
            "Long: cross + EMA200 | Short: cross + RSI<50\n"
            "SL:" + str(SL_ATR_MULT) + "x | TP:" + str(TP_ATR_MULT) + "x\n"
            "---\n" + "\n".join(results)
        )
        await u.message.reply_text(msg, parse_mode="Markdown")
    except Exception as e:
        await u.message.reply_text("Backtest failed: " + str(e))

async def main():
    global tg_app, signer_client, stats
    stats = load_stats()
    signer_client = lighter.SignerClient(
        url=BASE_URL,
        api_private_keys={API_KEY_INDEX: LIGHTER_PRIVATE_KEY},
        account_index=ACCOUNT_INDEX
    )
    tg_app = Application.builder().token(TELEGRAM_BOT_TOKEN).updater(None).build()
    for cmd, fn in [
        ("start",    cmd_start),
        ("status",   cmd_status),
        ("signal",   cmd_signal),
        ("stats",    cmd_stats),
        ("history",  cmd_history),
        ("balance",  cmd_balance),
        ("backtest", cmd_backtest),
    ]:
        tg_app.add_handler(CommandHandler(cmd, fn))

    await tg_app.initialize()
    await tg_app.start()
    loop = asyncio.get_event_loop()
    loop.create_task(strategy_loop())
    await tg_app.updater.start_polling(drop_pending_updates=True)
    try:
        await asyncio.Event().wait()
    finally:
        await tg_app.updater.stop()
        await tg_app.stop()
        await tg_app.shutdown()

if __name__ == "__main__":
    asyncio.run(main())
