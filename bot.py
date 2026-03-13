# -*- coding: utf-8 -*-
import os, asyncio, logging, json
import numpy as np
from datetime import datetime
import pandas as pd, httpx
from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes
import lighter

logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

TG_TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]
TG_CHAT  = os.environ["TELEGRAM_CHAT_ID"]
ACC_IDX  = int(os.environ["ACCOUNT_INDEX"])
KEY_IDX  = int(os.environ["API_KEY_INDEX"])
PRV_KEY  = os.environ["LIGHTER_PRIVATE_KEY"]

BASE_URL = "https://mainnet.zklighter.elliot.ai"
LEV      = 5
INTERVAL = 300
SL_MULT  = float(os.environ.get("SL_MULT", "2.0"))
TP_MULT  = float(os.environ.get("TP_MULT", "3.0"))
ADX_MIN  = 20.0
MIN_MAR  = 0.50
MKT_IDX  = 0
SYMBOL   = "ETH-USDT-SWAP"
STATS_F  = "stats.json"
START_M  = float(os.environ.get("ETH_MARGIN", "50"))
DECIMALS = 3
MIN_SIZE = 0.002

# Global state
cur_pos  = None
entry_px = 0.0
entry_sz = 0.0
entry_mg = 0.0
sl_px    = 0.0
tp_px    = 0.0
tg_app   = None
signer   = None
stats    = {}

def load_stats():
    try:
        with open(STATS_F) as f:
            return json.load(f)
    except Exception:
        return {
            "total_trades": 0, "wins": 0, "losses": 0,
            "total_pnl": 0.0, "current_margin": START_M,
            "peak_margin": START_M, "long_trades": 0, "short_trades": 0,
            "entry_price": 0.0, "entry_size": 0.0, "entry_margin": 0.0,
            "sl_price": 0.0, "tp_price": 0.0, "history": []
        }

def save_stats():
    with open(STATS_F, "w") as f:
        json.dump(stats, f)

def calc_alma(s, p, sigma=0.85, offset=0.85):
    m = offset * (p - 1)
    sv = p / sigma
    w = np.array([np.exp(-((i - m) ** 2) / (2 * sv ** 2)) for i in range(p)])
    w /= w.sum()
    r = pd.Series(np.nan, index=s.index)
    for i in range(p - 1, len(s)):
        r.iloc[i] = np.dot(w, s.iloc[i - p + 1:i + 1].values)
    return r

def calc_atr(df, p=10):
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - df["close"].shift()).abs(),
        (df["low"] - df["close"].shift()).abs()
    ], axis=1).max(axis=1)
    return tr.ewm(span=p, adjust=False).mean()

def calc_adx(df, p=14):
    tr = calc_atr(df, p)
    dp = (df["high"].diff()).clip(lower=0)
    dm = (-df["low"].diff()).clip(lower=0)
    dp = dp.where(dp > dm, 0)
    dm = dm.where(dm > dp, 0)
    dip = 100 * dp.ewm(span=p, adjust=False).mean() / tr.replace(0, np.nan)
    dim = 100 * dm.ewm(span=p, adjust=False).mean() / tr.replace(0, np.nan)
    dx = 100 * (dip - dim).abs() / (dip + dim).replace(0, np.nan)
    return dx.ewm(span=p, adjust=False).mean()

def calc_rsi(s, p=14):
    d = s.diff()
    g = d.clip(lower=0).ewm(span=p, adjust=False).mean()
    l = (-d.clip(upper=0)).ewm(span=p, adjust=False).mean()
    return 100 - (100 / (1 + g / l.replace(0, np.nan)))

def calc_indicators(df):
    df = df.copy()
    df["fast"] = calc_alma(df["close"], 13)
    df["slow"] = calc_alma(df["close"], 21)
    df["ema200"] = df["close"].ewm(span=200, adjust=False).mean()
    df["atr"] = calc_atr(df, 10)
    df["rsi"] = calc_rsi(df["close"], 14)
    df["adx"] = calc_adx(df, 14)
    return df

async def fetch_candles(lim=300):
    async with httpx.AsyncClient(timeout=15) as c:
        r = await c.get("https://www.okx.com/api/v5/market/candles",
            params={"instId": SYMBOL, "bar": "5m", "limit": str(lim)})
        data = r.json().get("data", [])
    df = pd.DataFrame(data)[[0, 2, 3, 4]].copy()
    df.columns = ["time", "high", "low", "close"]
    for col in ["high", "low", "close"]:
        df[col] = df[col].astype(float)
    df["time"] = pd.to_datetime(df["time"].astype(int), unit="ms")
    return df.sort_values("time").reset_index(drop=True)

async def place_order(side, size, price, reduce_only=False):
    slip = 0.002
    if side == "BUY":
        order_px = round(float(price) * (1 + slip), 2)
    else:
        order_px = round(float(price) * (1 - slip), 2)
    base_amt = int(float(size) * 10000)
    tx, txh, err = await signer.create_order(
        market_index=MKT_IDX,
        client_order_index=int(datetime.now().timestamp()),
        base_amount=base_amt,
        price=order_px,
        is_ask=(side == "SELL"),
        order_type=signer.ORDER_TYPE_MARKET,
        time_in_force=signer.ORDER_TIME_IN_FORCE_IMMEDIATE_OR_CANCEL,
        reduce_only=reduce_only,
        order_expiry=signer.DEFAULT_IOC_EXPIRY)
    if err:
        raise Exception(str(err))
    return txh

def calc_size(margin, price):
    return round(max((margin * LEV) / float(price), MIN_SIZE), DECIMALS)

def record_close(exit_px, reason, side):
    global stats
    if side == "LONG":
        pnl = round((exit_px - stats["entry_price"]) * stats["entry_size"], 4)
    else:
        pnl = round((stats["entry_price"] - exit_px) * stats["entry_size"], 4)
    new_m = round(max(stats["entry_margin"] + pnl, MIN_MAR), 4)
    stats["total_trades"] += 1
    stats["total_pnl"] = round(stats["total_pnl"] + pnl, 4)
    stats["current_margin"] = new_m
    if new_m > stats["peak_margin"]:
        stats["peak_margin"] = new_m
    if pnl >= 0:
        stats["wins"] += 1
    else:
        stats["losses"] += 1
    stats["history"].append({
        "no": stats["total_trades"], "side": side,
        "entry": stats["entry_price"], "exit": exit_px,
        "pnl": pnl, "reason": reason, "new_margin": new_m,
        "time": datetime.now().strftime("%Y-%m-%d %H:%M")
    })
    stats["history"] = stats["history"][-20:]
    save_stats()
    return pnl, new_m, "WIN" if pnl >= 0 else "LOSS"

async def send_tg(msg):
    try:
        await tg_app.bot.send_message(
            chat_id=TG_CHAT, text=msg, parse_mode="Markdown")
    except Exception as e:
        logger.error("TG error: %s", e)

async def strategy_loop():
    global cur_pos, entry_px, entry_sz, entry_mg, sl_px, tp_px, stats

    # Restore position on restart
    try:
        api = lighter.ApiClient(lighter.Configuration(host=BASE_URL))
        acc = lighter.AccountApi(api)
        r = await acc.account(str(ACC_IDX))
        await api.close()
        for p in (getattr(r, "positions", []) or []):
            if getattr(p, "market_index", -1) == MKT_IDX:
                s = float(getattr(p, "base_amount", 0) or 0)
                if s != 0:
                    cur_pos  = "LONG" if s > 0 else "SHORT"
                    entry_px = float(stats.get("entry_price", 0))
                    entry_sz = abs(s)
                    entry_mg = float(stats.get("entry_margin", 0))
                    sl_px    = float(stats.get("sl_price", 0))
                    tp_px    = float(stats.get("tp_price", 0))
                    logger.info("Restored: %s position", cur_pos)
    except Exception as e:
        logger.error("Restore error: %s", e)

    ps = cur_pos if cur_pos else "Wait"
    await send_tg(
        "*ALMA Bot Started*\n"
        "Margin: `$" + str(stats["current_margin"]) + "` | " + ps + "\n"
        "Long:  ALMA(13/21) + EMA200 + ADX>20\n"
        "Short: ALMA(13/21) + RSI<50 + ADX>20\n"
        "`" + str(LEV) + "x` | 5min | SL:" + str(SL_MULT) + "x TP:" + str(TP_MULT) + "x"
    )

    while True:
        try:
            df   = await fetch_candles(300)
            df   = calc_indicators(df)
            df   = df.dropna()
            curr = df.iloc[-1]
            prev = df.iloc[-2]

            price = float(curr["close"])
            atr   = float(curr["atr"])
            rsi   = float(curr["rsi"])
            adx   = float(curr["adx"])

            bull_cross = (curr["fast"] > curr["slow"]) and (prev["fast"] <= prev["slow"])
            bear_cross = (curr["fast"] < curr["slow"]) and (prev["fast"] >= prev["slow"])
            bull_trend = price > float(curr["ema200"])
            trending   = adx > ADX_MIN

            long_sig  = bull_cross and bull_trend and trending
            short_sig = bear_cross and (rsi < 50) and trending

            logger.info("ETH=$%.2f RSI=%.1f ADX=%.1f pos=%s", price, rsi, adx, cur_pos)

            if cur_pos is None:
                if long_sig and stats["current_margin"] >= MIN_MAR:
                    mg  = stats["current_margin"]
                    si  = calc_size(mg, price)
                    nsl = round(price - (SL_MULT * atr), 2)
                    ntp = round(price + (TP_MULT * atr), 2)
                    await send_tg(
                        "*ETH LONG Signal*\n"
                        "Price: `$" + str(round(price, 2)) + "` ADX: `" + str(round(adx, 1)) + "`\n"
                        "SL: `$" + str(nsl) + "` TP: `$" + str(ntp) + "`\n"
                        "Margin: `$" + str(mg) + "`"
                    )
                    try:
                        await place_order("BUY", si, price)
                        cur_pos = "LONG"; entry_px = price; entry_sz = si
                        entry_mg = mg; sl_px = nsl; tp_px = ntp
                        stats["entry_price"]  = price; stats["entry_size"]   = si
                        stats["entry_margin"] = mg;    stats["sl_price"]     = nsl
                        stats["tp_price"]     = ntp
                        stats["long_trades"]  = stats.get("long_trades", 0) + 1
                        save_stats()
                        await send_tg(
                            "*LONG Opened!*\n"
                            "Entry: `$" + str(round(price, 2)) + "` | `" + str(si) + " ETH`\n"
                            "SL: `$" + str(nsl) + "` TP: `$" + str(ntp) + "`"
                        )
                    except Exception as e:
                        cur_pos = None
                        await send_tg("LONG Failed: " + str(e))

                elif short_sig and stats["current_margin"] >= MIN_MAR:
                    mg  = stats["current_margin"]
                    si  = calc_size(mg, price)
                    nsl = round(price + (SL_MULT * atr), 2)
                    ntp = round(price - (TP_MULT * atr), 2)
                    await send_tg(
                        "*ETH SHORT Signal*\n"
                        "Price: `$" + str(round(price, 2)) + "` ADX: `" + str(round(adx, 1)) + "`\n"
                        "SL: `$" + str(nsl) + "` TP: `$" + str(ntp) + "`\n"
                        "Margin: `$" + str(mg) + "`"
                    )
                    try:
                        await place_order("SELL", si, price)
                        cur_pos = "SHORT"; entry_px = price; entry_sz = si
                        entry_mg = mg; sl_px = nsl; tp_px = ntp
                        stats["entry_price"]  = price; stats["entry_size"]   = si
                        stats["entry_margin"] = mg;    stats["sl_price"]     = nsl
                        stats["tp_price"]     = ntp
                        stats["short_trades"] = stats.get("short_trades", 0) + 1
                        save_stats()
                        await send_tg(
                            "*SHORT Opened!*\n"
                            "Entry: `$" + str(round(price, 2)) + "` | `" + str(si) + " ETH`\n"
                            "SL: `$" + str(nsl) + "` TP: `$" + str(ntp) + "`"
                        )
                    except Exception as e:
                        cur_pos = None
                        await send_tg("SHORT Failed: " + str(e))

            elif cur_pos == "LONG":
                unr = round((price - entry_px) * entry_sz, 4)
                reason = None
                if price >= tp_px:   reason = "TP"
                elif price <= sl_px: reason = "SL"
                elif bear_cross:     reason = "Cross"
                if reason:
                    await send_tg(
                        "*LONG " + reason + "*\n"
                        "Price: `$" + str(round(price, 2)) + "` Unrealized: `$" + str(unr) + "`"
                    )
                    try:
                        await place_order("SELL", entry_sz, price, reduce_only=True)
                        pnl, new_m, outcome = record_close(price, reason, "LONG")
                        wr  = round(stats["wins"] / max(stats["total_trades"], 1) * 100, 1)
                        sg  = "+" if pnl >= 0 else ""
                        await send_tg(
                            outcome + " *LONG #" + str(stats["total_trades"]) + "* [" + reason + "]\n"
                            "Entry: `$" + str(entry_px) + "` Exit: `$" + str(round(price, 2)) + "`\n"
                            "PnL: `" + sg + "$" + str(pnl) + "` WR: `" + str(wr) + "%`\n"
                            "Margin: `$" + str(new_m) + "`"
                        )
                        cur_pos = None
                    except Exception as e:
                        await send_tg("LONG Close Failed: " + str(e))

            elif cur_pos == "SHORT":
                unr = round((entry_px - price) * entry_sz, 4)
                reason = None
                if price <= tp_px:   reason = "TP"
                elif price >= sl_px: reason = "SL"
                elif bull_cross:     reason = "Cross"
                if reason:
                    await send_tg(
                        "*SHORT " + reason + "*\n"
                        "Price: `$" + str(round(price, 2)) + "` Unrealized: `$" + str(unr) + "`"
                    )
                    try:
                        await place_order("BUY", entry_sz, price, reduce_only=True)
                        pnl, new_m, outcome = record_close(price, reason, "SHORT")
                        wr  = round(stats["wins"] / max(stats["total_trades"], 1) * 100, 1)
                        sg  = "+" if pnl >= 0 else ""
                        await send_tg(
                            outcome + " *SHORT #" + str(stats["total_trades"]) + "* [" + reason + "]\n"
                            "Entry: `$" + str(entry_px) + "` Exit: `$" + str(round(price, 2)) + "`\n"
                            "PnL: `" + sg + "$" + str(pnl) + "` WR: `" + str(wr) + "%`\n"
                            "Margin: `$" + str(new_m) + "`"
                        )
                        cur_pos = None
                    except Exception as e:
                        await send_tg("SHORT Close Failed: " + str(e))

        except Exception as e:
            logger.error("Loop error: %s", e)
            await send_tg("Loop error: " + str(e))

        await asyncio.sleep(INTERVAL)

async def cmd_start(u, c):
    await u.message.reply_text(
        "*ALMA Bot Commands*\n"
        "/status - Price + ADX + Position\n"
        "/signal - Current signal\n"
        "/stats  - Trade statistics\n"
        "/history - Last 5 trades\n"
        "/balance - DEX balance\n"
        "/backtest - Run backtest",
        parse_mode="Markdown")

async def cmd_status(u, c):
    try:
        df   = await fetch_candles(300)
        df   = calc_indicators(df)
        df   = df.dropna()
        curr = df.iloc[-1]
        price  = float(curr["close"])
        ema200 = round(float(curr["ema200"]), 2)
        atr    = round(float(curr["atr"]), 2)
        rsi    = round(float(curr["rsi"]), 1)
        adx    = round(float(curr["adx"]), 1)
        trend  = "Bull" if price > ema200 else "Bear"
        adx_s  = "Trending" if adx > ADX_MIN else "Sideways-SKIP"
        ps     = cur_pos if cur_pos else "Waiting"
        extra  = ""
        if cur_pos and entry_px > 0:
            if cur_pos == "LONG":
                unr = round((price - entry_px) * entry_sz, 4)
            else:
                unr = round((entry_px - price) * entry_sz, 4)
            extra = (
                "\nUnrealized: `$" + str(unr) + "`"
                "\nSL: `$" + str(sl_px) + "` TP: `$" + str(tp_px) + "`"
            )
        await u.message.reply_text(
            "*ETH Status*\n"
            "Price: `$" + str(round(price, 2)) + "` | " + trend + "\n"
            "RSI: `" + str(rsi) + "` ATR: `$" + str(atr) + "`\n"
            "ADX: `" + str(adx) + "` [" + adx_s + "]\n"
            "EMA200: `$" + str(ema200) + "`\n"
            "Position: " + ps + extra + "\n"
            "Margin: `$" + str(stats["current_margin"]) + "`",
            parse_mode="Markdown")
    except Exception as e:
        await u.message.reply_text("Error: " + str(e))

async def cmd_signal(u, c):
    try:
        df   = await fetch_candles(300)
        df   = calc_indicators(df)
        df   = df.dropna()
        curr = df.iloc[-1]
        prev = df.iloc[-2]
        price = float(curr["close"])
        rsi   = round(float(curr["rsi"]), 1)
        adx   = round(float(curr["adx"]), 1)
        atr   = float(curr["atr"])
        bull_cross = (curr["fast"] > curr["slow"]) and (prev["fast"] <= prev["slow"])
        bear_cross = (curr["fast"] < curr["slow"]) and (prev["fast"] >= prev["slow"])
        bull_trend = price > float(curr["ema200"])
        trending   = adx > ADX_MIN
        adx_s = "Trending" if trending else "Sideways-SKIP"
        if bull_cross and bull_trend and trending:
            sig = "LONG NOW!\nSL: `$" + str(round(price - SL_MULT * atr, 2)) + "` TP: `$" + str(round(price + TP_MULT * atr, 2)) + "`"
        elif bear_cross and rsi < 50 and trending:
            sig = "SHORT NOW!\nSL: `$" + str(round(price + SL_MULT * atr, 2)) + "` TP: `$" + str(round(price - TP_MULT * atr, 2)) + "`"
        elif bull_cross and bull_trend and not trending:
            sig = "LONG blocked - ADX low"
        elif bear_cross and rsi < 50 and not trending:
            sig = "SHORT blocked - ADX low"
        elif bull_trend and curr["fast"] > curr["slow"]:
            sig = "Bullish - Wait cross"
        elif rsi < 50 and curr["fast"] < curr["slow"]:
            sig = "Bearish - Wait cross"
        else:
            sig = "No signal"
        await u.message.reply_text(
            "*Signal*\n"
            "Price: `$" + str(round(price, 2)) + "` RSI: `" + str(rsi) + "`\n"
            "ADX: `" + str(adx) + "` [" + adx_s + "]\n"
            + sig,
            parse_mode="Markdown")
    except Exception as e:
        await u.message.reply_text("Error: " + str(e))

async def cmd_stats(u, c):
    t   = stats["total_trades"]
    wr  = round(stats["wins"] / max(t, 1) * 100, 1)
    g   = round((stats["current_margin"] - START_M) / START_M * 100, 1)
    lt  = stats.get("long_trades", 0)
    sh  = stats.get("short_trades", 0)
    sg  = "+" if g >= 0 else ""
    await u.message.reply_text(
        "*Stats*\n"
        "Trades: `" + str(t) + "` (L:" + str(lt) + " S:" + str(sh) + ")\n"
        "WR: `" + str(wr) + "%` | `" + sg + str(g) + "%`\n"
        "PnL: `$" + str(round(stats["total_pnl"], 4)) + "`\n"
        "Margin: `$" + str(stats["current_margin"]) + "` Peak: `$" + str(stats["peak_margin"]) + "`",
        parse_mode="Markdown")

async def cmd_history(u, c):
    h = stats.get("history", [])
    if not h:
        await u.message.reply_text("No trades yet!")
        return
    lines = ["*Last trades:*"]
    for t in h[-5:]:
        sg = "+" if t["pnl"] >= 0 else ""
        lines.append(
            "#" + str(t["no"]) + " " + t["side"] +
            " [" + t["reason"] + "] `" + sg + "$" + str(t["pnl"]) +
            "` -> `$" + str(t["new_margin"]) + "`"
        )
    await u.message.reply_text("\n".join(lines), parse_mode="Markdown")

async def cmd_balance(u, c):
    try:
        api = lighter.ApiClient(lighter.Configuration(host=BASE_URL))
        acc = lighter.AccountApi(api)
        r   = await acc.account(str(ACC_IDX))
        await api.close()
        col  = getattr(r, "collateral", "?")
        upnl = getattr(r, "unrealized_pnl", "?")
        await u.message.reply_text(
            "*Balance*\n"
            "Collateral: `$" + str(col) + "`\n"
            "Unrealized: `$" + str(upnl) + "`",
            parse_mode="Markdown")
    except Exception as e:
        await u.message.reply_text("Error: " + str(e))

async def cmd_backtest(u, c):
    await u.message.reply_text("Backtest running... 30-60sec")
    try:
        results = []
        for days, label in [(90, "3M"), (365, "1Y")]:
            ac  = []
            cur = int(datetime.now().timesta