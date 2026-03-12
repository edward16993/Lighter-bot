import os, asyncio, logging, json, numpy as np
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
CHECK_INTERVAL = 300   # 5 min
SL_ATR_MULT    = 2.0   # SL = 2x ATR
TP_ATR_MULT    = 3.0   # TP = 3x ATR
ATR_PERIOD     = 10
ALMA_FAST      = 8
ALMA_SLOW      = 15
ALMA_SIGMA     = 0.85
ALMA_OFFSET    = 0.85
EMA200_PERIOD  = 200
MIN_MARGIN     = 0.50

ETH_MARKET = {
    "symbol":       "ETH-USDT-SWAP",
    "market_index": 0,
    "stats_file":   "eth_stats.json",
    "decimals":     3,
    "min_size":     0.002,
    "start_margin": float(os.environ.get("ETH_MARGIN", "50")),
}

position     = False
entry_price  = 0.0
entry_size   = 0.0
entry_margin = 0.0
sl_price     = 0.0
tp_price     = 0.0
stats        = {}
signer_client= None
tg_app       = None

def load_stats():
    try:
        if os.path.exists(ETH_MARKET["stats_file"]):
            with open(ETH_MARKET["stats_file"]) as f:
                return json.load(f)
    except: pass
    m = ETH_MARKET["start_margin"]
    return {"current_margin": m, "total_trades": 0, "wins": 0, "losses": 0,
            "total_pnl": 0.0, "peak_margin": m,
            "entry_price": 0.0, "entry_size": 0.0, "entry_margin": 0.0,
            "sl_price": 0.0, "tp_price": 0.0, "history": []}

def save_stats():
    with open(ETH_MARKET["stats_file"], "w") as f:
        json.dump(stats, f, indent=2)

# ─── INDICATORS ───────────────────────
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

def calc_indicators(df):
    df = df.copy()
    df["alma_fast"] = calc_alma(df["close"], ALMA_FAST, ALMA_SIGMA, ALMA_OFFSET)
    df["alma_slow"] = calc_alma(df["close"], ALMA_SLOW, ALMA_SIGMA, ALMA_OFFSET)
    df["ema200"]    = df["close"].ewm(span=EMA200_PERIOD, adjust=False).mean()
    df["atr"]       = calc_atr(df, ATR_PERIOD)
    return df

# ─── OKX DATA ─────────────────────────
async def fetch_candles(limit=250):
    async with httpx.AsyncClient(timeout=15) as c:
        r = await c.get(
            "https://www.okx.com/api/v5/market/candles",
            params={"instId": "ETH-USDT-SWAP", "bar": "5m", "limit": str(limit)}
        )
        data = r.json()
        if data.get("code") != "0" or not data.get("data"):
            raise Exception(f"OKX error: {data}")
        rows = list(reversed(data["data"]))
        df = pd.DataFrame(rows, columns=["time","open","high","low","close","vol","volCcy","volCcyQuote","confirm"])
        for col in ["high","low","close"]:
            df[col] = df[col].astype(float)
        return df

# ─── ORDER ────────────────────────────
async def place_order(side, size, price, reduce_only=False):
    if side == "BUY":
        order_price = int(price * 1.005 * 100)
    else:
        order_price = int(price * 0.995 * 100)
    base_amt = int(size * 10000)
    logger.info(f"Order: {side} {size} ETH @ ~{price:.2f}")
    tx, tx_hash, err = await signer_client.create_order(
        market_index=ETH_MARKET["market_index"],
        client_order_index=int(datetime.now().timestamp()),
        base_amount=base_amt,
        price=order_price,
        is_ask=(side == "SELL"),
        order_type=signer_client.ORDER_TYPE_MARKET,
        time_in_force=signer_client.ORDER_TIME_IN_FORCE_IMMEDIATE_OR_CANCEL,
        reduce_only=reduce_only,
        order_expiry=signer_client.DEFAULT_IOC_EXPIRY,
    )
    if err: raise Exception(str(err))
    return tx_hash

def calc_size(margin, price):
    size = (margin * LEVERAGE) / price
    size = max(size, ETH_MARKET["min_size"])
    return round(size, ETH_MARKET["decimals"])

def record_close(exit_price, reason):
    global stats
    pnl   = round((exit_price - stats["entry_price"]) * stats["entry_size"], 4)
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
        "no": stats["total_trades"], "entry": stats["entry_price"],
        "exit": exit_price, "pnl": pnl, "reason": reason,
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

# ─── MAIN LOOP ────────────────────────
async def strategy_loop():
    global position, entry_price, entry_size, entry_margin, sl_price, tp_price, stats

    # Startup position check
    try:
        api = lighter.ApiClient(lighter.Configuration(host=BASE_URL))
        acc = lighter.AccountApi(api)
        r   = await acc.account(str(ACCOUNT_INDEX))
        await api.close()
        for pos in (getattr(r, 'positions', []) or []):
            if getattr(pos, 'market_index', -1) == ETH_MARKET["market_index"]:
                size = float(getattr(pos, 'base_amount', 0) or 0)
                if size > 0:
                    position     = True
                    entry_price  = float(stats.get("entry_price", 0))
                    entry_size   = float(stats.get("entry_size", 0))
                    entry_margin = float(stats.get("entry_margin", 0))
                    sl_price     = float(stats.get("sl_price", 0))
                    tp_price     = float(stats.get("tp_price", 0))
                    logger.info("ETH position restored!")
    except Exception as e:
        logger.error(f"Startup check: {e}")

    await send_tg(
        "🤖 *ALMA Cross Bot Started!*\n"
        f"💰 Margin:`${stats['current_margin']}` | {'🟢 LONG' if position else '⚪ Wait'}\n"
        f"⚡ {LEVERAGE}x | 5min | OKX data\n"
        f"📈 ALMA({ALMA_FAST}/{ALMA_SLOW}) + EMA200\n"
        f"🛡️ SL:{SL_ATR_MULT}×ATR | 🎯 TP:{TP_ATR_MULT}×ATR\n"
        f"📊 Backtest: +430%/yr | +143%/3M 🚀"
    )

    while True:
        try:
            df = await fetch_candles(250)
            df = calc_indicators(df)
            df = df.dropna()

            curr = df.iloc[-1]
            prev = df.iloc[-2]
            price = curr["close"]
            atr   = curr["atr"]

            bull_cross = (curr["alma_fast"] > curr["alma_slow"]) and (prev["alma_fast"] <= prev["alma_slow"])
            bear_cross = (curr["alma_fast"] < curr["alma_slow"]) and (prev["alma_fast"] >= prev["alma_slow"])
            bull_trend = price > curr["ema200"]

            long_signal = bull_cross and bull_trend

            alma_f = round(curr["alma_fast"], 2)
            alma_s = round(curr["alma_slow"], 2)
            ema200 = round(curr["ema200"], 2)
            trend_icon = "🟢" if bull_trend else "🔴"

            logger.info(
                f"ETH=${price:.2f} ALMA_F={alma_f} ALMA_S={alma_s} "
                f"EMA200={ema200} {trend_icon} Bull={bull_trend} Pos={position} "
                f"SL={sl_price:.2f} TP={tp_price:.2f}"
            )

            # ── BUY ──
            if not position and long_signal:
                margin   = stats["current_margin"]
                size     = calc_size(margin, price)
                new_sl   = round(price - (SL_ATR_MULT * atr), 2)
                new_tp   = round(price + (TP_ATR_MULT * atr), 2)
                sl_pct   = round((price - new_sl) / price * 100, 2)
                tp_pct   = round((new_tp - price) / price * 100, 2)
                await send_tg(
                    f"📈 *ETH BUY Signal!*\n"
                    f"ALMA Fast crossed above Slow ✅\n"
                    f"Price > EMA200 {trend_icon}\n"
                    f"Price:`${price:.2f}` | ATR:`${atr:.2f}`\n"
                    f"🛡️ SL:`${new_sl}` (-{sl_pct}%)\n"
                    f"🎯 TP:`${new_tp}` (+{tp_pct}%)\n"
                    f"Size:`{size} ETH` | Margin:`${margin}` × `{LEVERAGE}x`"
                )
                try:
                    await place_order("BUY", size, price)
                    position     = True
                    entry_price  = price
                    entry_size   = size
                    entry_margin = margin
                    sl_price     = new_sl
                    tp_price     = new_tp
                    stats["entry_price"]  = price
                    stats["entry_size"]   = size
                    stats["entry_margin"] = margin
                    stats["sl_price"]     = new_sl
                    stats["tp_price"]     = new_tp
                    save_stats()
                    await send_tg(
                        f"✅ *ETH LONG Opened!*\n"
                        f"Entry:`${price:.2f}` | Size:`{size} ETH`\n"
                        f"🛡️ SL:`${new_sl}` | 🎯 TP:`${new_tp}`\n"
                        f"Compounding margin:`${margin}` 💰"
                    )
                except Exception as e:
                    position = False
                    await send_tg(f"❌ ETH BUY Failed: `{e}`")

            elif position:
                unrealized = round((price - entry_price) * entry_size, 4)

                # ── TP HIT ──
                if price >= tp_price:
                    await send_tg(
                        f"🎯 *ETH TP Hit!*\n"
                        f"Price:`${price:.2f}` ≥ TP:`${tp_price}`\n"
                        f"Unrealized:`${unrealized:+.4f}` 🚀"
                    )
                    try:
                        await place_order("SELL", entry_size, price, reduce_only=True)
                        position = False
                        pnl, new_m, outcome = record_close(price, "TP")
                        wr = round(stats["wins"] / max(stats["total_trades"], 1) * 100, 1)
                        await send_tg(
                            f"{outcome} *ETH TP #{stats['total_trades']}* 🎯\n"
                            f"Entry:`${entry_price:.2f}` → Exit:`${price:.2f}`\n"
                            f"PnL:`${pnl:+.4f}` | WR:`{wr}%`\n"
                            f"💰 Margin:`${new_m}` (compounded!)"
                        )
                    except Exception as e:
                        await send_tg(f"❌ TP Close Failed: `{e}`")

                # ── SL HIT ──
                elif price <= sl_price:
                    await send_tg(
                        f"🛑 *ETH SL Hit!*\n"
                        f"Price:`${price:.2f}` ≤ SL:`${sl_price}`\n"
                        f"Closing position! 🛡️"
                    )
                    try:
                        await place_order("SELL", entry_size, price, reduce_only=True)
                        position = False
                        pnl, new_m, outcome = record_close(price, "SL")
                        wr = round(stats["wins"] / max(stats["total_trades"], 1) * 100, 1)
                        await send_tg(
                            f"{outcome} *ETH SL #{stats['total_trades']}* 🛑\n"
                            f"Entry:`${entry_price:.2f}` → Exit:`${price:.2f}`\n"
                            f"PnL:`${pnl:+.4f}` | WR:`{wr}%`\n"
                            f"💰 Next Margin:`${new_m}`"
                        )
                    except Exception as e:
                        await send_tg(f"❌ SL Close Failed: `{e}`")

                # ── ALMA BEAR CROSS EXIT ──
                elif bear_cross:
                    await send_tg(
                        f"🔄 *ETH ALMA Cross Exit!*\n"
                        f"Fast crossed below Slow 📉\n"
                        f"Price:`${price:.2f}` | Unrealized:`${unrealized:+.4f}`"
                    )
                    try:
                        await place_order("SELL", entry_size, price, reduce_only=True)
                        position = False
                        pnl, new_m, outcome = record_close(price, "Cross")
                        wr = round(stats["wins"] / max(stats["total_trades"], 1) * 100, 1)
                        await send_tg(
                            f"{outcome} *ETH Cross #{stats['total_trades']}*\n"
                            f"Entry:`${entry_price:.2f}` → Exit:`${price:.2f}`\n"
                            f"PnL:`${pnl:+.4f}` | WR:`{wr}%`\n"
                            f"💰 Margin:`${new_m}` (compounded!)"
                        )
                    except Exception as e:
                        await send_tg(f"❌ Cross Close Failed: `{e}`")

        except Exception as e:
            logger.error(f"Loop error: {e}")

        await asyncio.sleep(CHECK_INTERVAL)

# ─── TELEGRAM COMMANDS ────────────────
async def cmd_start(u: Update, c: ContextTypes.DEFAULT_TYPE):
    await u.message.reply_text(
        f"🤖 *ALMA Cross Bot*\n"
        f"Margin:`${stats['current_margin']}` | {'🟢 LONG' if position else '⚪ Wait'}\n"
        f"📈 ALMA({ALMA_FAST}/{ALMA_SLOW}) + EMA200\n"
        f"🛡️ SL:{SL_ATR_MULT}×ATR | 🎯 TP:{TP_ATR_MULT}×ATR\n"
        f"📊 +430%/yr | +143%/3M\n"
        "/status /signal /stats /history /balance",
        parse_mode="Markdown")

async def cmd_status(u: Update, c: ContextTypes.DEFAULT_TYPE):
    try:
        df   = await fetch_candles(250)
        df   = calc_indicators(df)
        df   = df.dropna()
        curr = df.iloc[-1]
        price  = curr["close"]
        alma_f = round(curr["alma_fast"], 2)
        alma_s = round(curr["alma_slow"], 2)
        ema200 = round(curr["ema200"], 2)
        atr    = round(curr["atr"], 2)
        trend  = "🟢 Bull (>EMA200)" if price > ema200 else "🔴 Bear (<EMA200)"
        alma_icon = "📈 Bullish" if alma_f > alma_s else "📉 Bearish"
        pos_str = "🟢 LONG" if position else "⚪ Waiting"
        unrealized = ""
        if position and entry_price > 0:
            upnl = round((price - entry_price) * entry_size, 4)
            sl_dist = round(price - sl_price, 2)
            tp_dist = round(tp_price - price, 2)
            unrealized = (
                f"\nUnrealized:`${upnl:+.4f}`\n"
                f"SL:`${sl_price}` (dist:${sl_dist})\n"
                f"TP:`${tp_price}` (dist:${tp_dist})"
            )
        await u.message.reply_text(
            f"📊 *ETH Status*\n"
            f"Price:`${price:.2f}` | {trend}\n"
            f"ALMA Fast:`${alma_f}` Slow:`${alma_s}`\n"
            f"EMA200:`${ema200}` | ATR:`${atr}`\n"
            f"ALMA: {alma_icon}\n"
            f"Position: {pos_str}{unrealized}\n"
            f"💰 Margin:`${stats['current_margin']}`",
            parse_mode="Markdown")
    except Exception as e:
        await u.message.reply_text(f"❌ {e}")

async def cmd_signal(u: Update, c: ContextTypes.DEFAULT_TYPE):
    try:
        df   = await fetch_candles(250)
        df   = calc_indicators(df)
        df   = df.dropna()
        curr = df.iloc[-1]
        prev = df.iloc[-2]
        price = curr["close"]
        atr   = curr["atr"]
        bull_cross = (curr["alma_fast"] > curr["alma_slow"]) and (prev["alma_fast"] <= prev["alma_slow"])
        bear_cross = (curr["alma_fast"] < curr["alma_slow"]) and (prev["alma_fast"] >= prev["alma_slow"])
        bull_trend = price > curr["ema200"]
        if bull_cross and bull_trend:
            signal = "🚨 BUY Signal NOW!"
            potential_sl = round(price - (SL_ATR_MULT * atr), 2)
            potential_tp = round(price + (TP_ATR_MULT * atr), 2)
            extra = f"\nPotential SL:`${potential_sl}` TP:`${potential_tp}`"
        elif bear_cross:
            signal = "📉 SELL/Exit Signal"
            extra = ""
        elif bull_trend and curr["alma_fast"] > curr["alma_slow"]:
            signal = "🟡 Bullish - Wait for cross"
            extra = ""
        else:
            signal = "🔴 No signal - Wait"
            extra = ""
        await u.message.reply_text(
            f"🎯 *Signal Check*\n"
            f"Price:`${price:.2f}`\n"
            f"Trend: {'🟢 Bull' if bull_trend else '🔴 Bear'}\n"
            f"ALMA: {'📈 Fast>Slow' if curr['alma_fast']>curr['alma_slow'] else '📉 Fast<Slow'}\n"
            f"Signal: {signal}{extra}",
            parse_mode="Markdown")
    except Exception as e:
        await u.message.reply_text(f"❌ {e}")

async def cmd_stats(u: Update, c: ContextTypes.DEFAULT_TYPE):
    t  = stats["total_trades"]
    wr = round(stats["wins"] / max(t, 1) * 100, 1)
    sm = ETH_MARKET["start_margin"]
    g  = round((stats["current_margin"] - sm) / sm * 100, 1)
    await u.message.reply_text(
        f"📊 *ALMA Bot Stats*\n"
        f"Trades:`{t}` | WR:`{wr}%`\n"
        f"✅ Wins:`{stats['wins']}` | ❌ Losses:`{stats['losses']}`\n"
        f"Total PnL:`${stats['total_pnl']:.4f}`\n"
        f"💰 Margin:`${stats['current_margin']}` (+{g}%)\n"
        f"📈 Peak:`${stats['peak_margin']}`\n"
        f"🚀 Start:`${sm}`",
        parse_mode="Markdown")

async def cmd_history(u: Update, c: ContextTypes.DEFAULT_TYPE):
    h = stats.get("history", [])
    if not h:
        await u.message.reply_text("No trades yet!"); return
    msg = "📜 *Trade History (Last 5)*\n"
    for t in reversed(h[-5:]):
        msg += f"{t['result']} #{t['no']} `${t['pnl']:+.4f}` [{t.get('reason','')}]\n"
        msg += f"  `${t['old_margin']}→${t['new_margin']}` {t['time']}\n"
    await u.message.reply_text(msg, parse_mode="Markdown")

async def cmd_balance(u: Update, c: ContextTypes.DEFAULT_TYPE):
    try:
        async with httpx.AsyncClient(timeout=15) as cl:
            r = await cl.get(f"{BASE_URL}/api/v1/account",
                params={"account_index": ACCOUNT_INDEX})
            data = r.json()
            col  = data.get("account", {}).get("collateral", "N/A")
            upnl = data.get("account", {}).get("total_unrealized_pnl", "0")
            await u.message.reply_text(
                f"💰 *Balance*\n"
                f"Collateral:`${col}`\n"
                f"Unrealized PnL:`${upnl}`",
                parse_mode="Markdown")
    except Exception as e:
        await u.message.reply_text(f"❌ {e}")

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
        ("start",   cmd_start),
        ("status",  cmd_status),
        ("signal",  cmd_signal),
        ("stats",   cmd_stats),
        ("history", cmd_history),
        ("balance", cmd_balance),
    ]:
        tg_app.add_handler(CommandHandler(cmd, fn))

    await tg_app.initialize()
    await tg_app.start()

    asyncio.create_task(strategy_loop())

    offset = None
    while True:
        try:
            updates = await tg_app.bot.get_updates(
                offset=offset, t
