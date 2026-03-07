import os, asyncio, logging, json
from datetime import datetime
import httpx
import pandas as pd
from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes

logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

TELEGRAM_BOT_TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]
TELEGRAM_CHAT_ID   = os.environ["TELEGRAM_CHAT_ID"]
LIGHTER_API_KEY    = os.environ.get("LIGHTER_API_KEY", "")
LIGHTER_SECRET     = os.environ.get("LIGHTER_SECRET", "")
STARTING_MARGIN    = float(os.environ.get("STARTING_MARGIN", "5"))

LEVERAGE       = 6
RSI_BUY        = 35
RSI_CLOSE      = 70
RSI_PERIOD     = 14
CHECK_INTERVAL = 900

STATS_FILE  = "stats.json"
bot_app     = None
in_position = False

def load_stats():
    try:
        if os.path.exists(STATS_FILE):
            with open(STATS_FILE) as f:
                return json.load(f)
    except:
        pass
    return {
        "current_margin": STARTING_MARGIN,
        "total_trades": 0, "wins": 0, "losses": 0,
        "total_pnl": 0.0, "peak_margin": STARTING_MARGIN,
        "entry_price": 0.0, "entry_size": 0.0,
        "entry_margin": 0.0, "history": []
    }

def save_stats(s):
    try:
        with open(STATS_FILE, "w") as f:
            json.dump(s, f, indent=2)
    except Exception as e:
        logger.error(f"Save stats error: {e}")

stats = load_stats()

async def fetch_closes(limit=50):
    async with httpx.AsyncClient(timeout=15) as c:
        r = await c.get(
            "https://api.binance.com/api/v3/klines",
            params={"symbol": "ETHUSDT", "interval": "15m", "limit": limit}
        )
        return [float(x[4]) for x in r.json()]

def calc_rsi(closes, period=14):
    if len(closes) < period + 1:
        return 50.0
    try:
        s    = pd.Series(closes)
        d    = s.diff()
        gain = d.where(d > 0, 0.0).rolling(period).mean()
        loss = (-d.where(d < 0, 0.0)).rolling(period).mean()
        rs   = gain / loss
        rsi  = 100 - (100 / (1 + rs))
        val  = float(rsi.iloc[-1])
        return round(val, 2) if not pd.isna(val) else 50.0
    except:
        return 50.0

async def lighter_order(side, size, reduce_only=False):
    payload = {
        "market_index": 0,
        "base_amount": int(size * 1e6),
        "is_ask": (side == "SELL"),
        "order_type": "MARKET",
        "time_in_force": "IOC",
        "reduce_only": reduce_only,
        "leverage": LEVERAGE,
    }
    headers = {
        "Authorization": f"Bearer {LIGHTER_API_KEY}",
        "Content-Type": "application/json"
    }
    async with httpx.AsyncClient(timeout=15) as c:
        r = await c.post("https://mainnet.lighter.xyz/v1/orders", json=payload, headers=headers)
        return r.json()

async def lighter_balance():
    headers = {"Authorization": f"Bearer {LIGHTER_API_KEY}"}
    async with httpx.AsyncClient(timeout=15) as c:
        r = await c.get("https://mainnet.lighter.xyz/v1/account", headers=headers)
        return r.json()

def calc_size(margin, price):
    return round((margin * LEVERAGE) / price, 6)

def record_close(exit_price):
    entry = stats["entry_price"]
    size  = stats["entry_size"]
    old_m = stats["entry_margin"]
    pnl   = round((exit_price - entry) * size, 4)
    new_m = round(max(old_m + pnl, 1.0), 4)
    stats["total_trades"] += 1
    stats["total_pnl"]    += pnl
    stats["current_margin"] = new_m
    if new_m > stats["peak_margin"]:
        stats["peak_margin"] = new_m
    if pnl >= 0:
        stats["wins"] += 1
        result = "✅ WIN"
    else:
        stats["losses"] += 1
        result = "❌ LOSS"
    stats["history"].append({
        "no": stats["total_trades"], "entry": entry, "exit": exit_price,
        "pnl": pnl, "old_margin": old_m, "new_margin": new_m,
        "result": result, "time": datetime.now().strftime("%Y-%m-%d %H:%M")
    })
    stats["history"] = stats["history"][-10:]
    save_stats(stats)
    return pnl, new_m, result

async def send_tg(msg):
    if bot_app:
        try:
            await bot_app.bot.send_message(
                chat_id=TELEGRAM_CHAT_ID, text=msg, parse_mode="Markdown"
            )
        except Exception as e:
            logger.error(f"TG error: {e}")

async def strategy_loop():
    global in_position, stats
    logger.info("Strategy loop started!")
    await send_tg(
        "🤖 *Lighter RSI Bot Started!*\n"
        "━━━━━━━━━━━━━━━\n"
        f"📊 ETH | ⚡ {LEVERAGE}x | ⏱ 15min\n"
        f"🟢 BUY: RSI < {RSI_BUY}\n"
        f"🔴 CLOSE: RSI > {RSI_CLOSE}\n"
        f"💵 Margin: `${stats['current_margin']}`\n"
        "✅ Monitoring..."
    )
    while True:
        try:
            closes = await fetch_closes(50)
            price  = closes[-1]
            rsi    = calc_rsi(closes, RSI_PERIOD)
            ts     = datetime.now().strftime("%H:%M:%S")
            logger.info(f"{ts} ETH=${price:.2f} RSI={rsi} pos={in_position} margin=${stats['current_margin']}")

            if rsi < RSI_BUY and not in_position:
                margin = stats["current_margin"]
                size   = calc_size(margin, price)
                await send_tg(
                    f"🟢 *BUY SIGNAL*\n"
                    f"RSI: `{rsi}` | ETH: `${price:,.2f}`\n"
                    f"💵 `${margin}` × `{LEVERAGE}x` = `${margin*LEVERAGE:.2f}`\n"
                    f"📦 Size: `{size} ETH`"
                )
                try:
                    await lighter_order("BUY", size)
                    in_position = True
                    stats["entry_price"]  = price
                    stats["entry_size"]   = size
                    stats["entry_margin"] = margin
                    save_stats(stats)
                    await send_tg(f"✅ *LONG Opened!* Entry: `${price:,.2f}`")
                except Exception as e:
                    await send_tg(f"❌ BUY Failed: `{str(e)}`")

            elif rsi > RSI_CLOSE and in_position:
                await send_tg(f"🔴 *CLOSE SIGNAL*\nRSI: `{rsi}` | ETH: `${price:,.2f}`")
                try:
                    await lighter_order("SELL", stats["entry_size"], reduce_only=True)
                    in_position = False
                    pnl, new_m, outcome = record_close(price)
                    win_rate = round(stats["wins"] / max(stats["total_trades"], 1) * 100, 1)
                    await send_tg(
                        f"{outcome} *Trade #{stats['total_trades']} Closed!*\n"
                        f"Entry: `${stats['entry_price']:,.2f}` → Exit: `${price:,.2f}`\n"
                        f"PnL: `${pnl:+.4f}`\n"
                        f"━━━━━━━━━━━━━━━\n"
                        f"💵 New Margin: `${new_m}` _(next trade)_\n"
                        f"🏆 Win Rate: `{win_rate}%`"
                    )
                except Exception as e:
                    await send_tg(f"❌ CLOSE Failed: `{str(e)}`")

        except Exception as e:
            logger.error(f"Loop error: {e}")
        await asyncio.sleep(CHECK_INTERVAL)

async def cmd_start(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "🤖 *Lighter RSI Compounding Bot*\n"
        f"🟢 BUY RSI < {RSI_BUY} | 🔴 CLOSE RSI > {RSI_CLOSE}\n"
        f"⚡ {LEVERAGE}x | 💵 Margin: ${stats['current_margin']}\n"
        "/status /rsi /stats /history /balance",
        parse_mode="Markdown"
    )

async def cmd_status(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    pos = "🟢 LONG Open" if in_position else "⚪ Waiting"
    await update.message.reply_text(
        f"📈 *Status*\n{pos}\n"
        f"💵 Margin: `${stats['current_margin']}`\n"
        f"⏭ Next position: `${stats['current_margin'] * LEVERAGE:.2f}`",
        parse_mode="Markdown"
    )

async def cmd_rsi(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    try:
        closes = await fetch_closes(50)
        rsi    = calc_rsi(closes, RSI_PERIOD)
        price  = closes[-1]
        zone   = "🟢 BUY Zone" if rsi < RSI_BUY else ("🔴 CLOSE Zone" if rsi > RSI_CLOSE else "🟡 Neutral")
        await update.message.reply_text(
            f"📊 RSI: `{rsi}` | ETH: `${price:,.2f}`\n{zone}",
            parse_mode="Markdown"
        )
    except Exception as e:
        await update.message.reply_text(f"❌ {e}")

async def cmd_stats(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    w = stats["wins"]; l = stats["losses"]; t = stats["total_trades"]
    wr = round(w / max(t, 1) * 100, 1)
    growth = round(stats["current_margin"] - STARTING_MARGIN, 4)
    await update.message.reply_text(
        f"📊 *Stats*\n"
        f"Trades: `{t}` | ✅ `{w}` | ❌ `{l}` | WR: `{wr}%`\n"
        f"💵 Start: `${STARTING_MARGIN}` → Now: `${stats['current_margin']}`\n"
        f"📈 Growth: `${growth:+.4f}`\n"
        f"💹 PnL: `${stats['total_pnl']:+.4f}`",
        parse_mode="Markdown"
    )

async def cmd_history(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    h = stats.get("history", [])
    if not h:
        await update.message.reply_text("📭 No trades yet.")
        return
    msg = "📋 *Last 10 Trades*\n"
    for t in reversed(h):
        msg += f"{t['result']} #{t['no']} | PnL: `${t['pnl']:+.4f}` | Margin: `${t['old_margin']}→${t['new_margin']}`\n"
    await update.message.reply_text(msg, parse_mode="Markdown")

async def cmd_balance(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    try:
        d = await lighter_balance()
        await update.message.reply_text(
            f"💰 Collateral: `${float(d.get('collateral',0)):,.2f}`\n"
            f"📊 Equity: `${float(d.get('equity',0)):,.2f}`",
            parse_mode="Markdown"
        )
    except Exception as e:
        await update.message.reply_text(f"❌ {e}")

async def main():
    global bot_app
    bot_app = Application.builder().token(TELEGRAM_BOT_TOKEN).build()
    for cmd, handler in [
        ("start", cmd_start), ("status", cmd_status),
        ("rsi", cmd_rsi), ("stats", cmd_stats),
        ("history", cmd_history), ("balance", cmd_balance)
    ]:
        bot_app.add_handler(CommandHandler(cmd, handler))

    await bot_app.initialize()
    await bot_app.start()
    await bot_app.updater.start_polling(drop_pending_updates=True)

    asyncio.create_task(strategy_loop())
    await asyncio.Event().wait()

if __name__ == "__main__":
    asyncio.run(main())
  
