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

LEVERAGE       = 6
BB_PERIOD      = 20
BB_STD         = 2
RSI_PERIOD     = 14
RSI_BUY        = 45
CHECK_INTERVAL = 900

# ─── Markets Config ───────────────────────────────────────────
MARKETS = {
    "ETH": {
        "symbol":         "ETHUSDT",
        "market_index":   0,
        "stats_file":     "eth_stats.json",
        "decimals":       2,
        "start_margin":   float(os.environ.get("ETH_MARGIN", "5")),
    },
    "HYPE": {
        "symbol":         "HYPEUSDT",
        "market_index":   3,
        "stats_file":     "hype_stats.json",
        "decimals":       4,
        "start_margin":   float(os.environ.get("HYPE_MARGIN", "1")),
    }
}

bot_app   = None
positions = {"ETH": False, "HYPE": False}
all_stats = {}

def load_stats(token):
    f = MARKETS[token]["stats_file"]
    try:
        if os.path.exists(f):
            with open(f) as fp:
                return json.load(fp)
    except:
        pass
    start = MARKETS[token]["start_margin"]
    return {
        "current_margin": start,
        "total_trades": 0, "wins": 0, "losses": 0,
        "total_pnl": 0.0, "peak_margin": start,
        "entry_price": 0.0, "entry_size": 0.0,
        "entry_margin": 0.0, "history": []
    }

def save_stats(token, s):
    try:
        with open(MARKETS[token]["stats_file"], "w") as f:
            json.dump(s, f, indent=2)
    except Exception as e:
        logger.error(f"Save error: {e}")

for t in MARKETS:
    all_stats[t] = load_stats(t)

async def fetch_closes(symbol, limit=100):
    async with httpx.AsyncClient(timeout=15) as c:
        r = await c.get(
            "https://api.binance.com/api/v3/klines",
            params={"symbol": symbol, "interval": "15m", "limit": limit}
        )
        return [float(x[4]) for x in r.json()]

def calc_indicators(closes):
    s      = pd.Series(closes)
    sma    = s.rolling(BB_PERIOD).mean()
    std    = s.rolling(BB_PERIOD).std()
    upper  = round(float((sma + BB_STD * std).iloc[-1]), 4)
    middle = round(float(sma.iloc[-1]), 4)
    lower  = round(float((sma - BB_STD * std).iloc[-1]), 4)
    d      = s.diff()
    gain   = d.where(d > 0, 0.0).ewm(com=RSI_PERIOD-1, adjust=False).mean()
    loss   = (-d.where(d < 0, 0.0)).ewm(com=RSI_PERIOD-1, adjust=False).mean()
    rsi    = round(float((100 - (100 / (1 + gain / loss))).iloc[-1]), 2)
    price  = closes[-1]
    return upper, middle, lower, rsi, price <= lower and rsi < RSI_BUY, price >= upper

async def lighter_order(token, side, size, reduce_only=False):
    payload = {
        "market_index":  MARKETS[token]["market_index"],
        "base_amount":   int(size * 1e6),
        "is_ask":        (side == "SELL"),
        "order_type":    "MARKET",
        "time_in_force": "IOC",
        "reduce_only":   reduce_only,
        "leverage":      LEVERAGE,
    }
    headers = {
        "Authorization": f"Bearer {LIGHTER_API_KEY}",
        "Content-Type":  "application/json"
    }
    async with httpx.AsyncClient(timeout=15) as c:
        r = await c.post("https://mainnet.lighter.xyz/v1/orders", json=payload, headers=headers)
        return r.json()

def calc_size(token, margin, price):
    dec = MARKETS[token]["decimals"]
    return round((margin * LEVERAGE) / price, dec)

def record_close(token, exit_price):
    s     = all_stats[token]
    entry = s["entry_price"]
    size  = s["entry_size"]
    old_m = s["entry_margin"]
    pnl   = round((exit_price - entry) * size, 4)
    new_m = round(max(old_m + pnl, 0.5), 4)
    s["total_trades"] += 1
    s["total_pnl"]    += pnl
    s["current_margin"] = new_m
    if new_m > s["peak_margin"]:
        s["peak_margin"] = new_m
    result = "✅ WIN" if pnl >= 0 else "❌ LOSS"
    if pnl >= 0:
        s["wins"] += 1
    else:
        s["losses"] += 1
    s["history"].append({
        "no": s["total_trades"], "entry": entry,
        "exit": exit_price, "pnl": pnl,
        "old_margin": old_m, "new_margin": new_m,
        "result": result, "time": datetime.now().strftime("%Y-%m-%d %H:%M")
    })
    s["history"] = s["history"][-10:]
    save_stats(token, s)
    return pnl, new_m, result

async def send_tg(msg):
    if bot_app:
        try:
            await bot_app.bot.send_message(
                chat_id=TELEGRAM_CHAT_ID, text=msg, parse_mode="Markdown"
            )
        except Exception as e:
            logger.error(f"TG error: {e}")

async def token_loop(token):
    global positions
    cfg = MARKETS[token]
    icon = "🔷" if token == "ETH" else "🔶"
    logger.info(f"{token} loop started! Margin=${cfg['start_margin']}")

    while True:
        try:
            closes = await fetch_closes(cfg["symbol"])
            price  = closes[-1]
            upper, middle, lower, rsi, buy_signal, close_signal = calc_indicators(closes)
            dec    = cfg["decimals"]
            s      = all_stats[token]
            ts     = datetime.now().strftime("%H:%M:%S")

            logger.info(f"{ts} {token}=${price:.{dec}f} RSI={rsi} Buy={buy_signal} Close={close_signal} Margin=${s['current_margin']}")

            # ── BUY ───────────────────────────────────────────
            if buy_signal and not positions[token]:
                margin = s["current_margin"]
                size   = calc_size(token, margin, price)
                await send_tg(
                    f"{icon} *{token} BUY SIGNAL!*\n"
                    f"━━━━━━━━━━━━━━━\n"
                    f"📉 Price: `${price:.{dec}f}` ≤ Lower: `${lower}`\n"
                    f"📊 RSI: `{rsi}` _(Below {RSI_BUY})_\n"
                    f"💵 Margin: `${margin}` × `{LEVERAGE}x` = `${margin*LEVERAGE:.2f}`\n"
                    f"📦 Size: `{size} {token}`\n"
                    f"_Executing BUY..._"
                )
                try:
                    await lighter_order(token, "BUY", size)
                    positions[token]  = True
                    s["entry_price"]  = price
                    s["entry_size"]   = size
                    s["entry_margin"] = margin
                    save_stats(token, s)
                    await send_tg(f"✅ *{token} LONG Opened!*\nEntry: `${price:.{dec}f}` | Size: `{size} {token}`")
                except Exception as e:
                    positions[token] = False
                    await send_tg(f"❌ {token} BUY Failed: `{str(e)}`")

            # ── CLOSE ─────────────────────────────────────────
            elif close_signal and positions[token]:
                await send_tg(
                    f"{icon} *{token} CLOSE SIGNAL!*\n"
                    f"📈 Price: `${price:.{dec}f}` ≥ Upper: `${upper}`\n"
                    f"_Closing..._"
                )
                try:
                    await lighter_order(token, "SELL", s["entry_size"], reduce_only=True)
                    positions[token] = False
                    pnl, new_m, outcome = record_close(token, price)
                    win_rate = round(s["wins"] / max(s["total_trades"], 1) * 100, 1)
                    await send_tg(
                        f"{outcome} *{token} Trade #{s['total_trades']} Closed!*\n"
                        f"━━━━━━━━━━━━━━━\n"
                        f"Entry: `${s['entry_price']:.{dec}f}` → Exit: `${price:.{dec}f}`\n"
                        f"💰 PnL: `${pnl:+.4f}`\n"
                        f"💵 New Margin: `${new_m}` _(next trade)_\n"
                        f"🏆 Win Rate: `{win_rate}%`\n"
                        f"⏭ Next: `${new_m}` × `{LEVERAGE}x` 🚀"
                    )
                except Exception as e:
                    positions[token] = False
                    await send_tg(f"❌ {token} CLOSE Failed: `{str(e)}`")

        except Exception as e:
            logger.error(f"{token} error: {e}")
        await asyncio.sleep(CHECK_INTERVAL)

async def strategy_loop():
    await send_tg(
        "🤖 *Lighter DEX Multi-Token Bot!*\n"
        "━━━━━━━━━━━━━━━\n"
        f"🔷 ETH | 💵 Start: `${MARKETS['ETH']['start_margin']}`\n"
        f"🔶 HYPE | 💵 Start: `${MARKETS['HYPE']['start_margin']}`\n"
        f"⚡ {LEVERAGE}x | ⏱ 15min\n"
        f"🟢 BUY: Lower Band + RSI < {RSI_BUY}\n"
        f"🔴 CLOSE: Upper Band\n"
        "✅ Both tokens monitoring..."
    )
    await asyncio.gather(
        token_loop("ETH"),
        token_loop("HYPE")
    )

async def cmd_start(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "🤖 *Multi-Token Bot*\n"
        f"🔷 ETH Margin: `${all_stats['ETH']['current_margin']}`\n"
        f"🔶 HYPE Margin: `${all_stats['HYPE']['current_margin']}`\n"
        f"⚡ {LEVERAGE}x | Bollinger+RSI\n"
        "/status /bb /stats /history",
        parse_mode="Markdown"
    )

async def cmd_status(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    try:
        eth_c  = await fetch_closes("ETHUSDT")
        hype_c = await fetch_closes("HYPEUSDT")
        ep = eth_c[-1]; hp = hype_c[-1]
        _, _, _, er, _, _ = calc_indicators(eth_c)
        _, _, _, hr, _, _ = calc_indicators(hype_c)
        ep_str = "🟢 LONG" if positions["ETH"]  else "⚪ Wait"
        hp_str = "🟢 LONG" if positions["HYPE"] else "⚪ Wait"
    except:
        ep = hp = er = hr = 0
        ep_str = hp_str = "N/A"
    await update.message.reply_text(
        f"📊 *Status*\n"
        f"━━━━━━━━━━━━━━━\n"
        f"🔷 ETH: `${ep:,.2f}` | RSI:`{er}` | {ep_str}\n"
        f"💵 Margin: `${all_stats['ETH']['current_margin']}`\n"
        f"━━━━━━━━━━━━━━━\n"
        f"🔶 HYPE: `${hp:.4f}` | RSI:`{hr}` | {hp_str}\n"
        f"💵 Margin: `${all_stats['HYPE']['current_margin']}`",
        parse_mode="Markdown"
    )

async def cmd_bb(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    try:
        eth_c  = await fetch_closes("ETHUSDT")
        hype_c = await fetch_closes("HYPEUSDT")
        eu, em, el, er, eb, ec = calc_indicators(eth_c)
        hu, hm, hl, hr, hb, hc = calc_indicators(hype_c)
        ez = "🟢 BUY!" if eb else ("🔴 CLOSE!" if ec else "🟡 Wait")
        hz = "🟢 BUY!" if hb else ("🔴 CLOSE!" if hc else "🟡 Wait")
        await update.message.reply_text(
            f"📊 *Bollinger Bands*\n"
            f"━━━━━━━━━━━━━━━\n"
            f"🔷 *ETH* | RSI:`{er}` | {ez}\n"
            f"Upper:`${eu}` Mid:`${em}` Lower:`${el}`\n"
            f"━━━━━━━━━━━━━━━\n"
            f"🔶 *HYPE* | RSI:`{hr}` | {hz}\n"
            f"Upper:`${hu}` Mid:`${hm}` Lower:`${hl}`",
            parse_mode="Markdown"
        )
    except Exception as e:
        await update.message.reply_text(f"❌ {e}")

async def cmd_stats(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    msg = "📊 *Stats*\n━━━━━━━━━━━━━━━\n"
    for token in MARKETS:
        s    = all_stats[token]
        t    = s["total_trades"]
        wr   = round(s["wins"] / max(t, 1) * 100, 1)
        g    = round(s["current_margin"] - MARKETS[token]["start_margin"], 4)
        icon = "🔷" if token == "ETH" else "🔶"
        msg += (
            f"{icon} *{token}* | Trades:`{t}` | WR:`{wr}%`\n"
            f"Margin:`${s['current_margin']}` | Growth:`${g:+.4f}`\n"
            f"PnL:`${s['total_pnl']:+.4f}`\n"
            f"━━━━━━━━━━━━━━━\n"
        )
    await update.message.reply_text(msg, parse_mode="Markdown")

async def cmd_history(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    msg = "📋 *History*\n━━━━━━━━━━━━━━━\n"
    for token in MARKETS:
        icon = "🔷" if token == "ETH" else "🔶"
        h    = all_stats[token].get("history", [])
        msg += f"{icon} *{token}*\n"
        if not h:
            msg += "No trades yet\n"
        else:
            for t in reversed(h[-3:]):
                msg += f"{t['result']} #{t['no']} PnL:`${t['pnl']:+.4f}` `${t['old_margin']}→${t['new_margin']}`\n"
        msg += "━━━━━━━━━━━━━━━\n"
    await update.message.reply_text(msg, parse_mode="Markdown")

async def main():
    global bot_app
    bot_app = Application.builder().token(TELEGRAM_BOT_TOKEN).build()
    for cmd, handler in [
        ("start", cmd_start), ("status", cmd_status),
        ("bb", cmd_bb), ("stats", cmd_stats), ("history", cmd_history)
    ]:
        bot_app.add_handler(CommandHandler(cmd, handler))
    await bot_app.initialize()
    await bot_app.start()
    await bot_app.updater.start_polling(drop_pending_updates=True)
    asyncio.create_task(strategy_loop())
    await asyncio.Event().wait()

if __name__ == "__main__":
    asyncio.run(main())
