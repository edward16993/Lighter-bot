import os, asyncio, logging, json
from datetime import datetime
import pandas as pd
from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes
import lighter

logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

# ─── Environment Variables ────────────────────────────────────
TELEGRAM_BOT_TOKEN  = os.environ["TELEGRAM_BOT_TOKEN"]
TELEGRAM_CHAT_ID    = os.environ["TELEGRAM_CHAT_ID"]
ACCOUNT_INDEX       = int(os.environ["ACCOUNT_INDEX"])
API_KEY_INDEX       = int(os.environ["API_KEY_INDEX"])
LIGHTER_PRIVATE_KEY = os.environ["LIGHTER_PRIVATE_KEY"]

# ─── Config ───────────────────────────────────────────────────
BASE_URL       = "https://mainnet.zklighter.elliot.ai"
LEVERAGE       = 6
BB_PERIOD      = 20
BB_STD         = 2
RSI_PERIOD     = 14
RSI_BUY        = 45
CHECK_INTERVAL = 900  # 15 minutes

MARKETS = {
    "ETH": {
        "symbol":       "ETHUSDT",
        "market_index": 0,
        "stats_file":   "eth_stats.json",
        "decimals":     2,
        "size_decimals": 3,
        "start_margin": float(os.environ.get("ETH_MARGIN", "5")),
    },
    "HYPE": {
        "symbol":       "HYPEUSDT",
        "market_index": 3,
        "stats_file":   "hype_stats.json",
        "decimals":     4,
        "size_decimals": 2,
        "start_margin": float(os.environ.get("HYPE_MARGIN", "1")),
    }
}

bot_app      = None
positions    = {"ETH": False, "HYPE": False}
all_stats    = {}
signer_client = None

# ─── Stats ────────────────────────────────────────────────────
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
        "current_margin": start, "total_trades": 0,
        "wins": 0, "losses": 0, "total_pnl": 0.0,
        "peak_margin": start, "entry_price": 0.0,
        "entry_size": 0.0, "entry_margin": 0.0, "history": []
    }

def save_stats(token, s):
    try:
        with open(MARKETS[token]["stats_file"], "w") as f:
            json.dump(s, f, indent=2)
    except Exception as e:
        logger.error(f"Save error: {e}")

for t in MARKETS:
    all_stats[t] = load_stats(t)

# ─── Indicators ───────────────────────────────────────────────
async def fetch_closes(symbol, limit=100):
    import httpx
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

# ─── Lighter SDK Order ────────────────────────────────────────
async def place_order(token, side, size, price, reduce_only=False):
    cfg          = MARKETS[token]
    # price format: integer (e.g. $1980.50 → 198050)
    price_int    = int(price * 100)
    base_amount  = int(size * 1000)  # 0.001 ETH precision

    tx, tx_hash, err = await signer_client.create_order(
        market_index      = cfg["market_index"],
        client_order_index= int(datetime.now().timestamp()),
        base_amount       = base_amount,
        price             = price_int,
        is_ask            = (side == "SELL"),
        order_type        = signer_client.ORDER_TYPE_MARKET,
        time_in_force     = signer_client.ORDER_TIME_IN_FORCE_IMMEDIATE_OR_CANCEL,
        reduce_only       = reduce_only,
        order_expiry      = signer_client.DEFAULT_IOC_EXPIRY,
    )
    if err:
        raise Exception(f"Order error: {err}")
    return tx_hash

def calc_size(token, margin, price):
    return round((margin * LEVERAGE) / price, MARKETS[token]["size_decimals"])

def record_close(token, exit_price):
    s     = all_stats[token]
    pnl   = round((exit_price - s["entry_price"]) * s["entry_size"], 4)
    new_m = round(max(s["entry_margin"] + pnl, 0.5), 4)
    s["total_trades"] += 1
    s["total_pnl"]    += pnl
    s["current_margin"] = new_m
    if new_m > s["peak_margin"]:
        s["peak_margin"] = new_m
    result = "✅ WIN" if pnl >= 0 else "❌ LOSS"
    if pnl >= 0: s["wins"] += 1
    else:        s["losses"] += 1
    s["history"].append({
        "no": s["total_trades"], "entry": s["entry_price"],
        "exit": exit_price, "pnl": pnl,
        "old_margin": s["entry_margin"], "new_margin": new_m,
        "result": result, "time": datetime.now().strftime("%Y-%m-%d %H:%M")
    })
    s["history"] = s["history"][-10:]
    save_stats(token, s)
    return pnl, new_m, result

# ─── Telegram ─────────────────────────────────────────────────
async def send_tg(msg):
    if bot_app:
        try:
            await bot_app.bot.send_message(
                chat_id=TELEGRAM_CHAT_ID, text=msg, parse_mode="Markdown"
            )
        except Exception as e:
            logger.error(f"TG error: {e}")

# ─── Strategy Loop ────────────────────────────────────────────
async def token_loop(token):
    cfg  = MARKETS[token]
    icon = "🔷" if token == "ETH" else "🔶"

    while True:
        try:
            closes = await fetch_closes(cfg["symbol"])
            price  = closes[-1]
            upper, middle, lower, rsi, buy_signal, close_signal = calc_indicators(closes)
            dec    = cfg["decimals"]
            s      = all_stats[token]

            logger.info(f"{token}=${price:.{dec}f} RSI={rsi} Buy={buy_signal} Close={close_signal} Margin=${s['current_margin']}")

            # ── BUY ───────────────────────────────────────────
            if buy_signal and not positions[token]:
                margin = s["current_margin"]
                size   = calc_size(token, margin, price)
                await send_tg(
                    f"{icon} *{token} BUY SIGNAL!*\n"
                    f"━━━━━━━━━━━━━━━\n"
                    f"📉 Price:`${price:.{dec}f}` ≤ Lower:`${lower}`\n"
                    f"📊 RSI:`{rsi}`\n"
                    f"💵 `${margin}` × `{LEVERAGE}x` = `${margin*LEVERAGE:.2f}`\n"
                    f"📦 Size:`{size} {token}`\n_Executing..._"
                )
                try:
                    tx_hash = await place_order(token, "BUY", size, price)
                    positions[token]  = True
                    s["entry_price"]  = price
                    s["entry_size"]   = size
                    s["entry_margin"] = margin
                    save_stats(token, s)
                    await send_tg(
                        f"✅ *{token} LONG Opened!*\n"
                        f"Entry:`${price:.{dec}f}` | Size:`{size}`\n"
                        f"TX:`{tx_hash[:16]}...`"
                    )
                except Exception as e:
                    positions[token] = False
                    await send_tg(f"❌ {token} BUY Failed: `{str(e)}`")

            # ── CLOSE ─────────────────────────────────────────
            elif close_signal and positions[token]:
                await send_tg(
                    f"{icon} *{token} CLOSE!*\n"
                    f"Price:`${price:.{dec}f}` ≥ Upper:`${upper}`\n_Closing..._"
                )
                try:
                    tx_hash = await place_order(token, "SELL", s["entry_size"], price, reduce_only=True)
                    positions[token] = False
                    pnl, new_m, outcome = record_close(token, price)
                    wr = round(s["wins"] / max(s["total_trades"], 1) * 100, 1)
                    await send_tg(
                        f"{outcome} *{token} #{s['total_trades']} Closed!*\n"
                        f"━━━━━━━━━━━━━━━\n"
                        f"Entry:`${s['entry_price']:.{dec}f}` → Exit:`${price:.{dec}f}`\n"
                        f"💰 PnL:`${pnl:+.4f}` | Margin:`${new_m}`\n"
                        f"🏆 WR:`{wr}%` | Next:`${new_m}`×`{LEVERAGE}x` 🚀"
                    )
                except Exception as e:
                    positions[token] = False
                    await send_tg(f"❌ {token} CLOSE Failed: `{str(e)}`")

        except Exception as e:
            logger.error(f"{token} error: {e}")
        await asyncio.sleep(CHECK_INTERVAL)

async def strategy_loop():
    await send_tg(
        "🤖 *Multi-Token Bot Started!*\n"
        "━━━━━━━━━━━━━━━\n"
        f"🔷 ETH | 💵 `${MARKETS['ETH']['start_margin']}`\n"
        f"🔶 HYPE | 💵 `${MARKETS['HYPE']['start_margin']}`\n"
        f"⚡ {LEVERAGE}x | ⏱ 15min | BB+RSI\n"
        f"🔑 Account: `{ACCOUNT_INDEX}` | Key: `{API_KEY_INDEX}`\n"
        "✅ SDK Connected! Monitoring..."
    )
    await asyncio.gather(token_loop("ETH"), token_loop("HYPE"))

# ─── Telegram Commands ────────────────────────────────────────
async def cmd_start(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "🤖 *Multi-Token Bot*\n"
        f"🔷 ETH:`${all_stats['ETH']['current_margin']}`\n"
        f"🔶 HYPE:`${all_stats['HYPE']['current_margin']}`\n"
        f"⚡ {LEVERAGE}x | BB+RSI\n"
        "/status /bb /stats /history /balance",
        parse_mode="Markdown"
    )

async def cmd_status(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    try:
        ec = await fetch_closes("ETHUSDT")
        hc = await fetch_closes("HYPEUSDT")
        ep = ec[-1]; hp = hc[-1]
        _, _, _, er, _, _ = calc_indicators(ec)
        _, _, _, hr, _, _ = calc_indicators(hc)
        es = "🟢 LONG" if positions["ETH"]  else "⚪ Wait"
        hs = "🟢 LONG" if positions["HYPE"] else "⚪ Wait"
    except:
        ep = hp = er = hr = 0
        es = hs = "N/A"
    await update.message.reply_text(
        f"📊 *Status*\n━━━━━━━━━━━━━━━\n"
        f"🔷 ETH:`${ep:,.2f}` RSI:`{er}` {es}\n"
        f"💵 Margin:`${all_stats['ETH']['current_margin']}`\n"
        f"━━━━━━━━━━━━━━━\n"
        f"🔶 HYPE:`${hp:.4f}` RSI:`{hr}` {hs}\n"
        f"💵 Margin:`${all_stats['HYPE']['current_margin']}`",
        parse_mode="Markdown"
    )

async def cmd_bb(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    try:
        ec = await fetch_closes("ETHUSDT")
        hc = await fetch_closes("HYPEUSDT")
        eu,em,el,er,eb,ec2 = calc_indicators(ec)
        hu,hm,hl,hr,hb,hc2 = calc_indicators(hc)
        ez = "🟢 BUY!" if eb else ("🔴 CLOSE!" if ec2 else "🟡 Wait")
        hz = "🟢 BUY!" if hb else ("🔴 CLOSE!" if hc2 else "🟡 Wait")
        await update.message.reply_text(
            f"📊 *Bollinger Bands*\n━━━━━━━━━━━━━━━\n"
            f"🔷 *ETH* RSI:`{er}` {ez}\n"
            f"U:`${eu}` M:`${em}` L:`${el}`\n"
            f"━━━━━━━━━━━━━━━\n"
            f"🔶 *HYPE* RSI:`{hr}` {hz}\n"
            f"U:`${hu}` M:`${hm}` L:`${hl}`",
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
        msg += f"{icon} *{token}* Trades:`{t}` WR:`{wr}%`\nMargin:`${s['current_margin']}` Growth:`${g:+.4f}`\nPnL:`${s['total_pnl']:+.4f}`\n━━━━━━━━━━━━━━━\n"
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

async def cmd_balance(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    try:
        api_client  = lighter.ApiClient(lighter.Configuration(host=BASE_URL))
        account_api = lighter.AccountApi(api_client)
        resp        = await account_api.account(account_index=ACCOUNT_INDEX)
        await api_client.close()
        col = resp.collateral if hasattr(resp, 'collateral') else "N/A"
        await update.message.reply_text(
            f"💰 *Balance*\nCollateral:`${col}`",
            parse_mode="Markdown"
        )
    except Exception as e:
        await update.message.reply_text(f"❌ {e}")

# ─── Main ─────────────────────────────────────────────────────
async def main():
    global bot_app, signer_client

    # Init Lighter SDK
    signer_client = lighter.SignerClient(
        url              = BASE_URL,
        api_private_keys = {API_KEY_INDEX: LIGHTER_PRIVATE_KEY},
        account_index    = ACCOUNT_INDEX
    )
    logger.info("Lighter SDK initialized!")

    # Init Telegram
    bot_app = Application.builder().token(TELEGRAM_BOT_TOKEN).build()
    for cmd, handler in [
        ("start",   cmd_start),
        ("status",  cmd_status),
        ("bb",      cmd_bb),
        ("stats",   cmd_stats),
        ("history", cmd_history),
        ("balance", cmd_balance),
    ]:
        bot_app.add_handler(CommandHandler(cmd, handler))

    async with bot_app:
        await bot_app.initialize()
        await bot_app.start()
        asyncio.create_task(strategy_loop())
        await bot_app.updater.start_polling(drop_pending_updates=True)
        await asyncio.Event().wait()

if __name__ == "__main__":
    asyncio.run(main())
# Procfile hint: python bot.py
