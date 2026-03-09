import os, asyncio, logging, json
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
RSI_PERIOD     = 14
RSI_BUY        = 70       # RSI > 70 → BUY
RSI_CLOSE      = 40       # RSI < 40 → CLOSE
EMA_PERIOD     = 50       # Price > EMA50 → trend confirm
STOP_LOSS_PCT  = 0.03     # 3% stop loss
CHECK_INTERVAL = 900      # 15 min

ETH_MARKET = {
    "symbol":       "ETH-USDT-SWAP",
    "market_index": 0,
    "stats_file":   "eth_stats.json",
    "decimals":     2,
    "min_size":     0.002,
    "leverage":     5,
    "start_margin": float(os.environ.get("ETH_MARGIN", "8")),
}

position      = False
entry_price   = 0.0
entry_size    = 0.0
entry_margin  = 0.0
sl_price      = 0.0
stats         = {}
signer_client = None
tg_app        = None

def load_stats():
    try:
        if os.path.exists(ETH_MARKET["stats_file"]):
            with open(ETH_MARKET["stats_file"]) as f:
                return json.load(f)
    except: pass
    m = ETH_MARKET["start_margin"]
    return {"current_margin": m, "total_trades": 0, "wins": 0, "losses": 0,
            "total_pnl": 0.0, "peak_margin": m, "entry_price": 0.0,
            "entry_size": 0.0, "entry_margin": 0.0, "history": []}

def save_stats():
    with open(ETH_MARKET["stats_file"], "w") as f:
        json.dump(stats, f, indent=2)

async def fetch_closes(limit=100):
    async with httpx.AsyncClient(timeout=15) as c:
        r = await c.get(
            "https://www.okx.com/api/v5/market/candles",
            params={"instId": "ETH-USDT-SWAP", "bar": "15m", "limit": str(limit)}
        )
        data = r.json()
        if data.get("code") != "0" or not data.get("data"):
            raise Exception(f"OKX ETH error: {data}")
        return [float(x[4]) for x in reversed(data["data"])]

def calc_indicators(closes):
    s     = pd.Series(closes)
    # RSI
    delta = s.diff()
    gain  = delta.where(delta > 0, 0).ewm(com=RSI_PERIOD-1, adjust=False).mean()
    loss  = (-delta.where(delta < 0, 0)).ewm(com=RSI_PERIOD-1, adjust=False).mean()
    rsi   = round(float((100 - (100 / (1 + gain / loss))).iloc[-1]), 2)
    # EMA50
    ema50 = round(float(s.ewm(span=EMA_PERIOD, adjust=False).mean().iloc[-1]), 2)
    price = closes[-1]
    above_ema = price > ema50
    # Signals
    buy_sig   = rsi > RSI_BUY and above_ema
    close_sig = rsi < RSI_CLOSE
    return rsi, ema50, above_ema, buy_sig, close_sig

async def place_order(side, size, price, reduce_only=False):
    if side == "BUY":
        order_price = int(price * 1.05 * 100)
    else:
        order_price = int(price * 0.95 * 100)
    base_amt = int(size * 10000)
    logger.info(f"Order: {side} {size} ETH base={base_amt} price={order_price}")
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
    new_m = round(max(stats["entry_margin"] + pnl, 0.5), 4)
    stats["total_trades"] += 1
    stats["total_pnl"]    += pnl
    stats["current_margin"] = new_m
    if new_m > stats["peak_margin"]: stats["peak_margin"] = new_m
    result = "✅ WIN" if pnl >= 0 else "❌ LOSS"
    if pnl >= 0: stats["wins"] += 1
    else:        stats["losses"] += 1
    stats["history"].append({
        "no": stats["total_trades"], "entry": stats["entry_price"],
        "exit": exit_price, "pnl": pnl, "reason": reason,
        "old_margin": stats["entry_margin"], "new_margin": new_m,
        "result": result, "time": datetime.now().strftime("%Y-%m-%d %H:%M")
    })
    stats["history"] = stats["history"][-10:]
    save_stats()
    return pnl, new_m, result

async def send_tg(msg):
    try:
        await tg_app.bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=msg, parse_mode="Markdown")
    except Exception as e:
        logger.error(f"TG: {e}")

async def strategy_loop():
    global position, entry_price, entry_size, entry_margin, sl_price, stats

    # Check open positions on startup
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
                    sl_price     = entry_price * (1 - STOP_LOSS_PCT) if entry_price > 0 else 0
                    logger.info("ETH position found on startup!")
    except Exception as e:
        logger.error(f"Position check: {e}")

    eth_pos = "🟢 LONG" if position else "⚪ None"
    await send_tg(
        "🤖 *ETH Bot Started!*\n"
        f"🔷 ETH `${stats['current_margin']}` | {eth_pos}\n"
        f"⚡ {LEVERAGE}x | 15min | OKX data\n"
        f"📈 RSI>`{RSI_BUY}` + Price>EMA`{EMA_PERIOD}` → BUY\n"
        f"📉 RSI<`{RSI_CLOSE}` → CLOSE\n"
        f"🛑 SL `{int(STOP_LOSS_PCT*100)}%` ✅"
    )

    while True:
        try:
            closes = await fetch_closes()
            price  = closes[-1]
            rsi, ema50, above_ema, buy_sig, close_sig = calc_indicators(closes)
            ema_status = "✅" if above_ema else "❌"
            logger.info(f"ETH=${price:.2f} RSI={rsi} EMA50={ema50} Above={above_ema} Pos={position}")

            # BUY: RSI > 70 AND price > EMA50
            if not position and buy_sig:
                margin = stats["current_margin"]
                size   = calc_size(margin, price)
                await send_tg(
                    f"🔷 *ETH BUY!*\n"
                    f"RSI:`{rsi}` > `{RSI_BUY}` 📈\n"
                    f"Price:`${price:.2f}` > EMA50:`${ema50}` {ema_status}\n"
                    f"Size:`{size} ETH` | Margin:`${margin}` × `{LEVERAGE}x`\n"
                    f"SL:`${price*(1-STOP_LOSS_PCT):.2f}` (-{int(STOP_LOSS_PCT*100)}%)\n"
                    f"_Executing..._"
                )
                try:
                    await place_order("BUY", size, price)
                    position     = True
                    entry_price  = price
                    entry_size   = size
                    entry_margin = margin
                    sl_price     = price * (1 - STOP_LOSS_PCT)
                    stats["entry_price"]  = price
                    stats["entry_size"]   = size
                    stats["entry_margin"] = margin
                    save_stats()
                    await send_tg(
                        f"✅ *ETH LONG Opened!*\n"
                        f"Entry:`${price:.2f}` | Size:`{size}`\n"
                        f"SL:`${sl_price:.2f}`"
                    )
                except Exception as e:
                    position = False
                    await send_tg(f"❌ ETH BUY Failed: `{e}`")

            # CLOSE: RSI < 40
            elif position and close_sig:
                await send_tg(
                    f"🔷 *ETH CLOSE!*\n"
                    f"RSI:`{rsi}` < `{RSI_CLOSE}` 📉\n"
                    f"Price:`${price:.2f}`"
                )
                try:
                    await place_order("SELL", entry_size, price, reduce_only=True)
                    position = False
                    pnl, new_m, outcome = record_close(price, "RSI")
                    wr = round(stats["wins"] / max(stats["total_trades"], 1) * 100, 1)
                    await send_tg(
                        f"{outcome} *ETH #{stats['total_trades']}*\n"
                        f"Entry:`${entry_price:.2f}` → Exit:`${price:.2f}`\n"
                        f"PnL:`${pnl:+.4f}` | WR:`{wr}%`\n"
                        f"Margin:`${new_m}` × `{LEVERAGE}x` 🚀"
                    )
                except Exception as e:
                    position = False
                    await send_tg(f"❌ ETH CLOSE Failed: `{e}`")

            # STOP LOSS: -3%
            elif position and price <= sl_price:
                await send_tg(
                    f"🛑 *ETH STOP LOSS!*\n"
                    f"Price:`${price:.2f}` ≤ SL:`${sl_price:.2f}`\n"
                    f"Loss capped -{int(STOP_LOSS_PCT*100)}%!"
                )
                try:
                    await place_order("SELL", entry_size, price, reduce_only=True)
                    position = False
                    pnl, new_m, outcome = record_close(price, "SL")
                    wr = round(stats["wins"] / max(stats["total_trades"], 1) * 100, 1)
                    await send_tg(
                        f"{outcome} *ETH SL #{stats['total_trades']}*\n"
                        f"Entry:`${entry_price:.2f}` → Exit:`${price:.2f}`\n"
                        f"PnL:`${pnl:+.4f}` | WR:`{wr}%`\n"
                        f"Next Margin:`${new_m}` 🛡️"
                    )
                except Exception as e:
                    position = False
                    await send_tg(f"❌ ETH SL Failed: `{e}`")

            # No signal - log EMA status
            elif not position:
                if rsi > RSI_BUY and not above_ema:
                    logger.info(f"RSI>{RSI_BUY} but Price below EMA50 - waiting for trend!")

        except Exception as e:
            logger.error(f"ETH loop: {e}")
        await asyncio.sleep(CHECK_INTERVAL)

# ═══════════════════════════════════════
# TELEGRAM COMMANDS
# ═══════════════════════════════════════
async def cmd_start(u: Update, c: ContextTypes.DEFAULT_TYPE):
    await u.message.reply_text(
        f"🤖 *ETH RSI+EMA Bot*\n"
        f"Margin:`${stats['current_margin']}` | {'🟢 LONG' if position else '⚪ Wait'}\n"
        f"RSI>`{RSI_BUY}` + EMA`{EMA_PERIOD}` → BUY\n"
        f"RSI<`{RSI_CLOSE}` → CLOSE | SL`{int(STOP_LOSS_PCT*100)}%`\n"
        "/status /rsi /stats /history /balance",
        parse_mode="Markdown")

async def cmd_status(u: Update, c: ContextTypes.DEFAULT_TYPE):
    try:
        closes = await fetch_closes()
        price  = closes[-1]
        rsi, ema50, above_ema, buy_sig, close_sig = calc_indicators(closes)
        pos_status = "🟢 LONG" if position else "⚪ Wait"
        ema_icon   = "✅ Above" if above_ema else "❌ Below"
        signal = ""
        if buy_sig: signal = "\n🚨 BUY SIGNAL!"
        elif close_sig and position: signal = "\n🚨 CLOSE SIGNAL!"
        sl_info = f"\nSL:`${sl_price:.2f}`" if position else ""
        unrealized = ""
        if position and entry_price > 0:
            upnl = round((price - entry_price) * entry_size, 4)
            unrealized = f"\nUnrealized:`${upnl:+.4f}`"
        await u.message.reply_text(
            f"🔷 *ETH Status*\n"
            f"Price:`${price:.2f}` | RSI:`{rsi}`\n"
            f"EMA50:`${ema50}` {ema_icon}\n"
            f"Position: {pos_status}{sl_info}{unrealized}{signal}\n"
            f"Margin:`${stats['current_margin']}`",
            parse_mode="Markdown")
    except Exception as e:
        await u.message.reply_text(f"❌ {e}")

async def cmd_rsi(u: Update, c: ContextTypes.DEFAULT_TYPE):
    try:
        closes = await fetch_closes()
        price  = closes[-1]
        rsi, ema50, above_ema, buy_sig, close_sig = calc_indicators(closes)
        ema_icon = "✅" if above_ema else "❌"
        if buy_sig:     signal = "🚨 BUY Signal!"
        elif close_sig: signal = "📉 CLOSE Signal!"
        else:           signal = "🟡 Wait"
        await u.message.reply_text(
            f"🔷 *ETH Indicators*\n"
            f"Price:`${price:.2f}`\n"
            f"RSI:`{rsi}` (Buy>`{RSI_BUY}` Close<`{RSI_CLOSE}`)\n"
            f"EMA50:`${ema50}` {ema_icon}\n"
            f"Signal: {signal}",
            parse_mode="Markdown")
    except Exception as e:
        await u.message.reply_text(f"❌ {e}")

async def cmd_stats(u: Update, c: ContextTypes.DEFAULT_TYPE):
    t  = stats["total_trades"]
    wr = round(stats["wins"] / max(t, 1) * 100, 1)
    g  = round(stats["current_margin"] - ETH_MARKET["start_margin"], 4)
    await u.message.reply_text(
        f"📊 *ETH Stats*\n"
        f"Trades:`{t}` | WR:`{wr}%`\n"
        f"✅ Wins:`{stats['wins']}` | ❌ Losses:`{stats['losses']}`\n"
        f"Total PnL:`${stats['total_pnl']:.4f}`\n"
        f"Margin:`${stats['current_margin']}` Growth:`${g:+.4f}`\n"
        f"Peak:`${stats['peak_margin']}`",
        parse_mode="Markdown")

async def cmd_history(u: Update, c: ContextTypes.DEFAULT_TYPE):
    h = stats.get("history", [])
    if not h:
        await u.message.reply_text("No trades yet!"); return
    msg = "📜 *ETH Trade History*\n"
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
            col = data.get("account", {}).get("collateral", "N/A")
            upnl= data.get("account", {}).get("total_unrealized_pnl", "0")
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
        ("rsi",     cmd_rsi),
        ("stats",   cmd_stats),
        ("history", cmd_history),
        ("balance", cmd_balance),
    ]:
        tg_app.add_handler(CommandHandler(cmd, fn))

    await tg_app.initialize()
    await tg_app.start()

    offset = None
    asyncio.create_task(strategy_loop())
    while True:
        try:
            updates = await tg_app.bot.get_updates(
                offset=offset, timeout=10, allowed_updates=["message"])
            for update in updates:
                offset = update.update_id + 1
                await tg_app.process_update(update)
        except Exception as e:
            logger.error(f"Polling: {e}")
        await asyncio.sleep(1)

if __name__ == "__main__":
    asyncio.run(main())
    
