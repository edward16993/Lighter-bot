# -*- coding: utf-8 -*-
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
LEVERAGE       = 6
BB_PERIOD      = 20
BB_STD         = 2
RSI_PERIOD     = 14
RSI_BUY        = 45
CHECK_INTERVAL = 900

# Lighter API correct scaling (official docs):
# base_amount = size * 10000  (0.01 ETH â†’ 100)
# price_int   = price * 100   ($3100   â†’ 310000)
BASE_SCALE  = 10_000
PRICE_SCALE = 100

MARKETS = {
    "ETH": {
        "symbol":       "ETHUSDT",
        "market_index": 0,
        "stats_file":   "eth_stats.json",
        "decimals":     4,
        "icon":         "ðŸ”·",
        "start_margin": 5.0,
    },
    "HYPE": {
        "symbol":       "HYPEUSDT",
        "market_index": 3,
        "stats_file":   "hype_stats.json",
        "decimals":     4,
        "icon":         "ðŸ”¶",
        "start_margin": 1.0,
    },
}

positions     = {t: False for t in MARKETS}
all_stats     = {}
signer_client = None
tg_app        = None

# â”€â”€â”€ Stats â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def load_stats(token):
    try:
        if os.path.exists(MARKETS[token]["stats_file"]):
            with open(MARKETS[token]["stats_file"]) as f:
                return json.load(f)
    except:
        pass
    s = MARKETS[token]["start_margin"]
    return {"current_margin": s, "total_trades": 0, "wins": 0, "losses": 0,
            "total_pnl": 0.0, "peak_margin": s, "entry_price": 0.0,
            "entry_size": 0.0, "entry_margin": 0.0, "history": []}

def save_stats(token):
    with open(MARKETS[token]["stats_file"], "w") as f:
        json.dump(all_stats[token], f, indent=2)

for t in MARKETS:
    all_stats[t] = load_stats(t)

# â”€â”€â”€ Market data â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

async def fetch_closes(symbol, limit=100):
    endpoints = [
        "https://fapi.binance.com/fapi/v1/klines",  # Futures first
        "https://api.binance.com/api/v3/klines",     # Spot fallback
    ]
    async with httpx.AsyncClient(timeout=15) as c:
        for url in endpoints:
            try:
                r    = await c.get(url, params={"symbol": symbol, "interval": "15m", "limit": limit})
                data = r.json()
                if isinstance(data, list) and len(data) > 0:
                    return [float(x[4]) for x in data]
            except Exception:
                continue
    raise Exception(f"No data for {symbol}: not found on Futures or Spot")

# â”€â”€â”€ Indicators â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def calc_indicators(closes):
    s   = pd.Series(closes)
    sma = s.rolling(BB_PERIOD).mean()
    std = s.rolling(BB_PERIOD).std()
    upper  = round(float((sma + BB_STD * std).iloc[-1]), 4)
    middle = round(float(sma.iloc[-1]), 4)
    lower  = round(float((sma - BB_STD * std).iloc[-1]), 4)
    d    = s.diff()
    gain = d.where(d > 0, 0.0).ewm(com=RSI_PERIOD - 1, adjust=False).mean()
    loss = (-d.where(d < 0, 0.0)).ewm(com=RSI_PERIOD - 1, adjust=False).mean()
    rsi  = round(float((100 - (100 / (1 + gain / loss))).iloc[-1]), 2)
    p    = closes[-1]
    return upper, middle, lower, rsi, p <= lower and rsi < RSI_BUY, p >= upper

# â”€â”€â”€ Orders â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def calc_size(token, margin, price):
    """notional = margin x leverage -> size = notional / price"""
    notional = margin * LEVERAGE
    raw      = notional / price
    return round(raw, MARKETS[token]["decimals"])

async def place_order(token, side, size, price, reduce_only=False):
    base_amount = int(size * BASE_SCALE)
    price_int   = int(price * PRICE_SCALE)
    logger.info(
        f"ORDER {token} {side} | size={size} base_amount={base_amount} "
        f"price=${price} price_int={price_int} reduce_only={reduce_only}"
    )
    tx, tx_hash, err = await signer_client.create_order(
        market_index=MARKETS[token]["market_index"],
        client_order_index=int(datetime.now().timestamp()),
        base_amount=base_amount,
        price=price_int,
        is_ask=(side == "SELL"),
        order_type=signer_client.ORDER_TYPE_MARKET,
        time_in_force=signer_client.ORDER_TIME_IN_FORCE_IMMEDIATE_OR_CANCEL,
        reduce_only=reduce_only,
        order_expiry=signer_client.DEFAULT_IOC_EXPIRY,
    )
    if err:
        raise Exception(str(err))
    return tx_hash

def record_close(token, exit_price):
    s     = all_stats[token]
    pnl   = round((exit_price - s["entry_price"]) * s["entry_size"], 4)
    new_m = round(max(s["entry_margin"] + pnl, 0.5), 4)
    s["total_trades"] += 1
    s["total_pnl"]    += pnl
    s["current_margin"] = new_m
    if new_m > s["peak_margin"]:
        s["peak_margin"] = new_m
    result = "âœ… WIN" if pnl >= 0 else "âŒ LOSS"
    if pnl >= 0: s["wins"]   += 1
    else:        s["losses"] += 1
    s["history"].append({
        "no": s["total_trades"], "entry": s["entry_price"],
        "exit": exit_price, "pnl": pnl,
        "old_margin": s["entry_margin"], "new_margin": new_m,
        "result": result, "time": datetime.now().strftime("%Y-%m-%d %H:%M"),
    })
    s["history"] = s["history"][-10:]
    save_stats(token)
    return pnl, new_m, result

# â”€â”€â”€ Telegram â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

async def send_tg(msg):
    try:
        await tg_app.bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=msg, parse_mode="Markdown")
    except Exception as e:
        logger.error(f"TG: {e}")

# â”€â”€â”€ Trading loop â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

async def token_loop(token):
    icon = MARKETS[token]["icon"]
    while True:
        try:
            closes = await fetch_closes(MARKETS[token]["symbol"])
            price  = closes[-1]
            upper, middle, lower, rsi, buy_sig, close_sig = calc_indicators(closes)
            s = all_stats[token]
            logger.info(f"{token}=${price} RSI={rsi} Buy={buy_sig} Close={close_sig}")

            if buy_sig and not positions[token]:
                margin   = s["current_margin"]
                size     = calc_size(token, margin, price)
                notional = round(size * price, 2)
                await send_tg(
                    f"{icon} *{token} BUY!*\n"
                    f"Price:`${price}` <= Lower:`${lower}`\n"
                    f"RSI:`{rsi}` | Margin:`${margin}` x `{LEVERAGE}x`\n"
                    f"Size:`{size} {token}` (~`${notional}`) | Executing..."
                )
                try:
                    await place_order(token, "BUY", size, price)
                    positions[token]  = True
                    s["entry_price"]  = price
                    s["entry_size"]   = size
                    s["entry_margin"] = margin
                    save_stats(token)
                    await send_tg(f"âœ… *{token} LONG Opened!* Entry:`${price}`")
                except Exception as e:
                    positions[token] = False
                    await send_tg(f"âŒ {token} BUY Failed: `{e}`")

            elif close_sig and positions[token]:
                await send_tg(
                    f"{icon} *{token} CLOSE!* Price:`${price}` >= Upper:`${upper}`"
                )
                try:
                    await place_order(token, "SELL", s["entry_size"], price, reduce_only=True)
                    positions[token] = False
                    pnl, new_m, outcome = record_close(token, price)
                    wr = round(s["wins"] / max(s["total_trades"], 1) * 100, 1)
                    await send_tg(
                        f"{outcome} *{token} #{s['total_trades']}*\n"
                        f"PnL:`${pnl:+.4f}` | Margin:`${new_m}`\n"
                        f"WR:`{wr}%` | Next:`${new_m}` x `{LEVERAGE}x` ðŸš€"
                    )
                except Exception as e:
                    positions[token] = False
                    await send_tg(f"âŒ {token} CLOSE Failed: `{e}`")

        except Exception as e:
            logger.error(f"{token}: {e}")
        await asyncio.sleep(CHECK_INTERVAL)

async def strategy_loop():
    await send_tg(
        "ðŸ¤– *Bot Started!*\n"
        "ðŸ”· ETH `$5.0` | ðŸ”¶ HYPE `$1.0`\n"
        f"âš¡ {LEVERAGE}x | BB+RSI | 15min âœ…"
    )
    await asyncio.gather(token_loop("ETH"), token_loop("HYPE"))

# â”€â”€â”€ Commands â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

async def cmd_start(u: Update, c: ContextTypes.DEFAULT_TYPE):
    await u.message.reply_text(
        f"ðŸ¤– ETH:`${all_stats['ETH']['current_margin']}` | HYPE:`${all_stats['HYPE']['current_margin']}`\n"
        "/status /bb /stats /history /balance",
        parse_mode="Markdown"
    )

async def cmd_status(u: Update, c: ContextTypes.DEFAULT_TYPE):
    try:
        msg = ""
        for token in MARKETS:
            closes = await fetch_closes(MARKETS[token]["symbol"])
            price  = closes[-1]
            _,_,_,rsi,_,_ = calc_indicators(closes)
            status = "ðŸŸ¢ LONG" if positions[token] else "âšª Wait"
            icon   = MARKETS[token]["icon"]
            msg += (
                f"{icon} *{token}*: `${price}` RSI:`{rsi}` {status}\n"
                f"Margin:`${all_stats[token]['current_margin']}`\n"
                f"â”â”â”â”â”â”â”â”â”â”â”â”â”\n"
            )
        await u.message.reply_text(msg.strip(), parse_mode="Markdown")
    except Exception as e:
        await u.message.reply_text(f"âŒ Error: {e}")

async def cmd_bb(u: Update, c: ContextTypes.DEFAULT_TYPE):
    try:
        msg = ""
        for token in MARKETS:
            closes             = await fetch_closes(MARKETS[token]["symbol"])
            u2,m2,l2,rsi,buy,close = calc_indicators(closes)
            sig  = "ðŸŸ¢ BUY!" if buy else ("ðŸ”´ CLOSE!" if close else "ðŸŸ¡ Wait")
            icon = MARKETS[token]["icon"]
            msg += f"{icon} *{token}* RSI:`{rsi}` {sig}\nU:`${u2}` M:`${m2}` L:`${l2}`\n"
        await u.message.reply_text(msg, parse_mode="Markdown")
    except Exception as e:
        await u.message.reply_text(f"âŒ {e}")

async def cmd_stats(u: Update, c: ContextTypes.DEFAULT_TYPE):
    msg = ""
    for token in MARKETS:
        s  = all_stats[token]
        t  = s["total_trades"]
        wr = round(s["wins"] / max(t, 1) * 100, 1)
        g  = round(s["current_margin"] - MARKETS[token]["start_margin"], 4)
        msg += (
            f"{MARKETS[token]['icon']} *{token}* "
            f"T:`{t}` WR:`{wr}%` M:`${s['current_margin']}` G:`${g:+.4f}`\n"
        )
    await u.message.reply_text(msg, parse_mode="Markdown")

async def cmd_history(u: Update, c: ContextTypes.DEFAULT_TYPE):
    msg = ""
    for token in MARKETS:
        h    = all_stats[token].get("history", [])
        icon = MARKETS[token]["icon"]
        msg += f"{icon} *{token}*\n"
        if not h:
            msg += "No trades yet\n"
        else:
            for tr in reversed(h[-3:]):
                msg += (
                    f"{tr['result']} #{tr['no']} "
                    f"`${tr['pnl']:+.4f}` "
                    f"`${tr['old_margin']}â†’${tr['new_margin']}`\n"
                )
    await u.message.reply_text(msg, parse_mode="Markdown")

async def cmd_balance(u: Update, c: ContextTypes.DEFAULT_TYPE):
    try:
        api = lighter.ApiClient(lighter.Configuration(host=BASE_URL))
        acc = lighter.AccountApi(api)
        r   = await acc.account(account_index=ACCOUNT_INDEX)
        await api.close()
        await u.message.reply_text(
            f"ðŸ’° Collateral:`{r.collateral}`",
            parse_mode="Markdown"
        )
    except Exception as e:
        await u.message.reply_text(f"âŒ {e}")

# â”€â”€â”€ Main â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

async def main():
    global tg_app, signer_client

    signer_client = lighter.SignerClient(
        url=BASE_URL,
        api_private_keys={API_KEY_INDEX: LIGHTER_PRIVATE_KEY},
        account_index=ACCOUNT_INDEX,
    )

    tg_app = (
        Application.builder()
        .token(TELEGRAM_BOT_TOKEN)
        .updater(None)
        .build()
    )

    for cmd, fn in [
        ("start",   cmd_start),
        ("status",  cmd_status),
        ("bb",      cmd_bb),
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
                offset=offset, timeout=10, allowed_updates=["message"]
            )
            for update in updates:
                offset = update.update_id + 1
                await tg_app.process_update(update)
        except Exception as e:
            logger.error(f"Polling error: {e}")
        await asyncio.sleep(1)

if __name__ == "__main__":
    asyncio.run(main())
