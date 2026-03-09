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
BB_PERIOD      = 20
BB_STD         = 2
RSI_PERIOD     = 14
RSI_BUY        = 45
CHECK_INTERVAL = 900

MARKETS = {
    "ETH": {
        "symbol": "ETHUSDT", "market_index": 0,
        "stats_file": "eth_stats.json", "decimals": 2,
        "min_size": 0.002,
        "leverage": 5,
        "start_margin": float(os.environ.get("ETH_MARGIN", "5")),
    },
    "HYPE": {
        "symbol": "HYPEUSDT", "market_index": 24,
        "stats_file": "hype_stats.json", "decimals": 2,
        "min_size": 0.50,
        "leverage": 5,
        "start_margin": float(os.environ.get("HYPE_MARGIN", "1")),
    },
    "LIT": {
        "symbol": "LITUSDT", "market_index": 120,
        "stats_file": "lit_stats.json", "decimals": 2,
        "min_size": 2.0,
        "leverage": 3,
        "start_margin": float(os.environ.get("LIT_MARGIN", "1.5")),
    },
    "SOL": {
        "symbol": "SOLUSDT", "market_index": 2,
        "stats_file": "sol_stats.json", "decimals": 3,
        "min_size": 0.05,
        "leverage": 5,
        "start_margin": float(os.environ.get("SOL_MARGIN", "1")),
    }
}

# ETH uses Binance, HYPE uses OKX (no symbol override needed)

positions     = {"ETH": False, "HYPE": False, "LIT": False, "SOL": False}
all_stats     = {}
signer_client = None
tg_app        = None

def load_stats(token):
    try:
        if os.path.exists(MARKETS[token]["stats_file"]):
            with open(MARKETS[token]["stats_file"]) as f:
                return json.load(f)
    except: pass
    s = MARKETS[token]["start_margin"]
    return {"current_margin": s, "total_trades": 0, "wins": 0, "losses": 0,
            "total_pnl": 0.0, "peak_margin": s, "entry_price": 0.0,
            "entry_size": 0.0, "entry_margin": 0.0, "history": []}

def save_stats(token):
    with open(MARKETS[token]["stats_file"], "w") as f:
        json.dump(all_stats[token], f, indent=2)

for t in MARKETS:
    all_stats[t] = load_stats(t)

async def fetch_closes(symbol, limit=100):
    async with httpx.AsyncClient(timeout=15) as c:
        if symbol == "HYPEUSDT":
            # OKX API for HYPE
            r = await c.get(
                "https://www.okx.com/api/v5/market/candles",
                params={"instId": "HYPE-USDT", "bar": "15m", "limit": str(limit)}
            )
            data = r.json()
            if data.get("code") != "0" or not data.get("data"):
                raise Exception(f"OKX error for HYPE: {data}")
            return [float(x[4]) for x in reversed(data["data"])]
        elif symbol == "LITUSDT":
            # OKX API for LIT
            r = await c.get(
                "https://www.okx.com/api/v5/market/candles",
                params={"instId": "LIT-USDT", "bar": "15m", "limit": str(limit)}
            )
            data = r.json()
            if data.get("code") != "0" or not data.get("data"):
                raise Exception(f"OKX error for LIT: {data}")
            return [float(x[4]) for x in reversed(data["data"])]
        else:
            # Binance for ETH and others
            r = await c.get(
                "https://api.binance.com/api/v3/klines",
                params={"symbol": symbol, "interval": "15m", "limit": limit}
            )
            data = r.json()
            if not isinstance(data, list) or len(data) == 0:
                raise Exception(f"No data for {symbol}: {data}")
            return [float(x[4]) for x in data]

def calc_indicators(closes):
    s   = pd.Series(closes)
    sma = s.rolling(BB_PERIOD).mean()
    std = s.rolling(BB_PERIOD).std()
    upper  = round(float((sma + BB_STD * std).iloc[-1]), 4)
    middle = round(float(sma.iloc[-1]), 4)
    lower  = round(float((sma - BB_STD * std).iloc[-1]), 4)
    d    = s.diff()
    gain = d.where(d > 0, 0.0).ewm(com=RSI_PERIOD-1, adjust=False).mean()
    loss = (-d.where(d < 0, 0.0)).ewm(com=RSI_PERIOD-1, adjust=False).mean()
    rsi  = round(float((100 - (100 / (1 + gain / loss))).iloc[-1]), 2)
    p    = closes[-1]
    return upper, middle, lower, rsi, p <= lower and rsi < RSI_BUY, p >= upper

async def place_order(token, side, size, price, reduce_only=False):
    # Lighter precision: 1 ETH = 10 base units (from filled orders: 0.002 ETH = 20)
    # Wait - filled was 0.0020 with amount=20, so 1 ETH = 10000
    # But 0.0200 = 200 cancelled... price issue!
    # For market orders: use slippage price (BUY=high, SELL=low)
    if side == "BUY":
        order_price = int(price * 1.05 * 100)  # 5% slippage tolerance
    else:
        order_price = int(price * 0.95 * 100)  # 5% slippage tolerance

    base_amt = int(size * 10000)
    logger.info(f"Order: {side} {size} {token} base_amount={base_amt} price={order_price}")

    tx, tx_hash, err = await signer_client.create_order(
        market_index=MARKETS[token]["market_index"],
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

def calc_size(token, margin, price):
    lev = MARKETS[token].get("leverage", LEVERAGE)
    size = (margin * lev) / price
    min_size = MARKETS[token]["min_size"]
    size = max(size, min_size)
    return round(size, MARKETS[token]["decimals"])

def record_close(token, exit_price):
    s     = all_stats[token]
    pnl   = round((exit_price - s["entry_price"]) * s["entry_size"], 4)
    new_m = round(max(s["entry_margin"] + pnl, 0.5), 4)
    s["total_trades"] += 1; s["total_pnl"] += pnl
    s["current_margin"] = new_m
    if new_m > s["peak_margin"]: s["peak_margin"] = new_m
    result = "✅ WIN" if pnl >= 0 else "❌ LOSS"
    if pnl >= 0: s["wins"] += 1
    else:        s["losses"] += 1
    s["history"].append({"no": s["total_trades"], "entry": s["entry_price"],
        "exit": exit_price, "pnl": pnl, "old_margin": s["entry_margin"],
        "new_margin": new_m, "result": result,
        "time": datetime.now().strftime("%Y-%m-%d %H:%M")})
    s["history"] = s["history"][-10:]
    save_stats(token)
    return pnl, new_m, result

async def send_tg(msg):
    try:
        await tg_app.bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=msg, parse_mode="Markdown")
    except Exception as e:
        logger.error(f"TG: {e}")

async def token_loop(token):
    icon = "🔷" if token == "ETH" else "🔶"
    while True:
        try:
            closes = await fetch_closes(MARKETS[token]["symbol"])
            price  = closes[-1]
            upper, middle, lower, rsi, buy_sig, close_sig = calc_indicators(closes)
            dec = MARKETS[token]["decimals"]
            s   = all_stats[token]
            logger.info(f"{token}=${price:.{dec}f} RSI={rsi} Buy={buy_sig} Close={close_sig}")

            if buy_sig and not positions[token]:
                margin = s["current_margin"]
                size   = calc_size(token, margin, price)
                await send_tg(
                    f"{icon} *{token} BUY!*\n"
                    f"Price:`${price:.{dec}f}` ≤ Lower:`${lower}`\n"
                    f"RSI:`{rsi}` | `${margin}`×`{LEVERAGE}x`\n"
                    f"Size:`{size} {token}` | _Executing..._"
                )
                try:
                    tx = await place_order(token, "BUY", size, price)
                    positions[token] = True
                    s["entry_price"] = price; s["entry_size"] = size; s["entry_margin"] = margin
                    save_stats(token)
                    await send_tg(f"✅ *{token} LONG Opened!* Entry:`${price:.{dec}f}`")
                except Exception as e:
                    positions[token] = False
                    await send_tg(f"❌ {token} BUY Failed: `{e}`")

            elif close_sig and positions[token]:
                await send_tg(f"{icon} *{token} CLOSE!* Price:`${price:.{dec}f}` ≥ Upper:`${upper}`")
                try:
                    await place_order(token, "SELL", s["entry_size"], price, reduce_only=True)
                    positions[token] = False
                    pnl, new_m, outcome = record_close(token, price)
                    wr = round(s["wins"] / max(s["total_trades"], 1) * 100, 1)
                    await send_tg(
                        f"{outcome} *{token} #{s['total_trades']}*\n"
                        f"PnL:`${pnl:+.4f}` | Margin:`${new_m}`\n"
                        f"WR:`{wr}%` | Next:`${new_m}`×`{LEVERAGE}x` 🚀"
                    )
                except Exception as e:
                    positions[token] = False
                    await send_tg(f"❌ {token} CLOSE Failed: `{e}`")
        except Exception as e:
            logger.error(f"{token}: {e}")
        await asyncio.sleep(CHECK_INTERVAL)

async def check_open_positions():
    """Check Lighter DEX for open positions on startup"""
    try:
        api = lighter.ApiClient(lighter.Configuration(host=BASE_URL))
        acc = lighter.AccountApi(api)
        r   = await acc.account(str(ACCOUNT_INDEX))
        await api.close()

        open_positions = getattr(r, 'positions', []) or []
        for pos in open_positions:
            market_idx = getattr(pos, 'market_index', -1)
            size       = float(getattr(pos, 'base_amount', 0) or 0)
            if size > 0:
                if market_idx == MARKETS["ETH"]["market_index"]:
                    positions["ETH"] = True
                    logger.info("ETH position found on startup!")
                elif market_idx == MARKETS["HYPE"]["market_index"]:
                    positions["HYPE"] = True
                    logger.info("HYPE position found on startup!")
    except Exception as e:
        logger.error(f"Position check error: {e}")

async def strategy_loop():
    # Check existing positions before starting
    await check_open_positions()

    eth_pos  = "🟢 LONG" if positions["ETH"]  else "⚪ None"
    hype_pos = "🟢 LONG" if positions["HYPE"] else "⚪ None"

    lit_pos = "🟢 LONG" if positions.get("LIT") else "⚪ None"
    sol_pos = "🟢 LONG" if positions.get("SOL") else "⚪ None"
    await send_tg(
        "🤖 *Bot Started!*\n"
        f"🔷 ETH `${MARKETS['ETH']['start_margin']}` | {eth_pos}\n"
        f"🔶 HYPE `${MARKETS['HYPE']['start_margin']}` | {hype_pos}\n"
        f"🟣 LIT `${MARKETS['LIT']['start_margin']}` | {lit_pos}\n"
        f"🟤 SOL `${MARKETS['SOL']['start_margin']}` | {sol_pos}\n"
        f"⚡ {LEVERAGE}x | BB+RSI | 15min ✅"
    )
    await asyncio.gather(token_loop("ETH"), token_loop("HYPE"), token_loop("LIT"), token_loop("SOL"))

async def cmd_start(u: Update, c: ContextTypes.DEFAULT_TYPE):
    await u.message.reply_text(
        f"🤖 ETH:`${all_stats['ETH']['current_margin']}` HYPE:`${all_stats['HYPE']['current_margin']}`\n"
        "/status /bb /stats /history /balance", parse_mode="Markdown")

async def cmd_status(u: Update, c: ContextTypes.DEFAULT_TYPE):
    try:
        ec = await fetch_closes("ETHUSDT")
        hc = await fetch_closes("HYPEUSDT")
        ep = ec[-1]
        hp = hc[-1]
        _,_,_,er,_,_ = calc_indicators(ec)
        _,_,_,hr,_,_ = calc_indicators(hc)
        es = "🟢 LONG" if positions["ETH"] else "⚪ Wait"
        hs = "🟢 LONG" if positions["HYPE"] else "⚪ Wait"
        await u.message.reply_text(
            f"🔷 ETH:`${ep:,.2f}` RSI:`{er}` {es}\nMargin:`${all_stats['ETH']['current_margin']}`\n"
            f"━━━━━━━━━━━━━━━\n"
            f"🔶 HYPE:`${hp:.4f}` RSI:`{hr}` {hs}\nMargin:`${all_stats['HYPE']['current_margin']}`",
            parse_mode="Markdown")
    except Exception as e:
        await u.message.reply_text(f"❌ Error: {e}")

async def cmd_bb(u: Update, c: ContextTypes.DEFAULT_TYPE):
    try:
        lines = []
        emojis = {"ETH": "🔷", "HYPE": "🔶", "LIT": "🟣"}
        for token in MARKETS:
            symbol = MARKETS[token]["symbol"]
            closes = await fetch_closes(symbol)
            upper, mid, lower, rsi, buy_sig, close_sig = calc_indicators(closes)
            status = "🟢 BUY!" if buy_sig else ("🔴 CLOSE!" if close_sig else "🟡 Wait")
            emoji = emojis.get(token, "🔸")
            price = closes[-1]
            lines.append(
                f"{emoji} *{token}* RSI:`{rsi}` {status}\n"
                f"U:`${upper:.4f}` M:`${mid:.4f}` L:`${lower:.4f}`"
            )
        await u.message.reply_text("\n".join(lines), parse_mode="Markdown")
    except Exception as e:
        await u.message.reply_text(f"❌ {e}")


async def cmd_stats(u: Update, c: ContextTypes.DEFAULT_TYPE):
    msg = ""
    for token in MARKETS:
        s = all_stats[token]; t = s["total_trades"]
        wr = round(s["wins"]/max(t,1)*100,1)
        g  = round(s["current_margin"]-MARKETS[token]["start_margin"],4)
        icon = "🔷" if token=="ETH" else "🔶"
        msg += f"{icon} *{token}* T:`{t}` WR:`{wr}%` M:`${s['current_margin']}` G:`${g:+.4f}`\n"
    await u.message.reply_text(msg, parse_mode="Markdown")

async def cmd_history(u: Update, c: ContextTypes.DEFAULT_TYPE):
    msg = ""
    for token in MARKETS:
        icon = "🔷" if token=="ETH" else "🔶"
        h = all_stats[token].get("history",[])
        msg += f"{icon} *{token}*\n"
        if not h: msg += "No trades\n"
        else:
            for t in reversed(h[-3:]):
                msg += f"{t['result']} #{t['no']} `${t['pnl']:+.4f}` `${t['old_margin']}→${t['new_margin']}`\n"
    await u.message.reply_text(msg, parse_mode="Markdown")

async def cmd_balance(u: Update, c: ContextTypes.DEFAULT_TYPE):
    try:
        async with httpx.AsyncClient(timeout=15) as c2:
            r = await c2.get(
                f"{BASE_URL}/api/v1/account",
                params={"account_index": ACCOUNT_INDEX}
            )
            data = r.json()
            collateral = data.get("account", {}).get("collateral", "N/A")
            unrealized = data.get("account", {}).get("total_unrealized_pnl", "0")
            await u.message.reply_text(
                f"💰 *Balance*\n"
                f"Collateral: `${collateral}`\n"
                f"Unrealized PnL: `${unrealized}`",
                parse_mode="Markdown")
    except Exception as e:
        await u.message.reply_text(f"❌ {e}")

async def main():
    global tg_app, signer_client

    signer_client = lighter.SignerClient(
        url=BASE_URL,
        api_private_keys={API_KEY_INDEX: LIGHTER_PRIVATE_KEY},
        account_index=ACCOUNT_INDEX
    )

    tg_app = (
        Application.builder()
        .token(TELEGRAM_BOT_TOKEN)
        .updater(None)
        .build()
    )
    for cmd, fn in [("start",cmd_start),("status",cmd_status),("bb",cmd_bb),
                    ("stats",cmd_stats),("history",cmd_history),("balance",cmd_balance)]:
        tg_app.add_handler(CommandHandler(cmd, fn))

    await tg_app.initialize()
    await tg_app.start()
    # Manual polling loop
    offset = None
    asyncio.create_task(strategy_loop())
    while True:
        try:
            updates = await tg_app.bot.get_updates(offset=offset, timeout=10, allowed_updates=["message"])
            for update in updates:
                offset = update.update_id + 1
                await tg_app.process_update(update)
        except Exception as e:
            logger.error(f"Polling error: {e}")
        await asyncio.sleep(1)

if __name__ == "__main__":
    asyncio.run(main())
    
