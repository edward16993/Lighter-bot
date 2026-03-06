"""
Lighter DEX — RSI Compounding Auto Trade Bot
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Market  : ETH-USDC (Lighter DEX)
Strategy: RSI(14) on 15min candles
  RSI > 70 → LONG BUY (5x leverage)
  RSI < 30 → CLOSE position
Compound: Start $5, every trade profit adds to next margin
  Win  → margin + profit = next trade margin
  Loss → margin - loss   = next trade margin (never reset to $5)
"""

import os, asyncio, logging, json
from datetime import datetime
import httpx
import pandas as pd
from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

# ─── Environment Variables (Render.com-ல் set பண்ணுங்க) ──────
TELEGRAM_BOT_TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]
TELEGRAM_CHAT_ID   = os.environ["TELEGRAM_CHAT_ID"]
LIGHTER_API_KEY    = os.environ["LIGHTER_API_KEY"]
LIGHTER_SECRET     = os.environ["LIGHTER_SECRET"]

# ─── Strategy Config ──────────────────────────────────────────
TICKER          = "ETH"
LEVERAGE        = 5
RSI_PERIOD      = 14
RSI_BUY         = 70
RSI_CLOSE       = 30
STARTING_MARGIN = float(os.environ.get("STARTING_MARGIN", "5"))  # $5 default
CHECK_INTERVAL  = 60 * 15   # every 15 minutes

# ─── Persistent Stats ─────────────────────────────────────────
STATS_FILE = "stats.json"

def load_stats():
    if os.path.exists(STATS_FILE):
        with open(STATS_FILE) as f:
            return json.load(f)
    return {
        "current_margin": STARTING_MARGIN,  # starts at $5, grows with profit
        "total_trades":   0,
        "wins":           0,
        "losses":         0,
        "total_pnl":      0.0,
        "peak_margin":    STARTING_MARGIN,
        "entry_price":    0.0,
        "entry_size":     0.0,
        "entry_margin":   0.0,
        "history":        []   # last 10 trades
    }

def save_stats(s):
    with open(STATS_FILE, "w") as f:
        json.dump(s, f, indent=2)

stats   = load_stats()
bot_app = None
in_position = False

# ─── Lighter API Client ───────────────────────────────────────
class LighterClient:
    BASE = "https://mainnet.lighter.xyz/v1"

    def __init__(self, api_key, secret):
        self.api_key = api_key
        self.secret  = secret

    def _headers(self):
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type":  "application/json"
        }

    async def place_market_order(self, side: str, size: float, reduce_only=False):
        """
        side: "BUY" or "SELL"
        size: ETH amount
        """
        payload = {
            "market_index":  0,              # ETH-USDC
            "base_amount":   int(size * 1e6),# convert to micro units
            "is_ask":        side == "SELL",
            "order_type":    "MARKET",
            "time_in_force": "IOC",
            "reduce_only":   reduce_only,
            "leverage":      LEVERAGE,
        }
        async with httpx.AsyncClient() as c:
            r = await c.post(
                f"{self.BASE}/orders",
                json=payload,
                headers=self._headers(),
                timeout=10
            )
            return r.json()

    async def get_account(self):
        async with httpx.AsyncClient() as c:
            r = await c.get(
                f"{self.BASE}/account",
                headers=self._headers(),
                timeout=10
            )
            return r.json()

    async def get_positions(self):
        async with httpx.AsyncClient() as c:
            r = await c.get(
                f"{self.BASE}/positions",
                headers=self._headers(),
                timeout=10
            )
            return r.json()

lighter = LighterClient(LIGHTER_API_KEY, LIGHTER_SECRET)

# ─── Price & RSI ──────────────────────────────────────────────
async def fetch_closes(limit=50):
    """Binance free API — no key needed"""
    async with httpx.AsyncClient() as c:
        r = await c.get(
            "https://api.binance.com/api/v3/klines",
            params={"symbol": "ETHUSDT", "interval": "15m", "limit": limit},
            timeout=10
        )
        return [float(x[4]) for x in r.json()]

def calc_rsi(closes, period=14):
    if len(closes) < period + 1:
        return 50.0
    s    = pd.Series(closes)
    d    = s.diff()
    gain = d.where(d > 0, 0.0).rolling(period).mean()
    loss = (-d.where(d < 0, 0.0)).rolling(period).mean()
    return round(float((100 - 100 / (1 + gain / loss)).iloc[-1]), 2)

# ─── Compounding Core Logic ───────────────────────────────────
def calc_eth_size(margin_usd: float, eth_price: float) -> float:
    """
    ETH size = (margin × leverage) ÷ ETH price
    Example: ($5 × 5) ÷ $3000 = 0.00833 ETH
    """
    return round((margin_usd * LEVERAGE) / eth_price, 6)

def update_margin_after_close(exit_price: float) -> float:
    """
    PnL = (exit - entry) × eth_size
    New margin = old margin + pnl
    Compounding: profit adds, loss subtracts — never resets to $5
    """
    global stats

    entry    = stats["entry_price"]
    size     = stats["entry_size"]
    old_margin = stats["entry_margin"]

    pnl = round((exit_price - entry) * size, 4)  # in USDC

    new_margin = round(old_margin + pnl, 4)
    new_margin = max(new_margin, 1.0)  # minimum $1 (safety floor)

    # Update stats
    stats["total_trades"] += 1
    stats["total_pnl"]    += pnl
    stats["current_margin"] = new_margin

    if new_margin > stats["peak_margin"]:
        stats["peak_margin"] = new_margin

    if pnl >= 0:
        stats["wins"] += 1
        result = "✅ WIN"
    else:
        stats["losses"] += 1
        result = "❌ LOSS"

    trade = {
        "no":     stats["total_trades"],
        "entry":  entry,
        "exit":   exit_price,
        "size":   size,
        "pnl":    pnl,
        "old_margin": old_margin,
        "new_margin": new_margin,
        "result": result,
        "time":   datetime.now().strftime("%Y-%m-%d %H:%M")
    }
    stats["history"].append(trade)
    stats["history"] = stats["history"][-10:]
    save_stats(stats)

    return pnl, new_margin, result

# ─── Strategy Loop ────────────────────────────────────────────
async def strategy_loop():
    global in_position, stats

    await send_tg(
        "🤖 *Lighter DEX RSI Compounding Bot*\n"
        "━━━━━━━━━━━━━━━\n"
        f"📊 ETH-USDC | ⚡ {LEVERAGE}x | ⏱ 15min\n"
        f"🟢 BUY: RSI > {RSI_BUY}\n"
        f"🔴 CLOSE: RSI < {RSI_CLOSE}\n"
        f"💵 Starting Margin: `${STARTING_MARGIN}`\n"
        f"📈 Current Margin: `${stats['current_margin']}`\n"
        "━━━━━━━━━━━━━━━\n"
        "✅ Monitoring started!"
    )

    while True:
        try:
            ts     = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            closes = await fetch_closes(50)
            price  = closes[-1]
            rsi    = calc_rsi(closes, RSI_PERIOD)

            logger.info(f"{ts} | ETH=${price:,.2f} | RSI={rsi} | Margin=${stats['current_margin']} | pos={in_position}")

            # ── BUY SIGNAL ───────────────────────────────────
            if rsi > RSI_BUY and not in_position:

                margin   = stats["current_margin"]
                eth_size = calc_eth_size(margin, price)
                position_value = round(margin * LEVERAGE, 2)

                await send_tg(
                    f"🟢 *LONG SIGNAL!*\n"
                    f"━━━━━━━━━━━━━━━\n"
                    f"📈 RSI: `{rsi}` _(Above {RSI_BUY})_\n"
                    f"💰 ETH Price: `${price:,.2f}`\n"
                    f"💵 Margin: `${margin}` × `{LEVERAGE}x` = `${position_value}`\n"
                    f"📦 Size: `{eth_size} ETH`\n"
                    f"🔢 Trade #{stats['total_trades'] + 1}\n"
                    f"🕐 `{ts}`\n"
                    f"_Executing BUY..._"
                )

                try:
                    result = await lighter.place_market_order("BUY", eth_size)
                    in_position          = True
                    stats["entry_price"] = price
                    stats["entry_size"]  = eth_size
                    stats["entry_margin"] = margin
                    save_stats(stats)

                    await send_tg(
                        f"✅ *LONG Opened!*\n"
                        f"━━━━━━━━━━━━━━━\n"
                        f"💰 Entry: `${price:,.2f}`\n"
                        f"📦 `{eth_size} ETH` × `{LEVERAGE}x`\n"
                        f"💵 Margin used: `${margin}`\n"
                        f"🔗 TX: `{result.get('tx_hash', result.get('id', 'N/A'))}`"
                    )
                except Exception as e:
                    in_position = False
                    await send_tg(f"❌ *BUY Failed!*\n`{str(e)}`")
                    logger.error(f"BUY error: {e}")

            # ── CLOSE SIGNAL ─────────────────────────────────
            elif rsi < RSI_CLOSE and in_position:

                await send_tg(
                    f"🔴 *CLOSE SIGNAL!*\n"
                    f"━━━━━━━━━━━━━━━\n"
                    f"📉 RSI: `{rsi}` _(Below {RSI_CLOSE})_\n"
                    f"💰 ETH Price: `${price:,.2f}`\n"
                    f"🕐 `{ts}`\n"
                    f"_Closing position..._"
                )

                try:
                    eth_size = stats["entry_size"]
                    result   = await lighter.place_market_order("SELL", eth_size, reduce_only=True)
                    in_position = False

                    pnl, new_margin, outcome = update_margin_after_close(price)

                    win_rate = round(stats["wins"] / max(stats["total_trades"], 1) * 100, 1)
                    growth   = round(new_margin - STARTING_MARGIN, 4)

                    await send_tg(
                        f"{outcome} *Trade #{stats['total_trades']} Closed!*\n"
                        f"━━━━━━━━━━━━━━━\n"
                        f"📈 Entry: `${stats['entry_price']:,.2f}`\n"
                        f"📉 Exit:  `${price:,.2f}`\n"
                        f"💰 PnL:   `${pnl:+.4f} USDC`\n"
                        f"━━━━━━━━━━━━━━━\n"
                        f"📊 *Compounding Update:*\n"
                        f"💵 Old Margin: `${stats['entry_margin']}`\n"
                        f"💵 New Margin: `${new_margin}` _(next trade)_\n"
                        f"📈 Total Growth: `${growth:+.4f}`\n"
                        f"🏆 Win Rate: `{win_rate}%` ({stats['wins']}W / {stats['losses']}L)\n"
                        f"⏭ Next trade: `${new_margin}` × `{LEVERAGE}x` 🚀"
                    )

                except Exception as e:
                    in_position = False
                    await send_tg(f"❌ *CLOSE Failed!*\n`{str(e)}`")
                    logger.error(f"CLOSE error: {e}")

            else:
                pos = "🟢 In Position" if in_position else "⚪ Waiting"
                logger.info(f"No signal | {pos} | Margin=${stats['current_margin']}")

        except Exception as e:
            logger.error(f"Loop error: {e}")
            await send_tg(f"⚠️ *Error!*\n`{str(e)}`")

        await asyncio.sleep(CHECK_INTERVAL)

# ─── Telegram Helper ──────────────────────────────────────────
async def send_tg(msg):
    if bot_app:
        try:
            await bot_app.bot.send_message(
                chat_id=TELEGRAM_CHAT_ID, text=msg, parse_mode="Markdown"
            )
        except Exception as e:
            logger.error(f"TG error: {e}")

# ─── Telegram Commands ────────────────────────────────────────
async def cmd_start(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "🤖 *Lighter DEX RSI Compounding Bot*\n"
        "━━━━━━━━━━━━━━━\n"
        f"📊 ETH-USDC | ⚡ {LEVERAGE}x | ⏱ 15min\n"
        f"🟢 BUY: RSI > {RSI_BUY}\n"
        f"🔴 CLOSE: RSI < {RSI_CLOSE}\n"
        f"💵 Start Margin: `${STARTING_MARGIN}`\n"
        "📈 Profit compounds every trade!\n"
        "━━━━━━━━━━━━━━━\n"
        "/status /rsi /stats /history /balance",
        parse_mode="Markdown"
    )

async def cmd_status(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    pos = "🟢 LONG Open" if in_position else "⚪ No Position"
    entry = f"`${stats['entry_price']:,.2f}`" if in_position else "—"
    eth_size = f"`{stats['entry_size']} ETH`" if in_position else "—"
    await update.message.reply_text(
        f"📈 *Bot Status*\n"
        f"━━━━━━━━━━━━━━━\n"
        f"✅ Running: YES\n"
        f"📍 Position: {pos}\n"
        f"💰 Entry Price: {entry}\n"
        f"📦 Size: {eth_size}\n"
        f"💵 Current Margin: `${stats['current_margin']}`\n"
        f"⏭ Next Trade Value: `${round(stats['current_margin'] * LEVERAGE, 2)}`\n"
        f"⏱ Check: Every 15 mins",
        parse_mode="Markdown"
    )

async def cmd_rsi(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("_Fetching RSI..._", parse_mode="Markdown")
    try:
        closes = await fetch_closes(50)
        rsi    = calc_rsi(closes, RSI_PERIOD)
        price  = closes[-1]
        if rsi > RSI_BUY:
            zone = f"🟢 BUY Zone (RSI > {RSI_BUY})"
        elif rsi < RSI_CLOSE:
            zone = f"🔴 CLOSE Zone (RSI < {RSI_CLOSE})"
        else:
            zone = "🟡 Neutral — Waiting"
        await update.message.reply_text(
            f"📊 *Live RSI*\n"
            f"━━━━━━━━━━━━━━━\n"
            f"📈 RSI(14): `{rsi}`\n"
            f"💰 ETH: `${price:,.2f}`\n"
            f"📍 {zone}",
            parse_mode="Markdown"
        )
    except Exception as e:
        await update.message.reply_text(f"❌ {e}")

async def cmd_stats(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    total    = stats["total_trades"]
    wins     = stats["wins"]
    losses   = stats["losses"]
    pnl      = stats["total_pnl"]
    cur_m    = stats["current_margin"]
    peak_m   = stats["peak_margin"]
    growth   = round(cur_m - STARTING_MARGIN, 4)
    win_rate = round(wins / max(total, 1) * 100, 1)
    await update.message.reply_text(
        f"📊 *Compounding Stats*\n"
        f"━━━━━━━━━━━━━━━\n"
        f"🔢 Total Trades: `{total}`\n"
        f"✅ Wins: `{wins}` | ❌ Losses: `{losses}`\n"
        f"🏆 Win Rate: `{win_rate}%`\n"
        f"━━━━━━━━━━━━━━━\n"
        f"💵 Start Margin:   `${STARTING_MARGIN}`\n"
        f"💵 Current Margin: `${cur_m}`\n"
        f"🏅 Peak Margin:    `${peak_m}`\n"
        f"📈 Total Growth:   `${growth:+.4f}`\n"
        f"💹 Total PnL:      `${pnl:+.4f} USDC`",
        parse_mode="Markdown"
    )

async def cmd_history(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    history = stats.get("history", [])
    if not history:
        await update.message.reply_text("📭 No trades yet.")
        return
    msg = "📋 *Last 10 Trades*\n━━━━━━━━━━━━━━━\n"
    for t in reversed(history):
        msg += (
            f"{t['result']} Trade #{t['no']} | `{t['time']}`\n"
            f"   Entry: `${t['entry']:,.2f}` → Exit: `${t['exit']:,.2f}`\n"
            f"   PnL: `${t['pnl']:+.4f}` | Margin: `${t['old_margin']}` → `${t['new_margin']}`\n\n"
        )
    await update.message.reply_text(msg, parse_mode="Markdown")

async def cmd_balance(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    try:
        data = await lighter.get_account()
        await update.message.reply_text(
            f"💰 *Lighter Account*\n"
            f"━━━━━━━━━━━━━━━\n"
            f"💵 Collateral: `${float(data.get('collateral', 0)):,.2f}`\n"
            f"📊 Equity: `${float(data.get('equity', 0)):,.2f}`",
            parse_mode="Markdown"
        )
    except Exception as e:
        await update.message.reply_text(f"❌ {e}")

# ─── Main ─────────────────────────────────────────────────────
async def main():
    global bot_app
    bot_app = Application.builder().token(TELEGRAM_BOT_TOKEN).build()
    bot_app.add_handler(CommandHandler("start",   cmd_start))
    bot_app.add_handler(CommandHandler("status",  cmd_status))
    bot_app.add_handler(CommandHandler("rsi",     cmd_rsi))
    bot_app.add_handler(CommandHandler("stats",   cmd_stats))
    bot_app.add_handler(CommandHandler("history", cmd_history))
    bot_app.add_handler(CommandHandler("balance", cmd_balance))

    asyncio.create_task(strategy_loop())
    logger.info("🚀 Bot started!")
    await bot_app.run_polling(drop_pending_updates=True)

if __name__ == "__main__":
    asyncio.run(main())
