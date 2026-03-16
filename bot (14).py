import os,asyncio,logging,json
import numpy as np
from datetime import datetime
import pandas as pd,httpx
from telegram import Update
from telegram.ext import Application,CommandHandler,ContextTypes
import lighter

logging.basicConfig(format="%(asctime)s-%(levelname)s-%(message)s",level=logging.INFO)
logger=logging.getLogger(__name__)

TG_TOKEN=os.environ["TELEGRAM_BOT_TOKEN"]
TG_CHAT=os.environ["TELEGRAM_CHAT_ID"]
ACC_IDX=int(os.environ["ACCOUNT_INDEX"])
KEY_IDX=int(os.environ["API_KEY_INDEX"])
PRV_KEY=os.environ["LIGHTER_PRIVATE_KEY"]

BASE_URL="https://mainnet.zklighter.elliot.ai"
LEV=5
INTERVAL=300
SL_MULT=float(os.environ.get("SL_MULT","2.0"))
TP_MULT=float(os.environ.get("TP_MULT","3.0"))
ADX_MIN=20.0
MIN_MAR=0.50

MARKETS={
    "ETH":{"symbol":"ETH-USDT-SWAP","market_id":0,"stats_file":"eth_stats.json",
           "start_margin":float(os.environ.get("ETH_MARGIN","45")),
           "decimals":3,"min_size":0.002,"emoji":"ETH"},
    "BNB":{"symbol":"BNB-USDT-SWAP","market_id":25,"stats_file":"bnb_stats.json",
           "start_margin":float(os.environ.get("BNB_MARGIN","10")),
           "decimals":2,"min_size":0.02,"emoji":"BNB"}
}

state={}
tg_app=None
signer=None

def load_stats(coin):
    mkt=MARKETS[coin]
    try:
        with open(mkt["stats_file"]) as f: return json.load(f)
    except Exception:
        m=mkt["start_margin"]
        return {"total_trades":0,"wins":0,"losses":0,"total_pnl":0.0,
                "current_margin":m,"peak_margin":m,"long_trades":0,"short_trades":0,
                "entry_price":0.0,"entry_size":0.0,"entry_margin":0.0,
                "sl_price":0.0,"tp_price":0.0,"history":[]}

def save_stats(coin):
    with open(MARKETS[coin]["stats_file"],"w") as f:
        json.dump(state[coin]["stats"],f)

def calc_alma(s,p,sigma=0.85,offset=0.85):
    m=offset*(p-1); sv=p/sigma
    w=np.array([np.exp(-((i-m)**2)/(2*sv**2)) for i in range(p)])
    w/=w.sum()
    r=pd.Series(np.nan,index=s.index)
    for i in range(p-1,len(s)):
        r.iloc[i]=np.dot(w,s.iloc[i-p+1:i+1].values)
    return r

def calc_atr(df,p=10):
    tr=pd.concat([df["high"]-df["low"],
        (df["high"]-df["close"].shift()).abs(),
        (df["low"]-df["close"].shift()).abs()],axis=1).max(axis=1)
    return tr.ewm(span=p,adjust=False).mean()

def calc_adx(df,p=14):
    tr=calc_atr(df,p)
    dp=(df["high"].diff()).clip(lower=0)
    dm=(-df["low"].diff()).clip(lower=0)
    dp=dp.where(dp>dm,0); dm=dm.where(dm>dp,0)
    dip=100*dp.ewm(span=p,adjust=False).mean()/tr.replace(0,np.nan)
    dim=100*dm.ewm(span=p,adjust=False).mean()/tr.replace(0,np.nan)
    dx=100*(dip-dim).abs()/(dip+dim).replace(0,np.nan)
    return dx.ewm(span=p,adjust=False).mean()

def calc_rsi(s,p=14):
    d=s.diff()
    g=d.clip(lower=0).ewm(span=p,adjust=False).mean()
    l=(-d.clip(upper=0)).ewm(span=p,adjust=False).mean()
    return 100-(100/(1+g/l.replace(0,np.nan)))

def calc_indicators(df):
    df=df.copy()
    df["fast"]=calc_alma(df["close"],13)
    df["slow"]=calc_alma(df["close"],21)
    df["ema200"]=df["close"].ewm(span=200,adjust=False).mean()
    df["atr"]=calc_atr(df,10)
    df["rsi"]=calc_rsi(df["close"],14)
    df["adx"]=calc_adx(df,14)
    return df

async def fetch_candles(coin,lim=300):
    sym=MARKETS[coin]["symbol"]
    async with httpx.AsyncClient(timeout=15) as c:
        r=await c.get("https://www.okx.com/api/v5/market/candles",
            params={"instId":sym,"bar":"5m","limit":str(lim)})
        data=r.json().get("data",[])
    df=pd.DataFrame(data)[[0,2,3,4]].copy()
    df.columns=["time","high","low","close"]
    for col in ["high","low","close"]: df[col]=df[col].astype(float)
    df["time"]=pd.to_datetime(df["time"].astype(int),unit="ms")
    return df.sort_values("time").reset_index(drop=True)

async def place_order(coin,side,size,price,reduce_only=False):
    mkt=MARKETS[coin]
    slip=0.002
    if side=="BUY": op=round(float(price)*(1+slip),2)
    else: op=round(float(price)*(1-slip),2)
    ba=int(round(float(size)*10000))
    op_int=int(round(float(op)*100))
    tx,txh,err=await signer.create_order(
        market_index=int(mkt["market_id"]),
        client_order_index=int(datetime.now().timestamp()),
        base_amount=ba,price=op_int,is_ask=bool(side=="SELL"),
        order_type=signer.ORDER_TYPE_MARKET,
        time_in_force=signer.ORDER_TIME_IN_FORCE_IMMEDIATE_OR_CANCEL,
        reduce_only=bool(reduce_only),order_expiry=signer.DEFAULT_IOC_EXPIRY)
    if err: raise Exception(str(err))
    return txh

def calc_size(coin,margin,price):
    mkt=MARKETS[coin]
    return round(max((margin*LEV)/float(price),mkt["min_size"]),mkt["decimals"])

def record_close(coin,exit_px,reason,side):
    st=state[coin]["stats"]
    if side=="LONG": pnl=round((exit_px-st["entry_price"])*st["entry_size"],4)
    else: pnl=round((st["entry_price"]-exit_px)*st["entry_size"],4)
    new_m=round(max(st["entry_margin"]+pnl,MIN_MAR),4)
    st["total_trades"]+=1
    st["total_pnl"]=round(st["total_pnl"]+pnl,4)
    st["current_margin"]=new_m
    if new_m>st["peak_margin"]: st["peak_margin"]=new_m
    if pnl>=0: st["wins"]+=1
    else: st["losses"]+=1
    st["history"].append({"no":st["total_trades"],"side":side,
        "entry":st["entry_price"],"exit":exit_px,"pnl":pnl,
        "reason":reason,"new_margin":new_m,
        "time":datetime.now().strftime("%Y-%m-%d %H:%M")})
    st["history"]=st["history"][-20:]
    save_stats(coin)
    return pnl,new_m,"WIN" if pnl>=0 else "LOSS"

async def send_tg(msg):
    try: await tg_app.bot.send_message(chat_id=TG_CHAT,text=msg,parse_mode="Markdown")
    except Exception as e: logger.error("TG error: %s",e)

async def strategy_loop(coin):
    mkt=MARKETS[coin]; s=state[coin]; em=mkt["emoji"]
    try:
        api=lighter.ApiClient(lighter.Configuration(host=BASE_URL))
        acc=lighter.AccountApi(api)
        r=await acc.account(by="index",value=str(ACC_IDX)); await api.close()
        for pos2 in (getattr(r,"positions",[]) or []):
            if getattr(pos2,"market_index",-1)==mkt["market_id"]:
                sz=float(getattr(pos2,"base_amount",0) or 0)
                if sz!=0:
                    s["position"]="LONG" if sz>0 else "SHORT"
                    s["entry_px"]=float(s["stats"].get("entry_price",0))
                    s["entry_sz"]=abs(sz)
                    s["entry_mg"]=float(s["stats"].get("entry_margin",0))
                    s["sl_px"]=float(s["stats"].get("sl_price",0))
                    s["tp_px"]=float(s["stats"].get("tp_price",0))
                    logger.info("%s %s restored!",coin,s["position"])
    except Exception as e: logger.error("%s restore: %s",coin,e)

    while True:
        try:
            df=await fetch_candles(coin,300); df=calc_indicators(df); df=df.dropna()
            curr=df.iloc[-1]; prev=df.iloc[-2]
            price=float(curr["close"]); atr=float(curr["atr"])
            rsi=float(curr["rsi"]); adx=float(curr["adx"])
            bull_cross=(curr["fast"]>curr["slow"]) and (prev["fast"]<=prev["slow"])
            bear_cross=(curr["fast"]<curr["slow"]) and (prev["fast"]>=prev["slow"])
            bull_trend=price>float(curr["ema200"]); trending=adx>ADX_MIN
            long_sig=bull_cross and bull_trend and trending
            short_sig=bear_cross and (rsi<50) and trending
            logger.info("%s=$%.2f RSI=%.1f ADX=%.1f pos=%s",coin,price,rsi,adx,s["position"])

            if s["position"] is None:
                if long_sig and s["stats"]["current_margin"]>=MIN_MAR:
                    mg=s["stats"]["current_margin"]; si=calc_size(coin,mg,price)
                    nsl=round(price-(SL_MULT*atr),2); ntp=round(price+(TP_MULT*atr),2)
                    await send_tg(
                        "*"+em+" "+coin+" LONG Signal*\n"
                        "Price: $*"+str(round(price,2))+"* ADX: *"+str(round(adx,1))+"*\n"
                        "SL: $*"+str(nsl)+"* TP: $*"+str(ntp)+"*\n"
                        "Margin: $*"+str(mg)+"*")
                    try:
                        await place_order(coin,"BUY",si,price)
                        s["position"]="LONG"; s["entry_px"]=price
                        s["entry_sz"]=si; s["entry_mg"]=mg
                        s["sl_px"]=nsl; s["tp_px"]=ntp
                        s["stats"]["entry_price"]=price; s["stats"]["entry_size"]=si
                        s["stats"]["entry_margin"]=mg; s["stats"]["sl_price"]=nsl
                        s["stats"]["tp_price"]=ntp
                        s["stats"]["long_trades"]=s["stats"].get("long_trades",0)+1
                        save_stats(coin)
                        await send_tg(
                            "*"+em+" "+coin+" LONG Opened!*\n"
                            "Entry: $*"+str(round(price,2))+"* | *"+str(si)+" "+coin+"*\n"
                            "SL: $*"+str(nsl)+"* TP: $*"+str(ntp)+"*")
                    except Exception as e:
                        s["position"]=None
                        await send_tg(em+" "+coin+" LONG Failed: "+str(e))

                elif short_sig and s["stats"]["current_margin"]>=MIN_MAR:
                    mg=s["stats"]["current_margin"]; si=calc_size(coin,mg,price)
                    nsl=round(price+(SL_MULT*atr),2); ntp=round(price-(TP_MULT*atr),2)
                    await send_tg(
                        "*"+em+" "+coin+" SHORT Signal*\n"
                        "Price: $*"+str(round(price,2))+"* ADX: *"+str(round(adx,1))+"*\n"
                        "SL: $*"+str(nsl)+"* TP: $*"+str(ntp)+"*\n"
                        "Margin: $*"+str(mg)+"*")
                    try:
                        await place_order(coin,"SELL",si,price)
                        s["position"]="SHORT"; s["entry_px"]=price
                        s["entry_sz"]=si; s["entry_mg"]=mg
                        s["sl_px"]=nsl; s["tp_px"]=ntp
                        s["stats"]["entry_price"]=price; s["stats"]["entry_size"]=si
                        s["stats"]["entry_margin"]=mg; s["stats"]["sl_price"]=nsl
                        s["stats"]["tp_price"]=ntp
                        s["stats"]["short_trades"]=s["stats"].get("short_trades",0)+1
                        save_stats(coin)
                        await send_tg(
                            "*"+em+" "+coin+" SHORT Opened!*\n"
                            "Entry: $*"+str(round(price,2))+"* | *"+str(si)+" "+coin+"*\n"
                            "SL: $*"+str(nsl)+"* TP: $*"+str(ntp)+"*")
                    except Exception as e:
                        s["position"]=None
                        await send_tg(em+" "+coin+" SHORT Failed: "+str(e))

            elif s["position"]=="LONG":
                unr=round((price-s["entry_px"])*s["entry_sz"],4); reason=None
                if price>=s["tp_px"]: reason="TP"
                elif price<=s["sl_px"]: reason="SL"
                elif bear_cross: reason="Cross"
                if reason:
                    await send_tg("*"+em+" "+coin+" LONG "+reason+"*\nPrice: $*"+str(round(price,2))+"* Unrealized: $*"+str(unr)+"*")
                    try:
                        await place_order(coin,"SELL",s["entry_sz"],price,reduce_only=True)
                        pnl,new_m,outcome=record_close(coin,price,reason,"LONG")
                        wr=round(s["stats"]["wins"]/max(s["stats"]["total_trades"],1)*100,1)
                        sg="+" if pnl>=0 else ""
                        await send_tg(
                            outcome+" *"+em+" "+coin+" LONG #"+str(s["stats"]["total_trades"])+"* ["+reason+"]\n"
                            "Entry: $*"+str(s["entry_px"])+"* Exit: $*"+str(round(price,2))+"*\n"
                            "PnL: *"+sg+"$"+str(pnl)+"* WR: *"+str(wr)+"%*\n"
                            "Margin: $*"+str(new_m)+"*")
                        s["position"]=None
                    except Exception as e:
                        await send_tg(em+" "+coin+" LONG Close Failed: "+str(e))

            elif s["position"]=="SHORT":
                unr=round((s["entry_px"]-price)*s["entry_sz"],4); reason=None
                if price<=s["tp_px"]: reason="TP"
                elif price>=s["sl_px"]: reason="SL"
                elif bull_cross: reason="Cross"
                if reason:
                    await send_tg("*"+em+" "+coin+" SHORT "+reason+"*\nPrice: $*"+str(round(price,2))+"* Unrealized: $*"+str(unr)+"*")
                    try:
                        await place_order(coin,"BUY",s["entry_sz"],price,reduce_only=True)
                        pnl,new_m,outcome=record_close(coin,price,reason,"SHORT")
                        wr=round(s["stats"]["wins"]/max(s["stats"]["total_trades"],1)*100,1)
                        sg="+" if pnl>=0 else ""
                        await send_tg(
                            outcome+" *"+em+" "+coin+" SHORT #"+str(s["stats"]["total_trades"])+"* ["+reason+"]\n"
                            "Entry: $*"+str(s["entry_px"])+"* Exit: $*"+str(round(price,2))+"*\n"
                            "PnL: *"+sg+"$"+str(pnl)+"* WR: *"+str(wr)+"%*\n"
                            "Margin: $*"+str(new_m)+"*")
                        s["position"]=None
                    except Exception as e:
                        await send_tg(em+" "+coin+" SHORT Close Failed: "+str(e))

        except Exception as e:
            logger.error("%s loop error: %s",coin,e)
            await send_tg(em+" "+coin+" Error: "+str(e))
        await asyncio.sleep(INTERVAL)

async def cmd_start(u,c):
    msg="*ALMA Dual Bot*\n"
    for coin in MARKETS:
        s=state[coin]; em=MARKETS[coin]["emoji"]
        ps=s["position"] if s["position"] else "Wait"
        msg+=em+" *"+coin+"*: "+ps+" | $*"+str(s["stats"]["current_margin"])+"*\n"
    msg+="\n/status /signal /stats /history /balance"
    await u.message.reply_text(msg,parse_mode="Markdown")

async def cmd_status(u,c):
    msg="*Market Status*\n\n"
    for coin in MARKETS:
        mkt=MARKETS[coin]; em=mkt["emoji"]; s=state[coin]
        try:
            df=await fetch_candles(coin,300); df=calc_indicators(df); df=df.dropna(); curr=df.iloc[-1]
            price=float(curr["close"]); ema200=round(float(curr["ema200"]),2)
            atr=round(float(curr["atr"]),2); rsi=round(float(curr["rsi"]),1); adx=round(float(curr["adx"]),1)
            trend="Bull" if price>ema200 else "Bear"
            adx_s="Trending" if adx>ADX_MIN else "Sideways-SKIP"
            ps=s["position"] if s["position"] else "Wait"
            extra=""
            if s["position"] and s["entry_px"]>0:
                unr=round((price-s["entry_px"])*s["entry_sz"],4) if s["position"]=="LONG" else round((s["entry_px"]-price)*s["entry_sz"],4)
                extra="\nUnrealized: $*"+str(unr)+"*\nSL: $*"+str(s["sl_px"])+"* TP: $*"+str(s["tp_px"])+"*"
            msg+=(em+" *"+coin+"* $*"+str(round(price,2))+"* | "+trend+"\n"
                "RSI: *"+str(rsi)+"* ATR: $*"+str(atr)+"*\n"
                "ADX: *"+str(adx)+"* ["+adx_s+"]\n"
                "EMA200: $*"+str(ema200)+"*\n"
                "Position: "+ps+extra+"\n"
                "Margin: $*"+str(s["stats"]["current_margin"])+"*\n\n")
        except Exception as e:
            msg+=em+" *"+coin+"* Error: "+str(e)+"\n\n"
    await u.message.reply_text(msg,parse_mode="Markdown")

async def cmd_signal(u,c):
    msg="*Signals*\n\n"
    for coin in MARKETS:
        em=MARKETS[coin]["emoji"]
        try:
            df=await fetch_candles(coin,300); df=calc_indicators(df); df=df.dropna()
            curr=df.iloc[-1]; prev=df.iloc[-2]
            price=float(curr["close"]); rsi=round(float(curr["rsi"]),1)
            adx=round(float(curr["adx"]),1); atr=float(curr["atr"])
            bull_cross=(curr["fast"]>curr["slow"]) and (prev["fast"]<=prev["slow"])
            bear_cross=(curr["fast"]<curr["slow"]) and (prev["fast"]>=prev["slow"])
            bull_trend=price>float(curr["ema200"]); trending=adx>ADX_MIN
            adx_s="Trending" if trending else "Sideways-SKIP"
            if bull_cross and bull_trend and trending:
                sig="LONG NOW!\nSL: $*"+str(round(price-SL_MULT*atr,2))+"* TP: $*"+str(round(price+TP_MULT*atr,2))+"*"
            elif bear_cross and rsi<50 and trending:
                sig="SHORT NOW!\nSL: $*"+str(round(price+SL_MULT*atr,2))+"* TP: $*"+str(round(price-TP_MULT*atr,2))+"*"
            elif bull_cross and bull_trend and not trending: sig="LONG blocked - ADX low"
            elif bear_cross and rsi<50 and not trending: sig="SHORT blocked - ADX low"
            elif bull_trend and curr["fast"]>curr["slow"]: sig="Bullish - Wait cross"
            elif rsi<50 and curr["fast"]<curr["slow"]: sig="Bearish - Wait cross"
            else: sig="No signal"
            msg+=(em+" *"+coin+"* $*"+str(round(price,2))+"*\n"
                "RSI: *"+str(rsi)+"* ADX: *"+str(adx)+"* ["+adx_s+"]\n"
                +sig+"\n\n")
        except Exception as e:
            msg+=em+" *"+coin+"* Error: "+str(e)+"\n\n"
    await u.message.reply_text(msg,parse_mode="Markdown")

async def cmd_stats(u,c):
    msg="*Stats*\n\n"
    for coin in MARKETS:
        em=MARKETS[coin]["emoji"]; s=state[coin]["stats"]
        t=s["total_trades"]; wr=round(s["wins"]/max(t,1)*100,1)
        sm=MARKETS[coin]["start_margin"]
        g=round((s["current_margin"]-sm)/sm*100,1)
        lt=s.get("long_trades",0); sh=s.get("short_trades",0)
        sg="+" if g>=0 else ""
        msg+=(em+" *"+coin+"*\n"
            "Trades: *"+str(t)+"* (L:"+str(lt)+" S:"+str(sh)+")\n"
            "WR: *"+str(wr)+"%* | *"+sg+str(g)+"%*\n"
            "PnL: $*"+str(round(s["total_pnl"],4))+"*\n"
            "Margin: $*"+str(s["current_margin"])+"* Peak: $*"+str(s["peak_margin"])+"*\n\n")
    await u.message.reply_text(msg,parse_mode="Markdown")

async def cmd_history(u,c):
    msg="*History*\n\n"
    for coin in MARKETS:
        em=MARKETS[coin]["emoji"]
        h=state[coin]["stats"].get("history",[])
        msg+=em+" *"+coin+"*\n"
        if not h:
            msg+="No trades yet!\n\n"
        else:
            for t in reversed(h[-5:]):
                sg="+" if t["pnl"]>=0 else ""
                msg+="#"+str(t["no"])+" "+t["side"]+" ["+t["reason"]+"] *"+sg+"$"+str(t["pnl"])+"* -> $*"+str(t["new_margin"])+"*\n"
            msg+="\n"
    await u.message.reply_text(msg,parse_mode="Markdown")

async def cmd_balance(u,c):
    try:
        api=lighter.ApiClient(lighter.Configuration(host=BASE_URL))
        acc=lighter.AccountApi(api)
        r=await acc.account(by="index",value=str(ACC_IDX)); await api.close()
        col=getattr(r,"collateral","?"); upnl=getattr(r,"unrealized_pnl","?")
        msg="*Balance*\nCollateral: $*"+str(col)+"*\nUnrealized: $*"+str(upnl)+"*\n\n"
        for coin in MARKETS:
            s=state[coin]["stats"]; em=MARKETS[coin]["emoji"]
            msg+=em+" *"+coin+"* Margin: $*"+str(s["current_margin"])+"*\n"
        await u.message.reply_text(msg,parse_mode="Markdown")
    except Exception as e:
        await u.message.reply_text("Balance Error: "+str(e))

async def post_init(application):
    global signer
    signer=lighter.SignerClient(url=BASE_URL,api_private_keys={KEY_IDX:PRV_KEY},account_index=ACC_IDX)
    for coin in MARKETS:
        asyncio.get_event_loop().create_task(strategy_loop(coin))

def main():
    global tg_app,state
    for coin in MARKETS:
        state[coin]={"position":None,"entry_px":0.0,"entry_sz":0.0,
                     "entry_mg":0.0,"sl_px":0.0,"tp_px":0.0,
                     "stats":load_stats(coin)}
    tg_app=Application.builder().token(TG_TOKEN).post_init(post_init).build()
    for cmd,fn in [("start",cmd_start),("status",cmd_status),("signal",cmd_signal),
                   ("stats",cmd_stats),("history",cmd_history),("balance",cmd_balance)]:
        tg_app.add_handler(CommandHandler(cmd,fn))
    tg_app.run_polling(drop_pending_updates=True)

if __name__=="__main__":
    main()
