# -*- coding: utf-8 -*-
import os,asyncio,logging,json
import numpy as np
from datetime import datetime
import pandas as pd,httpx
from telegram import Update
from telegram.ext import Application,CommandHandler,ContextTypes
import lighter

logging.basicConfig(format="%(asctime)s-%(levelname)s-%(message)s",level=logging.INFO)
logger=logging.getLogger(__name__)

TG_TOKEN =os.environ["TELEGRAM_BOT_TOKEN"]
TG_CHAT  =os.environ["TELEGRAM_CHAT_ID"]
ACC_IDX  =int(os.environ["ACCOUNT_INDEX"])
KEY_IDX  =int(os.environ["API_KEY_INDEX"])
PRV_KEY  =os.environ["LIGHTER_PRIVATE_KEY"]

URL      ="https://mainnet.zklighter.elliot.ai"
LEV      =5
INTERVAL =300
SL_M     =float(os.environ.get("SL_MULT","2.0"))
TP_M     =float(os.environ.get("TP_MULT","3.0"))
ADX_T    =20.0
MIN_M    =0.50
MKT      =0
SYM      ="ETH-USDT-SWAP"
SF       ="stats.json"
START_M  =float(os.environ.get("ETH_MARGIN","50"))
DEC      =3
MIN_SZ   =0.002

pos=None; ep=0.0; esz=0.0; em=0.0; slp=0.0; tpp=0.0
app=None; sc=None; st={}

def load_st():
    try:
        with open(SF) as f: return json.load(f)
    except:
        return {"total_trades":0,"wins":0,"losses":0,"total_pnl":0.0,
                "current_margin":START_M,"peak_margin":START_M,
                "long_trades":0,"short_trades":0,"entry_price":0.0,
                "entry_size":0.0,"entry_margin":0.0,"sl_price":0.0,
                "tp_price":0.0,"history":[]}

def save_st():
    with open(SF,"w") as f: json.dump(st,f)

def alma(s,p,sig=0.85,off=0.85):
    m=off*(p-1); sv=p/sig
    w=np.array([np.exp(-((i-m)**2)/(2*sv**2)) for i in range(p)])
    w/=w.sum()
    r=pd.Series(np.nan,index=s.index)
    for i in range(p-1,len(s)):
        r.iloc[i]=np.dot(w,s.iloc[i-p+1:i+1].values)
    return r

def atr(df,p=10):
    tr=pd.concat([df["high"]-df["low"],
        (df["high"]-df["close"].shift()).abs(),
        (df["low"]-df["close"].shift()).abs()],axis=1).max(axis=1)
    return tr.ewm(span=p,adjust=False).mean()

def adx(df,p=14):
    tr=atr(df,p)
    dp=(df["high"].diff()).clip(lower=0)
    dm=(-df["low"].diff()).clip(lower=0)
    dp=dp.where(dp>dm,0); dm=dm.where(dm>dp,0)
    dip=100*dp.ewm(span=p,adjust=False).mean()/tr.replace(0,np.nan)
    dim=100*dm.ewm(span=p,adjust=False).mean()/tr.replace(0,np.nan)
    dx=100*(dip-dim).abs()/(dip+dim).replace(0,np.nan)
    return dx.ewm(span=p,adjust=False).mean()

def rsi(s,p=14):
    d=s.diff()
    g=d.clip(lower=0).ewm(span=p,adjust=False).mean()
    l=(-d.clip(upper=0)).ewm(span=p,adjust=False).mean()
    return 100-(100/(1+g/l.replace(0,np.nan)))

def indicators(df):
    df=df.copy()
    df["af"]=alma(df["close"],13)
    df["as"]=alma(df["close"],21)
    df["e2"]=df["close"].ewm(span=200,adjust=False).mean()
    df["at"]=atr(df,10)
    df["rs"]=rsi(df["close"],14)
    df["ax"]=adx(df,14)
    return df

async def candles(lim=300):
    async with httpx.AsyncClient(timeout=15) as c:
        r=await c.get("https://www.okx.com/api/v5/market/candles",
            params={"instId":SYM,"bar":"5m","limit":str(lim)})
        data=r.json().get("data",[])
    df=pd.DataFrame(data)[[0,2,3,4]].copy()
    df.columns=["time","high","low","close"]
    for col in ["high","low","close"]: df[col]=df[col].astype(float)
    df["time"]=pd.to_datetime(df["time"].astype(int),unit="ms")
    return df.sort_values("time").reset_index(drop=True)

async def order(side,size,price,ro=False):
    slp=0.002
    op=round(float(price)*(1+slp),2) if side=="BUY" else round(float(price)*(1-slp),2)
    ba=int(float(size)*10000)
    tx,txh,err=await sc.create_order(
        market_index=MKT,
        client_order_index=int(datetime.now().timestamp()),
        base_amount=ba,price=op,is_ask=(side=="SELL"),
        order_type=sc.ORDER_TYPE_MARKET,
        time_in_force=sc.ORDER_TIME_IN_FORCE_IMMEDIATE_OR_CANCEL,
        reduce_only=ro,order_expiry=sc.DEFAULT_IOC_EXPIRY)
    if err: raise Exception(str(err))
    return txh

def sz(margin,price):
    return round(max((margin*LEV)/price,MIN_SZ),DEC)

def close_trade(xp,reason,side):
    global st
    pnl=round(((xp-st["entry_price"])*st["entry_size"]) if side=="LONG"
               else ((st["entry_price"]-xp)*st["entry_size"]),4)
    nm=round(max(st["entry_margin"]+pnl,MIN_M),4)
    st["total_trades"]+=1; st["total_pnl"]+=round(pnl,4)
    st["current_margin"]=nm
    if nm>st["peak_margin"]: st["peak_margin"]=nm
    if pnl>=0: st["wins"]+=1
    else: st["losses"]+=1
    st["history"].append({"no":st["total_trades"],"side":side,
        "entry":st["entry_price"],"exit":xp,"pnl":pnl,"reason":reason,
        "new_margin":nm,"time":datetime.now().strftime("%Y-%m-%d %H:%M")})
    st["history"]=st["history"][-20:]
    save_st()
    return pnl,nm,"WIN" if pnl>=0 else "LOSS"

async def tg(msg):
    try: await app.bot.send_message(chat_id=TG_CHAT,text=msg,parse_mode="Markdown")
    except Exception as e: logger.error("TG:%s",e)

async def loop_main():
    global pos,ep,esz,em,slp,tpp,st
    try:
        api=lighter.ApiClient(lighter.Configuration(host=URL))
        acc=lighter.AccountApi(api)
        r=await acc.account(str(ACC_IDX)); await api.close()
        for p2 in (getattr(r,"positions",[]) or []):
            if getattr(p2,"market_index",-1)==MKT:
                s2=float(getattr(p2,"base_amount",0) or 0)
                if s2!=0:
                    pos="LONG" if s2>0 else "SHORT"
                    ep=float(st.get("entry_price",0))
                    esz=abs(s2); em=float(st.get("entry_margin",0))
                    slp=float(st.get("sl_price",0)); tpp=float(st.get("tp_price",0))
    except Exception as e: logger.error("Init:%s",e)

    ps=pos if pos else "Wait"
    await tg("*ALMA Bot Started*\n"
             "Margin:`$"+str(st["current_margin"])+"` | "+ps+"\n"
             "Long: ALMA(13/21)+EMA200+ADX>20\n"
             "Short: ALMA(13/21)+RSI<50+ADX>20\n"
             "`"+str(LEV)+"x` | 5min | SL:"+str(SL_M)+"x TP:"+str(TP_M)+"x")

    while True:
        try:
            df=await candles(300); df=indicators(df); df=df.dropna()
            cu=df.iloc[-1]; pv=df.iloc[-2]
            price=cu["close"]; at2=cu["at"]; rs2=cu["rs"]; ax2=cu["ax"]
            bc=(cu["af"]>cu["as"]) and (pv["af"]<=pv["as"])
            ec=(cu["af"]<cu["as"]) and (pv["af"]>=pv["as"])
            bt=price>cu["e2"]; tr=ax2>ADX_T
            ls=bc and bt and tr; ss=ec and (rs2<50) and tr
            logger.info("ETH=$%.2f RSI=%.1f ADX=%.1f pos=%s",price,rs2,ax2,pos)

            if pos is None:
                if ls and st["current_margin"]>=MIN_M:
                    mg=st["current_margin"]; si=sz(mg,price)
                    nsl=round(price-(SL_M*at2),2); ntp=round(price+(TP_M*at2),2)
                    await tg("*ETH LONG Signal*\nPrice:`$"+str(round(price,2))+"` ADX:`"+str(round(ax2,1))+"`\nSL:`$"+str(nsl)+"` TP:`$"+str(ntp)+"`\nMargin:`$"+str(mg)+"`")
                    try:
                        await order("BUY",si,price)
                        pos="LONG"; ep=price; esz=si; em=mg; slp=nsl; tpp=ntp
                        st["entry_price"]=price; st["entry_size"]=si
                        st["entry_margin"]=mg; st["sl_price"]=nsl; st["tp_price"]=ntp
                        st["long_trades"]=st.get("long_trades",0)+1; save_st()
                        await tg("*LONG Opened!*\nEntry:`$"+str(round(price,2))+"` | `"+str(si)+" ETH`\nSL:`$"+str(nsl)+"` TP:`$"+str(ntp)+"`")
                    except Exception as e:
                        pos=None; await tg("LONG Failed:"+str(e))

                elif ss and st["current_margin"]>=MIN_M:
                    mg=st["current_margin"]; si=sz(mg,price)
                    nsl=round(price+(SL_M*at2),2); ntp=round(price-(TP_M*at2),2)
                    await tg("*ETH SHORT Signal*\nPrice:`$"+str(round(price,2))+"` ADX:`"+str(round(ax2,1))+"`\nSL:`$"+str(nsl)+"` TP:`$"+str(ntp)+"`\nMargin:`$"+str(mg)+"`")
                    try:
                        await order("SELL",si,price)
                        pos="SHORT"; ep=price; esz=si; em=mg; slp=nsl; tpp=ntp
                        st["entry_price"]=price; st["entry_size"]=si
                        st["entry_margin"]=mg; st["sl_price"]=nsl; st["tp_price"]=ntp
                        st["short_trades"]=st.get("short_trades",0)+1; save_st()
                        await tg("*SHORT Opened!*\nEntry:`$"+str(round(price,2))+"` | `"+str(si)+" ETH`\nSL:`$"+str(nsl)+"` TP:`$"+str(ntp)+"`")
                    except Exception as e:
                        pos=None; await tg("SHORT Failed:"+str(e))

            elif pos=="LONG":
                un=round((price-ep)*esz,4); rs3=None
                if price>=tpp: rs3="TP"
                elif price<=slp: rs3="SL"
                elif ec: rs3="Cross"
                if rs3:
                    await tg("*LONG "+rs3+"*\nPrice:`$"+str(round(price,2))+"` Unr:`$"+str(un)+"`")
                    try:
                        await order("SELL",esz,price,ro=True)
                        pnl,nm,oc=close_trade(price,rs3,"LONG")
                        wr=round(st["wins"]/max(st["total_trades"],1)*100,1)
                        sg="+" if pnl>=0 else ""
                        await tg(oc+" *LONG #"+str(st["total_trades"])+"* ["+rs3+"]\nEntry:`$"+str(ep)+"` Exit:`$"+str(round(price,2))+"`\nPnL:`"+sg+"$"+str(pnl)+"` WR:`"+str(wr)+"%`\nMargin:`$"+str(nm)+"`")
                        pos=None
                    except Exception as e: await tg("LONG Close Failed:"+str(e))

            elif pos=="SHORT":
                un=round((ep-price)*esz,4); rs3=None
                if price<=tpp: rs3="TP"
                elif price>=slp: rs3="SL"
                elif bc: rs3="Cross"
                if rs3:
                    await tg("*SHORT "+rs3+"*\nPrice:`$"+str(round(price,2))+"` Unr:`$"+str(un)+"`")
                    try:
                        await order("BUY",esz,price,ro=True)
                        pnl,nm,oc=close_trade(price,rs3,"SHORT")
                        wr=round(st["wins"]/max(st["total_trades"],1)*100,1)
                        sg="+" if pnl>=0 else ""
                        await tg(oc+" *SHORT #"+str(st["total_trades"])+"* ["+rs3+"]\nEntry:`$"+str(ep)+"` Exit:`$"+str(round(price,2))+"`\nPnL:`"+sg+"$"+str(pnl)+"` WR:`"+str(wr)+"%`\nMargin:`$"+str(nm)+"`")
                        pos=None
                    except Exception as e: await tg("SHORT Close Failed:"+str(e))

        except Exception as e:
            logger.error("Loop:%s",e); await tg("Error:"+str(e))
        await asyncio.sleep(INTERVAL)

async def cmd_start(u,c):
    await u.message.reply_text("*ALMA Bot*\n/status /signal /stats /history /balance /backtest",parse_mode="Markdown")

async def cmd_status(u,c):
    try:
        df=await candles(300); df=indicators(df); df=df.dropna(); cu=df.iloc[-1]
        pr=cu["close"]; e2=round(cu["e2"],2); at2=round(cu["at"],2)
        rs2=round(cu["rs"],1); ax2=round(cu["ax"],1)
        tr="Bull" if pr>e2 else "Bear"
        ax_s="Trending" if ax2>ADX_T else "Sideways-SKIP"
        ps=pos if pos else "Waiting"
        ex=""
        if pos and ep>0:
            un=round((pr-ep)*esz,4) if pos=="LONG" else round((ep-pr)*esz,4)
            ex="\nUnrealized:`$"+str(un)+"`\nSL:`$"+str(slp)+"` TP:`$"+str(tpp)+"`"
        await u.message.reply_text(
            "*ETH Status*\nPrice:`$"+str(round(pr,2))+"` | "+tr+"\n"
            "RSI:`"+str(rs2)+"` ATR:`$"+str(at2)+"`\n"
            "ADX:`"+str(ax2)+"` ["+ax_s+"]\n"
            "EMA200:`$"+str(e2)+"`\nPos:"+ps+ex+"\n"
            "Margin:`$"+str(st["current_margin"])+"`",parse_mode="Markdown")
    except Exception as e: await u.message.reply_text("Error:"+str(e))

async def cmd_signal(u,c):
    try:
        df=await candles(300); df=indicators(df); df=df.dropna()
        cu=df.iloc[-1]; pv=df.iloc[-2]
        pr=cu["close"]; rs2=round(cu["rs"],1); ax2=round(cu["ax"],1); at2=cu["at"]
        bc=(cu["af"]>cu["as"]) and (pv["af"]<=pv["as"])
        ec=(cu["af"]<cu["as"]) and (pv["af"]>=pv["as"])
        bt=pr>cu["e2"]; tr=ax2>ADX_T
        if bc and bt and tr: sig="LONG NOW!\nSL:`$"+str(round(pr-SL_M*at2,2))+"` TP:`$"+str(round(pr+TP_M*at2,2))+"`"
        elif ec and rs2<50 and tr: sig="SHORT NOW!\nSL:`$"+str(round(pr+SL_M*at2,2))+"` TP:`$"+str(round(pr-TP_M*at2,2))+"`"
        elif bc and bt and not tr: sig="LONG blocked - ADX low"
        elif ec and rs2<50 and not tr: sig="SHORT blocked - ADX low"
        elif bt and cu["af"]>cu["as"]: sig="Bullish - Wait cross"
        elif rs2<50 and cu["af"]<cu["as"]: sig="Bearish - Wait cross"
        else: sig="No signal"
        ax_s="Trending" if tr else "Sideways-SKIP"
        await u.message.reply_text(
            "*Signal*\nPrice:`$"+str(round(pr,2))+"` RSI:`"+str(rs2)+"`\n"
            "ADX:`"+str(ax2)+"` ["+ax_s+"]\n"+sig,parse_mode="Markdown")
    except Exception as e: await u.message.reply_text("Error:"+str(e))

async def cmd_stats(u,c):
    t=st["total_trades"]; wr=round(st["wins"]/max(t,1)*100,1)
    g=round((st["current_margin"]-START_M)/START_M*100,1)
    lt=st.get("long_trades",0); sh=st.get("short_trades",0)
    sg="+" if g>=0 else ""
    await u.message.reply_text(
        "*Stats*\nTrades:`"+str(t)+"` (L:"+str(lt)+" S:"+str(sh)+")\n"
        "WR:`"+str(wr)+"%` | `"+sg+str(g)+"%`\n"
        "PnL:`$"+str(round(st["total_pnl"],4))+"`\n"
        "Margin:`$"+str(st["current_margin"])+"` Peak:`$"+str(st["peak_margin"])+"`",
        parse_mode="Markdown")

async def cmd_history(u,c):
    h=st.get("history",[])
    if not h: await u.message.reply_text("No trades!"); return
    lines=["*Last trades:*"]
    for t2 in h[-5:]:
        sg="+" if t2["pnl"]>=0 else ""
        lines.append("#"+str(t2["no"])+" "+t2["side"]+" ["+t2["reason"]+"] `"+sg+"$"+str(t2["pnl"])+"` ->`$"+str(t2["new_margin"])+"`")
    await u.message.reply_text("\n".join(lines),parse_mode="Markdown")

async def cmd_balance(u,c):
    try:
        api=lighter.ApiClient(lighter.Configuration(host=URL))
        acc=lighter.AccountApi(api); r=await acc.account(str(ACC_IDX)); await api.close()
        col=getattr(r,"collateral","?"); upnl=getattr(r,"unrealized_pnl","?")
        await u.message.reply_text("*Balance*\nCollateral:$"+str(col)+"\nUnrealized:$"+str(upnl),parse_mode="Markdown")
    except Exception as e: await u.message.reply_text("Error:"+str(e))

async def cmd_backtest(u,c):
    await u.message.reply_text("Backtest running... 30-60sec")
    try:
        results=[]
        for days,label in [(90,"3M"),(365,"1Y")]:
            ac=[]; cur=int(datetime.now().timestamp()*1000)
            st2=int((datetime.now().timestamp()-days*86400)*1000)
            async with httpx.AsyncClient(timeout=30) as cl:
                while cur>st2:
                    r=await cl.get("https://www.okx.com/api/v5/market/history-candles",
                        params={"instId":SYM,"bar":"5m","after":str(cur),"limit":"300"})
                    cn=r.json().get("data",[])
                    if not cn: break
                    ac.extend(cn); cur=int(cn[-1][0])-1
                    if len(cn)<300: break
                    await asyncio.sleep(0.1)
            df=pd.DataFrame(ac)[[0,2,3,4]].copy()
            df.columns=["time","high","low","close"]
            for col in ["high","low","close"]: df[col]=df[col].astype(float)
            df["time"]=pd.to_datetime(df["time"].astype(int),unit="ms")
            df=df.drop_duplicates().sort_values("time").reset_index(drop=True)
            df=indicators(df); df=df.dropna().reset_index(drop=True)
            m=8.0; pb=None; bep=bem=bsl=btp=0.0; tr2=[]; lx=sx=w=0
            for i in range(1,len(df)):
                cu=df.iloc[i]; pv=df.iloc[i-1]; pr=cu["close"]
                if any(pd.isna(cu[x]) for x in ["af","as","at","e2","rs","ax"]): continue
                bc=(cu["af"]>cu["as"]) and (pv["af"]<=pv["as"])
                ec=(cu["af"]<cu["as"]) and (pv["af"]>=pv["as"])
                bt=pr>cu["e2"]; tr3=cu["ax"]>ADX_T
                if pb=="LONG":
                    rs3=None
                    if pr>=btp: rs3="TP"
                    elif pr<=bsl: rs3="SL"
                    elif ec: rs3="X"
                    if rs3:
                        pnl=(pr-bep)/bep*bem*LEV; m=max(bem+pnl,0.5)
                        tr2.append(pnl);
                        if pnl>0: w+=1
                        pb=None
                elif pb=="SHORT":
                    rs3=None
                    if pr<=btp: rs3="TP"
                    elif pr>=bsl: rs3="SL"
                    elif bc: rs3="X"
                    if rs3:
                        pnl=(bep-pr)/bep*bem*LEV; m=max(bem+pnl,0.5)
                        tr2.append(pnl);
                        if pnl>0: w+=1
                        pb=None
                if pb is None and m>=0.5:
                    if bc and bt and tr3: pb="LONG"; bep=pr; bem=m; bsl=pr-SL_M*cu["at"]; btp=pr+TP_M*cu["at"]; lx+=1
                    elif ec and cu["rs"]<50 and tr3: pb="SHORT"; bep=pr; bem=m; bsl=pr+SL_M*cu["at"]; btp=pr-TP_M*cu["at"]; sx+=1
            tot=len(tr2); wr2=round(w/max(tot,1)*100,1); gr=round((m-8)/8*100,1)
            sg="+" if gr>=0 else ""
            results.append("*"+label+"*\nTrades:"+str(tot)+"(L:"+str(lx)+" S:"+str(sx)+") WR:"+str(wr2)+"%\n"+sg+str(gr)+"% $8->$"+str(round(m,2)))
        await u.message.reply_text("*Backtest* ALMA(13/21)+ADX>20\n---\n"+"\n\n".join(results),parse_mode="Markdown")
    except Exception as e: await u.message.reply_text("Backtest failed:"+str(e))

async def post_init(application):
    global sc,st
    sc=lighter.SignerClient(url=URL,api_private_keys={KEY_IDX:PRV_KEY},account_index=ACC_IDX)
    asyncio.get_event_loop().create_task(loop_main())

def main():
    global app,st
    st=load_st()
    app=Application.builder().token(TG_TOKEN).post_init(post_init).build()
    for cmd,fn in [("start",cmd_start),("status",cmd_status),("signal",cmd_signal),
                   ("stats",cmd_stats),("history",cmd_history),("balance",cmd_balance),("backtest",cmd_backtest)]:
        app.add_handler(CommandHandler(cmd,fn))
    app.run_polling(drop_pending_updates=True)

if __name__=="__main__":
    main()
