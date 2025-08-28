import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def gbm_prices(n=1000, s0=100.0, mu=0.08, sigma=0.25, seed=42):
    rng = np.random.default_rng(seed)
    dt = 1/252
    shocks = rng.normal((mu - 0.5*sigma**2)*dt, sigma*np.sqrt(dt), size=n)
    log_prices = np.cumsum(shocks) + np.log(s0)
    return pd.Series(np.exp(log_prices), name="Price")

def sma_crossover_backtest(prices, short=50, long=200):
    df = pd.DataFrame(prices)
    df["SMA_Short"] = df["Price"].rolling(short).mean()
    df["SMA_Long"]  = df["Price"].rolling(long).mean()
    df["Signal"] = 0
    df.loc[df["SMA_Short"] > df["SMA_Long"], "Signal"] = 1
    df.loc[df["SMA_Short"] < df["SMA_Long"], "Signal"] = -1
    df["Position"] = df["Signal"].shift(1).fillna(0)
    df["Ret"] = df["Price"].pct_change().fillna(0)
    df["StratRet"] = df["Position"] * df["Ret"]
    df["Cum_Market"] = (1+df["Ret"]).cumprod()
    df["Cum_Strategy"] = (1+df["StratRet"]).cumprod()
    return df

if __name__ == "__main__":
    prices = gbm_prices()
    bt = sma_crossover_backtest(prices)
    out_csv = "trading_baseline/sma_results.csv"
    out_png = "trading_baseline/sma_vs_market.png"
    bt[["Cum_Strategy", "Cum_Market"]].to_csv(out_csv)
    plt.figure(figsize=(10,5))
    plt.plot(bt["Cum_Market"], label="Market")
    plt.plot(bt["Cum_Strategy"], label="SMA Strategy")
    plt.title("SMA(50/200) Crossover — Strategy vs Market (Synthetic Data)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    print(f"Saved: {out_csv} and {out_png}")
