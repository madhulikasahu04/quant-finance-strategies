import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

def gbm_prices(n=1000, s0=100.0, mu=0.10, sigma=0.30, seed=123):
    rng = np.random.default_rng(seed)
    dt = 1/252
    shocks = rng.normal((mu - 0.5*sigma**2)*dt, sigma*np.sqrt(dt), size=n)
    log_prices = np.cumsum(shocks) + np.log(s0)
    return pd.Series(np.exp(log_prices), name="Price")

def rsi(series, window=14):
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(window).mean()
    loss = -delta.clip(upper=0).rolling(window).mean()
    rs = gain / (loss.replace(0, np.nan))
    return 100 - (100 / (1 + rs))

def build_features(prices):
    df = pd.DataFrame(prices)
    df["Ret"] = df["Price"].pct_change()
    df["SMA_10"] = df["Price"].rolling(10).mean()
    df["SMA_50"] = df["Price"].rolling(50).mean()
    df["Vol_20"] = df["Ret"].rolling(20).std()
    df["Mom_5"] = df["Price"].pct_change(5)
    df["RSI_14"] = rsi(df["Price"], 14)
    df["Target"] = (df["Ret"].shift(-1) > 0).astype(int)
    df = df.dropna()
    features = ["Ret", "SMA_10", "SMA_50", "Vol_20", "Mom_5", "RSI_14"]
    X, y = df[features], df["Target"]
    return df, X, y

if __name__ == "__main__":
    prices = gbm_prices()
    df, X, y = build_features(prices)
    split = int(len(X)*0.7)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]
    model = LogisticRegression(max_iter=200)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"Test Accuracy: {acc:.3f}")
    proba = model.predict_proba(X_test)[:,1]
    signal = np.where(proba >= 0.5, 1, -1)
    test_df = df.iloc[split:].copy()
    test_df["Position"] = np.r_[0, signal[:-1]]
    test_df["StratRet"] = test_df["Position"] * test_df["Ret"]
    test_df["Cum_Market"] = (1+test_df["Ret"]).cumprod()
    test_df["Cum_Strategy"] = (1+test_df["StratRet"]).cumprod()
    out_csv = "trading_ml/algorithmic_results.csv"
    out_png = "trading_ml/strategy_vs_market.png"
    test_df[["Cum_Strategy", "Cum_Market"]].to_csv(out_csv)
    plt.figure(figsize=(10,5))
    plt.plot(test_df["Cum_Market"], label="Market")
    plt.plot(test_df["Cum_Strategy"], label="ML Strategy")
    plt.title("ML (Logistic Regression) Strategy vs Market — Synthetic Data")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    print(f"Saved: {out_csv} and {out_png}")
