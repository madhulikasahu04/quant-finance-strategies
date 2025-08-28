import numpy as np
import matplotlib.pyplot as plt

def simulate_portfolio(n_sims=5000, horizon_days=252, mu=0.08, sigma=0.20, start_value=1.0, seed=7):
    rng = np.random.default_rng(seed)
    dt = 1/252
    sims = np.empty((horizon_days, n_sims), dtype=float)
    sims[0,:] = start_value
    for t in range(1, horizon_days):
        z = rng.normal(0, 1, size=n_sims)
        sims[t,:] = sims[t-1,:] * np.exp((mu - 0.5*sigma**2)*dt + sigma*np.sqrt(dt)*z)
    return sims

def var_es(ending_values, alpha=0.95):
    losses = 1.0 - ending_values
    q = np.quantile(losses, alpha)
    es = losses[losses >= q].mean()
    return q, es

if __name__ == "__main__":
    sims = simulate_portfolio()
    ending = sims[-1,:]
    VaR95, ES95 = var_es(ending, 0.95)
    print(f"1-year VaR(95%): {VaR95:.3%} | ES(95%): {ES95:.3%}")
    out_png = "portfolio_risk/ending_value_distribution.png"
    plt.figure(figsize=(8,5))
    plt.hist(ending, bins=50)
    plt.title("Ending Portfolio Value (1-year) — Monte Carlo (Synthetic)")
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    print(f"Saved: {out_png}")
