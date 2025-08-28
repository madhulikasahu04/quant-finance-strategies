# Quant Finance Strategies — Python

A compact, recruiter‑friendly set of quant projects showing practical Python for data handling, feature engineering, backtesting, and basic risk simulation. Uses synthetic data so it runs anywhere.

## Structure
```
quant-finance-strategies/
├── README.md
├── requirements.txt
├── trading_baseline/
│   └── moving_average.py
├── trading_ml/
│   └── algorithmic_trading.py
└── portfolio_risk/
    └── monte_carlo.py
```

## Quick start
```
pip install -r requirements.txt
python trading_baseline/moving_average.py
python trading_ml/algorithmic_trading.py
python portfolio_risk/monte_carlo.py
```

### Notes
- Replace synthetic prices with real OHLCV later.
- Try other models and walk‑forward validation.
- Add fees/slippage for realism.
