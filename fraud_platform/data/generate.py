"""Synthetic transaction dataset generator.

EVERYTHING PRODUCED HERE IS SYNTHETIC. No real cardholder or financial data is used.
The generator draws legitimate transactions from "normal" distributions and injects a
controllable fraction of fraudulent ones whose features are shifted in a few realistic
ways (unusual amounts, odd hours, far from home, more recent activity bursts). The shift
is deliberate but noisy, so models have signal to learn without the task being trivial.

Run:
    python -m fraud_platform.data.generate --rows 50000 --fraud-frac 0.01 --out artifacts/data/transactions.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from fraud_platform.config import DATA_DIR, RANDOM_SEED, TARGET

MERCHANT_CATEGORIES = [
    "grocery", "restaurant", "electronics", "travel",
    "fuel", "entertainment", "clothing", "online_services",
]
TRANSACTION_TYPES = ["purchase", "withdrawal", "transfer", "payment"]
DEVICE_TYPES = ["ios", "android", "web", "pos_terminal"]
COUNTRIES = ["GB", "US", "FR", "DE", "IN", "NG", "BR"]


def generate(
    n_rows: int = 50_000,
    fraud_frac: float = 0.01,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """Generate a labelled synthetic transaction dataframe.

    fraud_frac is the target fraction of rows labelled fraudulent (is_fraud=1).
    """
    if not 0 < fraud_frac < 0.5:
        raise ValueError("fraud_frac must be in (0, 0.5)")

    rng = np.random.default_rng(seed)
    n_fraud = int(round(n_rows * fraud_frac))
    n_legit = n_rows - n_fraud

    legit = _make_legit(n_legit, rng)
    fraud = _make_fraud(n_fraud, rng)

    df = pd.concat([legit, fraud], ignore_index=True)
    # shuffle so fraud isn't all at the bottom
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    # build a timestamp + id after shuffling
    base = pd.Timestamp("2026-01-01")
    minutes = rng.integers(0, 60 * 24 * 180, size=len(df))  # spread over ~6 months
    df["timestamp"] = base + pd.to_timedelta(minutes, unit="m")
    df["transaction_id"] = [f"tx_{i:08d}" for i in range(len(df))]
    # hour derived from timestamp so it's internally consistent
    df["hour"] = df["timestamp"].dt.hour

    cols = [
        "transaction_id", "timestamp", "amount", "account_age_days",
        "num_tx_last_24h", "avg_amount_last_30d", "distance_from_home_km",
        "hour", "merchant_category", "transaction_type", "device_type",
        "country", TARGET,
    ]
    return df[cols]


def _make_legit(n: int, rng: np.random.Generator) -> pd.DataFrame:
    # log-normal amounts (most spend is small, a few large)
    amount = np.round(rng.lognormal(mean=3.2, sigma=0.9, size=n), 2)
    account_age = rng.integers(30, 3650, size=n)
    num_tx_24h = rng.poisson(lam=3, size=n)
    avg_amount_30d = np.round(amount * rng.uniform(0.6, 1.4, size=n), 2)
    distance = np.abs(rng.normal(8, 12, size=n)).round(2)  # usually close to home
    hour = rng.integers(6, 23, size=n)  # mostly daytime, overwritten by ts hour later

    return pd.DataFrame({
        "amount": amount,
        "account_age_days": account_age,
        "num_tx_last_24h": num_tx_24h,
        "avg_amount_last_30d": avg_amount_30d,
        "distance_from_home_km": distance,
        "hour": hour,
        "merchant_category": rng.choice(MERCHANT_CATEGORIES, size=n,
                                        p=[0.22, 0.18, 0.08, 0.07, 0.12, 0.1, 0.13, 0.1]),
        "transaction_type": rng.choice(TRANSACTION_TYPES, size=n, p=[0.7, 0.1, 0.1, 0.1]),
        "device_type": rng.choice(DEVICE_TYPES, size=n, p=[0.35, 0.35, 0.2, 0.1]),
        "country": rng.choice(COUNTRIES, size=n, p=[0.45, 0.2, 0.1, 0.1, 0.08, 0.03, 0.04]),
        TARGET: np.zeros(n, dtype=int),
    })


def _make_fraud(n: int, rng: np.random.Generator) -> pd.DataFrame:
    """Fraud rows that OVERLAP heavily with legit behaviour.

    Real fraud doesn't separate cleanly, so we deliberately keep the distributions close to
    legit and only nudge them. We also make a chunk of fraud "stealth" — it looks almost
    exactly like legit activity. This keeps the task hard enough that models score in a
    realistic range and actually differ from each other (rather than every model hitting 1.0).
    """
    # fraud distributions sit close to legit, just shifted a bit
    amount = np.round(rng.lognormal(mean=3.6, sigma=1.0, size=n), 2)
    account_age = rng.integers(5, 2500, size=n)
    num_tx_24h = rng.poisson(lam=5, size=n)
    avg_amount_30d = np.round(amount * rng.uniform(0.4, 1.2, size=n), 2)
    distance = np.abs(rng.normal(35, 40, size=n)).round(2)
    hour = rng.integers(0, 24, size=n)

    # ~45% of fraud is "stealth": override the obvious tells with legit-like values
    stealth = rng.random(n) < 0.45
    amount[stealth] = np.round(rng.lognormal(mean=3.2, sigma=0.9, size=stealth.sum()), 2)
    distance[stealth] = np.abs(rng.normal(8, 12, size=stealth.sum())).round(2)
    num_tx_24h[stealth] = rng.poisson(lam=3, size=stealth.sum())

    return pd.DataFrame({
        "amount": amount,
        "account_age_days": account_age,
        "num_tx_last_24h": num_tx_24h,
        "avg_amount_last_30d": avg_amount_30d,
        "distance_from_home_km": distance,
        "hour": hour,
        # only mildly skewed category mix — still overlaps legit a lot
        "merchant_category": rng.choice(MERCHANT_CATEGORIES, size=n,
                                        p=[0.15, 0.12, 0.18, 0.12, 0.08, 0.1, 0.1, 0.15]),
        "transaction_type": rng.choice(TRANSACTION_TYPES, size=n, p=[0.55, 0.18, 0.17, 0.1]),
        "device_type": rng.choice(DEVICE_TYPES, size=n, p=[0.28, 0.3, 0.32, 0.1]),
        "country": rng.choice(COUNTRIES, size=n, p=[0.38, 0.18, 0.09, 0.09, 0.09, 0.1, 0.07]),
        TARGET: np.ones(n, dtype=int),
    })


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate SYNTHETIC transaction data")
    ap.add_argument("--rows", type=int, default=50_000)
    ap.add_argument("--fraud-frac", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=RANDOM_SEED)
    ap.add_argument("--out", type=str, default=str(DATA_DIR / "transactions.csv"))
    args = ap.parse_args()

    df = generate(args.rows, args.fraud_frac, args.seed)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    actual = df[TARGET].mean()
    print(f"[SYNTHETIC] wrote {len(df):,} rows to {out}")
    print(f"[SYNTHETIC] fraud rate = {actual:.4f} ({int(df[TARGET].sum())} fraud rows)")


if __name__ == "__main__":
    main()
