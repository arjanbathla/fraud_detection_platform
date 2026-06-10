"""Tests for the SQL EDA module that profiles the creditcard data straight from SQLite."""
from fraud_platform.data.eda import EDA_QUERIES, run_eda


def test_run_eda_returns_all_queries(sample_db):
    out = run_eda(sample_db)
    assert set(out) == set(EDA_QUERIES)
    for name, frame in out.items():
        assert not frame.empty, f"{name} returned no rows"


def test_class_balance_covers_every_row(sample_db, sample_df):
    out = run_eda(sample_db)
    cb = out["class_balance"]
    assert cb["n"].sum() == len(sample_df)
    # fractions add up to 1
    assert abs(cb["pct"].sum() - 100.0) < 1e-6


def test_fraud_rate_by_hour_is_a_rate(sample_db):
    by_hour = run_eda(sample_db)["fraud_rate_by_hour"]
    assert (by_hour["fraud_rate"] >= 0).all()
    assert (by_hour["fraud_rate"] <= 1).all()
    assert by_hour["hour"].between(0, 23).all()


def test_amount_stats_split_fraud_vs_legit(sample_db):
    stats = run_eda(sample_db)["amount_stats_by_class"]
    assert set(stats["Class"]) == {0, 1}
    assert (stats["avg_amount"] >= 0).all()
