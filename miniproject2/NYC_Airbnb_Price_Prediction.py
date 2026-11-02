import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

sns.set_style("whitegrid")
pd.set_option("display.max_columns", 30)
RANDOM_STATE = 42


def ensure_dataset(project_dir: Path) -> Path:
    dataset_path = project_dir / "AB_NYC_2019.csv"
    if dataset_path.exists():
        return dataset_path

    try:
        import kagglehub
    except ModuleNotFoundError:
        raise FileNotFoundError(
            "AB_NYC_2019.csv is missing and kagglehub is not installed. "
            "Download the dataset manually into the project folder."
        )

    print("Downloading Airbnb NYC dataset from Kaggle...")
    downloaded_dir = kagglehub.dataset_download("drgilermo/new-york-city-airbnb-open-data")
    csv_candidates = sorted(Path(downloaded_dir).glob("**/*.csv"))
    csv_file = next((p for p in csv_candidates if p.name == "AB_NYC_2019.csv"), None)
    if csv_file is None:
        raise FileNotFoundError(f"No AB_NYC_2019.csv found in {downloaded_dir}")

    shutil.copy2(csv_file, dataset_path)
    print(f"Dataset saved to: {dataset_path}")
    return dataset_path


def save_figure(fig, name: str, charts_dir: Path):
    fig.tight_layout()
    fig.savefig(charts_dir / f"{name}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    project_dir = Path(__file__).resolve().parent
    charts_dir = project_dir / "charts"
    charts_dir.mkdir(exist_ok=True)

    dataset_path = ensure_dataset(project_dir)
    df = pd.read_csv(dataset_path)
    print("Shape (rows, columns):", df.shape)
    print(df.head())

    df.info()

    num_cols = df.select_dtypes(include="number").columns.tolist()
    cat_cols = df.select_dtypes(include="object").columns.tolist()
    print("Numeric    :", num_cols)
    print("Object/text:", cat_cols)
    print("Unique values per categorical column:")
    print(df[["neighbourhood_group", "neighbourhood", "room_type"]].nunique())

    missing = pd.DataFrame({
        "missing_count": df.isnull().sum(),
        "missing_pct": (df.isnull().mean() * 100).round(2),
    })
    print(missing[missing.missing_count > 0].sort_values("missing_pct", ascending=False))

    print("Missing last_review:", df.last_review.isnull().sum())
    df["last_review"] = pd.to_datetime(df["last_review"])
    print("Date range:", df.last_review.min(), "->", df.last_review.max())

    df = df.drop(columns=["last_review"])

    no_rpm = df[df.reviews_per_month.isnull()]
    print("Rows with missing reviews_per_month:", len(no_rpm))
    print("number_of_reviews for those rows ->")
    print(no_rpm.number_of_reviews.describe())
    print("Any of them with reviews > 0 ?", (no_rpm.number_of_reviews > 0).any())

    df["reviews_per_month"] = df["reviews_per_month"].fillna(0)
    print("Remaining missing in reviews_per_month:", df.reviews_per_month.isnull().sum())

    df = df.drop(columns=["id", "name", "host_id", "host_name"])
    print("Listings with price == 0:", (df.price == 0).sum())
    df = df[df.price > 0].reset_index(drop=True)
    print("Shape now:", df.shape)
    print("Remaining missing values:", df.isnull().sum().sum())

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    sns.boxplot(x=df.price, ax=axes[0, 0], color="steelblue")
    axes[0, 0].set_title("price boxplot (raw)")
    sns.histplot(df.price, bins=100, ax=axes[0, 1], color="steelblue")
    axes[0, 1].set_title("price histogram (raw)")
    sns.boxplot(x=df.minimum_nights, ax=axes[1, 0], color="indianred")
    axes[1, 0].set_title("minimum_nights boxplot (raw)")
    sns.histplot(df.minimum_nights, bins=100, ax=axes[1, 1], color="indianred")
    axes[1, 1].set_title("minimum_nights histogram (raw)")
    save_figure(fig, "01_price_and_nights_distribution", charts_dir)

    print(df[["price", "minimum_nights"]].describe(percentiles=[0.5, 0.9, 0.95, 0.99]).round(1))

    p95_price = df.price.quantile(0.95)
    p95_nights = df.minimum_nights.quantile(0.95)
    print(f"95th percentile price         : ${p95_price:.0f}")
    print(f"95th percentile minimum_nights: {p95_nights:.0f} nights")

    before = len(df)
    df = df[(df.price <= p95_price) & (df.minimum_nights <= p95_nights)].reset_index(drop=True)
    print(f"Rows before: {before:,} | after: {len(df):,} | removed: {before - len(df):,} ({(before - len(df)) / before:.1%})")

    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    sns.histplot(df.price, bins=60, ax=axes[0], color="steelblue")
    axes[0].set_title("price after trimming")
    sns.histplot(df.minimum_nights, bins=30, ax=axes[1], color="indianred")
    axes[1].set_title("minimum_nights after trimming")
    save_figure(fig, "02_price_after_trimming", charts_dir)

    print("neighbourhood unique values      :", df.neighbourhood.nunique())
    print("neighbourhood_group unique values :", df.neighbourhood_group.nunique())

    df = df.drop(columns=["neighbourhood"])

    df_encoded = pd.get_dummies(df, columns=["neighbourhood_group", "room_type"], drop_first=True, dtype=int)
    print("Shape after encoding:", df_encoded.shape)
    print("Dropped baselines -> neighbourhood_group: Bronx | room_type: Entire home/apt")
    print(df_encoded.head())

    X = df_encoded.drop(columns=["price"])
    y = df_encoded["price"]
    print("X shape:", X.shape, "| y shape:", y.shape)
    print("Features:", list(X.columns))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE
    )
    print(f"Train: {X_train.shape[0]:,} rows ({len(X_train)/len(X):.0%}) | Test: {X_test.shape[0]:,} rows ({len(X_test)/len(X):.0%})")

    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)

    print(f"R-squared : {r2:.4f}  -> model explains {r2:.1%} of the variance in price")
    print(f"MAE       : ${mae:.2f}  -> average error in dollars")
    print(f"MSE       : {mse:,.2f}")
    print(f"RMSE      : ${rmse:.2f}  -> typical error, penalising big misses more")
    print(f"\nTrain R-squared: {model.score(X_train, y_train):.4f} (vs test {r2:.4f}) -> no sign of overfitting if close")

    coef = pd.Series(model.coef_, index=X.columns).sort_values()
    print("Intercept:", round(model.intercept_, 1))
    print(coef.round(2))

    fig, ax = plt.subplots(figsize=(8, 5))
    coef.plot(kind="barh", color=np.where(coef > 0, "seagreen", "indianred"), ax=ax)
    ax.set_title("Linear regression coefficients ($ effect on nightly price)")
    ax.set_xlabel("Coefficient (USD)")
    save_figure(fig, "03_model_coefficients", charts_dir)

    plt.figure(figsize=(7, 7))
    plt.scatter(y_test, y_pred, alpha=0.15, s=10, color="steelblue")
    lims = [0, max(y_test.max(), y_pred.max())]
    plt.plot(lims, lims, "r--", lw=2, label="Perfect prediction (45° line)")
    plt.xlabel("Actual price ($)")
    plt.ylabel("Predicted price ($)")
    plt.title("Actual vs. Predicted prices")
    plt.legend()
    save_figure(plt.gcf(), "04_actual_vs_predicted", charts_dir)

    residuals = y_test - y_pred
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].scatter(y_pred, residuals, alpha=0.15, s=10, color="purple")
    axes[0].axhline(0, color="red", ls="--")
    axes[0].set_xlabel("Predicted price ($)")
    axes[0].set_ylabel("Residual (actual - predicted)")
    axes[0].set_title("Residual plot")
    sns.histplot(residuals, bins=60, kde=True, ax=axes[1], color="purple")
    axes[1].set_title("Error distribution")
    axes[1].set_xlabel("Residual ($)")
    save_figure(fig, "05_residuals", charts_dir)

    res = pd.DataFrame({"actual": y_test.values, "pred": y_pred})
    res["abs_err"] = (res.actual - res.pred).abs()
    res["signed_err"] = res.actual - res.pred
    res["pct_err"] = res.abs_err / res.actual * 100
    res["price_band"] = pd.cut(
        res.actual,
        bins=[0, 75, 125, 200, 400],
        labels=["Budget (<=$75)", "Low-mid ($76-125)", "Upper-mid ($126-200)", "Expensive (>$200)"],
    )
    summary = res.groupby("price_band", observed=True).agg(
        listings=("actual", "size"),
        MAE_dollars=("abs_err", "mean"),
        avg_signed_error=("signed_err", "mean"),
        MAPE_pct=("pct_err", "mean"),
    ).round(1)
    print(summary)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    summary["MAE_dollars"].plot(kind="bar", ax=axes[0], color="steelblue")
    axes[0].set_title("MAE ($) by price band")
    axes[0].set_xlabel("")
    summary["MAPE_pct"].plot(kind="bar", ax=axes[1], color="indianred")
    axes[1].set_title("Average % error by price band")
    axes[1].set_xlabel("")
    for ax in axes:
        ax.tick_params(axis="x", rotation=20)
    save_figure(fig, "06_error_by_price_band", charts_dir)

    metrics = {
        "R2": r2,
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
    }
    metrics_path = project_dir / "model_metrics.txt"
    with open(metrics_path, "w", encoding="utf-8") as f:
        for k, v in metrics.items():
            f.write(f"{k}: {v}\n")

    print(f"\nSaved charts to: {charts_dir}")
    print(f"Saved metrics to: {metrics_path}")


if __name__ == "__main__":
    main()
