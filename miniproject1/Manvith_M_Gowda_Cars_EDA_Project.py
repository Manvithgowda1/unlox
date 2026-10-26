"""
Exploratory Data Analysis on a Cars Dataset
===========================================
Project : Unlox Student Project - Cars EDA
Dataset : "Car Price" used-car listings (Indian market, CarDekho-style data)
Source  : YBI Foundation public dataset repository on GitHub
          https://github.com/ybifoundation/Dataset  (file: Car Price.csv)

How to run
----------
    pip install pandas numpy matplotlib seaborn
    python StudentName_Cars_EDA_Project.py

The script first looks for "Car Price.csv" next to this file. If the file is
not found it downloads it from the URL above. Charts are saved in ./charts/.
"""

# =============================================================================
# 1. SETUP
# =============================================================================
import os
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", context="notebook")
pd.set_option("display.width", 160)
pd.set_option("display.max_columns", 30)

DATA_URL = ("https://raw.githubusercontent.com/ybifoundation/Dataset/"
            "main/Car%20Price.csv")
LOCAL_FILE = "Car Price.csv"
CHART_DIR = "charts"
os.makedirs(CHART_DIR, exist_ok=True)

PALETTE = "viridis"
BLUE, ORANGE = "#2a6fdb", "#f28e2b"


def save_chart(fig, name):
    """Tidy layout, save the chart as PNG and show it."""
    fig.tight_layout()
    fig.savefig(os.path.join(CHART_DIR, name), dpi=200, bbox_inches="tight")
    plt.show()
    plt.close(fig)


# =============================================================================
# 2. LOAD DATA AND INSPECT
# =============================================================================
source = LOCAL_FILE if os.path.exists(LOCAL_FILE) else DATA_URL
df_raw = pd.read_csv(source)
df = df_raw.copy()

rows_before, cols_before = df_raw.shape

print("=" * 70, "\n2. DATASET INSPECTION\n" + "=" * 70)
print(f"Shape: {rows_before} rows x {cols_before} columns\n")
print("Column names and data types:")
print(df_raw.dtypes, "\n")
print("Sample rows:")
print(df_raw.head(), "\n")

# Separate numerical and categorical variables
numeric_cols = df_raw.select_dtypes(include="number").columns.tolist()
categorical_cols = [c for c in df_raw.columns if c not in numeric_cols]
print("Numerical  :", numeric_cols)
print("Categorical:", categorical_cols)

# =============================================================================
# 3. DATA QUALITY CHECKS (before cleaning)
# =============================================================================
print("\n" + "=" * 70, "\n3. DATA QUALITY CHECKS\n" + "=" * 70)

missing_before = df_raw.isna().sum()
duplicates_before = int(df_raw.duplicated().sum())
print("Missing values per column:\n", missing_before, "\n")
print("Duplicate rows:", duplicates_before)

# Whitespace problems in text columns
whitespace_issues = {c: int((df_raw[c] != df_raw[c].str.strip()).sum())
                     for c in categorical_cols}
print("Rows with leading/trailing spaces:", whitespace_issues)

# Inconsistent category labels - review the brand list by eye
print("\nBrand labels:", sorted(df_raw["Brand"].unique()))
print("\nModel names that start with the odd brand labels:")
print(df_raw.loc[df_raw["Brand"].isin(["Land", "OpelCorsa"]), "Model"]
      .str.split().str[:2].str.join(" ").value_counts())

# Implausible odometer values (very low km on an old car)
REFERENCE_YEAR = int(df_raw["Year"].max())   # newest model year in the data
age_raw = REFERENCE_YEAR - df_raw["Year"]
suspicious_km = df_raw[(df_raw["KM_Driven"] < 100) & (age_raw >= 2)]
print(f"\nReference year used to compute car age: {REFERENCE_YEAR}")
print("Cars >= 2 years old showing < 100 km driven:", len(suspicious_km))

# =============================================================================
# 4. DATA CLEANING
# =============================================================================
print("\n" + "=" * 70, "\n4. DATA CLEANING\n" + "=" * 70)
cleaning_log = []

# 4.1 Trim spaces in text columns (safe, no information lost)
for c in categorical_cols:
    df[c] = df[c].str.strip()
cleaning_log.append("Trimmed leading/trailing spaces in all text columns "
                    "(none were found).")

# 4.2 Fix inconsistent brand labels so the same maker is not split/misnamed
brand_fix = {"Land": "Land Rover", "OpelCorsa": "Opel"}
fixed_labels = int(df["Brand"].isin(brand_fix).sum())
df["Brand"] = df["Brand"].replace(brand_fix)
cleaning_log.append(f"Corrected brand labels {brand_fix} "
                    f"({fixed_labels} rows affected).")

# 4.3 Remove exact duplicate listings
df = df.drop_duplicates().reset_index(drop=True)
cleaning_log.append(f"Dropped {duplicates_before} exact duplicate rows.")

# 4.4 Remove implausible odometer readings (data-entry errors)
mask_bad_km = (df["KM_Driven"] < 100) & ((REFERENCE_YEAR - df["Year"]) >= 2)
bad_km_removed = int(mask_bad_km.sum())
df = df[~mask_bad_km].reset_index(drop=True)
cleaning_log.append(f"Removed {bad_km_removed} rows where a car aged 2+ years "
                    f"shows under 100 km (likely entry error).")

# 4.5 Missing values: none found, so nothing is imputed
cleaning_log.append("No missing values were found, so no imputation was needed.")

# 4.6 Feature engineering / type improvements
df["Car_Age"] = REFERENCE_YEAR - df["Year"]                 # years
df["Price_Lakh"] = df["Selling_Price"] / 100_000            # 1 lakh = 100,000 INR
df["Log_Price"] = np.log10(df["Selling_Price"])             # for skewed price
owner_rank = {"Test Drive Car": 0, "First Owner": 1, "Second Owner": 2,
              "Third Owner": 3, "Fourth & Above Owner": 4}
df["Owner_Count"] = df["Owner"].map(owner_rank)
for c in ["Brand", "Fuel", "Seller_Type", "Transmission"]:
    df[c] = df[c].astype("category")
df["Owner"] = pd.Categorical(df["Owner"], categories=list(owner_rank),
                             ordered=True)
cleaning_log.append("Created Car_Age, Price_Lakh, Log_Price and Owner_Count; "
                    "converted text columns to category dtype.")

# 4.7 Outliers: detect with the IQR rule, but KEEP them (see explanation)
def iqr_outliers(series):
    q1, q3 = series.quantile([0.25, 0.75])
    iqr = q3 - q1
    return (series < q1 - 1.5 * iqr) | (series > q3 + 1.5 * iqr)

price_out = iqr_outliers(df["Selling_Price"])
km_out = iqr_outliers(df["KM_Driven"])
PREMIUM = ["Audi", "BMW", "Mercedes-Benz", "Jaguar", "Land Rover", "Volvo",
           "Jeep"]
premium_share_in_price_outliers = df.loc[price_out, "Brand"].isin(PREMIUM).mean()
cleaning_log.append(
    f"Flagged {int(price_out.sum())} price outliers and {int(km_out.sum())} "
    f"mileage outliers using the 1.5 x IQR rule but kept them: they are "
    f"genuine cars (e.g. luxury models), not errors. Skewness is handled with "
    f"medians and a log scale instead of deleting rows.")

rows_after, cols_after = df.shape
print("\n".join(f"- {line}" for line in cleaning_log))
print(f"\nDataset size BEFORE cleaning: {rows_before} rows x {cols_before} cols")
print(f"Dataset size AFTER  cleaning: {rows_after} rows x {cols_after} cols")
print("Missing values after cleaning:", int(df.isna().sum().sum()),
      "| Duplicates after cleaning:", int(df.duplicated().sum()))

# =============================================================================
# 5. DESCRIPTIVE STATISTICS
# =============================================================================
print("\n" + "=" * 70, "\n5. DESCRIPTIVE STATISTICS\n" + "=" * 70)

num_summary = df[["Year", "Selling_Price", "KM_Driven", "Car_Age"]].describe().T
num_summary["skew"] = df[["Year", "Selling_Price", "KM_Driven",
                          "Car_Age"]].skew()
print("Numerical columns:\n", num_summary.round(2), "\n")

cat_summary = pd.DataFrame({
    "unique": [df[c].nunique() for c in
               ["Brand", "Fuel", "Seller_Type", "Transmission", "Owner"]],
    "top": [df[c].mode()[0] for c in
            ["Brand", "Fuel", "Seller_Type", "Transmission", "Owner"]],
    "top_share_%": [round(df[c].value_counts(normalize=True).iloc[0] * 100, 1)
                    for c in ["Brand", "Fuel", "Seller_Type", "Transmission",
                              "Owner"]],
}, index=["Brand", "Fuel", "Seller_Type", "Transmission", "Owner"])
print("Categorical columns:\n", cat_summary, "\n")

for c in ["Fuel", "Transmission", "Seller_Type", "Owner"]:
    print(df[c].value_counts(), "\n")

# =============================================================================
# 6. VISUAL EDA  (8 charts, 5 chart types)
# =============================================================================
print("=" * 70, "\n6. VISUAL EDA\n" + "=" * 70)

# ---- Chart 1: Bar chart - most common brands (univariate, categorical) ----
top_brands = df["Brand"].value_counts().head(10)
fig, ax = plt.subplots(figsize=(8, 4))
sns.barplot(x=top_brands.values, y=top_brands.index.astype(str),
            palette=PALETTE, ax=ax)
for i, v in enumerate(top_brands.values):
    ax.text(v + 5, i, str(v), va="center", fontsize=9)
ax.set(title="Chart 1: Top 10 Car Brands by Number of Listings",
       xlabel="Number of listings", ylabel="Brand")
save_chart(fig, "01_top_brands.png")

# ---- Chart 2: Count plot - fuel type (univariate, categorical) ----
fuel_order = df["Fuel"].value_counts().index
fig, ax = plt.subplots(figsize=(8, 3.8))
sns.countplot(data=df, x="Fuel", order=fuel_order, palette="mako", ax=ax)
for p in ax.patches:
    ax.annotate(int(p.get_height()), (p.get_x() + p.get_width() / 2,
                p.get_height()), ha="center", va="bottom", fontsize=9)
ax.set(title="Chart 2: Listings by Fuel Type", xlabel="Fuel type",
       ylabel="Number of listings")
save_chart(fig, "02_fuel_count.png")

# ---- Chart 3: Histogram - price distribution (univariate, numerical) ----
fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
sns.histplot(df["Price_Lakh"], bins=50, color=BLUE, ax=axes[0])
axes[0].axvline(df["Price_Lakh"].median(), color="red", ls="--",
                label=f"Median = {df['Price_Lakh'].median():.1f}")
axes[0].axvline(df["Price_Lakh"].mean(), color="green", ls="--",
                label=f"Mean = {df['Price_Lakh'].mean():.1f}")
axes[0].set(title="Selling price (original scale)",
            xlabel="Price (INR lakh)", ylabel="Number of cars")
axes[0].legend()
sns.histplot(df["Log_Price"], bins=40, color=ORANGE, ax=axes[1])
axes[1].set(title="Selling price (log10 scale)",
            xlabel="log10(Price in INR)", ylabel="Number of cars")
fig.suptitle("Chart 3: Distribution of Selling Price", y=1.02,
             fontweight="bold")
save_chart(fig, "03_price_hist.png")

# ---- Chart 4: Box plot - price by transmission (bivariate) ----
fig, ax = plt.subplots(figsize=(8, 4))
sns.boxplot(data=df, x="Transmission", y="Price_Lakh", palette=[BLUE, ORANGE],
            ax=ax)
ax.set_yscale("log")
ax.set(title="Chart 4: Selling Price by Transmission Type (log scale)",
       xlabel="Transmission", ylabel="Price (INR lakh, log scale)")
save_chart(fig, "04_price_by_transmission.png")

# ---- Chart 5: Scatter - car age vs price (bivariate) ----
fig, ax = plt.subplots(figsize=(8, 4.2))
sns.scatterplot(data=df, x="Car_Age", y="Price_Lakh", alpha=0.35, s=18,
                color=BLUE, ax=ax)
med_by_age = df.groupby("Car_Age")["Price_Lakh"].median()
ax.plot(med_by_age.index, med_by_age.values, color="red", marker="o", ms=4,
        label="Median price for each age")
ax.set_yscale("log")
ax.legend()
ax.set(title="Chart 5: Car Age vs Selling Price (log scale)",
       xlabel="Car age (years)", ylabel="Price (INR lakh, log scale)")
save_chart(fig, "05_age_vs_price.png")

# ---- Chart 6: Scatter - km driven vs price, coloured by age band
#      (bivariate + multivariate) ----
df["Age_Band"] = pd.cut(df["Car_Age"], bins=[-1, 3, 7, 12, 40],
                        labels=["0-3 yrs", "4-7 yrs", "8-12 yrs", "13+ yrs"])
fig, ax = plt.subplots(figsize=(8, 4.2))
sns.scatterplot(data=df, x="KM_Driven", y="Price_Lakh", hue="Age_Band",
                palette="viridis", alpha=0.5, s=18, ax=ax)
ax.set_yscale("log")
ax.set(title="Chart 6: Kilometres Driven vs Selling Price, by Age Band",
       xlabel="Kilometres driven", ylabel="Price (INR lakh, log scale)")
ax.legend(title="Car age", fontsize=8)
save_chart(fig, "06_km_vs_price.png")

# ---- Chart 7: Bar chart - median price by brand (bivariate) ----
brand_stats = (df.groupby("Brand", observed=True)["Price_Lakh"]
                 .agg(["median", "count"]))
brand_stats = brand_stats[brand_stats["count"] >= 20].sort_values(
    "median", ascending=False)
top_price_brands = brand_stats.head(10)
fig, ax = plt.subplots(figsize=(8, 4.2))
sns.barplot(x=top_price_brands["median"].values,
            y=top_price_brands.index.astype(str), palette="rocket", ax=ax)
for i, v in enumerate(top_price_brands["median"].values):
    ax.text(v + 0.3, i, f"{v:.1f}", va="center", fontsize=9)
ax.set(title="Chart 7: Median Selling Price by Brand (brands with 20+ listings)",
       xlabel="Median price (INR lakh)", ylabel="Brand")
save_chart(fig, "07_median_price_brand.png")

# ---- Chart 8: Heatmap - correlation matrix (multivariate) ----
corr_cols = ["Selling_Price", "Car_Age", "KM_Driven", "Owner_Count"]
corr = df[corr_cols].corr(method="spearman")
fig, ax = plt.subplots(figsize=(6.2, 4.6))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1,
            square=True, linewidths=0.5, ax=ax)
ax.set_title("Chart 8: Spearman Correlation Heatmap")
save_chart(fig, "08_correlation_heatmap.png")

# =============================================================================
# 7. EXTRA ANALYSIS USED FOR THE FINDINGS
# =============================================================================
fuel_share = df["Fuel"].value_counts(normalize=True) * 100
brand_share = df["Brand"].value_counts(normalize=True) * 100
top2_brand_share = brand_share.head(2).sum()
petrol_diesel_share = fuel_share[["Petrol", "Diesel"]].sum()

median_price_trans = df.groupby("Transmission", observed=True)[
    "Selling_Price"].median()
median_age_trans = df.groupby("Transmission", observed=True)["Car_Age"].median()
premium_share_trans = (df.assign(is_premium=df["Brand"].isin(PREMIUM))
                       .groupby("Transmission", observed=True)["is_premium"]
                       .mean() * 100)

median_by_band = df.groupby("Age_Band", observed=True)["Price_Lakh"].median()
corr_age = corr.loc["Selling_Price", "Car_Age"]
corr_km = corr.loc["Selling_Price", "KM_Driven"]
corr_age_km = corr.loc["Car_Age", "KM_Driven"]

# Does mileage still matter once age is similar? (cars aged 4-7 years)
mid_age = df[df["Age_Band"] == "4-7 yrs"]
corr_km_mid_age = mid_age[["Selling_Price", "KM_Driven"]].corr(
    method="spearman").iloc[0, 1]

price_q1, price_q3 = df["Selling_Price"].quantile([0.25, 0.75])
price_out_threshold = price_q3 + 1.5 * (price_q3 - price_q1)   # IQR upper fence
non_premium_outlier_brands = (df.loc[price_out & ~df["Brand"].isin(PREMIUM),
                                     "Brand"].astype(str)
                              .value_counts().index.tolist())

mean_price, median_price = df["Selling_Price"].mean(), df["Selling_Price"].median()
price_skew = df["Selling_Price"].skew()

# =============================================================================
# 8. KEY FINDINGS (generated from the numbers above)
# =============================================================================
findings = [
    (f"Market is concentrated: {brand_share.index[0]} and "
     f"{brand_share.index[1]} together make up {top2_brand_share:.0f}% of "
     f"listings, and petrol + diesel cars account for "
     f"{petrol_diesel_share:.1f}% of all cars (CNG, LPG and electric are "
     f"each under 2%)."),
    (f"Prices are strongly right-skewed: the mean price is "
     f"Rs {mean_price/1e5:.2f} lakh but the median is only "
     f"Rs {median_price/1e5:.2f} lakh (skewness {price_skew:.1f}). Because of "
     f"a few luxury cars, the median is a better 'typical price' than the "
     f"mean."),
    (f"Car age is the strongest numerical link to price: Spearman "
     f"correlation = {corr_age:.2f}. The median price falls from "
     f"Rs {median_by_band.iloc[0]:.1f} lakh for 0-3 year-old cars to "
     f"Rs {median_by_band.iloc[-1]:.1f} lakh for cars older than 12 years."),
    (f"Mileage is only moderately related to price (correlation "
     f"{corr_km:.2f}) and is itself tied to age (age vs km = "
     f"{corr_age_km:.2f}), so part of its apparent effect is really an age "
     f"effect. Among 4-7 year-old cars the km-price correlation is "
     f"{corr_km_mid_age:.2f}."),
    (f"Automatic cars have a much higher median price than manual cars "
     f"(Rs {median_price_trans['Automatic']/1e5:.2f} lakh vs "
     f"Rs {median_price_trans['Manual']/1e5:.2f} lakh), but this is not a "
     f"pure transmission effect: {premium_share_trans['Automatic']:.0f}% of "
     f"automatics are from premium brands against "
     f"{premium_share_trans['Manual']:.1f}% of manuals, and automatics are "
     f"also newer (median age {median_age_trans['Automatic']:.0f} vs "
     f"{median_age_trans['Manual']:.0f} years)."),
    (f"Brand matters: among brands with 20+ listings, "
     f"{brand_stats.index[0]} has the highest median price "
     f"(Rs {brand_stats['median'].iloc[0]:.1f} lakh), while the best-selling "
     f"brand {brand_share.index[0]} has a median of "
     f"Rs {brand_stats.loc[brand_share.index[0], 'median']:.1f} lakh."),
    (f"Price outliers are genuine high-end cars rather than errors: "
     f"{int(price_out.sum())} cars ({price_out.mean()*100:.1f}%) sell above "
     f"Rs {price_out_threshold/1e5:.1f} lakh, their median age is only "
     f"{df.loc[price_out, 'Car_Age'].median():.0f} years, and "
     f"{premium_share_in_price_outliers*100:.0f}% belong to premium brands "
     f"(the rest are mostly {non_premium_outlier_brands[0]} and "
     f"{non_premium_outlier_brands[1]}). They were therefore kept."),
]

limitations = [
    "The dataset is a single snapshot of listings with no listing date or "
    "city, so price trends over time or by location cannot be studied.",
    "Selling_Price is an asking/selling price of the listing, not a verified "
    "transaction price.",
    "Important price drivers such as car condition, trim/variant details, "
    "engine size and horsepower are missing or only hidden in the Model text.",
    "Car age depends on the assumed reference year "
    f"({REFERENCE_YEAR}, the newest model year in the data).",
    "Relationships shown are associations, not proof of cause and effect.",
    "Several categories are very small (e.g. 1 electric car, a few brands "
    "with fewer than 10 listings) so conclusions about them are unreliable.",
]

print("\n" + "=" * 70, "\n8. KEY FINDINGS\n" + "=" * 70)
for i, f in enumerate(findings, 1):
    print(f"{i}. {f}\n")
print("LIMITATIONS")
for l in limitations:
    print("-", l)

print("\nDone. Charts saved in the ./charts folder.")
