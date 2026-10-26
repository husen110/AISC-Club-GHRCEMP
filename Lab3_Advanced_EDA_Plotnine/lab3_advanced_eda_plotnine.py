# =====================================================================
# Lab Assignment 3: Advanced Univariate, Bivariate, and Multivariate
#                   Analysis
# ---------------------------------------------------------------------
# Objective : Exploratory data analysis (EDA) across 1, 2 and 3+
#             features using Plotnine (ggplot2 syntax in Python) and
#             stacked visualizations.
# Dataset   : "diamonds" (ships with plotnine: plotnine.data.diamonds)
# Run with  : python lab3_advanced_eda_plotnine.py
# Requires  : pip install plotnine pandas seaborn matplotlib
# Output    : Step-by-step console log + PNG plots in ./plots/
# =====================================================================

import os
import warnings
from datetime import datetime

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotnine
import seaborn as sns
from plotnine import (aes, element_text, facet_wrap, geom_bar, geom_histogram,
                      geom_point, geom_vline, ggplot, labs, position_fill,
                      scale_color_cmap_d, scale_fill_brewer,
                      scale_y_continuous, theme, theme_bw, theme_minimal)
from plotnine.data import diamonds

warnings.filterwarnings("ignore")  # keep the console log clean


# ---------- Helper: print a clearly visible step header --------------
def step(n, title):
    print("\n" + "=" * 70)
    print(f"STEP {n} : {title}")
    print("=" * 70)


OUT_DIR = "plots"
os.makedirs(OUT_DIR, exist_ok=True)


def save_gg(plot, file, width=8, height=5):
    path = os.path.join(OUT_DIR, file)
    plot.save(path, width=width, height=height, dpi=110, verbose=False)
    print(">> Plot saved to:", path)


pd.set_option("display.width", 110)
print("Lab Assignment 3 - Advanced Univariate, Bivariate & Multivariate Analysis")
print("Python libraries : plotnine", plotnine.__version__, "| pandas", pd.__version__,
      "| seaborn", sns.__version__)
print("Run on           :", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))


# =====================================================================
# STEP 1 : Load and inspect the dataset
# =====================================================================
step(1, "Load the 'diamonds' dataset")

df = diamonds.copy()
print(">> Dataset loaded successfully.")
print(">> Shape (rows x columns):", df.shape)

print("\n>> Column data types:")
print(df.dtypes)

print("\n>> First 5 rows:")
print(df.head())

print("\n>> Missing values in the whole dataset:", int(df.isnull().sum().sum()))

print("\n>> Summary of key continuous features:")
print(df[["carat", "depth", "table", "price"]].describe().round(2))

print("\n>> Levels of the categorical features (ordered worst -> best):")
for col in ["cut", "color", "clarity"]:
    print(f"   {col:<8}: {list(df[col].cat.categories)}")


# =====================================================================
# STEP 2 : Univariate analysis - Plotnine histogram of price
# =====================================================================
step(2, "Univariate: Plotnine Histogram of 'price' (bin width + theme)")

print(">> Effect of bin width on the number of bins:")
price_range = df["price"].max() - df["price"].min()
for bw in [250, 500, 1000]:
    print(f"   binwidth = {bw:>4}  ->  about {int(np.ceil(price_range / bw))} bins")

BINWIDTH = 500
print(f">> Chosen binwidth = {BINWIDTH} (fine enough to show shape, not too noisy)")
print(f">> Mean price   = ${df['price'].mean():,.2f}")
print(f">> Median price = ${df['price'].median():,.2f}")
print(f">> Skewness     = {df['price'].skew():.3f}  (> 0 means right-skewed)")

hist_plot = (
    ggplot(df, aes(x="price"))
    + geom_histogram(binwidth=BINWIDTH, fill="#2c7fb8", color="white", boundary=0)
    + geom_vline(xintercept=df["price"].mean(), color="red",
                 linetype="dashed", size=1)
    + geom_vline(xintercept=df["price"].median(), color="darkgreen",
                 linetype="dotted", size=1)
    + labs(title="Distribution of Diamond Prices",
           subtitle=f"binwidth = {BINWIDTH};  red dashed = mean,  green dotted = median",
           x="Price (USD)", y="Number of diamonds")
    + theme_minimal()
    + theme(plot_title=element_text(size=14, weight="bold"))
)
save_gg(hist_plot, "1_univariate_histogram_price.png")

print(">> Interpretation: price is strongly right-skewed - most diamonds cost",
      "under $5,000,\n   with a long tail of expensive stones up to $18,823.")


# =====================================================================
# STEP 3 : Bivariate analysis - Stacked bar plot of cut vs clarity
# =====================================================================
step(3, "Bivariate: Stacked Bar Plot of 'cut' vs 'clarity' (relative %)")

print(">> Counts (crosstab of cut x clarity):")
print(pd.crosstab(df["cut"], df["clarity"]))

print("\n>> Relative distribution - row percentages (each cut sums to 100%):")
print((pd.crosstab(df["cut"], df["clarity"], normalize="index") * 100).round(1))

stacked_plot = (
    ggplot(df, aes(x="cut", fill="clarity"))
    + geom_bar(position=position_fill(), color="white", size=0.2)
    + scale_y_continuous(labels=lambda v: [f"{x:.0%}" for x in v])
    + scale_fill_brewer(type="div", palette="RdYlBu")
    + labs(title="Relative Distribution of Clarity within each Cut",
           x="Cut quality", y="Share of diamonds", fill="Clarity")
    + theme_bw()
    + theme(plot_title=element_text(size=14, weight="bold"))
)
save_gg(stacked_plot, "2_bivariate_stacked_bar_cut_clarity.png")

ideal_if = (pd.crosstab(df["cut"], df["clarity"], normalize="index") * 100).loc["Ideal", "IF"]
fair_if = (pd.crosstab(df["cut"], df["clarity"], normalize="index") * 100).loc["Fair", "IF"]
print(">> Interpretation: better cuts have a larger share of high-clarity stones -\n"
      f"   e.g. IF clarity is {ideal_if:.1f}% of Ideal cuts but only {fair_if:.1f}% of Fair cuts.")


# =====================================================================
# STEP 4 : Multivariate analysis - faceted scatter (4 variables)
# =====================================================================
step(4, "Multivariate: Faceted Scatter - carat vs price, color = clarity, facet = cut")

sample = df[df["carat"] <= 3].sample(n=5000, random_state=42)
print(">> Using a random sample of 5,000 diamonds up to 3 carat (random_state=42)\n"
      "   to avoid overplotting.")
print(">> Variables encoded: x = carat, y = price, color = clarity, facet = cut")

print("\n>> Correlation of carat and price within each cut:")
print(df.groupby("cut", observed=True)
        .apply(lambda g: g["carat"].corr(g["price"]), include_groups=False)
        .round(3).rename("corr(carat, price)"))

print("\n>> Interaction effect - median price of ~1 carat diamonds (0.95-1.05 ct)")
print("   by cut and clarity (holding carat fixed shows the clarity effect):")
one_ct = df[df["carat"].between(0.95, 1.05)]
print(one_ct.pivot_table(values="price", index="cut", columns="clarity",
                         aggfunc="median", observed=True).round(0))

facet_plot = (
    ggplot(sample, aes(x="carat", y="price", color="clarity"))
    + geom_point(alpha=0.6, size=1.2)
    + facet_wrap("~cut", ncol=3)
    + scale_color_cmap_d("viridis")
    + labs(title="Carat vs Price by Clarity, faceted by Cut",
           x="Carat", y="Price (USD)", color="Clarity")
    + theme_bw()
    + theme(plot_title=element_text(size=14, weight="bold"))
)
save_gg(facet_plot, "3_multivariate_faceted_scatter.png", width=10, height=6.5)

print(">> Interpretation: price rises steeply with carat in every cut; within the\n"
      "   same carat, higher clarity (yellow/green) stones sit above lower clarity\n"
      "   (purple) ones, so carat and clarity interact to drive price in every cut.")


# =====================================================================
# STEP 5 : Multivariate analysis - pair plot with hue
# =====================================================================
step(5, "Multivariate: Pair Plot of carat, depth, table, price (hue = cut)")

pair_vars = ["carat", "depth", "table", "price"]
print(">> Correlation matrix of the pair-plot variables:")
print(df[pair_vars].corr().round(3))

small = df.sample(n=1500, random_state=1)
g = sns.pairplot(small, vars=pair_vars, hue="cut", palette="viridis",
                 plot_kws={"alpha": 0.5, "s": 12}, corner=True, height=2.1)
g.figure.suptitle("Pair Plot of Diamond Features (hue = cut, n = 1,500 sample)",
                  y=1.02, fontsize=13, fontweight="bold")
path = os.path.join(OUT_DIR, "4_multivariate_pairplot.png")
g.savefig(path, dpi=100, bbox_inches="tight")
plt.close(g.figure)
print(">> Plot saved to:", path)

print(">> Interpretation: carat and price are strongly correlated (r = "
      f"{df['carat'].corr(df['price']):.3f});\n"
      "   depth and table show little relation to price.")


# =====================================================================
# Done
# =====================================================================
step(6, "Lab Assignment 3 completed")
print("Files generated:")
for f in sorted(os.listdir(OUT_DIR)):
    print("  -", os.path.join(OUT_DIR, f))
