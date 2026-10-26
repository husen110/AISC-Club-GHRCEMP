# =====================================================================
# Lab Assignment 2: Python Data Visualization with Matplotlib, Pandas,
#                   and Seaborn
# ---------------------------------------------------------------------
# Objective : Hands-on experience using core Python visual libraries
#             for univariate and bivariate analysis.
# Dataset   : Seaborn's built-in "tips" dataset
# Run with  : python lab2_python_data_visualization.py
# Requires  : pip install matplotlib pandas seaborn
# Output    : Step-by-step console log + 3 PNG plots in ./plots/
# =====================================================================

import os
from datetime import datetime

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


# ---------- Helper: print a clearly visible step header --------------
def step(n, title):
    print("\n" + "=" * 70)
    print(f"STEP {n} : {title}")
    print("=" * 70)


# Save each figure as PNG; show it on screen too when a display exists
OUT_DIR = "plots"
os.makedirs(OUT_DIR, exist_ok=True)


def save_plot(fig, file):
    path = os.path.join(OUT_DIR, file)
    fig.savefig(path, dpi=110, bbox_inches="tight")
    print(">> Plot saved to:", path)
    if matplotlib.get_backend().lower() != "agg":
        plt.show()
    plt.close(fig)


pd.set_option("display.width", 100)
print("Lab Assignment 2 - Python Data Visualization (Matplotlib, Pandas, Seaborn)")
print("Python libraries : matplotlib", matplotlib.__version__,
      "| pandas", pd.__version__, "| seaborn", sns.__version__)
print("Run on           :", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))


# =====================================================================
# STEP 1 : Load the dataset
# =====================================================================
step(1, "Load the 'tips' dataset from Seaborn")

tips = sns.load_dataset("tips")
print(">> Dataset loaded successfully.")
print(">> Shape (rows x columns):", tips.shape)

print("\n>> Column data types:")
print(tips.dtypes)

print("\n>> First 5 rows:")
print(tips.head())

print("\n>> Missing values per column:")
print(tips.isnull().sum())

print("\n>> Summary of continuous variables:")
print(tips[["total_bill", "tip", "size"]].describe().round(2))


# =====================================================================
# STEP 2 : Matplotlib - Scatter plot of total_bill vs tip
# =====================================================================
step(2, "Matplotlib Scatter Plot: total_bill vs tip (colored by sex)")

r = tips["total_bill"].corr(tips["tip"])
print(f">> Pearson correlation between total_bill and tip: r = {r:.4f}")

# Customization: a different color and marker style for each group
styles = {"Male":   {"color": "royalblue", "marker": "o"},
          "Female": {"color": "crimson",   "marker": "^"}}
print(">> Plot customization used:")
for sex, st in styles.items():
    n = (tips["sex"] == sex).sum()
    print(f"   {sex:<6} -> color = {st['color']:<9}, marker = '{st['marker']}', points = {n}")

fig, ax = plt.subplots(figsize=(8, 5.5))
for sex, st in styles.items():
    subset = tips[tips["sex"] == sex]
    ax.scatter(subset["total_bill"], subset["tip"],
               c=st["color"], marker=st["marker"], s=45,
               alpha=0.7, edgecolors="white", linewidths=0.5,
               label=f"{sex} (n={len(subset)})")
ax.set_title("Total Bill vs Tip Amount (Tips dataset)", fontsize=13, fontweight="bold")
ax.set_xlabel("Total Bill (USD)", fontsize=11)
ax.set_ylabel("Tip (USD)", fontsize=11)
ax.grid(True, linestyle="--", alpha=0.4)
ax.legend(title="Customer Sex", loc="upper left")
save_plot(fig, "1_matplotlib_scatter_bill_vs_tip.png")

print(">> Interpretation: positive relationship - larger bills generally receive",
      "larger tips.")


# =====================================================================
# STEP 3 : Pandas - Group by a categorical variable + horizontal bars
# =====================================================================
step(3, "Pandas groupby('day') + Horizontal Bar Chart (.plot(kind='barh'))")

grouped = tips.groupby("day", observed=True).agg(
    num_bills=("total_bill", "count"),
    avg_total_bill=("total_bill", "mean"),
    avg_tip=("tip", "mean"),
)
print(">> Code: tips.groupby('day').agg(count, mean total_bill, mean tip)")
print(">> Grouped result:")
print(grouped.round(2))

avg_tip_by_day = grouped["avg_tip"]
print("\n>> Series plotted (average tip per day):")
print(avg_tip_by_day.round(2))

ax = avg_tip_by_day.plot(kind="barh",
                         color=["#4c72b0", "#55a868", "#c44e52", "#8172b2"],
                         edgecolor="black", figsize=(8, 5))
ax.set_title("Average Tip by Day of the Week", fontsize=13, fontweight="bold")
ax.set_xlabel("Average Tip (USD)", fontsize=11)
ax.set_ylabel("Day", fontsize=11)
for i, v in enumerate(avg_tip_by_day):
    ax.text(v + 0.03, i, f"${v:.2f}", va="center")
ax.set_xlim(0, avg_tip_by_day.max() * 1.15)
save_plot(ax.get_figure(), "2_pandas_barh_avg_tip_by_day.png")

print(f">> Interpretation: highest average tip on {avg_tip_by_day.idxmax()} "
      f"(${avg_tip_by_day.max():.2f}), lowest on {avg_tip_by_day.idxmin()} "
      f"(${avg_tip_by_day.min():.2f}).")


# =====================================================================
# STEP 4 : Seaborn - Boxplot of total_bill by day with hue = sex
# =====================================================================
step(4, "Seaborn Boxplot: total_bill by day, hue = sex (3rd dimension)")

print(">> Median total_bill for each day and sex:")
print(tips.pivot_table(values="total_bill", index="day", columns="sex",
                       aggfunc="median", observed=True).round(2))

# Outliers using the same 1.5 * IQR rule the boxplot whiskers use
print("\n>> Outliers detected (beyond 1.5 * IQR) in each day/sex group:")
found = False
for (day, sex), g in tips.groupby(["day", "sex"], observed=True):
    q1, q3 = g["total_bill"].quantile([0.25, 0.75])
    iqr = q3 - q1
    out = g[(g["total_bill"] < q1 - 1.5 * iqr) | (g["total_bill"] > q3 + 1.5 * iqr)]
    if len(out):
        found = True
        print(f"   {day:<4} {sex:<6}: {sorted(out['total_bill'].round(2).tolist())}")
if not found:
    print("   No outliers found.")

fig, ax = plt.subplots(figsize=(9, 5.5))
sns.boxplot(data=tips, x="day", y="total_bill", hue="sex",
            palette={"Male": "#6fa8dc", "Female": "#ea9999"},
            flierprops={"marker": "o", "markerfacecolor": "red", "markersize": 5},
            ax=ax)
ax.set_title("Total Bill by Day, split by Sex", fontsize=13, fontweight="bold")
ax.set_xlabel("Day of the Week", fontsize=11)
ax.set_ylabel("Total Bill (USD)", fontsize=11)
ax.legend(title="Sex")
save_plot(fig, "3_seaborn_boxplot_bill_by_day_sex.png")

print(">> Interpretation: weekend (Sat/Sun) bills are higher and more spread out;\n"
      "   male customers have higher median bills on 3 of the 4 days.")


# =====================================================================
# Done
# =====================================================================
step(5, "Lab Assignment 2 completed")
print("Files generated:")
for f in sorted(os.listdir(OUT_DIR)):
    print("  -", os.path.join(OUT_DIR, f))
