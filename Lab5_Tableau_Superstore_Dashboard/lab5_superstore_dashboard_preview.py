# =====================================================================
# Lab Assignment 5: Interactive Visual Analytics using Tableau
#                   -- Python companion script --
# ---------------------------------------------------------------------
# The dashboard itself is built in Tableau (see the PDF for the steps).
# This script inspects the same "Sample - Superstore.csv" file and
# computes every number each Tableau worksheet should show, so the
# Tableau results can be checked. It also draws preview images of the
# worksheets and of the dashboard before / after the Region filter.
#
# Dataset   : Sample - Superstore.csv (9,994 order lines, 2015-2018)
# Run with  : python lab5_superstore_dashboard_preview.py
# Requires  : pip install pandas matplotlib
# Output    : Step-by-step console log + PNG previews in ./plots/
# =====================================================================

import os
from datetime import datetime

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd


# ---------- Helper: print a clearly visible step header --------------
def step(n, title):
    print("\n" + "=" * 70)
    print(f"STEP {n} : {title}")
    print("=" * 70)


OUT_DIR = "plots"
os.makedirs(OUT_DIR, exist_ok=True)


def save_plot(fig, file):
    path = os.path.join(OUT_DIR, file)
    fig.savefig(path, dpi=110, bbox_inches="tight")
    print(">> Plot saved to:", path)
    if matplotlib.get_backend().lower() != "agg":
        plt.show()
    plt.close(fig)


money = mticker.FuncFormatter(lambda v, _: f"{'-' if v < 0 else ''}${abs(v) / 1000:,.0f}K")
REGION_COLORS = {"Central": "#4e79a7", "East": "#f28e2b", "South": "#e15759", "West": "#76b7b2"}
SEGMENT_COLORS = {"Consumer": "#4e79a7", "Corporate": "#f28e2b", "Home Office": "#59a14f"}

pd.set_option("display.width", 110)
pd.set_option("display.max_columns", None)
print("Lab Assignment 5 - Tableau Superstore Dashboard (Python companion)")
print("Python libraries : pandas", pd.__version__, "| matplotlib", matplotlib.__version__)
print("Run on           :", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))


# =====================================================================
# STEP 1 : Import the dataset and inspect data types
# =====================================================================
step(1, "Import 'Sample - Superstore.csv' and inspect data types")

df = pd.read_csv("Sample - Superstore.csv", parse_dates=["Order Date", "Ship Date"])
print(">> Dataset loaded successfully.")
print(">> Shape (rows x columns):", df.shape)

# How Tableau will classify each column on its Data Source page
def tableau_type(col):
    if col in ("Country", "State", "City"):
        return "Abc  String (geographic role)", "Dimension"
    if col == "Postal Code":
        return "Abc  String (set geo role: ZIP Code)", "Dimension"
    if pd.api.types.is_datetime64_any_dtype(df[col]):
        return "Date", "Dimension"
    if col == "Row ID":
        return "#    Number (whole)", "Dimension"
    if col == "Quantity":
        return "#    Number (whole)", "Measure"
    if pd.api.types.is_numeric_dtype(df[col]):
        return "#    Number (decimal)", "Measure"
    return "Abc  String", "Dimension"

types = pd.DataFrame({
    "pandas dtype": df.dtypes.astype(str),
    "Tableau data type": [tableau_type(c)[0] for c in df.columns],
    "Tableau role": [tableau_type(c)[1] for c in df.columns],
    "unique values": df.nunique(),
})
print("\n>> Column data types (pandas vs Tableau):")
print(types)

print("\n>> Missing values (non-zero only):")
print(df.isnull().sum()[lambda s: s > 0])
print(f"\n>> Order Date range : {df['Order Date'].min():%d-%b-%Y} to "
      f"{df['Order Date'].max():%d-%b-%Y}")
print(f">> Total Sales      : ${df['Sales'].sum():,.2f}")
print(f">> Total Profit     : ${df['Profit'].sum():,.2f}")
print(f">> Profit ratio     : {df['Profit'].sum() / df['Sales'].sum():.2%}")


# =====================================================================
# STEP 2 : Worksheet 1 - Bar chart of Sales and Profit by Region
# =====================================================================
step(2, "Worksheet 1 - Bar chart: total Sales and Profit by Region")

region = (df.groupby("Region")[["Sales", "Profit"]].sum()
            .sort_values("Sales", ascending=False))
region["Profit Ratio %"] = region["Profit"] / region["Sales"] * 100
print(">> SUM(Sales) and SUM(Profit) per Region (what Tableau should show):")
print(region.round(2))


def draw_region_bars(ax, data, selected=None):
    # selected = region clicked on the dashboard; other bars are faded
    x = range(len(data))
    alpha = [1.0 if selected in (None, r) else 0.25 for r in data.index]
    for i, r in enumerate(data.index):
        ax.bar(i - 0.2, data.loc[r, "Sales"], width=0.4, label="Sales" if i == 0 else None,
               color=REGION_COLORS[r], edgecolor="black", alpha=alpha[i])
        ax.bar(i + 0.2, data.loc[r, "Profit"], width=0.4, label="Profit" if i == 0 else None,
               color="#bab0ac", edgecolor="black", hatch="//", alpha=alpha[i])
    for i, (s, p) in enumerate(zip(data["Sales"], data["Profit"])):
        ax.text(i - 0.2, s, f"${s / 1000:,.0f}K", ha="center", va="bottom", fontsize=8)
        ax.text(i + 0.2, p, f"${p / 1000:,.0f}K", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(list(x))
    ax.set_xticklabels(data.index)
    ax.yaxis.set_major_formatter(money)
    ax.set_title("Sales & Profit by Region", fontweight="bold")
    ax.legend(loc="upper right", fontsize=8)


fig, ax = plt.subplots(figsize=(8, 4.8))
draw_region_bars(ax, region)
save_plot(fig, "1_sheet_region_bar.png")
print(f">> Interpretation: {region.index[0]} has the highest sales and profit; "
      f"{region['Profit Ratio %'].idxmin()} has the lowest profit ratio "
      f"({region['Profit Ratio %'].min():.1f}%).")


# =====================================================================
# STEP 3 : Worksheet 2 - Line chart of monthly Sales (time series)
# =====================================================================
step(3, "Worksheet 2 - Line chart: Sales by month and year")

yearly = df.groupby(df["Order Date"].dt.year)[["Sales", "Profit"]].sum()
yearly.index.name = "Year"
yearly["Sales growth %"] = yearly["Sales"].pct_change() * 100
print(">> SUM(Sales) and SUM(Profit) per YEAR(Order Date):")
print(yearly.round(2))

monthly = df.set_index("Order Date")["Sales"].resample("MS").sum()
print("\n>> Continuous MONTH(Order Date) series: 48 months; peaks and lows:")
print(f"   Highest month : {monthly.idxmax():%b %Y} (${monthly.max():,.2f})")
print(f"   Lowest month  : {monthly.idxmin():%b %Y} (${monthly.min():,.2f})")

by_month = (df.assign(Year=df["Order Date"].dt.year, Month=df["Order Date"].dt.month)
              .pivot_table(values="Sales", index="Month", columns="Year", aggfunc="sum"))
print("\n>> Sales by month (rows) and year (columns), in USD:")
print(by_month.round(0))


def draw_monthly_line(ax, series):
    ax.plot(series.index, series.values, color="#4e79a7", marker="o", markersize=3)
    ax.yaxis.set_major_formatter(money)
    ax.set_title("Monthly Sales Trend (2015-2018)", fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.4)


fig, ax = plt.subplots(figsize=(9, 4.5))
draw_monthly_line(ax, monthly)
ax.set_xlabel("Month of Order Date")
ax.set_ylabel("Sales")
save_plot(fig, "2_sheet_monthly_sales_line.png")
print(">> Interpretation: sales dipped 2.8% in 2016, then grew strongly in 2017 and 2018.\n"
      "   Clear seasonal pattern: low in Jan/Feb, highest in Sep, Nov and Dec.")


# =====================================================================
# STEP 4 : Worksheet 3 - Map (Sales by State) and segment scatter
# =====================================================================
step(4, "Worksheet 3 - Geographic (State map) and Segment (scatter) breakdowns")

state = df.groupby("State")[["Sales", "Profit"]].sum().sort_values("Sales", ascending=False)
print(f">> Filled map values: {len(state)} states. Top 10 states by SUM(Sales):")
print(state.head(10).round(2))
print("\n>> 5 states with the largest losses (shown red if Profit is on Color):")
print(state.sort_values("Profit").head(5).round(2))

segment = df.groupby("Segment")[["Sales", "Profit"]].sum()
segment["Customers"] = df.groupby("Segment")["Customer ID"].nunique()
print("\n>> Segment breakdown:")
print(segment.round(2))

customers = df.groupby(["Customer Name", "Segment"])[["Sales", "Profit"]].sum().reset_index()
print(f"\n>> Scatter plot: one mark per customer ({len(customers)} marks),"
      " x = SUM(Sales), y = SUM(Profit), color = Segment")

fig, ax = plt.subplots(figsize=(8, 4.8))
top10 = state.head(10)[::-1]
ax.barh(top10.index, top10["Sales"],
        color=["#59a14f" if p > 0 else "#e15759" for p in top10["Profit"]], edgecolor="black")
ax.xaxis.set_major_formatter(money)
ax.set_title("Top 10 States by Sales (values shown on the Tableau map)\n"
             "green = profitable, red = loss-making", fontweight="bold")
save_plot(fig, "3_sheet_state_sales_top10.png")


def draw_segment_scatter(ax, data):
    for seg, g in data.groupby("Segment"):
        ax.scatter(g["Sales"], g["Profit"], s=14, alpha=0.65,
                   color=SEGMENT_COLORS[seg], label=seg)
    ax.axhline(0, color="grey", linewidth=0.8)
    ax.xaxis.set_major_formatter(money)
    ax.yaxis.set_major_formatter(money)
    ax.set_xlabel("Sales per customer")
    ax.set_ylabel("Profit per customer")
    ax.set_title("Customer Sales vs Profit by Segment", fontweight="bold")
    ax.legend(fontsize=8, title="Segment", title_fontsize=8)


fig, ax = plt.subplots(figsize=(8, 5))
draw_segment_scatter(ax, customers)
save_plot(fig, "4_sheet_segment_scatter.png")
print(f">> Interpretation: {state.index[0]} and {state.index[1]} lead sales; "
      f"{state['Profit'].idxmin()} loses the most money.")


# =====================================================================
# STEP 5 : Dashboard preview and the Region filter action
# =====================================================================
step(5, "Dashboard preview - all regions vs Region = 'West' selected")


def draw_dashboard(data, title, file, selected=None):
    # The source sheet (Region bars) keeps all regions; only the targets are filtered
    fig = plt.figure(figsize=(13, 8))
    grid = fig.add_gridspec(2, 2, height_ratios=[1, 1.05], hspace=0.35, wspace=0.22)
    draw_region_bars(fig.add_subplot(grid[0, 0]), region, selected)
    draw_segment_scatter(fig.add_subplot(grid[0, 1]),
                         data.groupby(["Customer Name", "Segment"])[["Sales", "Profit"]]
                             .sum().reset_index())
    ax = fig.add_subplot(grid[1, :])
    draw_monthly_line(ax, data.set_index("Order Date")["Sales"].resample("MS").sum())
    fig.suptitle(title, fontsize=15, fontweight="bold")
    save_plot(fig, file)


draw_dashboard(df, "Superstore Sales Dashboard - all regions", "5_dashboard_all_regions.png")

west = df[df["Region"] == "West"]
print("\n>> After clicking the 'West' bar (Filter action), the other sheets show:")
print(f"   Order lines       : {len(west):,}")
print(f"   Total Sales       : ${west['Sales'].sum():,.2f}")
print(f"   Total Profit      : ${west['Profit'].sum():,.2f}")
print(f"   Customers in West : {west['Customer ID'].nunique()}")
print(f"   States in West    : {west['State'].nunique()}")
draw_dashboard(west, "Superstore Sales Dashboard - filtered by clicking Region = West",
               "6_dashboard_filtered_west.png", selected="West")


# =====================================================================
# Done
# =====================================================================
step(6, "Lab Assignment 5 companion script completed")
print("Files generated:")
for f in sorted(os.listdir(OUT_DIR)):
    print("  -", os.path.join(OUT_DIR, f))
