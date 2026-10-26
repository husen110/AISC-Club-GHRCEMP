# =====================================================================
# Lab Assignment 1: Basic Statistics & Visualization in R
# ---------------------------------------------------------------------
# Objective : Calculate descriptive statistics and generate
#             foundational graphics in R.
# Dataset   : Built-in R dataset `mtcars`
# Run with  : Rscript lab1_basic_stats_visualization.R
#             (or source() it inside RStudio)
# Output    : Step-by-step console log + 3 PNG plots in ./plots/
# =====================================================================

# ---------- Helper: print a clearly visible step header --------------
step <- function(n, title) {
  cat("\n", strrep("=", 70), "\n", sep = "")
  cat(sprintf("STEP %s : %s\n", n, title))
  cat(strrep("=", 70), "\n", sep = "")
}

# Save each plot as PNG; when running interactively also draw on screen
out_dir <- "plots"
if (!dir.exists(out_dir)) dir.create(out_dir)

save_plot <- function(file, draw) {
  path <- file.path(out_dir, file)
  png(path, width = 900, height = 650, res = 110)
  draw()
  dev.off()
  if (interactive()) draw()
  cat(">> Plot saved to:", path, "\n")
}

cat("Lab Assignment 1 - Basic Statistics & Visualization in R\n")
cat("R version :", R.version.string, "\n")
cat("Run on    :", format(Sys.time(), "%Y-%m-%d %H:%M:%S"), "\n")


# =====================================================================
# STEP 1 : Load the dataset
# =====================================================================
step(1, "Load the built-in 'mtcars' dataset")

data(mtcars)
cat(">> Dataset loaded successfully.\n")
cat(">> Dimensions (rows x columns):", nrow(mtcars), "x", ncol(mtcars), "\n\n")

cat(">> Structure of the dataset:\n")
str(mtcars)

cat("\n>> First 6 rows:\n")
print(head(mtcars))

cat("\n>> Missing values per column:\n")
print(colSums(is.na(mtcars)))


# =====================================================================
# STEP 2 : Summary statistics for key numerical attributes
# =====================================================================
step(2, "Summary statistics (Mean, Median, SD, Variance, IQR)")

key_vars <- c("mpg", "hp", "wt", "disp", "qsec")
cat(">> Key numerical attributes selected:", paste(key_vars, collapse = ", "), "\n")
cat("   mpg  = Miles per gallon\n")
cat("   hp   = Gross horsepower\n")
cat("   wt   = Weight (1000 lbs)\n")
cat("   disp = Displacement (cu. in.)\n")
cat("   qsec = 1/4 mile time (seconds)\n")

cat("\n>> 2a. Built-in summary() output:\n")
print(summary(mtcars[, key_vars]))

cat("\n>> 2b. Calculating each statistic individually (example: mpg):\n")
cat("   mean(mtcars$mpg)   =", round(mean(mtcars$mpg), 3), "\n")
cat("   median(mtcars$mpg) =", round(median(mtcars$mpg), 3), "\n")
cat("   sd(mtcars$mpg)     =", round(sd(mtcars$mpg), 3), "\n")
cat("   var(mtcars$mpg)    =", round(var(mtcars$mpg), 3), "\n")
cat("   IQR(mtcars$mpg)    =", round(IQR(mtcars$mpg), 3), "\n")

cat("\n>> 2c. Descriptive statistics table for all key attributes:\n")
stats_table <- data.frame(
  Mean     = sapply(mtcars[, key_vars], mean),
  Median   = sapply(mtcars[, key_vars], median),
  Std_Dev  = sapply(mtcars[, key_vars], sd),
  Variance = sapply(mtcars[, key_vars], var),
  IQR      = sapply(mtcars[, key_vars], IQR),
  Min      = sapply(mtcars[, key_vars], min),
  Max      = sapply(mtcars[, key_vars], max)
)
print(round(stats_table, 3))

write.csv(round(stats_table, 3), "summary_statistics.csv")
cat("\n>> Statistics table saved to: summary_statistics.csv\n")


# =====================================================================
# STEP 3 : Histogram - distribution of a single continuous variable
# =====================================================================
step(3, "Histogram of Miles Per Gallon (mpg)")

h <- hist(mtcars$mpg, breaks = 8, plot = FALSE)
cat(">> Histogram bins (class intervals) and frequencies:\n")
print(data.frame(
  Bin       = paste0("(", head(h$breaks, -1), ", ", h$breaks[-1], "]"),
  Frequency = h$counts
))

save_plot("1_histogram_mpg.png", function() {
  hist(mtcars$mpg,
       breaks = 8,
       col    = "skyblue",
       border = "white",
       main   = "Histogram of Miles Per Gallon (mtcars)",
       xlab   = "Miles Per Gallon (mpg)",
       ylab   = "Frequency")
  abline(v = mean(mtcars$mpg),   col = "red",       lwd = 2, lty = 2)
  abline(v = median(mtcars$mpg), col = "darkgreen", lwd = 2, lty = 3)
  legend("topright",
         legend = c(sprintf("Mean = %.2f",   mean(mtcars$mpg)),
                    sprintf("Median = %.2f", median(mtcars$mpg))),
         col = c("red", "darkgreen"), lty = c(2, 3), lwd = 2, bty = "n")
})

cat(">> Interpretation: mean (", round(mean(mtcars$mpg), 2),
    ") > median (", median(mtcars$mpg),
    "), so the mpg distribution is slightly right-skewed.\n", sep = "")


# =====================================================================
# STEP 4 : Boxplot - identify potential outliers across groups
# =====================================================================
step(4, "Boxplot of Horsepower (hp) grouped by Number of Cylinders (cyl)")

cat(">> Number of cars in each cylinder group:\n")
print(table(cyl = mtcars$cyl))

cat("\n>> Five-number summary of hp for each cylinder group:\n")
print(tapply(mtcars$hp, mtcars$cyl, fivenum))

b <- boxplot(hp ~ cyl, data = mtcars, plot = FALSE)

cat("\n>> Outliers detected (values beyond 1.5 * IQR from the box):\n")
if (length(b$out) > 0) {
  out_cyl  <- as.numeric(b$names[b$group])
  out_rows <- mtcars[paste(mtcars$cyl, mtcars$hp) %in% paste(out_cyl, b$out), ]
  print(out_rows[, c("cyl", "hp", "mpg", "wt")])
} else {
  cat("   No outliers found.\n")
}

save_plot("2_boxplot_hp_by_cyl.png", function() {
  boxplot(hp ~ cyl, data = mtcars,
          col  = c("lightgreen", "khaki", "salmon"),
          main = "Horsepower by Number of Cylinders (mtcars)",
          xlab = "Number of Cylinders",
          ylab = "Gross Horsepower (hp)",
          outpch = 19, outcol = "red", outcex = 1.3)
  if (length(b$out) > 0) {
    text(b$group, b$out, labels = rownames(out_rows),
         pos = 4, cex = 0.8, col = "red")
  }
})


# =====================================================================
# STEP 5 : Scatter plot with trend line
# =====================================================================
step(5, "Scatter Plot of Weight (wt) vs Miles Per Gallon (mpg) + Trend Line")

r <- cor(mtcars$wt, mtcars$mpg)
cat(">> Pearson correlation between wt and mpg: r =", round(r, 4), "\n")

model <- lm(mpg ~ wt, data = mtcars)
cat("\n>> Fitting linear regression model: lm(mpg ~ wt)\n")
print(summary(model))

coefs <- coef(model)
cat(sprintf(">> Trend line equation: mpg = %.3f + (%.3f) * wt\n", coefs[1], coefs[2]))
cat(sprintf(">> R-squared = %.4f\n", summary(model)$r.squared))

save_plot("3_scatter_wt_vs_mpg.png", function() {
  plot(mtcars$wt, mtcars$mpg,
       pch  = 19,
       col  = "steelblue",
       main = "Car Weight vs Fuel Efficiency (mtcars)",
       xlab = "Weight (1000 lbs)",
       ylab = "Miles Per Gallon (mpg)")
  abline(model, col = "red", lwd = 2)
  legend("topright",
         legend = c("Observed cars",
                    sprintf("Trend: mpg = %.2f %.2f*wt", coefs[1], coefs[2]),
                    sprintf("r = %.3f,  R^2 = %.3f", r, summary(model)$r.squared)),
         col = c("steelblue", "red", NA), pch = c(19, NA, NA),
         lty = c(NA, 1, NA), lwd = 2, bty = "n")
})

cat(">> Interpretation: strong negative relationship - heavier cars give",
    "lower mpg.\n   Each extra 1000 lbs reduces mpg by about",
    round(abs(coefs[2]), 2), "miles per gallon.\n")


# =====================================================================
# Done
# =====================================================================
step("6", "Lab Assignment 1 completed")
cat("Files generated:\n")
cat("  - summary_statistics.csv\n")
cat(paste0("  - ", file.path(out_dir, list.files(out_dir))), sep = "\n")
