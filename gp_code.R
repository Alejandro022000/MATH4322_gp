# Load necessary libraries
library(tidyverse)
library(caret)
library(randomForest)

# Read the dataset
url <- "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv"
white_wine <- read_delim(url, col_names = TRUE, delim = ";")

# Replace spaces in column names with underscores
colnames(white_wine) <- make.names(colnames(white_wine), unique = TRUE)

# Set seed for reproducibility
set.seed(123)

# Train/Test split
train_idx <- createDataPartition(white_wine$quality, p = 0.8, list = FALSE)
train_data <- white_wine[train_idx, ]
test_data <- white_wine[-train_idx, ]

# Linear Regression
lm_model <- lm(quality ~ ., data = train_data)
lm_summary <- summary(lm_model)

# Random Forest
rf_model <- randomForest(quality ~ ., data = train_data)

# Evaluate models on test data
lm_pred <- predict(lm_model, test_data)
lm_rmse <- sqrt(mean((test_data$quality - lm_pred)^2))

rf_pred <- predict(rf_model, test_data)
rf_rmse <- sqrt(mean((test_data$quality - rf_pred)^2))

# Variable Importance for Random Forest
var_importance <- importance(rf_model)
var_importance_plot <- varImpPlot(rf_model)


# Results and interpretation
cat("Linear Regression Model Summary:\n")
print(lm_summary)

cat("\nRandom Forest Model Summary:\n")
print(rf_model)

cat("\nLinear Regression RMSE:", lm_rmse)
cat("\nRandom Forest RMSE:", rf_rmse)

cat("\nVariable Importance Plot for Random Forest:\n")
print(var_importance_plot)



