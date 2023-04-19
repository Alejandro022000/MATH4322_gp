# Load necessary libraries
library(tidyverse)
library(caret)
library(randomForest)
library(rpart)
library(rpart.plot)
library(ipred)
library(gbm)
library(ggplot2)

# Read the dataset
url <- "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv"
white_wine <- read_delim(url, col_names = TRUE, delim = ";")

# Replace spaces in column names with underscores
colnames(white_wine) <- make.names(colnames(white_wine), unique = TRUE)

# Set seed for reproducibility
set.seed(123)

# Initialize variables to store MSE values
unpruned_mse_values <- numeric(10)
pruned_mse_values <- numeric(10)
bagging_mse_values <- numeric(10)
rf_mse_values <- numeric(10)
boosting_mse_values <- numeric(10)

# Perform train-test split and model evaluation 10 times
for (i in 1:10) {
  # Train/Test split
  train_idx <- createDataPartition(white_wine$quality, p = 0.8, list = FALSE)
  train_data <- white_wine[train_idx, ]
  test_data <- white_wine[-train_idx, ]
  
  # Unpruned decision tree
  unpruned_tree <- rpart(quality ~ ., data = train_data)
  unpruned_pred <- predict(unpruned_tree, test_data)
  unpruned_mse <- mean((test_data$quality - unpruned_pred)^2)
  unpruned_mse_values[i] <- unpruned_mse
  
  # Pruned decision tree
  control <- rpart.control(cp = 0.01) # Adjust cp value to control pruning
  pruned_tree <- rpart(quality ~ ., data = train_data, control = control)
  pruned_pred <- predict(pruned_tree, test_data)
  pruned_mse <- mean((test_data$quality - pruned_pred)^2)
  pruned_mse_values[i] <- pruned_mse
  
  # Bagging
  bagging_model <- bagging(quality ~ ., data = train_data, nbagg = 100)
  bagging_pred <- predict(bagging_model, test_data)
  bagging_mse <- mean((test_data$quality - bagging_pred)^2)
  bagging_mse_values[i] <- bagging_mse
  
  # Random Forest
  rf_model <- randomForest(quality ~ ., data = train_data)
  rf_pred <- predict(rf_model, test_data)
  rf_mse <- mean((test_data$quality - rf_pred)^2)
  rf_mse_values[i] <- rf_mse
  
  # Boosting
  boosting_model <- gbm(quality ~ ., data = train_data, distribution = "gaussian", n.trees = 100, shrinkage = 0.1, interaction.depth = 3)
  boosting_pred <- predict(boosting_model, test_data, n.trees = 100)
  boosting_mse <- mean((test_data$quality - boosting_pred)^2)
  boosting_mse_values[i] <- boosting_mse
}

# Create a table with the MSE values for all 10 iterations
mse_table <- data.frame(
  Iteration = 1:10,
  Unpruned = unpruned_mse_values,
  Pruned = pruned_mse_values,
  Bagging = bagging_mse_values,
  Random_Forest = rf_mse_values,
  Boosting = boosting_mse_values
)

# Print the table
print(mse_table)

# Reshape the mse_table into a long format
mse_table_long <- mse_table %>%
  pivot_longer(
    cols = -Iteration,
    names_to = "Model",
    values_to = "MSE"
  )

# Create a line plot of mse_table
mse_plot <- ggplot(mse_table_long, aes(x = Iteration, y = MSE, color = Model, group = Model)) +
  geom_line() +
  geom_point() +
  theme_minimal() +
  labs(
    title = "MSE",
    x = "Iteration",
    y = "MSE"
  )

# Display the plot
print(mse_plot)

# Train/Test split
train_idx <- createDataPartition(white_wine$quality, p = 0.8, list = FALSE)
train_data <- white_wine[train_idx, ]
test_data <- white_wine[-train_idx, ]


# Pruned decision tree
control <- rpart.control(cp = 0.01) # Adjust cp value to control pruning
pruned_tree <- rpart(quality ~ ., data = train_data, control = control)

# Random Forest
rf_model <- randomForest(quality ~ ., data = train_data)

# Create a decision tree plot for the pruned decision tree
rpart.plot(pruned_tree, main = "Pruned Decision Tree")


# Create a variable importance plot for the random forest model
varImpPlot(rf_model, main = "Variable Importance - Random Forest")
