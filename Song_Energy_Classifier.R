
#Importing the necessary libraries

library(caret)
library(class)

#Loading the dataset

rm(list=ls())

df <- read.csv('spotify_songs.csv')

df$energy <- ifelse(df$energy < 0.65, 0, ifelse(df$energy > 0.65, 1, 0))
df$energy <- as.factor(df$energy)

df

min_max_scale <- function(x) {
  (x - min(x)) / (max(x) - min(x))
}

# Select columns to normalize
columns_to_normalize <- c("loudness", "speechiness", "acousticness", "tempo")

normalized_data <- df

# Apply min-max scaling to selected columns
normalized_data[columns_to_normalize] <- lapply(normalized_data[columns_to_normalize], min_max_scale)

df <- normalized_data

df


#Splitting the dataset into training and testing partition

trainIndex <- createDataPartition(df$energy, p = 0.8, 
                                  list = FALSE, 
                                  times = 1)
data_train <- df[ trainIndex,]
data_test <- df[-trainIndex,]

predictors <- data_train [, c("loudness", "speechiness", "acousticness", "valence", "tempo")]
target <- data_train[, "energy"]

model <- train(
  x = predictors,
  y = target,
  method = "knn",
  trControl = trainControl(method = "cv"), # Cross-validation
  tuneGrid = expand.grid(k = 1:10), # Hyperparameter grid for 'k'
  metric = "Accuracy" # Evaluation metric
)

model$results

# Best model with tuned hyperparameters
best_k <- model$bestTune$k

best_k

best_model <- train(
  x = predictors,
  y = target,
  method = "knn",
  trControl = trainControl(method = "cv"),
  tuneGrid = data.frame(k = best_k),
  metric = "Accuracy"
)

knn_predictions <- predict(best_model, newdata = data_test)

#Building and printing the confusion matrix

knn_confusion_matrix <- table(knn_predictions, data_test$energy)

print(knn_confusion_matrix)

#Calculate Accuracy

knn_accuracy <- sum(diag(knn_confusion_matrix)) / sum(knn_confusion_matrix)

knn_true_positives <- knn_confusion_matrix[1, 1]
knn_false_positives <- knn_confusion_matrix[2, 1]
knn_false_negatives <- knn_confusion_matrix[1, 2]

# Calculate Precision
knn_precision <- knn_true_positives / (knn_true_positives + knn_false_positives)

# Calculate Recall
knn_recall <- knn_true_positives / (knn_true_positives + knn_false_negatives)

# Calculate F1 Score
knn_f1 <- 2 * (knn_precision * knn_recall) / (knn_precision + knn_recall)

#Results

print(knn_accuracy)
print(knn_precision)
print(knn_recall)
print(knn_f1)

install.packages("rpart")
install.packages("rpart.plot")

library(rpart)
library(rpart.plot)
library(caret)
library(e1071)

hyper_grid <- expand.grid(cp = seq(0.01, 0.1, by = 0.01))

cart_model <- train(x = predictors,
                    y = target,
                    method = "rpart",
                    trControl = trainControl(method = "cv",  # Cross-validation method
                                                     number = 5,    # Number of folds for cross-validation
                                                     verboseIter = TRUE),
                    tuneGrid = hyper_grid,
                    tuneLength = nrow(hyper_grid),
                    metric = "Accuracy")

cart_model$bestTune

cart_model$results

rpart.plot(cart_model$finalModel, uniform = TRUE, main = "CART Model")

cart_predictions <- predict(cart_model, newdata = data_test)

#Building and printing the confusion matrix

cart_confusion_matrix <- table(cart_predictions, data_test$energy)

print(cart_confusion_matrix)

#Calculate Accuracy

cart_accuracy <- sum(diag(cart_confusion_matrix)) / sum(cart_confusion_matrix)

cart_true_positives <- cart_confusion_matrix[1, 1]
cart_false_positives <- cart_confusion_matrix[2, 1]
cart_false_negatives <- cart_confusion_matrix[1, 2]

# Calculate Precision
cart_precision <- cart_true_positives / (cart_true_positives + cart_false_positives)

# Calculate Recall
cart_recall <- cart_true_positives / (cart_true_positives + cart_false_negatives)

# Calculate F1 Score
cart_f1 <- 2 * (cart_precision * cart_recall) / (cart_precision + cart_recall)

#Results

print(cart_accuracy)
print(cart_precision)
print(cart_recall)
print(cart_f1)
