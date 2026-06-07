#Course: CS 513
#First Name: Nishank
#Last Name: Shetty
#CWID: 20025517
#Project Name:

library(readr)
library(class)
library(ggplot2)
library(GGally)
library(caTools)
library(gmodels)
library(e1071)
library(naivebayes)
library(caret)



#Loading Dataset
songs <- read.csv("Projects/CS513/spotify_songs.csv", header = TRUE)

#Listing variables
print(names(songs))

feat <- songs[,c("loudness","speechiness","tempo","valence","acousticness")]
target<-songs[,c("energy")]

#Listing Features
print(names(feat))


data<- songs[,c("loudness","speechiness","tempo","valence","acousticness","energy")]
data
mean(data$energy)
sd(data$energy)


#Normalizing Data
col_norm<- c("loudness","speechiness","tempo","acousticness")

normalize<- function(x){
  (x - min(x, na.rm = TRUE)) / (max(x, na.rm = TRUE) - min(x, na.rm = TRUE))
}


data[col_norm]<- lapply(data[col_norm],normalize)


threshold<- 0.7

data$energy <- ifelse(data$energy > threshold, 1, 0)

data$energy <- as.factor(data$energy)


set.seed(123)  # Setting a seed for reproducibility
n <- nrow(data) # calculating rows in dataset
train_boundary <- sample(1:n, 0.7 * n) #spliting data into train & test 
train_boundary # for my reference
training_data <- data[train_boundary, ]
test_data <- data[-train_boundary, ]



#1.Support Vector Machine

svm_model <- svm(energy ~ loudness + speechiness + tempo + valence + acousticness, data = training_data)# fitting data into Support Vector Machine


# Making predictions 
predictions <- predict(svm_model, newdata = test_data)


# Confusion Matrix
conf_matrix <- table(predictions, test_data$energy)
print(conf_matrix)

# Accuracy
accuracy <- sum(diag(conf_matrix)) / sum(conf_matrix)
cat("Accuracy:", accuracy, "\n")

# Precision, Recall, F1 Score
precision <- conf_matrix[2, 2] / sum(conf_matrix[, 2])
recall <- conf_matrix[2, 2] / sum(conf_matrix[2, ])
f1 <- 2 * (precision * recall) / (precision + recall)
accuracy <- sum(diag(conf_matrix)) / sum(conf_matrix)
cat("Precision:", precision, "\n")
cat("Recall:", recall, "\n")
cat("F1 Score:", f1, "\n")

#2. Naive Bayes


nb_model <- naiveBayes(energy ~ loudness + speechiness + tempo + valence + acousticness, data = training_data) # fitting the data into the Naive Bayes Classification model 


# Make predictions
predictions <- predict(nb_model, test_data) 


# Confusion Matrix
conf_matrix <- confusionMatrix(predictions, test_data$energy)
print(conf_matrix)

# Accuracy
accuracy <- conf_matrix$overall["Accuracy"]
cat("Accuracy:", accuracy, "\n")

# Precision, Recall, F1 Score
precision <- conf_matrix$byClass["Precision"]
recall <- conf_matrix$byClass["Recall"]
f1 <- conf_matrix$byClass["F1"]
cat("Accuracy:", accuracy, "\n")
cat("Precision:", precision, "\n")
cat("Recall:", recall, "\n")
cat("F1 Score:", f1, "\n")


# Plot Confusion Matrix
conf_matrix_plot <- confusionMatrix(predictions, test_data$energy)
plot(conf_matrix_plot$table, col = conf_matrix_plot$byClass, 
     main = paste("Confusion Matrix\nAccuracy:", round(accuracy, 2)),
     sub = paste("Precision:", round(precision, 2), " Recall:", round(recall, 2), " F1 Score:", round(f1, 2)),
     cex.main = 1.2, cex.sub = 1.2, cex.axis = 1.2, cex.lab = 1.2)
confusionMatrix(predictions, test_data$energy)





rm(list = ls())
dev.off()

