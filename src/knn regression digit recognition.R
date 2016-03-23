#Using the standard data libraries as well as FNN for the KNN algorithm

library(readr)
library(caret)
library(FNN)

# Import the input data files
train <- read_csv("train.csv")
test <- read_csv("test.csv")

# Write to the log:
cat(sprintf("Training set has %d rows and %d columns\n", nrow(train), ncol(train)))
cat(sprintf("Test set has %d rows and %d columns\n", nrow(test), ncol(test)))

# For testing, can uncomment this to only check a sample of the data
# 
# trainIndex <- createDataPartition(train$label, p = .01,
#                                   list = FALSE,
#                                   times = 1)
# 
# Train <- train[ trainIndex,]
# Test  <- train[-trainIndex,]
#
# The full regression will just set Train <- train, Test <- test

cat("Get the count ", "\n")
print(table(Train$label))

cat("\n")
cat("Implementing the K-Nearest Neighbor Algorithm", "\n")

k_model <- FNN::knn(Train[,-1], Test[,-1], Train$label, k=10, algorithm = "cover_tree")

cat("\n")
cat("Evaluating accuracy of the KNN predictions: ","\n")

print(confusionMatrix(model.fnn,Test$label))












