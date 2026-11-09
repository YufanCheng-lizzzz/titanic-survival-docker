# Titanic Survival Prediction in R

# 1. Load the data
data_dir <- "/app/data"

if (!dir.exists(data_dir)) {
  data_dir <- "src/data"
}


train <- read.csv(
  file.path(data_dir, "train.csv"),
  na.strings = c("", "NA")
)

cat("Training data shape:\n")
print(dim(train))

cat("\nFirst five rows:\n")
print(head(train, 5))

cat("\nMissing values:\n")
print(colSums(is.na(train)))


# 2. Explore the data
cat("\nSurvival rate by sex:\n")
print(aggregate(Survived ~ Sex, data = train, mean))

cat("\nSurvival rate by passenger class:\n")
print(aggregate(Survived ~ Pclass, data = train, mean))


# 3. Clean the data
features <- c("Pclass", "Sex", "Age", "SibSp",
              "Parch", "Fare", "Embarked")

get_mode <- function(x) {
  names(sort(table(x), decreasing = TRUE))[1]
}

clean_data <- function(df, age_fill, fare_fill, embarked_fill) {
  df <- df[, features, drop = FALSE]

  df$Age[is.na(df$Age)] <- age_fill
  df$Fare[is.na(df$Fare)] <- fare_fill
  df$Embarked[is.na(df$Embarked)] <- embarked_fill

  df$Sex <- factor(df$Sex, levels = c("female", "male"))
  df$Embarked <- factor(df$Embarked, levels = c("C", "Q", "S"))

  return(df)
}

age_median <- median(train$Age, na.rm = TRUE)
fare_median <- median(train$Fare, na.rm = TRUE)
embarked_mode <- get_mode(train$Embarked)

X <- clean_data(train, age_median, fare_median, embarked_mode)

cat("\nData after cleaning:\n")
print(head(X, 5))

cat("\nMissing values after cleaning:\n")
print(colSums(is.na(X)))


# 4. Train logistic regression
model_formula <- Survived ~ Pclass + Sex + Age +
  SibSp + Parch + Fare + Embarked

X$Survived <- train$Survived

model <- glm(
  model_formula,
  data = X,
  family = binomial()
)

predict_binary <- function(model, data) {
  probabilities <- predict(model, newdata = data, type = "response")
  as.integer(probabilities >= 0.5)
}

train_predictions <- predict_binary(model, X)
train_accuracy <- mean(train_predictions == train$Survived)

cat("\nLogistic Regression Results:\n")
cat(sprintf("Training accuracy: %.4f\n", train_accuracy))


# 5. Evaluate on a held-out test set
set.seed(42)

train_idx <- sample(
  seq_len(nrow(train)),
  size = floor(0.8 * nrow(train))
)

train_split <- train[train_idx, ]
val_split <- train[-train_idx, ]

# Use training split statistics to prevent data leakage
split_age <- median(train_split$Age, na.rm = TRUE)
split_fare <- median(train_split$Fare, na.rm = TRUE)
split_embarked <- get_mode(train_split$Embarked)

X_train <- clean_data(
  train_split, split_age, split_fare, split_embarked
)

X_val <- clean_data(
  val_split, split_age, split_fare, split_embarked
)

X_train$Survived <- train_split$Survived

validation_model <- glm(
  model_formula,
  data = X_train,
  family = binomial()
)

train_pred <- predict_binary(validation_model, X_train)
val_pred <- predict_binary(validation_model, X_val)

cat("\nModel Validation Results:\n")
cat(sprintf(
  "Training accuracy: %.4f\n",
  mean(train_pred == train_split$Survived)
))
cat(sprintf(
  "Held-out test accuracy (20%% of train.csv): %.4f\n",
  mean(val_pred == val_split$Survived)
))


# 6. Predict on Kaggle test data
test <- read.csv(
  file.path(data_dir, "test.csv"),
  na.strings = c("", "NA")
)

cat("\nTest data shape:\n")
print(dim(test))

cat("\nMissing values in test data:\n")
print(colSums(is.na(test)))

X_test <- clean_data(
  test, age_median, fare_median, embarked_mode
)

cat("\nMissing values after test data cleaning:\n")
print(colSums(is.na(X_test)))

test_predictions <- predict_binary(model, X_test)

results <- data.frame(
  PassengerId = test$PassengerId,
  Survived = test_predictions
)

cat("\nFirst 10 test predictions:\n")
print(head(results, 10))

cat("\nPredicted survival counts:\n")
print(table(results$Survived))

cat("\nKaggle test accuracy cannot be calculated.\n")
cat("The Kaggle test.csv does not contain survival labels.\n")
cat("Model performance is evaluated using a held-out 20% split of train.csv.\n")
