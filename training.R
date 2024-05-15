# This is an example script to train your model given the (cleaned) input dataset.
# 
# This script will not be run on the holdout data, 
# but the resulting model model.joblib will be applied to the holdout data.
# 
# It is important to document your training steps here, including seed, 
# number of folds, model, et cetera

library(caret)
library(boot)
library(glmnet)
library(MLmetrics)
library(dplyr) 
library(tidyr)

train_save_model <- function(cleaned_df, outcome_df) {
  # Trains a model using the cleaned dataframe and saves the model to a file.

  # Parameters:
  # cleaned_df (dataframe): The cleaned data from clean_df function to be used for training the model.
  # outcome_df (dataframe): The data with the outcome variable (e.g., from PreFer_train_outcome.csv or PreFer_fake_outcome.csv).

  # Set the seed for reproducibility
  set.seed(08540)
  
  cleaned_df = cleaned_df %>% merge(outcome_df, by = "nomem_encr") %>% 
    drop_na(new_child) %>% 
    mutate(new_child = as.factor(new_child))
  
  # Create a vector with the class labels
  class_labels <- cleaned_df$new_child
  
  # Split the data into training and testing sets, keeping the same proportion of classes
  train_indices <- createDataPartition(class_labels, p = 0.80, list = FALSE)
  
  # Subset the data based on the training indices
  train_set <- cleaned_df[train_indices, ] %>% select("age", "has_partner", "religious2018", "net_household_income2020",
                     "urban2020", "new_child")
  test_set <- cleaned_df[-train_indices, ] %>% select("age", "has_partner", "religious2018", "net_household_income2020",
                     "urban2020", "new_child")
  
  # Oversample the minority class in the training set
  train_data_minority <- train_set %>% filter(new_child == "1")
  train_data_majority <- train_set %>% filter(new_child == "0")
  oversampled_minority <- upSample(x = train_data_minority %>% select(-new_child), 
                                   y = train_data_minority$new_child) %>% 
    rename("new_child" = "Class")
  
  # Combine oversampled minority class with majority class
  train_data_balanced <- rbind(train_data_majority, oversampled_minority)
  
  # Shuffle the rows to mix minority and majority class samples
  train_data_balanced <- train_data_balanced[sample(nrow(train_data_balanced)), ]
  
  # Separate predictors and target variable
  X_train <- train_data_balanced %>% select(-new_child)
  y_train <- train_data_balanced$new_child
  X_test <- test_set %>% select(-new_child)
  y_test <- test_set$new_child
  
  # Define the training control
  train_control <- trainControl(method = "cv", number = 10)
  
  # Train the logistic regression model with 10-fold cross-validation
  model <- train(new_child ~ age + has_partner + religious2018 + net_household_income2020 + urban2020, 
                 data = train_data_balanced, method = "glm", trControl = train_control, family = "binomial")
  
  # Print the model results
  print(model)
  
  # Access performance metrics
  summary(model)
  
  predicted_probs = predict(model, newdata = test_set, type = "prob")
  predicted_classes = ifelse(predicted_probs[2] > 0.25, "1", "0")
  true_labels = test_set$new_child
  
  # Calculate accuracy
  accuracy <- mean(predicted_classes == true_labels)
  cat("Accuracy:", accuracy, "\n")
  
  # Calculate F1 score
  f1_score <- F1_Score(y_pred = predicted_classes, y_true = true_labels)
  cat("F1 Score:", f1_score, "\n")
  
  # Calculate sensitivity
  sensitivity <- Sensitivity(y_pred = predicted_classes, y_true = true_labels)
  cat("Sensitivity:", sensitivity, "\n")
  
  # Calculate specificity
  specificity <- Specificity(y_pred = predicted_classes, y_true = true_labels)
  cat("Specificity:", specificity, "\n")

  saveRDS(model, file ="model.rds")
}
