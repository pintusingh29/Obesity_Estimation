library(tidyverse)
library(ggplot2)
library(randomForest)
library(psych)
library(caret)      
library(dplyr)
library(nnet)       
library(corrplot)  
library(RColorBrewer)  
library(ggthemes)      
library(pheatmap)      
library(GGally)        
library(pROC)          
library(doParallel)


data <- read.csv("Est_ObesityDataSet.csv", stringsAsFactors = TRUE) 

str(data)
summary(data)
head(data)
describe(data)

cat("Total Missing Values:", sum(is.na(data)), "\n")


data <- data %>% 
  mutate(across(where(is.numeric), ~ifelse(is.na(.), median(., na.rm = TRUE), .)))
         

data$Gender <- as.factor(data$Gender)
data$Obesity_Level <- as.factor(data$Obesity_Level)  
   

data <- data %>%
mutate(BMI = Weight / (Height^2))

data$BMI <- as.numeric(as.character(data$BMI))

data %>%
  group_by(Gender) %>%
  summarise(
    Mean_Age = mean(Age, na.rm = TRUE),
    Median_Weight = median(Weight, na.rm = TRUE),
    SD_Height = sd(Height, na.rm = TRUE)
  )

ggplot(data, aes(x = Age)) +
  geom_histogram(binwidth = 5, fill = "blue", color = "black") +
  labs(title = "Distribution of Age", x = "Age", y = "Count")

ggplot(data, aes(x = Gender, y = Weight, fill = Gender)) +
  geom_boxplot() +
  labs(title = "Weight Distribution by Gender", x = "Gender", y = "Weight")

cl <- makePSOCKcluster(4)  
registerDoParallel(cl)
     

ggplot(data, aes(x = Obesity_Level, fill = Obesity_Level)) +
geom_bar() +
scale_fill_brewer(palette = "Set3") +
theme_minimal() +
labs(title = "Distribution of Obesity Levels",
x = "Obesity Level",
y = "Count") +
     theme(legend.position = "none", 
     plot.title = element_text(size = 16, face = "bold"))
         

numeric_data <- data %>% 
select(where(is.numeric))
cor_matrix <- cor(numeric_data, use = "complete.obs")
print(cor_matrix)

corrplot(cor_matrix, method = "color", col = brewer.pal(n = 9, name = "Blues"),
         addCoef.col = "black", tl.col = "black", tl.srt = 45, number.cex = 0.7)
         

set.seed(123)  
trainIndex <- createDataPartition(data$Obesity_Level, p = 0.8, list = FALSE)
trainData <- data[trainIndex, ]
testData <- data[-trainIndex, ]
         

use_upsampling <- TRUE  # Set to FALSE to use down-sampling
         
if (use_upsampling) {
    trainData_balanced <- upSample(x = trainData[-ncol(trainData)], y = trainData$Obesity_Level)
    trainData_balanced$Class <- NULL  # Remove extra column added by upSample
      } else {
           trainData_balanced <- downSample(x = trainData[-ncol(trainData)], y = trainData$Obesity_Level)
           trainData_balanced$Class <- NULL  # Remove extra column added by downSample
         }
         

tune_grid <- expand.grid(mtry = c(2, 4, 6, 8))
rf_model <- train(Obesity_Level ~ ., data = trainData_balanced, method = "rf", 
            ntree = 500, importance = TRUE, tuneGrid = tune_grid,
            trControl = trainControl(method = "cv", number = 5))
         
print(rf_model)
         

importance_df <- data.frame(Feature = rownames(rf_model$finalModel$importance),
      Importance = rf_model$finalModel$importance[, 1]) %>%
      arrange(desc(Importance))
         
ggplot(importance_df, aes(x = reorder(Feature, Importance), y = Importance, fill = Importance)) +
    geom_bar(stat = "identity") +
    coord_flip() +
    scale_fill_gradient(low = "lightblue", high = "darkblue") +
    theme_minimal() +
    labs(title = "Feature Importance in Random Forest Model",
         x = "Features",
         y = "Importance") +
    theme(plot.title = element_text(size = 16, face = "bold"))
         

rf_pred <- predict(rf_model, testData)
conf_matrix_rf <- confusionMatrix(rf_pred, testData$Obesity_Level)
print(conf_matrix_rf)
         

pheatmap(as.matrix(conf_matrix_rf$table), cluster_rows = FALSE, cluster_cols = FALSE,
        display_numbers = TRUE, color = colorRampPalette(c("white", "blue"))(100),
        main = "Confusion Matrix - Random Forest")
         

normalize <- function(x) {
    return ((x - min(x)) / (max(x) - min(x)))
}
         
trainData_norm <- trainData_balanced %>%
    mutate(across(where(is.numeric), normalize))
    testData_norm <- testData %>%
    mutate(across(where(is.numeric), normalize))
         

nn_model <- nnet(Obesity_Level ~ ., data = trainData_norm, size = 10, maxit = 300, decay = 0.01, 
    trace = TRUE, MaxNWts = 1000, abstol = 1.0e-4, reltol = 1.0e-8)

nn_pred <- predict(nn_model, testData_norm, type = "raw")
nn_pred_class <- apply(nn_pred, 1, which.max)  
levels_list <- levels(trainData_norm$Obesity_Level)
nn_pred_factor <- factor(levels_list[nn_pred_class], levels = levels_list)
         

conf_matrix_nn <- confusionMatrix(nn_pred_factor, testData_norm$Obesity_Level)
print(conf_matrix_nn)
         

pheatmap(as.matrix(conf_matrix_nn$table), cluster_rows = FALSE, cluster_cols = FALSE,
        display_numbers = TRUE, color = colorRampPalette(c("white", "red"))(100),
        main = "Confusion Matrix - Neural Network")
     

if (length(levels(data$Obesity_Level)) == 2) {
roc_curve <- roc(testData$Obesity_Level, as.numeric(rf_pred))
plot(roc_curve, main = "ROC Curve for Random Forest")
auc(roc_curve)
}
         

rf_acc <- conf_matrix_rf$overall["Accuracy"]
nn_acc <- conf_matrix_nn$overall["Accuracy"]
         
cat("Random Forest Accuracy:", rf_acc, "\n")
cat("Neural Network Accuracy:", nn_acc, "\n")
         

mcnemar_test <- mcnemar.test(table(rf_pred, nn_pred_factor))
print(mcnemar_test)

stopCluster(cl)

predictions <- predict(rf_model, testData)
confusionMatrix(predictions, testData$Obesity_Level)

importance(rf_model$finalModel)
varImpPlot(rf_model$finalModel)


