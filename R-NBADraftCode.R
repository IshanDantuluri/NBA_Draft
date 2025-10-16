#####################################Data#########################################

# Install required packages if not already installed
packages <- c("randomForest", "nnet", "ggplot2", "dplyr")

# Check for missing packages and install them
installed_packages <- rownames(installed.packages())
for (pkg in packages) {
  if (!(pkg %in% installed_packages)) {
    install.packages(pkg)
  }
}

# Load the packages
lapply(packages, library, character.only = TRUE)

#Load your data
nba_draft <- read.csv("C:/Users/ishan/Downloads/NBA_DRAFT_FINALDATA (1).csv", stringsAsFactors = FALSE)

#Calculate performance score using more features
nba_draft$performance_score <- with(nba_draft,
  points_per_game * 1.0 +                 # scoring
  average_total_rebounds * 0.8 +          # rebounding
  average_assists * 0.9 +                 # playmaking
  win_shares_per_48_minutes * 1.5 +       # efficiency
  box_plus_minus * 1.2 +                  # advanced impact
  field_goal_percentage * 0.5 +           # shooting efficiency
  X3_point_percentage * 0.4 +             # 3PT shooting
  free_throw_percentage * 0.3 +           # FT shooting
  average_minutes_played * 0.2 +          # usage
  minutes_played * 0.05 +                 # total minutes
  games * 0.05                            # experience
)

#Create pick_quality factor based on quantiles
score_quantiles <- quantile(nba_draft$performance_score, probs = c(0, 0.33, 0.66, 1), na.rm = TRUE)

nba_draft$pick_quality <- cut(
  nba_draft$performance_score,
  breaks = score_quantiles,
  labels = c("Poor", "Average", "Good"),
  include.lowest = TRUE,
  right = FALSE
)

#Show distribution of classes
table(nba_draft$pick_quality)

#View sample data
head(nba_draft)

#Numeric columns to contaminate
numeric_cols <- c("points_per_game","average_total_rebounds","average_assists","win_shares_per_48_minutes","box_plus_minus","field_goal_percentage","X3_point_percentage","free_throw_percentage","average_minutes_played","minutes_played","games")

contamination_rate <- runif(1, min=0.1,max=0.2)
total_contam_test <- floor(nrow(nba_draft) * contamination_rate)
num_outliers_test <- ceiling(total_contam_test * 0.8)
num_nas_test <- total_contam_test - num_outliers_test

#Outliers: multiply by 10 for severity
outlier_rows_test <- sample(1:nrow(nba_draft), num_outliers_test)
for (row in outlier_rows_test) {
  col <- sample(numeric_cols, 1)
  val <- nba_draft[row, col]
  if (!is.na(val)) {
    nba_draft[row, col] <- round(val * 10, 2)
  }
}

#Inject NAs in test set
na_rows_test <- sample(setdiff(1:nrow(nba_draft), outlier_rows_test), num_nas_test)
for (row in na_rows_test) {
  col <- sample(numeric_cols, 1)
  nba_draft[row, col] <- NA
}

#Clean Test
clean_test <- nba_draft
for (col in numeric_cols) {
  Q1 <- quantile(clean_test[[col]], 0.25, na.rm = TRUE)
  Q3 <- quantile(clean_test[[col]], 0.75, na.rm = TRUE)
  IQR <- Q3 - Q1
  lower_bound <- Q1 - 1.5 * IQR
  upper_bound <- Q3 + 1.5 * IQR
  clean_test <- clean_test[(is.na(clean_test[[col]]) | (clean_test[[col]] >= lower_bound & clean_test[[col]] <= upper_bound)), ]
}
nba_draft <- na.omit(clean_test)

###############################Models#########################################

###Random Forest

library(randomForest)

nba_draft <- na.omit(nba_draft)

nba_draft$pick_quality <- as.factor(nba_draft$pick_quality)

train_indicesR <- sample(1:nrow(nba_draft),round(0.8*nrow(nba_draft)))

train_dataR <- nba_draft[train_indicesR,]
test_dataR <- nba_draft[-train_indicesR,]

RF_model <- randomForest(pick_quality ~ ., data=train_dataR,ntree=500,mtry=sqrt(ncol(train_dataR)-1))

predictions_RF <- predict(RF_model, newdata = test_dataR)

confusion_matrix_RF <- table(predictions_RF, test_dataR$pick_quality)
print(confusion_matrix_RF)

accuracy_RF <- sum(diag(confusion_matrix_RF)) / sum(confusion_matrix_RF)

###Linear Regression

tdataClean <- na.omit(nba_draft)

tdataClean$pick_quality_numeric <- as.numeric(tdataClean$pick_quality)

labels <- levels(tdataClean$pick_quality)

random_indices <- sample(nrow(tdataClean))
randTdata <- tdataClean[random_indices,]

accScores <- c()

for (i in 1:4) {
  test_indices <- ((i-1)*(nrow(randTdata)/4)+1):(i*(nrow(randTdata)/4))
  
  test_m <- randTdata[test_indices, ]
  train_m <- randTdata[-test_indices, ]

  multiplelrm_m <- lm(pick_quality_numeric ~ points_per_game+average_total_rebounds+average_assists+
                        win_shares_per_48_minutes+box_plus_minus+field_goal_percentage+
                        X3_point_percentage+free_throw_percentage+average_minutes_played+
                        minutes_played+games,
                      data = train_m)

  prediction_m <- predict(multiplelrm_m, test_m)
  prediction_m_rounded <- pmin(pmax(round(prediction_m), 1), 3)

  predicted_labels <- factor(labels[prediction_m_rounded], levels = labels)
  actual_labels <- factor(test_m$pick_quality, levels = labels)

  confusion_matrix_m <- table(Actual = actual_labels, Predicted = predicted_labels)
  
  print(confusion_matrix_m)

  acc <- sum(diag(confusion_matrix_m)) / sum(confusion_matrix_m)
  accScores[i] <- acc
}

total_accuracy <- mean(accScores)

###Logistic Regression
library(nnet) 

tdataClean_l <- na.omit(nba_draft)

random_indices_l <- sample(nrow(tdataClean_l))
randTdata_l <- tdataClean_l[random_indices_l,]

accScores_l <- c()

for (i in 1:4) {
  test_indices_l <- ((i-1)*(nrow(randTdata_l)/4)+1):(i*(nrow(randTdata_l)/4))
  
  test_l <- randTdata_l[test_indices_l, ]
  train_l <- randTdata_l[-test_indices_l, ]

  logRegModel <- multinom(pick_quality ~ points_per_game + average_total_rebounds + average_assists +
                           win_shares_per_48_minutes + box_plus_minus + field_goal_percentage +
                           X3_point_percentage + free_throw_percentage + average_minutes_played +
                           minutes_played + games,
                         data = train_l, trace = FALSE)
  
  prediction_l <- predict(logRegModel, newdata = test_l)
  
  confusion_matrix_l <- table(Actual = test_l$pick_quality, Predicted = prediction_l)
  print(confusion_matrix_l)
  
  acc_l <- sum(diag(confusion_matrix_l)) / sum(confusion_matrix_l)
  accScores_l[i] <- acc_l
}

total_accuracy_l <- mean(accScores_l)

cat(sprintf("Random Forest: %.2f%%\n", accuracy_RF * 100))
cat(sprintf("Linear Regression: %.2f%%\n", total_accuracy * 100))
cat(sprintf("Logistic Regression: %.2f%%\n", total_accuracy_l * 100))


###################################Graphs#####################################
library(ggplot2)

###Side-By-Side Graph
accuracy_df <- data.frame(
  Model = c("Random Forest", "Linear Regression", "Logistic Regression"),
  Accuracy = c(accuracy_RF * 100, total_accuracy * 100, total_accuracy_l * 100)
)

ggplot(accuracy_df, aes(x = Model, y = Accuracy, fill = Model)) +
  geom_bar(stat = "identity", width = 0.6, show.legend = FALSE) +
  geom_text(aes(label = paste0(round(Accuracy, 2), "%")), vjust = -0.5, size = 5) +
  ylim(0, 100) +
  scale_y_continuous(limits = c(0, 100), breaks = seq(0, 100, 10)) +
  labs(title = "Model Accuracy Comparison", y = "Accuracy (%)", x = "Model")

dev.new()

###Range of Model Accuracies
library(dplyr)

simulate_accuracies <- function(mean_acc, n=20, sd=0.02) {
  pmax(pmin(rnorm(n, mean_acc, sd), 1), 0) 
}

rf_sim <- simulate_accuracies(accuracy_RF, 20)
lin_sim <- simulate_accuracies(total_accuracy, 20)
log_sim <- simulate_accuracies(total_accuracy_l, 20)

accuracy_long_sim <- data.frame(
  Accuracy = c(rf_sim, lin_sim, log_sim) * 100,
  Model = rep(c("Random Forest", "Linear Regression", "Logistic Regression"), each = 20)
)

ggplot(accuracy_long_sim, aes(x = Model, y = Accuracy, color = Model)) +
  geom_jitter(width = 0.2, size = 3, alpha = 0.7) +
  stat_summary(fun = mean, geom = "point", shape = 18, size = 5, color = "black") + 
  scale_y_continuous(limits = c(85, 100), breaks = seq(85, 100, 1)) +    
  labs(title = "Simulated Distribution of Model Accuracies", y = "Accuracy (%)", x = "Model") +
  theme(legend.position = "none")

dev.new()

###Random Forest Graph
rf_conf_matrix_df <- as.data.frame(confusion_matrix_RF)
colnames(rf_conf_matrix_df) <- c("Predicted", "Actual", "Freq")

ggplot(rf_conf_matrix_df, aes(x = Actual, y = Predicted, fill = Freq)) +
  geom_tile(color = "white") +
  geom_text(aes(label = Freq), color = "white", size = 5) +
  scale_fill_gradient(low = "blue", high = "blue4") +
  labs(title = "Random Forest Confusion Matrix", x = "Actual Pick Quality", y = "Predicted Pick Quality")

dev.new()

###Linear Regression Graph
lin_conf_matrix_df <- as.data.frame(confusion_matrix_m)
colnames(lin_conf_matrix_df) <- c("Predicted", "Actual", "Freq")

ggplot(lin_conf_matrix_df, aes(x = Actual, y = Predicted, fill = Freq)) +
  geom_tile(color = "white") +
  geom_text(aes(label = Freq), color = "white", size = 5) +
  scale_fill_gradient(low = "red", high = "red4") +
  labs(title = "Linear Regression Confusion Matrix", x = "Actual Pick Quality", y = "Predicted Pick Quality")

dev.new()
###Logistic Regression Graph
log_conf_matrix_df <- as.data.frame(confusion_matrix_l)
colnames(log_conf_matrix_df) <- c("Predicted", "Actual", "Freq")

ggplot(log_conf_matrix_df, aes(x = Actual, y = Predicted, fill = Freq)) +
  geom_tile(color = "white") +
  geom_text(aes(label = Freq), color = "white", size = 5) +
  scale_fill_gradient(low = "green", high = "green4") +
  labs(title = "Logistic Regression Confusion Matrix", x = "Actual Pick Quality", y = "Predicted Pick Quality")

dev.new()

###Feature Importance on RF
importance_df <- as.data.frame(importance(RF_model))
importance_df$Feature <- rownames(importance_df)

importance_df <- importance_df[order(importance_df$MeanDecreaseGini, decreasing = TRUE), ]

ggplot(importance_df, aes(x = reorder(Feature, MeanDecreaseGini), y = MeanDecreaseGini, fill = Feature)) +
  geom_bar(stat = "identity") +
  coord_flip() +
  scale_y_continuous(expand = expansion(mult = c(0, 0.05))) +  
  labs(title = "Random Forest Feature Importance",
       x = "Feature",
       y = "Importance") +
  theme(legend.position = "none")

dev.new()

###Points Effect on Pick Quality (Scatter Plot)
ggplot(nba_draft, aes(x = pick_quality, y = points_per_game, color = pick_quality)) +
  geom_jitter(width = 0.2, alpha = 0.6) +
  labs(title = "Points Per Game vs Pick Quality",
       x = "Pick Quality",
       y = "Points Per Game") +
  scale_y_continuous(limits = c(0, 20), breaks = seq(0, 20, 2.5)) +
  theme(legend.position = "none")

dev.new()

###Rebounds Effect on Pick Quality (Violin Plot)
ggplot(nba_draft, aes(x = pick_quality, y = average_total_rebounds, fill = pick_quality)) +
  geom_violin(alpha = 0.7) +
  labs(title = "Average Total Rebounds Distribution by Pick Quality",
       x = "Pick Quality",
       y = "Average Total Rebounds") +
  scale_y_continuous(limits = c(0, 8), breaks = seq(0, 8, 1)) +
  theme(legend.position = "none")

dev.new()

###Box Plus Minus Effect on Pick Quality (Box Plot)
ggplot(nba_draft, aes(x = pick_quality, y = box_plus_minus, fill = pick_quality)) +
  geom_boxplot(alpha = 0.7) +
  labs(title = "Box Plus/Minus by Pick Quality",
       x = "Pick Quality",
       y = "Box Plus/Minus") +
  scale_y_continuous(limits = c(-7.5, 5), breaks = seq(-7.5, 5, 2.5)) +
  theme(legend.position = "none")


View(nba_draft)