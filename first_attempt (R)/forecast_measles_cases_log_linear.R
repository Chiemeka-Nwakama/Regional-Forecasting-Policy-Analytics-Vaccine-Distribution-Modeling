
# Loop through each country
for (country in unique(MCV1_Coverage_LnReg$Location)) {
  # Filter data for the current country of interest
  country_data <- MCV1_Coverage_LnReg %>% filter(Location == country)
  
  # Log-transform MeaslesCases (adding 1 to avoid log(0) which is undef)
  country_data <- country_data %>% mutate(LogCases = log(MeaslesCases + 1))
  
  # Fits linear regression model on the log-transformed data
  model <- lm(LogCases ~ Year, data = country_data)
  
  # Creates a data frame for future years for the current country
  future_data <- data.frame(Year = future_years)
  
  # Predicts future measle cases in underlined by a log function
  future_data$LogCases <- predict(model, newdata = future_data)
  
  # Back-transforms predictions to original scale by exponetiating it
  future_data$MeaslesCases <- exp(future_data$LogCases) - 1  # Subtract 1 to reverse the offset (that was used to avoid undefined log 0)
  
  # Assigns the current country to the predictions
  future_data$Country <- country
  
  # Combines results of the current country with the rest of the predicted country cases
  future_predictions_mv1 <- rbind(future_predictions_mv1, future_data)
}

# shows our future predictions for the next 10 years in each country
print(future_predictions_mv1)


#---------------------------------------------
# Loop through each country
for (country in unique(MCV1_Coverage_LnReg$Location)) {
  # Filter data for the current country of interest
  country_data <- MCV1_Coverage_LnReg %>% filter(Location == country)
  
  
  # Fits linear regression model on the log-transformed data
  model <- lm(MCV1_Coverage ~ Year, data = country_data)
  
  # Creates a data frame for future years for the current country
  future_data <- data.frame(Year = future_years)
  
  # Predicts future measle cases in underlined by a log function
  future_data$LogCases <- predict(model, newdata = future_data)
  
  #p
  future_data$MeaslesCases <- future_data$LogCases
  
  # Assigns the current country to the predictions
  future_data$Country <- country
  
  # Combines results of the current country with the rest of the predicted country cases
  future_predictions_mv1 <- rbind(future_predictions_mv1, future_data)
}

# shows our future predictions for the next 10 years in each country
print(future_predictions_mv1)

