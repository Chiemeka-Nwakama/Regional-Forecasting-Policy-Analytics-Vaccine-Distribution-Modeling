install.packages("readxl")
library(readxl)
library(tidyr)
library(dplyr)

#surveyData <- read_excel(survey-data_wuenic2023rev.xlsx', sheet = "Data")
measlesReportedCases <- read.csv('measles_WHO.csv') # measels reported cases from 1974-2023 for 213 countries

MCV2_Coverage = read.csv('MCV2.csv') # MCV2 vaccine coverage rate from 2000-2023 for 189 countries
# print(MCV1_Coverage)
#Data Preparation and Cleanup
#Removes all years for measles cases before 2000 to match the other measles datasets

#Measles Reported Cases


measlesReportedCases = measlesReportedCases[,1:25]

#Fills in missing case reports in with 0s as nothing was reported
for(i in 1:nrow(measlesReportedCases)){
  for(j in 1:ncol(measlesReportedCases[1,])){
    
    if(is.na(measlesReportedCases[i,j]) || measlesReportedCases[i,j] == ""){
      measlesReportedCases[i,j] = 0 #change NA's to 0
    }
    #if there is a space in number like 11 456 (11,456) remove it
    measlesReportedCases[i, j] <- gsub(" ", "", measlesReportedCases[i, j])
  }
}

#removes extra garbage rows
measlesReportedCases = measlesReportedCases[1:214,] 


#converts the cases to numeric values
measlesReportedCases[,2:ncol(measlesReportedCases)] <- apply(measlesReportedCases[,2:ncol(measlesReportedCases)], 2, as.numeric)


# Calculate row means, excluding NA values
average_values <- rowMeans(measlesReportedCases[,2:ncol(measlesReportedCases)], na.rm = TRUE)

# Combine row names (assumed to be country names) with the average values
results <- data.frame(Country = measlesReportedCases[,1], Average = average_values)

# Sort by average values in descending order
sorted_results <- results[order(-results$Average), ]


# Get the top 5 and bottom 5 countries
top_5 <- head(sorted_results, 5)
#bottom_5 <- tail(sorted_results, 5)

# Return the top 5 measles case number countries as a list
list(Top5 = top_5)

#Will include:
#United States, UK, Finland, Argentina, Germany

#remove the X in front of each year
colnames(measlesReportedCases) <- gsub("^X", "", colnames(measlesReportedCases))

countriesMeaslesReported = c("UnitedStatesofAmerica", "UnitedKingdomofGreatBritainandNorthernIreland", "Finland", "Argentina", "Germany", "DemocraticRepublicoftheCongo", "China", "Nigeria", "India",  "Madagascar")


#filters out the countries we are not interested in
measlesReportedCases <- measlesReportedCases %>%
  filter(Location %in% countriesMeaslesReported)


#Refactors the reported case matrix with the filtered countries to be used
measlesReportedCases_LnReg <- measlesReportedCases %>%
  pivot_longer(cols = -Location, names_to = "Year", values_to = "MeaslesCases") %>%
  mutate(Year = as.numeric(Year))  # Convert Year to numeric

future_predictions_cases <- data.frame()

# Specify the years you want to predict
future_years <- c(2024, 2025, 2026, 2027, 2028, 2029, 2030, 2031, 2032, 2033, 2034)

# Loop through each country
for (country in unique(measlesReportedCases_LnReg$Location)) {
  # Filter data for the current country of interest
  country_data <- measlesReportedCases_LnReg %>% filter(Location == country)
  
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
  future_predictions_cases <- rbind(future_predictions_cases, future_data)
}

# shows our future predictions for the next 10 years in each country
print(future_predictions_cases)

#-------------------------------------------------------

#MCV1 Coverage

MCV1_Coverage = read.csv('MCV1.csv') # MCV1 vaccine coverage rate from 2000-2023 for 194 countries


#Remove columns 1, 2 and 4
MCV1_Coverage = MCV1_Coverage[,c(-1,-2,-4)]

#Renames the country column to "Location"

colnames(MCV1_Coverage)[colnames(MCV1_Coverage) == 'country'] <- 'Location'

#Fills in missing case reports in with 0s as nothing was reported
for(i in 1:nrow(MCV1_Coverage)){
  for(j in 1:ncol(MCV1_Coverage[1,])){
    
    if(is.na(MCV1_Coverage[i,j]) || MCV1_Coverage[i,j] == ""){
      MCV1_Coverage[i,j] = 0 #change NA's to 0
    }
    #if there is a space in number like 11 456 (11,456) remove it
    MCV1_Coverage[i, j] <- gsub(" ", "", MCV1_Coverage[i, j])
  }
}

#removes extra garbage rows
MCV1_Coverage = MCV1_Coverage[1:214,] 


#converts the cases to numeric values
MCV1_Coverage[,2:ncol(MCV1_Coverage)] <- apply(MCV1_Coverage[,2:ncol(MCV1_Coverage)], 2, as.numeric)



#Will include:
#United States, UK, Finland, Argentina, Germany

countriesMCV1_Coverage = c("UnitedStates", "UnitedKingdom", "Finland", "Argentina", "Germany", "DemocraticRepublicoftheCongo", "China", "Nigeria", "India",  "Madagascar")

#remove the X in front of each year
colnames(MCV1_Coverage) <- gsub("^X", "", colnames(MCV1_Coverage))


#filters out the countries we are not interested in
MCV1_Coverage <- MCV1_Coverage %>%
  filter(Location %in% countriesMCV1_Coverage)


#Refactors the reported case matrix with the filtered countries to be used
MCV1_Coverage_LnReg <- MCV1_Coverage %>%
  pivot_longer(cols = -Location, names_to = "Year", values_to = "MCV1_Coverage") %>%
  mutate(Year = as.numeric(Year))  # Convert Year to numeric

future_predictions_mv1 <- data.frame()

# Specify the years you want to predict
future_years <- c(2024, 2025, 2026, 2027, 2028, 2029, 2030, 2031, 2032, 2033, 2034)
# Loop through each country
for (country in unique(MCV1_Coverage_LnReg$Location)) {
  # Filter data for the current country of interest
  country_data <- MCV1_Coverage_LnReg %>% filter(Location == country)
  
  
  # Fits linear regression model on the log-transformed data
  model <- lm(MCV1_Coverage ~ Year, data = country_data)
  
  # Creates a data frame for future years for the current country
  future_data <- data.frame(Year = future_years)
  
  # Predicts future measle cases in underlined by a log function
  future_data$Coverage <- predict(model, newdata = future_data)
  
  #limits coverage to only be able to be at 99% max
  future_data$Coverage <- ifelse(future_data$Coverage > 99, 99, future_data$Coverage)
  
  
 
  
  # Assigns the current country to the predictions
  future_data$Country <- country
  
  # Combines results of the current country with the rest of the predicted country cases
  future_predictions_mv1 <- rbind(future_predictions_mv1, future_data)
}

# shows our future predictions for the next 10 years in each country
print(future_predictions_mv1)

#---------------------------------------------------

#MCV2 Coverage

MCV2_Coverage = read.csv('MCV2.csv') # MCV2 vaccine coverage rate from 2000-2023 for 194 countries


#Remove columns 1, 2 and 4
MCV2_Coverage = MCV2_Coverage[,c(-1,-2,-4)]

#Renames the country column to "Location"

colnames(MCV2_Coverage)[colnames(MCV2_Coverage) == 'country'] <- 'Location'

#Fills in missing case reports in with 0s as nothing was reported
for(i in 1:nrow(MCV2_Coverage)){
  for(j in 1:ncol(MCV2_Coverage[1,])){
    
    if(is.na(MCV2_Coverage[i,j]) || MCV2_Coverage[i,j] == ""){
      MCV2_Coverage[i,j] = 0 #change NA's to 0
    }
    #if there is a space in number like 11 456 (11,456) remove it
    MCV2_Coverage[i, j] <- gsub(" ", "", MCV2_Coverage[i, j])
  }
}

#removes extra garbage rows
MCV2_Coverage = MCV2_Coverage[1:214,] 


#converts the cases to numeric values
MCV2_Coverage[,2:ncol(MCV2_Coverage)] <- apply(MCV2_Coverage[,2:ncol(MCV2_Coverage)], 2, as.numeric)



#Will include:
#United States, UK, Finland, Argentina, Germany

countriesMCV2_Coverage = c("UnitedStates", "UnitedKingdom", "Finland", "Argentina", "Germany", "DemocraticRepublicoftheCongo", "China", "Nigeria", "India",  "Madagascar")

#remove the X in front of each year
colnames(MCV2_Coverage) <- gsub("^X", "", colnames(MCV2_Coverage))


#filters out the countries we are not interested in
MCV2_Coverage <- MCV2_Coverage %>%
  filter(Location %in% countriesMCV2_Coverage)


#Refactors the reported case matrix with the filtered countries to be used
MCV2_Coverage_LnReg <- MCV2_Coverage %>%
  pivot_longer(cols = -Location, names_to = "Year", values_to = "MCV2_Coverage") %>%
  mutate(Year = as.numeric(Year))  # Convert Year to numeric

future_predictions_mv2 <- data.frame()

# Specify the years you want to predict
future_years <- c(2024, 2025, 2026, 2027, 2028, 2029, 2030, 2031, 2032, 2033, 2034)
# Loop through each country
for (country in unique(MCV2_Coverage_LnReg$Location)) {
  # Filter data for the current country of interest
  country_data <- MCV2_Coverage_LnReg %>% filter(Location == country)
  
  
  # Fits linear regression model on the log-transformed data
  model <- lm(MCV2_Coverage ~ Year, data = country_data)
  
  # Creates a data frame for future years for the current country
  future_data <- data.frame(Year = future_years)
  
  # Predicts future measle cases in underlined by a log function
  future_data$Coverage <- predict(model, newdata = future_data)
  
  #limits coverage to only be able to be at 99% max
  future_data$Coverage <- ifelse(future_data$Coverage > 99, 99, future_data$Coverage)
  
  
  
  
  # Assigns the current country to the predictions
  future_data$Country <- country
  
  # Combines results of the current country with the rest of the predicted country cases
  future_predictions_mv2 <- rbind(future_predictions_mv2, future_data)
}

# shows our future predictions for the next 10 years in each country
print(future_predictions_mv2)


