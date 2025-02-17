# Install necessary packages (if not already installed)
required_packages <- c("readxl", "tidyverse", "forecast", "tseries", "TTR", "zoo")
installed_packages <- rownames(installed.packages())
to_install <- setdiff(required_packages, installed_packages)
if (length(to_install) > 0) install.packages(to_install)

# Load required libraries
library(readxl)     # For reading Excel files
library(tidyverse)  # For data manipulation and visualization
library(forecast)   # For ARIMA and SARIMA modeling
library(tseries)    # For ADF test
library(TTR)        # For trend analysis using SMA
library(zoo)        # For linear interpolation

# Load the dataset
file_path <- "Vital statistics in the UK.xlsx"  # Replace with the actual file path
birth_data <- read_excel(file_path, sheet = "Birth", skip = 5)

# Clean column names and retain relevant columns
colnames(birth_data) <- c(
  "Year", "Live_Births_UK", "Live_Births_England_Wales", "Live_Births_England",
  "Live_Births_Wales", "Live_Births_Scotland", "Live_Births_Northern_Ireland",
  "Fertility_Rate_UK", "Fertility_Rate_England_Wales", "Fertility_Rate_England",
  "Fertility_Rate_Wales", "Fertility_Rate_Scotland", "Fertility_Rate_Northern_Ireland"
)

birth_data <- birth_data %>%
  select(Year, Live_Births_UK) %>%
  mutate(
    Year = as.numeric(Year),
    Live_Births_UK = as.numeric(Live_Births_UK)
  ) %>%
  filter(!is.na(Live_Births_UK))

# Step 1: Ensure data is sorted
birth_data <- birth_data %>%
  arrange(Year)  # Sort by Year in ascending order

# Step 2: Create time series object with correct start year
births_ts <- ts(birth_data$Live_Births_UK, start = min(birth_data$Year), frequency = 1)

# Step 3: Plot again to confirm
plot(births_ts, main = "Live Births in the UK", ylab = "Number of Births", xlab = "Year")

# Step 4: Trend Analysis Using Simple Moving Average (SMA)
library(TTR)  # For SMA
births_sma <- SMA(births_ts, n = 5)  # 5-year moving average
plot(births_ts, main = "Live Births in the UK with Trend (SMA)", ylab = "Number of Births", xlab = "Year", col = "blue", lwd = 2)
lines(births_sma, col = "red", lwd = 2)
legend("topright", legend = c("Actual Data", "5-Year SMA Trend"), col = c("blue", "red"), lwd = 2)

# Step 5: Perform Augmented Dickey-Fuller (ADF) Test for Stationarity
adf_result <- adf.test(births_ts)
print(adf_result)

# Step 6: If non-stationary, apply differencing
if (adf_result$p.value > 0.05) {
  # Apply differencing
  births_diff <- diff(births_ts)
  
  # Perform ADF test on the differenced series
  adf_diff_result <- adf.test(births_diff)
  print(adf_diff_result)
  
  # Suppress scientific notation for the plot
  options(scipen = 10)
  
  # Plot the differenced time series
  plot(births_diff, main = "Differenced Time Series", 
       ylab = "Differenced Births", xlab = "Year", 
       col = "blue", lwd = 2)
}



# Step 7: Fit ARIMA Model
arima_model <- auto.arima(births_ts, seasonal = FALSE, stepwise = FALSE, approximation = FALSE)
summary(arima_model)

# Check Residuals for ARIMA Model
checkresiduals(arima_model)

# Forecast with ARIMA Model
arima_forecast <- forecast(arima_model, h = 10)
plot(arima_forecast, main = "ARIMA Forecast for Live Births in the UK")

# Step 8: Fit ETS Model
ets_model <- ets(births_ts)
summary(ets_model)

# Check Residuals for ETS Model
checkresiduals(ets_model)

# Forecast with ETS Model
ets_forecast <- forecast(ets_model, h = 10)
plot(ets_forecast, main = "ETS Forecast for Live Births in the UK")

# Step 9: Fit NNAR Model
nnar_model <- nnetar(births_ts)
summary(nnar_model)

# Check Residuals for NNAR Model
checkresiduals(nnar_model)

# Forecast with NNAR Model
nnar_forecast <- forecast(nnar_model, h = 10)
plot(nnar_forecast, main = "NNAR Forecast for Live Births in the UK")

# Step 10: Train-Test Split and Validation
train_data <- window(births_ts, end = c(2010))
test_data <- window(births_ts, start = c(2011))

# Train and Forecast with ETS Model on Training Data
ets_train_model <- ets(train_data)
ets_train_forecast <- forecast(ets_train_model, h = length(test_data))

# Train and Forecast with ARIMA Model on Training Data
arima_train_model <- auto.arima(train_data, seasonal = FALSE, stepwise = FALSE, approximation = FALSE)
arima_train_forecast <- forecast(arima_train_model, h = length(test_data))

# Train and Forecast with NNAR Model on Training Data
nnar_train_model <- nnetar(train_data)
nnar_train_forecast <- forecast(nnar_train_model, h = length(test_data))

# Calculate Test Accuracy for ARIMA, ETS, and NNAR
test_accuracy_arima <- accuracy(arima_train_forecast$mean, test_data)
test_accuracy_ets <- accuracy(ets_train_forecast$mean, test_data)
test_accuracy_nnar <- accuracy(nnar_train_forecast$mean, test_data)

# Step 11: Compare Models Based on Test RMSE
comparison <- data.frame(
  Model = c("ARIMA", "ETS", "NNAR"),
  AIC = c(arima_train_model$aic, ets_train_model$aic, NA), # NNAR does not provide AIC
  RMSE = c(
    test_accuracy_arima["Test set", "RMSE"],
    test_accuracy_ets["Test set", "RMSE"],
    test_accuracy_nnar["Test set", "RMSE"]
  )
)
print("Model Comparison Based on Test RMSE and AIC:")
print(comparison)

# Step 12: Recommend Best Model Based on RMSE
best_model <- comparison[which.min(comparison$RMSE), "Model"]
cat("The best model for forecasting is:", best_model, "\n")
