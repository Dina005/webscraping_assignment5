# eBay Deal Discount Prediction - Assignmet 6

## Overview
The objective of this assignment is to predict the discount percentage variable using linear regression under two seperate secnarions:
1. **Already scraped discount price:** analyzing the relationship between product features and the existing discount percentage to predict future data
2. **Predicted discount price:** predicted discount percentages using incomplete or hidden data

## Methodology
After importing and loading the necessary libraries and datasets, and following the steps of lab 12, the following steps were undertaken:
1. Handling missing values in key columns: price, original price, shipping, and discount percentage
2. Converting non-numerice values in the shipping columns to numeric values by assigning numeric represnetation
3. Adjusting certain aspects of the code including:
    * creating conditions for the new column (discount_bins)
    * Seperating each column for easier counting
    * Using a balancing via undersampling technique


## Analysis
# Discount Percentage Distribution
The histogram of the distribution of the discount percentage variable revealed a higher frequency of high discounts in the dataset (over 30%). Specifically, over 2,000 (3226) discounts were categorized as high, mostly ranging between 40% and 60%.

This was further supported during the binning process. The counts for each discount category were: 261 for low discounts, 488 for medium discounts, and 3226 for high discounts. This significant class imbalance, with a much larger number of high discount deals, proved the importance of class balancing. Undersampling was used to reduce the number of samples in each category to the size of the 261 to prevent the model from being biased towards the majority class (high discounts).

# Scatter Plot (Actual vs. Predicted)
 Comparing the actual and predicted discount percentages did not show a strong linear correlation. The significant scatter of points indicates a difference between the model's predictions and the actual discount values.

# Residual Plot
Displaying the errors (actual - predicted) against the predicted discount percentages, exhibited a non-random distribution. Key observations include:

* The residuals did not appear to be randomly scattered , suggesting potential biases in the model's predictions.
* The presence of outliers, indicated the model's predictions were significantly inaccurate.
* The variance of the residuals did not appear to be constant across different predicted discount levels which violates a key assumption of linear regression and can affect the reliability of the model

These observations  suggest that the linear regression model might not be a good fit for predicting the discount percentage.

# Evaluation Metrics
1.  **Mean Absolute Error (MAE) = 11.36**
    The MAE measures the average absolute difference between the predicted and actual discount percentages. An MAE of 11.36 indicates that, on average, the model's predictions are off by 11.36 percentage points. A lower MAE generally indicates a better-performing model. 

2.  **Mean Squared Error (MSE): 219.37**
    The MSE measures the average of the squared differences between the predicted and actual discount percentages. Squaring the errors gives more weight to larger errors. An MSE of 219.37 indicates there is an error in the model as it significantly large.

3.  **Root Mean Squared Error (RMSE): 14.81**
    The RMSE is the square root of the MSE but in percentage. An RMSE of 14.81 indicates that, on average, the magnitude of the prediction errors is 14.81%, which can be interpreted similarly to the MSE.

4.  **R-Squared Score: 0.59**
    The R-squared score, also known as the coefficient of determination, represents the proportion of the variance in the discount percentage that is predictable from the price, original price, and shipping. A value of 0.59 suggests that approximately 59% of the variance in the discount percentage is explained by the model which is moderate. 