### Import Libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, root_mean_squared_error
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt


### Step 1: Data Prepration
## Load the cleaned dataset
df = pd.read_csv("cleaned_ebay_deals.csv")
print("Loaded rows:", len(df))

## Remove Rows With Missing Values
df_cleaned = df.dropna(subset=["price", "original_price", "shipping", "discount_percentage"]).copy()
print("Rows after cleaning:", len(df))

## Plot Histogram of Discount Percentage
plt.figure(figsize=(10, 5))
sns.histplot(df['discount_percentage'], bins=10, kde=True, color='green')
plt.xlabel("Discount (%)")
plt.title("Distribution of Discount Percentages")
plt.show()

### Step 2: Binning and Balancing Data
## Create a new column: discount_bin 
conditions = [
    (df_cleaned['discount_percentage'] >= 0) & (df_cleaned['discount_percentage'] <= 10),
    (df_cleaned['discount_percentage'] >= 10) & (df_cleaned['discount_percentage'] <=30),
    (df_cleaned['discount_percentage'] > 30)
]

choices = ['Low Discount', 'Medium Discount', 'High Discount']
df_cleaned['discount_bin'] = np.select(conditions, choices, default='Unknown')


## Count - Sample in Each category
# Seperate each Category
df_low = df_cleaned[df_cleaned['discount_bin'] == 'Low Discount']
df_medium = df_cleaned[df_cleaned['discount_bin'] == 'Medium Discount']
df_high = df_cleaned[df_cleaned['discount_bin'] == 'High Discount']

# Count Samples in Each
low_count = len(df_low)
medium_count = len(df_medium)
high_count = len(df_high)
print(f"Low Discount: {low_count}")
print(f"Medium Discount: {medium_count}")
print(f"High Discount: {high_count}")


## Balance Using Undersampling
min_count = min(low_count, medium_count, high_count)
if min_count> 0:
    df_low_balanced = df_low.sample(min_count, random_state=42, replace=False)
    df_medium_balanced = df_medium.sample(min_count, random_state=42, replace=False)
    df_high_balanced = df_high.sample(min_count, random_state=42, replace=False)

df_balanced = pd.concat([df_low_balanced, df_medium_balanced, df_high_balanced])
print(df_balanced['discount_bin'].value_counts())

## Remove New Column
df_balanced = df_balanced.drop(columns=['discount_bin'])

## Training
# Features
X = df_balanced[['price', 'original_price', 'shipping']]
y = df_balanced['discount_percentage']
print("Final dataset shape (X):", X.shape)
print("Final labels shape (y):", y.shape)


### Step 3: Regression Modeling
## 80% training, 20% testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

## Train Linear Regression Model
# Convert Colums to Numeric - X_train
X_train['shipping'] = X_train['shipping'].replace(['Free shipping '], 1.0)
X_train['shipping'] = X_train['shipping'].replace(['Shipping info inavailable'], 2.0)
X_train['shipping'] = pd.to_numeric(X_train['shipping'], errors='coerce')
X_train.fillna(0, inplace=True)

# Convert Colums to Numeric - X_test
X_test['shipping'] = X_test['shipping'].replace(['Free shipping '], 1.0)
X_test['shipping'] = X_test['shipping'].replace(['Shipping info inavailable'], 2.0)
X_test['shipping'] = pd.to_numeric(X_test['shipping'], errors='coerce')
X_test.fillna(0, inplace=True)

#Train Model
model_b = LinearRegression()
model_b.fit(X_train, y_train)

## Predictions
y_pred_b = model_b.predict(X_test)

## Evaluate Model
# Mean Absolute Error
mae = mean_absolute_error(y_test, y_pred_b)

# Mean Squared Error
mse = mean_squared_error(y_test, y_pred_b)

# Root Mean Squared Error
rmse = np.sqrt(mse)

# R Squared Score
r2 = r2_score(y_test, y_pred_b)

print(f"Mean Absolute Error: {mae:.2f}")
print(f"Mean Squared Error: {mse:.2f}")
print(f"Root Mean Squared Error: {rmse:.2f}")
print(f"R Squared Score: {r2:.2f}")

### Step 4: Visual Evaluation
## Scatter Plot - Predicted vs. Actual discounted prices
plt.figure(figsize = (8, 6))
sns.scatterplot(x=y_test, y=y_pred_b)
plt.xlabel('Actual Discounted Percentage')
plt.ylabel('Predicted Discount Percentage')
plt.title('Actual vs. Predicted Discounted Percenatge')
plt.show()

## Plot Residuals
error = y_test - y_pred_b
plt.figure(figsize = (8, 6))
sns.scatterplot(x=y_pred_b, y=error)
plt.xlabel('Predicted Discount Percentage')
plt.ylabel('Error (Actual - Predicted)')
plt.title ('Error (Residual) Plot')
plt.show()

### Step 5: Applying Model to Incomplete Data

## Remove Discount Percentage Column
df_incomplete = df.drop(columns=['discount_percentage'])


## Select 20 Products with Specific Columns
df_sample = df_incomplete.sample(n=20, random_state=42)

## Train Model
# Features
X_new = df_sample[['price', 'original_price', 'shipping']]

#Convert Shipping into Numeric
X_new['shipping'] = X_new['shipping'].replace(['Free shipping '], 1.0)
X_new['shipping'] = X_new['shipping'].replace(['Shipping info inavailable'], 2.0)
X_new['shipping'] = pd.to_numeric(X_new['shipping'], errors='coerce')
X_new.fillna(0, inplace=True)


# Predict
predicted_discounts = model_b.predict(X_new)

## Present Results in Table
df_results = pd.DataFrame({
    'Title': df_sample['title'],
    'Price': df_sample['price'],
    'Original Price': df_sample['original_price'],
    'Shipping': df_sample['shipping'],
    'Predicted Discount (%)': predicted_discounts 
})
print(df_results)

