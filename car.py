import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load dataset
df = pd.read_csv("car data.csv")

# Data preprocessing
df['Car_Age'] = 2025 - df['Year']  # Creating a new feature for car age
df.drop(columns=['Car_Name', 'Year'], inplace=True)  # Dropping unnecessary columns

# Encoding categorical variables
categorical_features = ['Fuel_Type', 'Selling_type', 'Transmission']
ohe = OneHotEncoder(drop='first', sparse_output=False)
categorical_encoded = ohe.fit_transform(df[categorical_features])
categorical_df = pd.DataFrame(categorical_encoded, columns=ohe.get_feature_names_out())
df = pd.concat([df.drop(columns=categorical_features), categorical_df], axis=1)

# Splitting data into training and testing sets
X = df.drop(columns=['Selling_Price'])
y = df['Selling_Price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scaling numerical features
scaler = StandardScaler()
X_train[['Present_Price', 'Driven_kms', 'Car_Age']] = scaler.fit_transform(X_train[['Present_Price', 'Driven_kms', 'Car_Age']])
X_test[['Present_Price', 'Driven_kms', 'Car_Age']] = scaler.transform(X_test[['Present_Price', 'Driven_kms', 'Car_Age']])

# Training the model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Making predictions
y_pred = model.predict(X_test)

# Evaluating the model
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

# Printing evaluation metrics
print(f"Mean Absolute Error: {mae:.2f}")
print(f"Mean Squared Error: {mse:.2f}")
print(f"Root Mean Squared Error: {rmse:.2f}")
print(f"R-squared Score: {r2:.2f}")

# Feature importance plot
feature_importance = model.feature_importances_
features = X.columns
sns.barplot(x=feature_importance, y=features)
plt.xlabel("Feature Importance")
plt.ylabel("Features")
plt.title("Feature Importance of Car Price Prediction Model")
