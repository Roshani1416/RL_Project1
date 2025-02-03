# Exploratory Data Analysis (EDA) for Dynamic Pricing

# Install Dependencies (if needed)
# !pip install pandas matplotlib seaborn

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load Dataset
df = pd.read_csv("woolballhistory.csv")  # Change to your dataset path
df['Report Date'] = pd.to_datetime(df['Report Date'])

def basic_info():
    print("\nDataset Overview:\n")
    print(df.info())
    print("\nFirst Few Rows:\n", df.head())
    print("\nMissing Values:\n", df.isnull().sum())
    print("\nSummary Statistics:\n", df.describe())

basic_info()

# Visualizing Price Distribution
plt.figure(figsize=(8, 5))
sns.histplot(df['Product Price'], bins=30, kde=True, color='blue')
plt.title("Product Price Distribution")
plt.xlabel("Price")
plt.ylabel("Frequency")
plt.show()

# Time Series Plot - Sales over Time
plt.figure(figsize=(12, 6))
sns.lineplot(x=df['Report Date'], y=df['Total Sales'], label='Total Sales', color='green')
plt.title("Total Sales Over Time")
plt.xlabel("Date")
plt.ylabel("Total Sales")
plt.xticks(rotation=45)
plt.show()

# Price vs. Sales Relationship
plt.figure(figsize=(8, 6))
sns.scatterplot(x=df['Product Price'], y=df['Total Sales'], alpha=0.5)
plt.title("Price vs Total Sales")
plt.xlabel("Product Price")
plt.ylabel("Total Sales")
plt.show()

# Correlation Heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title("Feature Correlation Heatmap")
plt.show()

print("EDA Completed Successfully!")