import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set style and seed
sns.set(style="whitegrid")
np.random.seed(42)

# 1. Generate synthetic data
n = 500
data = {
    "Customer_ID": np.arange(1001, 1001 + n),
    "Age": np.random.randint(18, 65, size=n),
    "Gender": np.random.choice(["Male", "Female"], size=n),
    "Category": np.random.choice(["Electronics", "Clothing", "Home", "Sports"], size=n),
    "Purchase_Amount": np.round(np.abs(np.random.normal(3000, 1500, size=n)), 2),
    "Rating": np.round(np.random.uniform(1, 5, size=n), 1)
}
df = pd.DataFrame(data)

# 2. Display summary statistics
print("=== Summary Statistics ===\n")
print(df.describe(include='all'))

# 3. Visualizations

# a. Purchase Amount Distribution
plt.figure(figsize=(10, 6))
sns.histplot(df["Purchase_Amount"], bins=30, kde=True, color='skyblue')
plt.title("Purchase Amount Distribution")
plt.xlabel("Purchase Amount")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()

# b. Gender vs Average Purchase
plt.figure(figsize=(8, 5))
sns.barplot(x="Gender", y="Purchase_Amount", data=df, estimator=np.mean, palette="pastel")
plt.title("Average Purchase Amount by Gender")
plt.ylabel("Average Purchase")
plt.tight_layout()
plt.show()

# c. Category-wise Purchase Distribution
plt.figure(figsize=(10, 6))
sns.boxplot(x="Category", y="Purchase_Amount", data=df, palette="Set2")
plt.title("Category-wise Purchase Distribution")
plt.tight_layout()
plt.show()

# d. Rating Distribution
plt.figure(figsize=(8, 5))
sns.histplot(df["Rating"], bins=20, kde=True, color='orange')
plt.title("Product Rating Distribution")
plt.xlabel("Rating")
plt.ylabel("Count")
plt.tight_layout()
plt.show()

# e. Correlation Heatmap
plt.figure(figsize=(6, 4))
sns.heatmap(df[["Age", "Purchase_Amount", "Rating"]].corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.show()
