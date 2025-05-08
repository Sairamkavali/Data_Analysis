import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set style and seed
sns.set(style="darkgrid")
np.random.seed(7)

# 1. Generate synthetic data
n = 300
data = {
    "Student_ID": np.arange(1, n + 1),
    "Gender": np.random.choice(["Male", "Female"], size=n),
    "Hours_Studied": np.round(np.random.normal(5, 2, size=n), 1),
    "Parental_Education": np.random.choice(["High School", "Bachelor's", "Master's"], size=n),
    "Test_Preparation": np.random.choice(["Completed", "None"], size=n, p=[0.4, 0.6]),
    "Score": np.clip(np.round(np.random.normal(65, 15, size=n), 1), 0, 100)
}
df = pd.DataFrame(data)

# Adjust scores slightly based on hours studied
df["Score"] += (df["Hours_Studied"] - df["Hours_Studied"].mean()) * 2
df["Score"] = df["Score"].clip(0, 100)

# 2. Summary statistics
print("=== Student Performance Summary ===\n")
print(df.describe(include='all'))

# 3. Visualizations

# a. Score Distribution
plt.figure(figsize=(8, 5))
sns.histplot(df["Score"], bins=20, kde=True, color='lightgreen')
plt.title("Exam Score Distribution")
plt.xlabel("Score")
plt.ylabel("Number of Students")
plt.tight_layout()
plt.show()

# b. Gender vs Average Score
plt.figure(figsize=(6, 4))
sns.barplot(x="Gender", y="Score", data=df, estimator=np.mean, palette="Set2")
plt.title("Average Score by Gender")
plt.tight_layout()
plt.show()

# c. Hours Studied vs Score (Trend Line)
plt.figure(figsize=(8, 5))
sns.regplot(x="Hours_Studied", y="Score", data=df, scatter_kws={'alpha':0.5}, line_kws={"color":"red"})
plt.title("Effect of Study Hours on Score")
plt.tight_layout()
plt.show()

# d. Test Preparation vs Score
plt.figure(figsize=(6, 4))
sns.boxplot(x="Test_Preparation", y="Score", data=df, palette="pastel")
plt.title("Test Preparation Impact")
plt.tight_layout()
plt.show()

# e. Parental Education Impact
plt.figure(figsize=(8, 5))
sns.boxplot(x="Parental_Education", y="Score", data=df, palette="Set3")
plt.title("Parental Education vs Student Score")
plt.tight_layout()
plt.show()
