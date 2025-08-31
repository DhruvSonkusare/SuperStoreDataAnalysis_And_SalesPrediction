import pandas as pd 
import numpy as np
from matplotlib import pyplot as plt

df=pd.read_csv("superstore/superstore.csv")

print(df.head())


print(df.columns)

print(df.info())

print("Missing values:\n", df.isnull().sum())

print(df.shape)

df_cleaned=df.drop_duplicates()



df_cleaned=df_cleaned.dropna()


print(df_cleaned.shape)

numeric_cols=df_cleaned.select_dtypes(include=[np.number]).columns

for col in numeric_cols:
    q1=df_cleaned[col].quantile(0.25)
    q3=df_cleaned[col].quantile(0.75)
    iqr=q3-q1
    lb=q1-1.5*iqr
    ub=q3+1.5*iqr
    df_cleaned = df_cleaned[(df_cleaned[col] >= lb) & (df_cleaned[col] <= ub)]



print(df_cleaned.shape)

stats = df_cleaned.describe().T[['mean', '50%', 'std']]  # 50% = median
stats['var'] = df_cleaned.var(numeric_only=True)
stats.rename(columns={'50%': 'median'}, inplace=True)
print("\nStatistical Summary:\n", stats)

corr_matrix = df.corr(numeric_only=True)
print("\nCorrelation Matrix:\n", corr_matrix)



df[numeric_cols].hist(figsize=(12, 10), bins=20)
plt.suptitle("Histograms of Numerical Features")
plt.show()

plt.figure(figsize=(12, 8))  
for i, col in enumerate(numeric_cols, 1):  
    plt.subplot(len(numeric_cols) // 3 + 1, 3, i)  
    plt.boxplot(df[col].dropna())  
    plt.title(col)  
plt.tight_layout()  
plt.show()

corr_matrix = df[numeric_cols].corr()  # Get correlation matrix

plt.figure(figsize=(10, 6))
plt.imshow(corr_matrix, cmap="coolwarm", interpolation="nearest")
plt.colorbar(label="Correlation")
plt.xticks(range(len(corr_matrix)), corr_matrix.columns, rotation=45, ha="right")
plt.yticks(range(len(corr_matrix)), corr_matrix.columns)
for i in range(len(corr_matrix)):
    for j in range(len(corr_matrix)):
        plt.text(j, i, f"{corr_matrix.iloc[i, j]:.2f}",
                 ha="center", va="center", color="black", fontsize=8)

plt.title("Correlation Heatmap (Matplotlib)")
plt.tight_layout()
plt.show()