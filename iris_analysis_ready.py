import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris

sns.set(style="whitegrid")

try:
    iris_data = load_iris()
    df = pd.DataFrame(data=iris_data.data, columns=iris_data.feature_names)
    df['species'] = pd.Categorical.from_codes(iris_data.target, iris_data.target_names)

    print(df.head())
    print(df.info())
    print(df.isnull().sum())

except Exception as e:
    print("Error loading or processing the dataset:", e)

try:
    print(df.describe())
    grouped = df.groupby('species').mean()
    print(grouped)
    print("Observations:")
    print("1. Species differ mainly in petal length and width.")
    print("2. Sepal length and width vary less between species.")
    print("3. Petal differences can be used for classification.")

except Exception as e:
    print("Error during basic data analysis:", e)

try:
    plt.figure(figsize=(8,5))
    plt.plot(df['sepal length (cm)'][:50], marker='o', linestyle='-')
    plt.title("Sepal Length Trend (first 50 samples)")
    plt.xlabel("Samples")
    plt.ylabel("Sepal Length (cm)")
    plt.show()

    plt.figure(figsize=(8,5))
    grouped['petal length (cm)'].plot(kind='bar', color='skyblue')
    plt.title("Average Petal Length by Species")
    plt.xlabel("Species")
    plt.ylabel("Petal Length (cm)")
    plt.show()

    plt.figure(figsize=(8,5))
    plt.hist(df['sepal width (cm)'], bins=10, color='lightgreen', edgecolor='black')
    plt.title("Sepal Width Distribution")
    plt.xlabel("Sepal Width (cm)")
    plt.ylabel("Frequency")
    plt.show()

    plt.figure(figsize=(8,5))
    sns.scatterplot(x='sepal length (cm)', y='petal length (cm)',
                    hue='species', data=df, s=100)
    plt.title("Sepal Length vs Petal Length")
    plt.xlabel("Sepal Length (cm)")
    plt.ylabel("Petal Length (cm)")
    plt.legend(title="Species")
    plt.show()

except Exception as e:
    print("Error during plotting:", e)

print("Conclusion:")
print("- Exploratory analysis and visualizations show clear patterns among species.")
print("- Petal lengths and widths are the main indicators to differentiate species.")
print("- Plots help identify relationships and trends in the data.")
