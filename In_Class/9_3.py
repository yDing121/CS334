import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


if __name__ == "__main__":
    df = pd.read_csv(f"{__file__}/../../Data/iris_cs334.csv")
    print()
    print(df.head())
    print('--'*10)

    print(f"Median petal length:\t{round(df['petal_length'].median(), 2)}")
    print(f"Mean sepal width of Virginica:\t{round(df.loc[df['variety']=='Virginica', ['sepal_width']].mean(), 2)}")
    print(f"SD of sepal lengths of Setosa:\t{round(df.loc[df['variety']=='Setosa', ['sepal_length']].std(), 2)}")
    # print(df.loc[df['variety']=='Virginica', ['sepal_width']].mean())x

    # plt.figure()
    sns.scatterplot(df, x='petal_width', y='petal_length', hue='variety')
    plt.show()
