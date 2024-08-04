import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_histograms(df, num_cols):
    df[num_cols].hist(figsize=(14, 14))
    plt.show()

def plot_categorical_analysis(df, cat_cols):
    for i in cat_cols:
        print(f'{i} Value Counts:')
        print(df[i].value_counts(normalize=True))
        print('*' * 40)
    
    for i in cat_cols:
        if i != 'Attrition':
            (pd.crosstab(df[i], df['Attrition'], normalize='index') * 100).plot(kind='bar', figsize=(8, 4), stacked=True)
            plt.ylabel('Percentage Attrition %')
            plt.show()

def plot_correlation_matrix(df, num_cols):
    plt.figure(figsize=(15, 8))
    sns.heatmap(df[num_cols].corr(), annot=True, fmt='0.2f', cmap='YlGnBu')
    plt.show()

def plot_scatter(df, x_col, y_col, hue_col):
    sns.scatterplot(data=df, x=x_col, y=y_col, hue=hue_col)
    plt.show()