import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from graphviz import Source
from sklearn.tree import export_graphviz


# Pandas df exploration functions

def print_null_ct(df):
    # print how many null values each column has, if any:
        print('Count of Null Values per Column, if any:\n\n{}'.format(df.isnull().sum()[df.isnull().sum() > 0]))

def print_unique_ct(df):
    # print how many unique values each column has:
    print('Count of Unique Values per Column:\n')
    for col in df.columns:
        print('{}: {}'.format(col, len(df[col].unique())))

def get_cols_of_type(df, type):
    # print names of columns of given type
    cols = list(df.select_dtypes(type).columns)
    print('{} Columns ({}): \n{}'.format(type, len(cols), cols))
    return cols


# Pipeline functions

def encode_sex(df):
    # encoder = OneHotEncoder(sparse=False)
    # encoder.fit_transform(df[['sex', 'embarked']])
    # encoder.categories_
    df['Sex'] = df['Sex'].map({'male': 1, 'female': 0})
    return df

def pipeline(df):
    updated_df = df.copy()
    updated_df = encode_sex(updated_df)
    return updated_df


# Plotting functions

def plot_pie(series, fig, ax):
    # fig, ax = plt.subplots(figsize=(8,8))

    series.value_counts().plot.pie(ax=ax, autopct='%1.2f%%')

    plt.rcParams['font.size'] = 18
    
    fig.tight_layout()
    return fig, ax

def plot_counts_bygroup(df, features, groupby, fig, axs):
    # fig, axs = plt.subplots(6, 4, figsize=(14,18))

    for feature, ax in zip(features, axs.flatten()[:len(features)]):
        ax = sns.countplot(data=df, x=feature, hue=groupby, ax=ax)
        ax.legend_.remove()

    fig.tight_layout()
    return fig, axs

def plot_topN_features(feature_importances, feature_list, N):
    # Plot the feature importance
    idxes = np.argsort(-feature_importances)
    feature_list[idxes]
    rev_sort_feature_importances = feature_importances[idxes]
    rev_sort_feature_cols = feature_list[idxes]

    feat_scores = pd.DataFrame({'Fraction of Samples Affected' : rev_sort_feature_importances[:N]},
                               index=rev_sort_feature_cols[:N])
    feat_scores = feat_scores.sort_values(by='Fraction of Samples Affected')
    feat_scores.plot(kind='barh')
    
    plt.title('Feature Importances', size=25)
    plt.ylabel('Features', size=25)
    return plt

def plot_tree(tree, feature_list, out_file=None):
    # Source(plot_tree(tree, feature_list, out_file=None)) to print in Jupyter nb
    return export_graphviz(tree, out_file=out_file, feature_names=feature_list)