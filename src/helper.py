import matplotlib.pyplot as plt


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

def plot_pie(series):
    fig, ax = plt.subplots(figsize=(8,8))

    series.value_counts().plot.pie(ax=ax, autopct='%1.2f%%')

    plt.rcParams['font.size'] = 18
    plt.setp(series.index, fontsize=15)
    fig.tight_layout()
    return fig, ax
