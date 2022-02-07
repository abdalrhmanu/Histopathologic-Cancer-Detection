import pandas as pd
import matplotlib.pyplot as plt

def check_null_values(df):
    null_df = pd.DataFrame({'columns': df.columns,
                            'percent_null': df.isnull().sum() * 100 / len(df),
                            'percent_zero': df.isin([0]).sum() * 100 / len(df),
                            'percent_one': df.isin([1]).sum() * 100 / len(df)
                            })
    return null_df

def count_values(df, label):
    return df[label].value_counts()


def GeneratePlot(figsize, df, kind):
    plt.figure(figsize=figsize)
    df.value_counts().plot(kind=kind)
