import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px

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


def plot_img(x, y):

    npimg = x.numpy()  # convert tensor to numpy array
    npimg_tr = np.transpose(npimg, (1, 2, 0))  # Convert to H*W*C shape
    fig = px.imshow(npimg_tr)
    fig.update_layout(coloraxis_showscale=False, title=str(y_grid_train))
    fig.update_xaxes(showticklabels=False)
    fig.update_layout(template='plotly_white', height=200)
    fig.update_layout(margin={"r": 0, "t": 60, "l": 0, "b": 0})
    fig.update_layout(title={'text': str(y), 'y': 0.9,
                      'x': 0.5, 'xanchor': 'center', 'yanchor': 'top'})

    fig.show()
