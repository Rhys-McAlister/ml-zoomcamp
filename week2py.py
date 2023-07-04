import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

url = "https://raw.githubusercontent.com/alexeygrigorev/datasets/master/housing.csv"


def fetch_data(url):
    df = pd.read_csv(url)
    return df

df = fetch_data(url=url)

sns.histplot(df.median_house_value, bins=50)
plt.show()

