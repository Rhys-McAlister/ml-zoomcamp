import pandas as pd
import numpy as np
from typing import List
from sklearn.metrics import mutual_info_score
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression

def fetch_data(url: str) -> pd.DataFrame:
    df = pd.read_csv(url)
    return df


def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = df.columns.str.lower().str.replace(' ', '_')
    return df


def clean_numerical_columns(df: pd.DataFrame) -> pd.DataFrame:
    df.totalcharges = pd.to_numeric(df.totalcharges, errors='coerce').fillna(0)
    df.churn = (df.churn == 'yes').astype(int)
    return df


def data_cleaning_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    df = clean_columns(df)
    df = clean_categorical_columns(df)
    df = clean_numerical_columns(df)
    # df = clean_date_columns(df)
    return df

def one_hot_encoding(df: pd.DataFrame, cols_to_transform: List[str]) -> pd.DataFrame:
    """
    Function to one-hot encode a dataframe.

    Args:
        df (pd.DataFrame): dataframe to one-hot encode
        cols_to_transform (List[str]): list of columns to one-hot encode

    Returns:
        pd.DataFrame: one-hot encoded dataframe
    """
    vec_enc = DictVectorizer(sparse=False)
    x_train = vec_enc.fit_transform(df[cols_to_transform].to_dict(orient='records'))
    return x_train, vec_enc






# def calculate_mutual_info(df: pd.DataFrame, target: str, comparsion_columns: list[str]) -> pd.Series:
#     """
#     Function to calculate mutual information between a target and all other columns in a dataframe.
#
#     Args:
#         df (pd.DataFrame): dataframe to calculate mutual information for
#         target (str): target column name
#
#     Returns:
#         pd.DataFrame: dataframe with mutual information values for each column
#     """
#
#     mi_scores = []
#     for col in comparsion_columns:
#         mi_score = mutual_info_score(df[col], df[target])
#         mi_scores.append(mi_score)
#
#     mi_scores = pd.Series(mi_scores, index=comparsion_columns)
#     mi_scores = mi_scores.sort_values(ascending=False)
#     return mi_scores


def calculate_mutual_info(df: pd.DataFrame, target: str, comparsion_columns: list[str]) -> pd.DataFrame:
    """
    Function to calculate mutual information between a target and all other columns in a dataframe.

    Args:
        df (pd.DataFrame): dataframe to calculate mutual information for
        target (str): target column name

    Returns:
        pd.Series: series with mutual information values for each column
    """
    mi_scores = pd.Series([mutual_info_score(df[col], df[target]) for col in comparsion_columns],
                          index=comparsion_columns).sort_values(ascending=False)


    return mi_scores

    # mi_scores = []
    # for col in df.columns:
    #     mi_score = mutual_info_score(df[col], df[target])
    #     mi_scores.append(mi_score)
    #
    # mi_scores = pd.Series(mi_scores, index=df.columns)
    # mi_scores = mi_scores.sort_values(ascending=False)
    # return mi_scores


def clean_categorical_columns(df: pd.DataFrame) -> pd.DataFrame:
    categorical_columns = list(df.dtypes[df.dtypes == 'object'].index)

    for col in categorical_columns:
        df[col] = df[col].str.lower().str.replace(' ', '_')

    return df


def find_string_cols(df: pd.DataFrame) -> List[str]:
    return df.select_dtypes(include='object').columns.tolist()


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = clean_columns(df)
    string_cols = find_string_cols(df)
    for col in string_cols:
        df[col] = df[col].str.lower().str.replace(' ', '_')

    return df


def log_variables(df: pd.DataFrame, cols_to_transform: List[str]) -> pd.DataFrame:
    for col in cols_to_transform:
        df[col] = np.log1p(df[col])
    return df


def train_test_val_split(df: pd.DataFrame,
                         val_split: float = 0.2,
                         test_split: float = 0.2) -> pd.DataFrame:
    """
    Function to split a dataframe into train, test, and validation sets.

    
    Args:
        df (pd.DataFrame): dataframe to split
        val_split (float): percentage of data to use for validation
        test_split (float): percentage of data to use for testing

    Returns:
        pd.DataFrame: train, test, and validation sets

    """
    assert val_split + test_split < 1, "val_split + test_split must be less than 1"

    n_train = int(len(df) * (1 - val_split - test_split))
    n_val = int(len(df) * val_split)
    n_test = len(df) - n_train - n_val

    train_set = df[:n_train]
    test_set = df[n_train:(n_train + n_test)]
    val_set = df[(n_train + n_test):]

    return train_set, test_set, val_set


def impute_missing_values(df: pd.DataFrame, strategy: str) -> pd.DataFrame:
    """
    Function to impute missing values in a dataframe.

    Args:
        df (pd.DataFrame): dataframe to impute
        strategy (str): strategy to use for imputation. 
        Must be one of 'median', 'mean', or 'zero'

    Returns:
        pd.DataFrame: dataframe with imputed values
    """

    missing_cols = df.columns[df.isnull().any()].tolist()

    if strategy == 'median':
        for col in missing_cols:
            df.loc[:, col] = df[col].fillna(df[col].median())
        return df

    elif strategy == 'mean':
        for col in missing_cols:
            df.loc[:, col] = df[col].fillna(df[col].mean())
        return df

    elif strategy == 'zero':
        for col in missing_cols:
            df.loc[:, col] = df[col].fillna(0)
        return df
    else:
        raise ValueError("strategy must be one of 'median', 'mean', or 'zero'")


