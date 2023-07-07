import pandas as pd
import numpy as np
from typing import List


def fetch_data(url: str) -> pd.DataFrame:
    df = pd.read_csv(url)
    return df

def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = df.columns.str.lower().str.replace(' ', '_')
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
    

