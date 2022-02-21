import numpy as np
import pandas as pd
import yaml
from datetime import timedelta
import matplotlib.pyplot as plt

def yaml_loader(filepath):
    with open(filepath, 'r') as file_descriptor:
        data = yaml.full_load(file_descriptor)
    return data


def data_check(df):
    column_names = ['Column Name', 'Unique entries', 'Unique entries percent', 'Missing values', 'Missing values percent']
    all_rows = []
    total_rows = df.shape[0]
    columns = df.columns
    for column in columns:
        unique_entries = df[column].nunique()
        unique_entries_percent = round(100 * (unique_entries / total_rows), 1)
        missing_entries = df[column].isna().sum()
        missing_entries_percent = round(100 * (missing_entries / total_rows), 1)
        all_rows.append([column, unique_entries, unique_entries_percent, missing_entries, missing_entries_percent])
    df_checked = pd.DataFrame(all_rows, columns=column_names)
    return df_checked


def list_diff(list1, list2):
    return list(set(list1) - set(list2)) + list(set(list2) - set(list1))


def reduce_cardinality(df, column, threshold):
    categories_renamed = []
    total = 0
    threshold_value = round(threshold * df.shape[0])
    # Convert Pandas series of categories and their frequency (value_counts) to a Python dicitonary
    df_dict = df[column].value_counts().to_dict()
    # Iterate over the dictionary and add the categories (keys) until their frequencies (values) exceed the threshold
    for k, v in df_dict.items():
        total += v
        if total <= threshold_value:
            categories_renamed.append(k)
    return categories_renamed


def create_one_hot_dataframe(df, categorical_list):
    for feature in categorical_list:
        tmp_df = pd.get_dummies(df[feature], prefix=feature)
        # Merge one-hot encoded columns with original DataFrame
        df = pd.merge(left=df,
                      right=tmp_df,
                      left_index=True,
                      right_index=True)
        # Drop the feature that has just been encoded
        df = df.drop(columns=feature)
    return df


def optimal_number_of_trees(results, metric):
    # results contains an ordered dictionary
    # validation_0 is the training loss
    # validation_1 is the validation loss
    d = results["validation_1"][metric]  # list of validation logloss values
    min_value = min(d)
    min_index = d.index(min_value)
    return min_index
