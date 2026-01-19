# Import Libraries
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def data_load(path):
    '''
    Method to load excel file
    
    :param path: File Path
    '''
    # Code to check if path exist and is excel file
    if path and path.endswith('.xlsx') and os.path.exists(path):
        df = pd.read_excel(path)
        clean_df = data_clean(df)
        fix_df = fill_missing_values(clean_df)

        return fix_df
    else:
        print("File path not found")

def data_clean(df):
    '''
    Method to remove extra columns and renaming the wrong columns
    
    :param df: Data frame
    :return df: New/Changed data frame
    '''
    # Drop extra columns
    df = df.drop(columns=['Unnamed: 0', '_id', 'crawled_at', 'images', 'pid', 'product_details', 'url'], errors='ignore')
    
    # Swap columns heading
    df.rename(columns={'description':'discount', 'discount':'description'}, inplace=True)
    
    # Drop text columns
    df=df.drop(columns=['description','title', 'sub_category'], errors='ignore')
    
    if 'discount' in df.columns:
        df['discount'] = df['discount'].str.replace('% off', '', regex=False)

    return df

def fill_missing_values(df):
    """
    Fill missing values in numeric and categorical columns and visualize distributions 
    of numeric columns to check skewness before imputation.

    - Converts numeric columns to appropriate types.
    - Plots log-transformed histograms for numeric columns
    - Fills missing numeric values:
        * 'actual_price' with mean and 'average_rating','discount', 'selling_price' with median
    - Fills missing categorical values:
        * 'brand' & 'seller' with 'Unknown'

    :param df: pandas DataFrame containing product data
    :return: DataFrame with missing values filled
    """
    print("Missing Values: Before \n", df.isnull().sum())

    # Typecast for histogram plotting
    if 'actual_price' in df.columns:
        df['actual_price'] = df['actual_price'].str.replace(',', '').astype(float)

    df['selling_price'] = pd.to_numeric(df['selling_price'], errors='coerce')
    df['discount'] = pd.to_numeric(df['discount'], errors='coerce')

    # Visualizing distributions to check skewness
    num_cols = ['actual_price', 'average_rating', 'discount', 'selling_price']

    plt.figure(figsize=(15,4))
    for i, col in enumerate(num_cols):
        plt.subplot(1,4,i+1)
        sns.histplot(np.log1p(df[col].dropna()), bins=30, kde=True)
        plt.title(f'Log Distribution of {col}')
        plt.xlabel(col)
        plt.ylabel('Count')

    plt.show()

    # Filling numeric and categorical missing values
    df.fillna(
    {
        'actual_price': df['actual_price'].mean(), 
        'average_rating': df['average_rating'].median(), 
        'discount': df['discount'].median(), 
        'selling_price': df['selling_price'].median(),
        'brand': 'Unknown',
        'seller': 'Unknown'
    }, inplace=True)

    print("Missing Values: After \n", df.isnull().sum())

    return df
