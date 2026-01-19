# Import Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def show_graphs(df):
    display_box_plot(df)
    display_count_plot(df)
    df = merge_category(df)
    display_binned_price_relationship(df)

    return df

def display_box_plot(df):
    """
    Method to check outliers using boxplot
    :param df: Dataframe
    """

    num_cols = ['actual_price', 'selling_price', 'discount', 'average_rating']

    plt.figure(figsize=(12,6))  
    for i, col in enumerate(num_cols):
        plt.subplot(1, len(num_cols), i+1)
        sns.boxplot(y=df[col])
        plt.title(f'Boxplot of {col}')
        
    plt.tight_layout()
    plt.show()

def display_count_plot(df):
    """
    Plotting count plot for categorical columns
    :param df: Data Frame
    """

    categorical_cols = ['category', 'out_of_stock']

    plt.figure(figsize=(15,6))
    for i, col in enumerate(categorical_cols):
        plt.subplot(1, 2, i+1)
        sns.countplot(data=df, x=col)
        plt.xticks(rotation=45)
        plt.title(col)
    plt.tight_layout()
    plt.show()

def merge_category(df):
    """
    Merging product categories which has less values into one
    Mapping out_of_stock column values to binary
    :param df: Data Frame
    """
    df['category'] = df['category'].replace(['Footwear', 'Bags, Wallets & Belts', 'Toys'], 'Other')
    df['out_of_stock'] = df['out_of_stock'].map({False:0, True:1})  

    print("Count of category \n", df['category'].value_counts())
    print("\n Count of out of stock \n", df['out_of_stock'].value_counts())

    return df

def display_binned_price_relationship(df):
    """
    Checking relation between Discount and Actual Price
    Checking relation between Selling Price and Actual Price
    :param df: Data Frame
    """
    # create temporary bins
    bins = np.linspace(df['actual_price'].min(), df['actual_price'].max(), 20)

    # use bins directly in groupby
    bin_avg_discount = (
        df.groupby(pd.cut(df['actual_price'], bins), observed=True)['discount']
        .mean()
        .reset_index()
    )

    bin_avg_selling = (
        df.groupby(pd.cut(df['actual_price'], bins), observed=True)['selling_price']
        .mean()
        .reset_index()
    )

    # plots
    plt.figure(figsize=(15,6))

    plt.subplot(1, 2, 1)
    plt.plot(bin_avg_discount.iloc[:, 0].astype(str), bin_avg_discount['discount'], marker='o')
    plt.xticks(rotation=45)
    plt.title('Average Discount vs Actual Price')
    plt.xlabel('Actual Price Bin')
    plt.ylabel('Average Discount')

    plt.subplot(1, 2, 2)
    plt.plot(bin_avg_selling.iloc[:, 0].astype(str), bin_avg_selling['selling_price'], marker='o')
    plt.xticks(rotation=45)
    plt.title('Average Selling Price vs Actual Price')
    plt.xlabel('Actual Price Bin')
    plt.ylabel('Average Selling Price')

    plt.tight_layout()
    plt.show()

def model_output_plot(model_name, y_test, y_pred):
    """
    Plotted model output
    :param y_test: "Actual Selling Price"
    :param y_pred: "Predicted Selling Price"
    """
    plt.figure(figsize=(8,6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.xlabel("Actual Selling Price")
    plt.ylabel("Predicted Selling Price")
    plt.title(model_name + ": Actual vs Predicted Selling Price")
    plt.show()
    