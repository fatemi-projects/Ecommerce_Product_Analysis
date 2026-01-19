# Import Libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

def encode_features(df):
    """
    Method to encode the categorical features into numeric for model training
    
    :param df: Data Frame
    """
    # Binary encoding for category
    df['category'] = df['category'].apply(lambda x: 1 if x == 'Clothing and Accessories' else 0)

    # Top N brands (e.g., top 10), rest as 'Other'
    top_brands = df['brand'].value_counts().nlargest(10).index
    df['brand'] = df['brand'].apply(lambda x: x if x in top_brands else 'Other')
    brand_dummies = pd.get_dummies(df['brand'], prefix='brand', drop_first=True)

    # Top N sellers (e.g., top 10), rest as 'Other'
    top_sellers = df['seller'].value_counts().nlargest(10).index
    df['seller'] = df['seller'].apply(lambda x: x if x in top_sellers else 'Other')
    seller_dummies = pd.get_dummies(df['seller'], prefix='seller', drop_first=True)

    # Combine dummies with original df
    df_encoded = pd.concat([df.drop(['brand','seller'], axis=1), brand_dummies, seller_dummies], axis=1)

    return df_encoded

def split_data(df_encoded):
    """
    Method to split encoded data into train and test for model training
    
    :param df_encoded: Encoded Data Frame
    """
    X = df_encoded.drop('selling_price', axis=1) 
    y = df_encoded["selling_price"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    return X_train, X_test, y_train, y_test

def standardization(scaler, X_train, X_test):
    """
    Standardize features by removing the mean and scaling to unit variance.
    
    :param scaler: StandardScaler object
    :param X_train: Training feature set
    :param X_test: Test feature set 
    :return: Scaled Train and Test Data
    """
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Convert scaled array to DataFrame
    X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=X_train.columns)
    X_train_scaled_df.head(3)

    return X_train_scaled, X_test_scaled

def model_linear_regression(model, X_train_scaled, X_test_scaled, y_train, y_test):
    """
    Train and evaluate a Linear Regression model.

    :param model: LinearRegression object
    :param X_train_scaled: Scaled training features
    :param X_test_scaled: Scaled test features
    :param y_train: Training target values
    :param y_test: Test target values
    :return: y_test and predicted values y_pred
    """
    # Fit on training data
    model.fit(X_train_scaled, y_train)

    # Predict on test data
    y_pred_lr = model.predict(X_test_scaled)

    # Evaluate
    mse_lr = mean_squared_error(y_test, y_pred_lr)
    rmse_lr = np.sqrt(mse_lr)
    r2_lr = r2_score(y_test,y_pred_lr)

    print(f"LR Mean Squared Error: {mse_lr:.2f}")
    print(f"LR Root Mean Squared Error: {rmse_lr:.2f}")
    print(f"LR R-squared: {r2_lr:.2f}")

    return y_test, y_pred_lr

def model_svr(model, X_train_scaled, X_test_scaled, y_train, y_test):
    """
    Train and evaluate a Support Vector Regressor model.

    :param model: SVR object
    :param X_train_scaled: Scaled training features
    :param X_test_scaled: Scaled test features
    :param y_train: Training target values
    :param y_test: Test target values
    :return: y_test and predicted values y_pred
    """
    # Fit on training data
    model.fit(X_train_scaled, y_train)

    # Predict on test data
    y_pred_svr_untuned = model.predict(X_test_scaled)

    # Evaluate
    mse_svm = mean_squared_error(y_test, y_pred_svr_untuned)
    rmse_svm = np.sqrt(mse_svm)
    r2_svm = r2_score(y_test, y_pred_svr_untuned)

    print(f"SVM Mean Squared Error: {mse_svm:.2f}")
    print(f"SVM Root Mean Squared Error: {rmse_svm:.2f}")
    print(f"SVM R-squared: {r2_svm:.2f}")

    return y_test, y_pred_svr_untuned

def tuning_svr(grid_search_obj, X_train_scaled, X_test_scaled, y_train, y_test):
    """
    Perform hyperparameter tuning for an SVR model using a GridSearchCV object 
    and evaluate the best model on test data.

    :param grid_search_obj: Initialized GridSearchCV object for SVR
    :param X_train_scaled: Scaled training features
    :param X_test_scaled: Scaled test features
    :param y_train: Training target values
    :param y_test: Test target values
    :return: y_test and predicted values from the best tuned SVR
    """
    # Fit on training data
    grid_search_obj.fit(X_train_scaled, y_train)

    # Best parameters
    print("Best parameters:", grid_search_obj.best_params_)

    # Predict on test data using best estimator
    best_svm = grid_search_obj.best_estimator_
    y_pred_svm_tuned = best_svm.predict(X_test_scaled)

    # Evaluate
    mse_svm = mean_squared_error(y_test, y_pred_svm_tuned)
    rmse_svm = np.sqrt(mse_svm)
    r2_svm = r2_score(y_test, y_pred_svm_tuned)

    print(f"Tuned SVM Mean Squared Error: {mse_svm:.2f}")
    print(f"SVM Root Mean Squared Error: {rmse_svm:.2f}")
    print(f"Tuned SVM R-squared: {r2_svm:.2f}")

    return y_test, y_pred_svm_tuned

def model_knn(model, X_train_scaled, X_test_scaled, y_train, y_test):
    """
    Train and evaluate a K Nearest Neighbor Model.

    :param model: KNN object
    :param X_train_scaled: Scaled training features
    :param X_test_scaled: Scaled test features
    :param y_train: Training target values
    :param y_test: Test target values
    :return: y_test and predicted values y_pred
    """
    # Fit on training data
    model.fit(X_train_scaled, y_train)

    # Predict on test data
    y_pred_knn = model.predict(X_test_scaled)

    # Evaluate
    mse_knn = mean_squared_error(y_test, y_pred_knn)
    rmse_knn = np.sqrt(mse_knn)
    r2_knn = r2_score(y_test, y_pred_knn)

    print(f"KNN Mean Squared Error: {mse_knn:.2f}")
    print(f"KNN Root Mean Squared Error: {rmse_knn:.2f}")
    print(f"KNN R-squared: {r2_knn:.2f}")

    return y_test, y_pred_knn
