# Import Libraries
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
import eda, visualization, model_building   

def main():
    df = eda.data_load("../flipkart_products_data.xlsx")
    df = visualization.show_graphs(df)
    
    df_encoded = model_building.encode_features(df)
    X_train, X_test, y_train, y_test = model_building.split_data(df_encoded)

    scaler = StandardScaler()
    X_train_scaled, X_test_scaled = model_building.standardization(scaler, X_train, X_test)

    # Print feature names used for scaling
    print("Features used in scaler:")
    print(X_train.columns.tolist())

    # Calling Logistic Regression Model(LR)
    lr_model = LinearRegression()
    y_test, y_pred_lr = model_building.model_linear_regression(lr_model, X_train_scaled, X_test_scaled, y_train, y_test)
    visualization.model_output_plot('Logistic Regression', y_test, y_pred_lr)

    # Calling Support Vector Regression Model(SVR)
    svr_model = SVR(kernel='rbf')
    y_test, y_pred_svr_untuned = model_building.model_svr(svr_model, X_train_scaled, X_test_scaled, y_train, y_test)
    visualization.model_output_plot('Untuned Support Vector Regression', y_test, y_pred_svr_untuned)

    # Define parameter grid
    param_grid = {
        'C': [0.1, 1, 10, 100],
        'gamma': ['scale', 0.01, 0.1, 1],
        'epsilon': [0.01, 0.1, 1]
    }
    # Grid search with 5-fold cross-validation
    grid_search = GridSearchCV(estimator=svr_model, param_grid=param_grid, cv=5, scoring='r2', n_jobs=-1, verbose=2)

    # Calling Support Vector Regression Model(SVR) after tuning
    y_test, y_pred_svr_tuned = model_building.tuning_svr(grid_search, X_train_scaled, X_test_scaled, y_train, y_test)
    visualization.model_output_plot('Tuned Support Vector Regression', y_test, y_pred_svr_tuned)

    # Calling K Nearest Neighbor Model(KNN)
    knn_model = KNeighborsRegressor(n_neighbors=5)
    y_test, y_pred_knn = model_building.model_knn(knn_model, X_train_scaled, X_test_scaled, y_train, y_test)
    visualization.model_output_plot('K Nearest Neighbor Model', y_test, y_pred_knn)

    # Save models and scaler
    joblib.dump(scaler, "models/scaler.pkl")
    joblib.dump(lr_model, "models/lr_model.pkl")
    joblib.dump(svr_model, "models/svr_model.pkl")
    joblib.dump(knn_model, "models/knn_model.pkl")

if __name__ == "__main__":
    main()
