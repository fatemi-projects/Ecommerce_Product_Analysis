from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import joblib
import os

app = Flask(__name__)

# ---------------- PATHS ----------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "python_files", "models")

# ---------------- LOAD MODELS ----------------
scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
lr_model = joblib.load(os.path.join(MODEL_DIR, "lr_model.pkl"))
svr_model = joblib.load(os.path.join(MODEL_DIR, "svr_model.pkl"))
knn_model = joblib.load(os.path.join(MODEL_DIR, "knn_model.pkl"))

# ---------------- FEATURE LIST (ORDER MATTERS) ----------------
FEATURE_COLUMNS = [
    'actual_price', 'average_rating', 'category', 'discount', 'out_of_stock',
    'brand_Amp', 'brand_Black Beat', 'brand_ECKO Unl', 'brand_Free Authori',
    'brand_Keo', 'brand_Other', 'brand_Pu', 'brand_REEB', 'brand_True Bl', 'brand_Unknown',
    'seller_AMALGUS ENTERPRISE', 'seller_ARBOR', 'seller_ArvindTrueBlue',
    'seller_BioworldMerchandising', 'seller_Black Beatle',
    'seller_Keoti', 'seller_Other', 'seller_RetailNet',
    'seller_SandSMarketing', 'seller_Unknown'
]

# Dropdown options
BRANDS = [
    'Amp', 'Black Beat', 'ECKO Unl', 'Free Authori', 'Keo', 'Other', 'Pu', 'REEB', 'True Bl', 'Unknown'
]

SELLERS = [
    'AMALGUS ENTERPRISE', 'ARBOR', 'ArvindTrueBlue','BioworldMerchandising', 'Black Beatle',
    'Keoti', 'Other', 'RetailNet', 'SandSMarketing', 'Unknown'
]

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    model_name = None
    input_values = {}

    if request.method == "POST":
        # -------- FORM VALUES --------
        actual_price = float(request.form["actual_price"])
        average_rating = float(request.form["average_rating"])
        discount = float(request.form["discount"])
        category_text = request.form["category"]
        out_of_stock = int(request.form.get("out_of_stock", 0))
        brand = request.form["brand"]
        seller = request.form["seller"]

        category = 1 if category_text == "Clothing and Accessories" else 0

        # -------- BUILD FEATURE VECTOR --------
        input_dict = {col: 0 for col in FEATURE_COLUMNS}

        input_dict["actual_price"] = actual_price
        input_dict["average_rating"] = average_rating
        input_dict["discount"] = discount
        input_dict["category"] = category
        input_dict["out_of_stock"] = out_of_stock

        # Brand encoding
        brand_col = f"brand_{brand}"
        if brand_col in input_dict:
            input_dict[brand_col] = 1
        else:
            input_dict["brand_Other"] = 1

        # Seller encoding
        seller_col = f"seller_{seller}"
        if seller_col in input_dict:
            input_dict[seller_col] = 1
        else:
            input_dict["seller_Other"] = 1

        # -------- SCALE INPUT --------
        input_df = pd.DataFrame([input_dict])
        input_scaled = scaler.transform(input_df)

        # -------- PREDICT --------
        model_selected = request.form["model"]

        if model_selected == "lr":
            prediction = lr_model.predict(input_scaled)[0]
            model_name = "Linear Regression"

        elif model_selected == "svr":
            prediction = svr_model.predict(input_scaled)[0]
            model_name = "Support Vector Regression"

        elif model_selected == "knn":
            prediction = knn_model.predict(input_scaled)[0]
            model_name = "K-Nearest Neighbors"

        # -------- DISPLAY INPUT VALUES --------
        input_values = {
            "Actual Price": actual_price,
            "Average Rating": average_rating,
            "Discount (%)": discount,
            "Category": category_text,
            "Out of Stock": "Yes" if out_of_stock else "No",
            "Brand": brand,
            "Seller": seller
        }

    return render_template(
        "index.html",
        prediction=prediction,
        model_name=model_name,
        input_values=input_values,
        brands=BRANDS,
        sellers=SELLERS
    )

if __name__ == "__main__":
    app.run(debug=True)
