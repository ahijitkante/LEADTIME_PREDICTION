from flask import Flask, render_template, request
import joblib
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

app = Flask(__name__)

# Load the pre-trained model
model = joblib.load('linear_regression_model.sav')

# Define the product options and mapping to features
products = {
    '1': {'SH_CD': 1, 'CATALOG_NO': 1},
    '2': {'SH_CD': 2, 'CATALOG_NO': 2},
    '3': {'SH_CD': 3, 'CATALOG_NO': 3},
    '4': {'SH_CD': 4, 'CATALOG_NO': 4},
    '5': {'SH_CD': 5, 'CATALOG_NO': 5},
}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    product_id = request.form['product']
    indent_date = request.form['indent_date']

    indent_date_obj = datetime.strptime(indent_date, '%Y-%m-%d')
    indent_year = indent_date_obj.year
    indent_month = indent_date_obj.month
    indent_day = indent_date_obj.day

    # Generate random values for other features
    random_values = {
        'INSUR_ITEM_IND': np.random.randint(0, 2),
        'AR_ITEM_IND': np.random.randint(0, 2),
        'SPC_ITEM_IND': np.random.randint(0, 2),
        'QUALITY_IND': np.random.randint(0, 2),
        'time_line': np.random.random()
    }

    # Prepare the feature array
    features = [
        products[product_id]['SH_CD'],
        products[product_id]['CATALOG_NO'],
        random_values['INSUR_ITEM_IND'],
        random_values['AR_ITEM_IND'],
        random_values['SPC_ITEM_IND'],
        random_values['QUALITY_IND'],
        random_values['time_line'],
        indent_year,
        indent_month,
        indent_day
    ]

    # Convert to DataFrame
    X_new = pd.DataFrame([features], columns=['SH_CD', 'CATALOG_NO', 'INSUR_ITEM_IND', 'AR_ITEM_IND', 'SPC_ITEM_IND', 'QUALITY_IND', 'time_line', 'INDENT_YEAR', 'INDENT_MONTH', 'INDENT_DAY'])

    # Predict lead time
    predicted_lead_time = model.predict(X_new)[0]

    # Calculate the receipt date
    receipt_date = indent_date_obj + timedelta(days=predicted_lead_time)

    return render_template('index.html', prediction=predicted_lead_time, receipt_date=receipt_date.strftime('%Y-%m-%d'))

if __name__ == '__main__':
    app.run(debug=True)
