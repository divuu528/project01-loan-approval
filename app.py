import config
import pickle
import pandas as pd
from flask import Flask, render_template, request, redirect, url_for

app = Flask(__name__)

# --- Load Model and Scaler ---
# Note: Assuming config.py and the artifacts directory are correctly set up.
try:
    with open(config.model_path, "rb") as f:
        model = pickle.load(f)
except FileNotFoundError:
    print(f"Error: Model file not found at {config.model_path}. Please check config.py.")
    model = None

try:
    with open(config.model_path_scaler, "rb") as f:
        model_scaler = pickle.load(f)
except FileNotFoundError:
    print(f"Error: Scaler file not found at {config.model_path_scaler}. Please check config.py.")
    model_scaler = None

# Encoding/Decoding maps
dec_target = {1: 'Approved', 0: 'Rejected'}

# --- Prediction Logic ---
@app.route('/', methods=['GET'])
def home():
    """Renders the input form."""
    # Render with no result or input data initially
    return render_template('loan_data_dashboard.html', result=None, input_data={})

@app.route('/predict', methods=['POST'])
def predict():
    """Handles the form submission, predicts, and returns the prediction along with input data."""
    if not model or not model_scaler:
        return render_template("loan_data_dashboard.html", result="Error: ML Model or Scaler not loaded.", input_data={})
        
    try:
        # 1. Get data from standard HTML form (request.form) and store it in a dictionary
        # We read all values as strings first to easily pass back to the HTML input fields
        form_data_str = {
            'no_of_dependents': request.form.get('no_of_dependents', ''),
            'education': request.form.get('education', ''),
            'self_employed': request.form.get('self_employed', ''),
            'income_annum': request.form.get('income_annum', ''),
            'loan_amount': request.form.get('loan_amount', ''),
            'loan_term': request.form.get('loan_term', ''),
            'cibil_score': request.form.get('cibil_score', ''),
            'residential_assets_value': request.form.get('residential_assets_value', ''),
            'commercial_assets_value': request.form.get('commercial_assets_value', ''),
            'luxury_assets_value': request.form.get('luxury_assets_value', ''),
            'bank_asset_value': request.form.get('bank_asset_value', '')
        }

        # 2. Prepare data for model prediction (convert to numerical/int types)
        # We use a separate dictionary for prediction after conversion
        form_data_num = {k: int(v) for k, v in form_data_str.items() if v}

        column = list(form_data_num.keys())
        
        # 3. Convert to DataFrame and Scale
        df = pd.DataFrame([form_data_num], columns=column)
        user_scaled = model_scaler.transform(df)

        # 4. Predict and Decode
        predicted_value = int(model.predict(user_scaled)[0])
        predicted_status = dec_target.get(predicted_value, "Unknown Status")

        # 5. Render template, passing BOTH the result AND the original string input data
        return render_template(
            "loan_data_dashboard.html", 
            result=predicted_status, 
            input_data=form_data_str
        )

    except Exception as e:
        print(f"Prediction Error: {e}")
        # Pass back the input data even if an error occurs
        return render_template(
            "loan_data_dashboard.html", 
            result=f"Error processing input: {e}", 
            input_data=form_data_str 
        )


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)
