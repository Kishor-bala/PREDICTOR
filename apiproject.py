from flask import Flask, request, jsonify, render_template
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from flask_cors import CORS
import os

app = Flask(__name__)
CORS(app)

# --------------------------
# Load dataset and prepare encoders/options
# --------------------------
DATA_PATH = "DataSet.csv"
data = pd.read_csv(DATA_PATH)

categorical_columns = ['divisions', 'States']
options = {}
encoder_dict = {}

# Keep original (cleaned) string options, then encode columns for model
for col in categorical_columns:
    # Make sure values are strings and stripped of extra whitespace
    col_series = data[col].dropna().astype(str).str.strip()
    unique_vals = sorted(col_series.unique().tolist())
    options[col] = unique_vals

    encoder = LabelEncoder()
    # fit encoder on cleaned strings
    encoder.fit(col_series)
    # replace column in dataframe with encoded ints (use cleaned version)
    data[col] = encoder.transform(data[col].astype(str).str.strip())
    encoder_dict[col] = encoder

# --------------------------
# Prepare features/labels and train model once
# --------------------------
if 'label' not in data.columns:
    raise RuntimeError("DataSet.csv must contain a 'label' column for target.")

X = data.drop('label', axis=1)
y = data['label']

# Train a simple Decision Tree (trained once at startup)
model = DecisionTreeClassifier()
model.fit(X, y)


# --------------------------
# Routes
# --------------------------
@app.route('/')
def home():
    return render_template("project.html")


@app.route('/options', methods=['GET'])
def get_options():
    # Return the valid options for dropdowns (strings exactly as in dataset)
    return jsonify(options)


@app.route('/predict_crop', methods=['POST'])
def predict_crop():
    try:
        user_input = request.get_json()
        if not user_input:
            return jsonify({"error": "Invalid JSON input"}), 400

        # Validate and convert numeric fields
        for num in ['temperature', 'humidity', 'ph', 'rainfall']:
            if num not in user_input:
                return jsonify({"error": f"Missing field '{num}'"}), 400
            try:
                user_input[num] = float(user_input[num])
            except Exception:
                return jsonify({"error": f"Field '{num}' must be numeric"}), 400

        # Encode categorical inputs using fitted encoders (use stripped string)
        for col in categorical_columns:
            if col not in user_input:
                return jsonify({"error": f"Missing field '{col}'"}), 400
            val = str(user_input[col]).strip()
            try:
                user_input[col] = int(encoder_dict[col].transform([val])[0])
            except ValueError:
                # value not seen in training set
                return jsonify({"error": f"Unseen label '{val}' for '{col}'"}), 400

        # Build DataFrame in the same column order as training X
        user_df = pd.DataFrame([user_input])
        user_df = user_df[X.columns]  # this ensures correct column order

        # Predict
        predicted_label = model.predict(user_df)[0]
        return jsonify({"predicted_crop_label": str(predicted_label)})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# --------------------------
# Run (for local dev); Render provides PORT env var
# --------------------------
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
