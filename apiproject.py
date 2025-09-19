from flask import Flask, request, jsonify, render_template
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from flask_cors import CORS
import os

app = Flask(__name__)
CORS(app)

# --------------------------
# Load dataset
# --------------------------
data = pd.read_csv("DataSet.csv")

# Encode categorical columns
encoder_dict = {}
categorical_columns = ["divisions", "States"]

for col in categorical_columns:
    encoder = LabelEncoder()
    data[col] = encoder.fit_transform(data[col])
    encoder_dict[col] = encoder

# Split dataset
X = data.drop("label", axis=1)
y = data["label"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model once
model = DecisionTreeClassifier()
model.fit(X_train, y_train)


# --------------------------
# Routes
# --------------------------
@app.route("/")
def home():
    return render_template("project.html")


@app.route("/predict_crop", methods=["POST"])
def predict_crop():
    try:
        user_input = request.get_json()

        # Encode categorical inputs
        for col in categorical_columns:
            if col in user_input:
                try:
                    user_input[col] = encoder_dict[col].transform([user_input[col]])[0]
                except ValueError:
                    return jsonify({"error": f"Unseen label '{user_input[col]}' for '{col}'"})

        # Convert to DataFrame
        user_df = pd.DataFrame([user_input])

        # Predict
        predicted_label = model.predict(user_df)[0]
        return jsonify({"predicted_crop_label": str(predicted_label)})

    except Exception as e:
        return jsonify({"error": str(e)})


# --------------------------
# Run on Render
# --------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
