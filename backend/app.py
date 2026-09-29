from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import joblib

app = Flask(__name__, static_folder="../frontend")
CORS(app)

# Load trained AI model
model = joblib.load("model/spam_model.pkl")
vectorizer = joblib.load("model/tfidf_vectorizer.pkl")


@app.route("/")
def home():
    return send_from_directory("../frontend", "index.html")


@app.route("/<path:path>")
def frontend_files(path):
    return send_from_directory("../frontend", path)


@app.route("/predict", methods=["POST"])
def predict():

    data = request.get_json()

    email = data.get("email", "")

    if not email:
        return jsonify({
            "error": "Email text is required"
        }), 400

    # Convert email text into numbers
    email_vector = vectorizer.transform([email])

    # Predict
    prediction = model.predict(email_vector)[0]

    # Get prediction probability
    probabilities = model.predict_proba(email_vector)[0]
    confidence = max(probabilities) * 100

    # Determine result and risk
    if prediction == "spam":
        result = "SPAM"
        risk = "HIGH"
    else:
        result = "HAM"
        risk = "LOW"

    # Send result to frontend
    return jsonify({
        "prediction": result,
        "risk": risk,
        "confidence": round(confidence, 2)
    })


if __name__ == "__main__":
    app.run(
        host="127.0.0.1",
        port=5000,
        debug=False
    )