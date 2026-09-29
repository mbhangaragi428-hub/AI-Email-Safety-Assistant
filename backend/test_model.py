import joblib

# Load the trained model
model = joblib.load("model/spam_model.pkl")

# Load the TF-IDF vectorizer
vectorizer = joblib.load("model/tfidf_vectorizer.pkl")

# Ask user for an email
email = input("\nEnter an email to check: ")

# Convert email into numbers
email_vectorized = vectorizer.transform([email])

# Make prediction
prediction = model.predict(email_vectorized)[0]

# Display result
print("\n-----------------------------")
print("AI EMAIL SAFETY RESULT")
print("-----------------------------")

if prediction == "spam":
    print("Prediction: SPAM")
    print("Risk Level: HIGH")
else:
    print("Prediction: HAM")
    print("Risk Level: LOW")

print("-----------------------------")