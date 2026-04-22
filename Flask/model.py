import joblib

# 1. Load the trained model into memory
# (Make sure the filename matches what you actually saved it as)
try:
    model = joblib.load('iris_logistic_model.joblib')
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")

# 2. Create a function to handle the prediction
def predict_iris(sepal_len, sepal_width, petal_len, petal_width):
    # Scikit-learn models expect a 2D array (a list inside a list) for predictions
    features = [[sepal_len, sepal_width, petal_len, petal_width]]
    
    # The .predict() method returns an array of predictions.
    # Since we are only passing one flower, we grab the first result at index [0]
    prediction = model.predict(features)[0]
    
    # Convert it to a standard Python integer (useful for JSON responses or SQL inserts)
    return int(prediction)


