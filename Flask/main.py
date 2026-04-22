from flask import Flask, request, jsonify
from model import predict_iris      
from db_config import save_prediction 
import os
from flask_cors import CORS
from dotenv import load_dotenv

load_dotenv()  

app = Flask(__name__)
CORS(app)



@app.route("/")
def intro():
    return jsonify({"msg": "we are on flask server" , 
                    "PORT" : 5000
                    })

@app.route("/predict", methods=["POST"])
def get_prediction():
    # 1. Get the JSON data sent by the user
    data = request.get_json()
    
    # Check if data was provided
    if not data:
        return jsonify({"error": "No JSON data provided"}), 400

    # Extract the values from the JSON dictionary
    sl = data.get("sepal_len")
    sw = data.get("sepal_width")
    pl = data.get("petal_len")
    pw = data.get("petal_width")
    
    # 2. Get the prediction from your model.py
    result = predict_iris(sl, sw, pl, pw)
    
    # 3. Save to MySQL using your db_config.py
    db_success = save_prediction(sl, sw, pl, pw, result)
    
    # 4. Return the response
    return jsonify({
        "success": True,
        "prediction_class": result,
        "saved_to_db": db_success
    })

if __name__ == "__main__":
    # Run the standard Flask development server

    port = int(os.getenv("FLASK_PORT", 5000))

    app.run( port=port , debug=True)