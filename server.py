from flask import Flask, request, jsonify, send_file
import pandas as pd
from flask_cors import CORS
import numpy as np
from sklearn.linear_model import LinearRegression
from dotenv import load_dotenv
import os
import io
from huggingface_hub import InferenceClient
from transformers import pipeline

# Load environment variables from .env
load_dotenv()

app = Flask(__name__)
CORS(app, origins="*")

sheet_id = '1VUOiDUrvtUge8StvOoPecLe0aV4vU_3gYTQH8OAdphw'
url = f'https://docs.google.com/spreadsheets/d/{sheet_id}/gviz/tq?tqx=out:csv'
df = pd.read_csv(url)
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
HF_TOKEN = os.getenv("HF_TOKEN")
MODEL_ID = os.getenv("MODEL_ID")
classifier = pipeline('zero-shot-classification', model='facebook/bart-large-mnli')    

client = InferenceClient(provider="together", api_key=HF_TOKEN)

def detect_ai_or_human(text):
  labels=['Human-Written','AI-Written']
  result = classifier(text, candidate_labels = labels)
  return result['labels'][0], result['scores'][0]


@app.route("/generate-image", methods=["POST"])
def generate_image():
    try:
        # Get the prompt from the request JSON body
        data = request.get_json()
        text = data.get("text", "")
        if not text:
            return jsonify({"error": "Missing 'text' in request"}), 400

        # Generate the image
        image = client.text_to_image(
            text,
            model=MODEL_ID  # Make sure this is set in your .env
        )

        # Save image to in-memory buffer
        img_io = io.BytesIO()
        image.save(img_io, format='PNG')
        img_io.seek(0)

        return send_file(img_io, mimetype='image/png')

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
    
@app.route("/ai-or-human-text", methods=["POST"])
def ai_or_human():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
            
        text = data.get("text", "")
        if not text:
            return jsonify({"error": "Missing 'text' in request"}), 400
            
        label, score = detect_ai_or_human(text)

        # Fix the label comparison - match the actual labels from detect_ai_or_human
        if label == 'Human-Written' and round(score, 2) <= 0.75:
            label = 'AI-Written'

        report = f"The text is most likely {label} with a confidence of {score:.2f}"
        return jsonify({"result": report, "label": label, "confidence": round(score, 2)}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    

@app.route("/predictprice/<locality>/<noOfYears>")
def predictPrice(locality:str, noOfYears:int):
    try:
        # Check if locality exists in the dataframe
        if locality not in df["Location"].values:
            return jsonify({"error": f"Locality '{locality}' not found in database"}), 404
            
        action_area_1 = df[df["Location"] == locality].iloc[0, 2:].astype(float)
        action_area_1 = action_area_1.interpolate()
        train_data = action_area_1.dropna()
        
        if len(train_data) < 2:
            return jsonify({"error": "Insufficient data for prediction"}), 400
            
        train_data.index = train_data.index.astype(int)
        
        lastYear = int(train_data.index[-1])
        X_train = train_data.index.values.reshape(-1, 1)  # Years
        y_train = train_data.values  # Prices
        model = LinearRegression()
        model.fit(X_train, y_train)
        X_predict = np.array([(lastYear + int(i)) for i in range(1, int(noOfYears) + 1)]).reshape(-1, 1)
        predictions = model.predict(X_predict)
        # Create a new Series for the predictions
        rangerstr = [str(lastYear + int(i)) for i in range(1, int(noOfYears) + 1)] 
        predictions_series = pd.Series(predictions.flatten(), index=rangerstr)
        predictions_series = {year: round(value, 2) for year, value in predictions_series.items()}
        # Combine the original data with predictions
        new_data = {'current_year': str(lastYear), 'current_price': round(action_area_1[-1], 2)}
        new_data.update(predictions_series)
        return jsonify(new_data), 200
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/")
def home():
    return "This API is Written in Python and is working just fine. "

@app.route("/get-user/<user_id>")
def get_user(user_id):
    user_data = {
        "user_id": user_id,
        "name": "John Doe",
        "email": "job2@gmail.com"
    }
    extra = request.args.get("extra")
    if extra:
        user_data["extra"]= extra

    return jsonify(user_data),200

@app.route("/create-user", methods=["POST"])
def create_user():
    # if request.method == 'POST'
    data = request.get_json()
    return jsonify(data), 201

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=8000)
