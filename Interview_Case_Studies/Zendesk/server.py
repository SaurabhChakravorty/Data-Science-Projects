# -*- coding: utf-8 -*-

from flask import Flask, request, jsonify
from intent_classifier import IntentClassifier, clean_text
from flasgger import Swagger

app = Flask(__name__)
swagger = Swagger(app)  # Add Swagger UI

model = IntentClassifier()
model_loaded = False


@app.route('/ready')
def ready():
    """
    Health check to see if the model is loaded.
    ---
    responses:
      200:
        description: Model is ready
      423:
        description: Model not ready
    """
    if model_loaded and model.is_ready():
        return 'OK', 200
    return 'Not ready', 423


@app.route('/intent', methods=['POST'])
def intent():
    """
    Predict intent from input text.
    ---
    parameters:
      - name: body
        in: body
        required: true
        schema:
          type: object
          required:
            - text
          properties:
            text:
              type: string
              example: Book a flight to New York
    responses:
      200:
        description: Prediction successful
      400:
        description: Missing or invalid input
      500:
        description: Server error
    """
    if not model_loaded or not model.is_ready():
        return jsonify({"error": "Model not loaded"}), 503

    data = request.get_json()
    if not data or 'text' not in data:
        return jsonify({"error": "Missing 'text' in request"}), 400

    input_text = clean_text(data['text'])

    try:
        prediction = model.predict(
            text=input_text,
            lstm_model_path="saved_model",
            transformer_model_path="saved_model",
            method="voting"
        )
        return jsonify({"intents": prediction}), 200    #--> returns can also be logged in logs.py(for later....)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


def main():
    global model_loaded

    # Load the model before starting the server
    model.load(
        file_path="data/atis/train.tsv",               # ---> just to show the model is initialised
        train=False,                                   # ----> set to true in case you data and labels are loaded. The model will detect and train
        lstm_model_path="saved_model",
        transformer_model_path="saved_model"
    )
    model_loaded = True
    #print("Model is ready. Starting server...")

    app.run(host='127.0.0.1', port=8080)


if __name__ == '__main__':
    main()
