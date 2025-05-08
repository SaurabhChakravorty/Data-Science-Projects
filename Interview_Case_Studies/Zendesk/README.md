# AI Agents - ML Engineer Coding Challenge
We're excited that you want to join the AI Agents team.  If you have any questions regarding this task, please don't hesitate to ask.

We understand that this challenge involves significant effort, and some parts may not be fully completed. However, our main focus is on understanding your thought process, decision-making, and approach, especially how you would tackle the problem in a real production setting. Clearly documenting your choices and reasoning is key.

## Brief

Your task is to implement an intent classifier that can be used to provide inference services via an HTTP service. The boilerplate for the service is implemented in the `server.py` file, and you'll need to implement the API function for inference according to the API documentation provided below. You may either implement and train a neural network-based intent classification model or use an external LLM as a classifier.

For the neural network-based model, the boilerplate interface has been defined in `intent_classifier.py`. You can add any methods and functionality to this class as you deem necessary. You may use any deep learning library (TensorFlow, Keras, PyTorch, etc.) you wish, and you can also use pre-existing components for building the network architecture if they would be useful in real-life production systems. Provide tooling and instructions for training the network from scratch.

If you prefer to use an external LLM, you should adopt a Zero-Shot or Few-Shot classification paradigm, and your prompts should be properly encapsulated to create a modular classifier.



## Implementation Notes / Requirements
- ATIS data can be used for training and developing the network. You'll find the data files in `data/atis` directory. Files are TSV files where the first column is the text and the second column is the intent label. ATIS data is in English only but extra points are given for language-agnostic implementation.
- The given codebase contains one bug (that we know of). You need to find and fix this bug.
- Your service needs to adopt the following API Documentation.
- The modularity of your system is crucial, new model types can easily be integrated to your service with minimal changes.


## API Documentation
API documentation for intent classification service.

### `GET /ready`
Returns HTTP status code 200 with response body `"OK"` when the server is running, model has been loaded and is ready to
serve infer requests and 423 with response body `"Not ready"` when the model has not been loaded.

### `POST /intent`
Responds intent classification results for the given query utterance.

#### Request
JSON request with MIME-Type of `application/json` and body:
- **text** `string`: Input sentence intent classification

Example request
```json
{
  "text": "find me a flight that flies from Memphis to tacoma"
}
```

#### Response
JSON response with body:
- **intents** `[Prediction]`: An array of top 3 intent prediction results. See `Prediction` type below.

`Prediction` is a JSON object with fields:
- **label** `string`: Intent label name
- **confidence** `float`: Probability for the predicted intent

Example response
```json
{
  "intents": [{
    "label": "flight",
    "confidence": 0.73
  }, {
    "label": "aircraft",
    "confidence": 0.12
  }, {
    "label": "capacity",
    "confidence": 0.03
  }]
}
```

#### Exceptions
All exceptions are JSON responses with HTTP status code other than 2XX, error label and human readable error message.



##### 400 Text is empty
Given when the text in the body is an empty string.
```json
{
  "label": "TEXT_EMPTY",
  "message": "\"text\" is empty."
}
```

##### 500 Internal error
Given with any other exception. Human readable message includes the exception text.
```json
{
  "label": "INTERNAL_ERROR",
  "message": "<ERROR_MESSAGE>"
}
```


## Evaluation
- **Scenario fitness:** How does your solution meet the requirements?
- **Modularity:** Can your code easily be modified? How much effort is needed to add a new kind of ML model to your inference service?
- **Research & Experimentation:** What kind of experiments you did to select best model and features?
- **Code readability and comments:** Is your code easily comprehensible and structured for maintainability?
- **Robustness:** Does your solution demonstrate reliability and consideration for edge cases?
- **Bonus:** Any additional creative features: Docker files, architectural diagrams for model or service, Swagger, model performance metrics etc. 


## Structure of files

├── intent_classifier.py    
├── model.py                
├── server.py               
├── test_and_try.ipynb     :   
├── requirements.txt        
├── saved_model            
│   ├── lstm_model.pt
│   ├── vocab.json
│   ├── bert_transformer
│       ├── config.json
│       ├── pytorch_model.bin
│       ├── tokenizer.json
│       └── tokenizer_config.json
├── test.py
└── README.md


>This project integrates multiple intent classification models and combines them using an ensemble voting strategy to improve prediction accuracy and robustness. Here's a breakdown of the models used and the rationale behind each:

## Models Used

### Zero-Shot Classifier (BART-MNLI) --> baseline model

1. Uses facebook/bart-large-mnli for classification without fine-tuning. 
2. Allows inference on unseen labels using natural language descriptions. 
3. Particularly helpful when label sets are dynamic or limited labeled data exists.

### LSTM (Long Short-Term Memory)    --> used with embeddings

1. A bidirectional LSTM network trained on tokenized sequences.
2. Handles contextual dependencies in sequence data.
3. Useful for domain-specific data with limited samples. 
4. Trained using PyTorch with class weights to mitigate label imbalance.

### Transformer (BERT)          --> wanted to see effect of full transformer based model
1. Fine-tuned version of bert-base-uncased using Hugging Face Transformers. 
2. Captures deep semantic representations of text. 
3. Good general performance across various NLP tasks. 
4. Trained with evaluation strategy and early stopping.


### Simple Voting Classifier   

1. Aggregates predictions from LSTM, Transformer, and Zero-shot. 
2. Combines normalized confidence scores from all models. 
3. Returns top k labels based on the total score (default k=3). 
4. Ensures better generalization and improves single-model weaknesses.

## Documentation of code

### Scenario Fitness
1. The codebase effectively solves multi-intent classification using both supervised (LSTM, Transformer) and unsupervised (Zero-shot) methods. 
2. Implemented a simple_voting_classifier to aggregate predictions from multiple models, ensuring robustness even when one model underperforms. 
3. A RESTful Flask API (/intent) is exposed for real-time predictions, enabling production deployment. 
4. Batch inference and support for multiple top predictions (top-3) aligns with real-world intent classification scenarios.

### Modularity
The project is highly modular:
1. model.py: encapsulates all training and inference logic for LSTM, BERT, Zero-Shot, and voting ensemble. 
2. intent_classifier.py: handles orchestration and loading, separated from training. If new class is discovered set `train` parameter as True to re-train the models. 
3. Each model (LSTM, Transformer, Zero-Shot) has its own train, predict, and normalize methods. 
4. Adding a new model (e.g., SVM, or GPT-based) only requires writing a new method in models.py and integrating it into the voting logic if needed. The predict method can easily be added in the `model` class

### Research & Experimentation

The notebook __(test_and_try.ipynb)__ and also .pdf version demonstrates experiments, batch prediction, and metric collection. The notebook is used to train the models and consistency is maintained why each and every model is used.
>  Please follow the Notebook  to understand all the documentation of code and the experimentation done.

### Robustness

1. Files follow PEP8 styling, and imports are grouped logically. 
2. Code was iteratively refactored to reduce duplication and centralize logic (e.g., normalization is done uniformly).

## Architecture 
                                 ┌────────────────────────────┐
                                 │      Training Pipeline     │
                                 └────────────────────────────┘
                                           ▲
                                           │
                          ┌────────────────┼────────────────┐
                          │                │                │
                          ▼                ▼                ▼
                ┌────────────────┐ ┌────────────────┐ ┌────────────────────┐
                │     LSTM       │ │   Transformer  │ │   Zero-Shot Model  │ 
                │  train_lstm()  │ │train_transform │ │  Pretrained HF API │       -----> Model training and conterization
                └────────────────┘ └────────────────┘ └────────────────────┘
                          │                │
                          ▼                ▼
                 saved_model/      saved_model/bert_transformer

                                 ┌────────────────────────────┐
                                 │   Inference REST Server    │
                                 │        (Flask API)         │                  --------> Model deployed at endpoints for inference
                                 └────────────────────────────┘
                                           ▲
                     ┌─────────────────────┼──────────────────────┐
                     │                     │                      │
                     ▼                     ▼                      ▼
              /intent endpoint    /ready endpoint           Batch evaluation

                            ┌────────────────────────────┐
                            │ IntentClassifier.predict() │  ----> Predicts class
                            └────────────────────────────┘
                                           ▲
                                           │
                        ┌──────────────────┴──────────────────┐
                        │          Prediction Methods         │
                        └─────────────────────────────────────┘
                            │     │       │       │
                            ▼     ▼       ▼       ▼
                     LSTM       BERT   Zero-Shot   VotingClassifier
                 (lstm_predict) (predict_transformer)      (combines all 3)


## Instructions
### Installation

- pip install -r requirements.txt  : All libs with version are mentioned

### Running the server
- python server.py --port 8080

### Making request
- curl -X POST http://localhost:8080/intent -H "Content-Type: application/json" -d '{"text": "book me a flight to New York"}'
- You can run `test.py` with some example use cases to check the results

## Limitations and Future Work
```
## Limitations
- The Zero-Shot model is slower on CPU; could be optimized using GPU batching.
- Currently supports English only; multilingual support can be added.

## Future Improvements
- Add support for Few-Shot learning via prompt-tuning.
- Integrate more advanced models like RoBERTa or DeBERTa.
- Add Swagger/OpenAPI UI for endpoint testing.
```