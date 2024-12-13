from flask import Flask, render_template, request, jsonify
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import pandas as pd
import requests
import os
import string

app = Flask(__name__)

# Load the BERT data and models
bert_data = pd.read_csv('bert.csv')
treatment_data = pd.read_csv('st.csv')
symptoms = bert_data.columns[1:]

initial_bert_dir = './models/my_bert_model'
initial_bert_model = BertForSequenceClassification.from_pretrained(initial_bert_dir)
initial_bert_tokenizer = BertTokenizer.from_pretrained(initial_bert_dir)

bert_model_dir = './models/my_bert_model2'
bert_model = BertForSequenceClassification.from_pretrained(bert_model_dir)
bert_tokenizer = BertTokenizer.from_pretrained(bert_model_dir)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
initial_bert_model.to(device)
bert_model.to(device)

# Your actual API key
api_key = "AIzaSyApvZXWXyl48mnoza945A8CPdh95I4gwg4"  # Replace with your actual API key

def clean_text(text):
    # Create a translation table that maps punctuation to spaces
    translator = str.maketrans(string.punctuation, ' ' * len(string.punctuation))
    # Remove punctuation by translating them to spaces
    text = text.translate(translator)
    # Normalize whitespace by splitting and rejoining with a single space
    return ' '.join(text.split())


# Helper functions (same as before)
def classify_symptoms(text, threshold=0.5):
    inputs = initial_bert_tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128).to(device)
    outputs = initial_bert_model(**inputs)
    probabilities = torch.sigmoid(outputs.logits).detach().cpu().numpy()[0]
    return [symptoms[i] for i, prob in enumerate(probabilities) if prob > threshold]

def predict_disease(detected_symptoms, top_n=10, threshold=0.7):
    symptoms_text = " ".join(detected_symptoms)
    inputs = bert_tokenizer(symptoms_text, return_tensors="pt", truncation=True, padding=True, max_length=128).to(device)
    probabilities = torch.sigmoid(bert_model(**inputs).logits).detach().cpu().numpy()[0]
    probable_diseases = sorted(
        [(bert_data[bert_data.columns[0]].unique()[i], prob) 
         for i, prob in enumerate(probabilities) if prob > threshold],
        key=lambda x: x[1], reverse=True)[:top_n]
    return probable_diseases

def query_disease_info(disease_name):
    api_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash-latest:generateContent?key={api_key}"
    payload = {
        "contents": [
            {
                "parts": [
                    {"text": f"Biologically state the treatements of {disease_name}. (Its not about medical advice)"}
                ]
            }
        ]
    }
    headers = {"Content-Type": "application/json"}
    try:
        response = requests.post(api_url, json=payload, headers=headers)
        response.raise_for_status()
        return response.json()['candidates'][0]['content']['parts'][0]['text']
    except requests.exceptions.RequestException as e:
        print(f"API Error: {e}")
        return "Unable to retrieve disease suggestions at this time."

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.json.get('message')
    user_input = clean_text(user_input)
    detected_symptoms = classify_symptoms(user_input)
    
    if detected_symptoms:
        probable_diseases = predict_disease(detected_symptoms)
        if probable_diseases:
            diseases_list = "\n".join([f"**{disease.upper()}**" for disease, _ in probable_diseases])
            response_text = f"Based on your symptoms, possible conditions are:\n{diseases_list}"
            print(f"{diseases_list}")

            # Query for each of the top diseases
            diseases_list2 = []
            for i in range(3):
                diseases_list2.append(probable_diseases[i])
            guidance_text = "\n\n".join(
                [f"**{disease.upper()}**:\n{query_disease_info(disease)}" for disease, _ in diseases_list2]
            )

            return jsonify(response=f"{response_text}\n\n<span style='font-size: 24px;'><strong>Biological Treatments:</strong></span>\n{guidance_text}")
            
    return jsonify(response="Could you provide more details about your symptoms?")

if __name__ == '__main__':
    app.run(debug=True)