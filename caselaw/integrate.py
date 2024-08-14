import requests
import torch
from transformers import BertTokenizer

# Load the pre-trained BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def get_case_law(query):
    url = 'http://localhost:5000/case_law'
    params = {'query': query}
    response = requests.get(url, params=params)
    return response.json()

def process_user_input(input_text):
    inputs = tokenizer.encode_plus(
        input_text,
        add_special_tokens=True,
        max_length=512,
        return_attention_mask=True,
        return_tensors='pt'
    )

    outputs = model(inputs['input_ids'], attention_mask=inputs['attention_mask'])
    logits = outputs.last_hidden_state[:, 0, :]

    # Get the top 5 most relevant case law results
    query = logits.topk(5).indices
    results = get_case_law(query)

    return results

# Test the process_user_input function
input_text = 'This is a test affidavit.'
results = process_user_input(input_text)
print(results)