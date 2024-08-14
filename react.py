import pandas as pd
import torch
from transformers import RagTokenizer, RagModel

# Load the legal document
doc = pd.read_word('legal_document.docx')

# Preprocess the text
text = doc['Text']
text = text.lower()
text = text.replace('\n','')

# Tokenize the text
tokenizer = RagTokenizer.from_pretrained('facebook/rag-tokenizer')
inputs = tokenizer.encode_plus(
    text,
    add_special_tokens=True,
    max_length=512,
    return_attention_mask=True,
    return_tensors='pt'
)

# Create a RAG model
model = RagModel.from_pretrained('facebook/rag-model')

# Define a custom dataset class for RAG
class RagDataset(torch.utils.data.Dataset):
    def __init__(self, inputs, labels):
        self.inputs = inputs
        self.labels = labels

    def __getitem__(self, idx):
        input_ids = self.inputs['input_ids'][idx]
        attention_mask = self.inputs['attention_mask'][idx]
        labels = self.labels[idx]

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }

    def __len__(self):
        return len(self.inputs['input_ids'])

# Create a dataset and data loader for RAG
dataset = RagDataset(inputs, labels=[0]*len(inputs['input_ids']))
data_loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

# Train the RAG model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

for epoch in range(5):
    model.train()
    total_loss = 0
    for batch in data_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad()

        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f'Epoch {epoch+1}, Loss: {total_loss / len(data_loader)}')

model.eval()