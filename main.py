import pandas as pd
import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from wordcloud import WordCloud
import networkx as nx
import matplotlib.pyplot as plt
from fact_checking import FactChecker

# Load the legal document in Word format
doc = pd.read_word('legal_document.docx')

# Preprocess the text
text = doc['Text']
text = text.lower()
text = text.replace('\n','')

# Tokenize the text
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
inputs = tokenizer.encode_plus(
    text,
    add_special_tokens=True,
    max_length=512,
    return_attention_mask=True,
    return_tensors='pt'
)

# Convert the text into a numerical representation
model = AutoModel.from_pretrained('bert-base-uncased')
outputs = model(inputs['input_ids'], attention_mask=inputs['attention_mask'])

# Perform Latent Semantic Analysis (LSA)
lsa_model = TruncatedSVD(n_components=100)
lsa_matrix = lsa_model.fit_transform(outputs.last_hidden_state[:, 0, :])

# Calculate Information Entropy
entropy = np.sum(-lsa_matrix * np.log2(lsa_matrix), axis=1)

# Generate Word Cloud
wordcloud = WordCloud(width=800, height=400).generate(text)
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()

# Create Knowledge Graph
entities = []
relationships = []
for sentence in text.split('.'):
    entities.extend([entity for entity in sentence.split() if entity.isalpha()])
    relationships.extend([(entity1, entity2) for entity1, entity2 in zip(entities, entities[1:])])
G = nx.Graph()
G.add_nodes_from(entities)
G.add_edges_from(relationships)
pos = nx.spring_layout(G)
nx.draw_networkx(G, pos, with_labels=True, node_color='lightblue', edge_color='gray')
plt.show()

# Perform Fact Checking
fact_checker = FactChecker()
claims = []
for sentence in text.split('.'):
    claims.append((sentence, fact_checker.check(sentence)))
print(claims)
