import pandas as pd
import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from textblob import TextBlob
from wordcloud import WordCloud
import networkx as nx
import matplotlib.pyplot as plt
from docx import Document
from io import StringIO
import re
from PyPDF2 import PdfReader, PdfFileReader
import read_pdf

def extractPdfInformation(pdf_path):
    with open(pdf_path, 'rb') as f:
        pdf = PdfFileReader(f)
        information = pdf.getDocumentInfo()
        number_of_pages = pdf.getNumPages()

    txt = f"""
    Information about {pdf_path}: 

    Author: {information.author}
    Creator: {information.creator}
    Producer: {information.producer}
    Subject: {information.subject}
    Title: {information.title}
    Number of pages: {number_of_pages}
    """

    print(txt)
    return information


# Load the legal document
def getPdfText(filename):
   fullText = list()
   reader = PdfReader(filename) 
   for page in range(len(reader.pages)): 
      txtObj = reader.pages[page]
      print(f'{txtObj}')
      fullText.append(txtObj.extract_text())
   txt = '\n'.join(fullText)
   return re.sub(r'\n{2,}', '\n', txt)

# Load the legal document in Word format
# doc = pd.read_word('2023.docx')
info = extractPdfInformation('affidavit.pdf')
text = read_pdf.extractTextPdf('affidavit.pdf')
text = text.lower()
text = text.replace('\n','')

with torch.no_grad():
   # Tokenize the text
   tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
   inputs = tokenizer.encode_plus(
      text,
      add_special_tokens=True,
      max_length=512,
      return_attention_mask=True,
      return_tensors='pt'
   )

   # Perform LSA
   lsa_model = TruncatedSVD(n_components=100)
   lsa_matrix = lsa_model.fit_transform(inputs['input_ids'])

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

   # Calculate word stats
   word_freq = {}
   word_cooc = {}
   for sentence in text.split('.'):
      words = sentence.split()
      for word in words:
         if word not in word_freq:
               word_freq[word] = 0
         word_freq[word] += 1
         for other_word in words:
               if other_word!= word:
                  if (word, other_word) not in word_cooc:
                     word_cooc[(word, other_word)] = 0
                  word_cooc[(word, other_word)] += 1

   # Perform sentiment analysis
   sia = SentimentIntensityAnalyzer()
   sentiment = sia.polarity_scores(text)

   # Detect strong arguments vs weak arguments
   claims = []
   premises = []
   conclusions = []
   for sentence in text.split('.'):
      if sentence.startswith('Therefore'):
         conclusions.append(sentence)
      elif sentence.startswith('Because'):
         premises.append(sentence)
      else:
         claims.append(sentence)

   # Print the results
   print('Latent Semantic Analysis:')
   print(lsa_matrix)
   print('Informatin Entropy:')
   print(entropy)
   print('Word Stats:')
   print(word_freq)
   print(word_cooc)
   print('Sentiment Analysis:')
   print(sentiment)
   print('Strong Arguments vs Weak Arguments:')
   print(claims)
   print(premises)
   print(conclusions)