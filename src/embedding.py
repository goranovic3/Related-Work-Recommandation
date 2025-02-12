import sqlite3
import json
from tqdm import tqdm
import numpy as np
import torch
from transformers import DistilBertTokenizer, DistilBertModel
from sentence_transformers import SentenceTransformer
from gensim.models import Word2Vec
import pandas as pd

def initialize_database(db_name):
    """Initialize the database if it doesn't exist."""
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS abstracts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            abstract TEXT UNIQUE,
            embedding TEXT
        )
    ''')
    conn.commit()
    conn.close()

def fetch_existing_embeddings(abstracts, db_name, chunk_size=900):
    """Fetch existing embeddings from the database."""
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()
    existing_abstracts = {}

    for i in tqdm(range(0, len(abstracts), chunk_size), desc="Fetching existing embeddings"):
        batch = abstracts[i:i + chunk_size]
        placeholders = ','.join(['?'] * len(batch))
        cursor.execute(f"SELECT abstract, embedding FROM abstracts WHERE abstract IN ({placeholders})", batch)
        for abstract, embedding in cursor.fetchall():
            existing_abstracts[abstract] = json.loads(embedding)
    
    conn.close()
    return existing_abstracts

def store_new_embeddings(abstract_embeddings, db_name):
    """Store new abstract embeddings in the database."""
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()
    
    for abstract, embedding in abstract_embeddings.items():
        cursor.execute("INSERT INTO abstracts (abstract, embedding) VALUES (?, ?)", (abstract, json.dumps(embedding)))
    
    conn.commit()
    conn.close()

def generate_embeddings(abstracts, method, model_name, batch_size):
    """Generate embeddings using the specified method."""
    if method == 'sentence-transformer':
        model = SentenceTransformer(model_name)
        return {abstract: emb.tolist() for abstract, emb in 
                zip(abstracts, model.encode(abstracts, convert_to_tensor=False))}

    elif method == 'word2vec':
        sentences = [abstract.split() for abstract in abstracts]
        w2v_model = Word2Vec(sentences, vector_size=300, window=5, min_count=1, workers=4)
        return {abstract: np.mean([w2v_model.wv[word] for word in abstract.split() 
                if word in w2v_model.wv] or [np.zeros(300)], axis=0).tolist() for abstract in abstracts}

    elif method == 'distilbert':
        tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
        model = DistilBertModel.from_pretrained("distilbert-base-uncased")
        encoded_input = tokenizer(abstracts, padding=True, truncation=True, return_tensors="pt")
        with torch.no_grad():
            batch_embeddings = model(**encoded_input).last_hidden_state[:, 0, :].numpy()
        return {abstract: emb.tolist() for abstract, emb in zip(abstracts, batch_embeddings)}

def transform_abstracts_to_embeddings(df, method='sentence-transformer', model_name='all-MiniLM-L6-v2', batch_size=1024):
    """Transform abstracts into embeddings using the chosen method."""
    db_name = f"databases/abstracts_{method}.db"
    initialize_database(db_name)

    df['abstract'] = df['abstract'].fillna("")
    
    existing_embeddings = fetch_existing_embeddings(df['abstract'].tolist(), db_name)
    new_abstracts = [abstract for abstract in df['abstract'] if abstract not in existing_embeddings]

    if new_abstracts:
        for i in tqdm(range(0, len(new_abstracts), batch_size), desc="Generating embeddings"):
            batch = new_abstracts[i:i + batch_size]
            batch_embeddings = generate_embeddings(batch, method, model_name, batch_size)
            store_new_embeddings(batch_embeddings, db_name)
            existing_embeddings.update(batch_embeddings)  # Ensure the new embeddings are also added

    df['abstract_embeddings'] = df['abstract'].map(existing_embeddings)

    # Ensure that all abstracts have embeddings; fallback to a zero vector if missing
    embedding_dim = len(next(iter(existing_embeddings.values()), [0]))  # Get dimension from first embedding
    df['abstract_embeddings'] = df['abstract_embeddings'].apply(
        lambda x: x if isinstance(x, list) else [0] * embedding_dim
    )

    return df
