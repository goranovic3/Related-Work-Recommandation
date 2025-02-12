import logging
import sys
from tqdm import tqdm
import faiss
import numpy as np
import pandas as pd
import ast

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s — %(name)s — %(levelname)s — %(message)s',
    stream=sys.stdout
)
logger = logging.getLogger(__name__)

def build_faiss_index(embeddings: np.ndarray, use_gpu: bool = False) -> faiss.IndexFlatL2:
    """
    Build a FAISS index for fast similarity search.
    """
    dim = embeddings.shape[1]
    logger.info("Building FAISS index with dimension %d", dim)
    
    index = faiss.IndexFlatL2(dim)  # Default CPU-based index
    
    logger.info("Adding %d embeddings to the FAISS index...", embeddings.shape[0])
    index.add(embeddings)
    
    logger.info("FAISS index built successfully with %d vectors", index.ntotal)
    return index

def get_top_n_neighbors(index: faiss.IndexFlatL2, query_embeddings: np.ndarray, top_n: int = 10) -> np.ndarray:
    """
    Retrieve the top N most similar papers for each query.
    """
    logger.info("Retrieving top-%d neighbors for %d query embeddings...", top_n, query_embeddings.shape[0])
    distances, indices = index.search(query_embeddings, top_n)
    logger.info("Completed retrieving top-%d neighbors.", top_n)
    return indices

def parse_embeddings(df: pd.DataFrame) -> np.ndarray:
    """
    Convert embedding strings into NumPy arrays.
    """
    logger.info("Parsing and converting embeddings...")
    try:
        embeddings_list = [
            ast.literal_eval(emb) if isinstance(emb, str) else emb
            for emb in tqdm(df['abstract_embeddings'])
        ]
        return np.array(embeddings_list, dtype='float32')
    except Exception as e:
        logger.error("Error parsing embeddings: %s", str(e))
        raise

def compute_citation_analytics(df: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    """
    Compute citation statistics based on nearest neighbors.
    """
    logger.info("Starting citation analytics computation...")
    
    df['id'] = df['id'].astype(str)
    paper_ids = df['id'].to_numpy()
    embeddings = parse_embeddings(df)
    
    if embeddings.shape[0] == 0:
        logger.error("No embeddings found in the dataset.")
        return df
    
    index = build_faiss_index(embeddings)
    top_neighbors = get_top_n_neighbors(index, embeddings, top_n)
    
    citation_counts, citation_ratios = [], []
    
    logger.info("Analyzing citations and computing statistics...")
    for i in tqdm(range(len(df)), desc="Analyzing citations"):
        query_paper_id = paper_ids[i]
        
        cited_papers = set(df.loc[df['id'] == query_paper_id, 'filtered_references'].values[0])
        retrieved_paper_ids = set(paper_ids[top_neighbors[i]])
        
        cited_neighbors = cited_papers.intersection(retrieved_paper_ids)
        citation_counts.append(len(cited_neighbors))
        citation_ratios.append(len(cited_neighbors) / top_n if top_n > 0 else 0)
    
    df['num_cited_neighbors'] = citation_counts
    df['citation_ratio'] = citation_ratios
    
    logger.info("Average cited papers in top-%d: %.2f", top_n, np.mean(citation_counts))
    logger.info("Average citation ratio in top-%d: %.2f%%", top_n, np.mean(citation_ratios) * 100)
    logger.info("Citation analytics computation completed.")
    
    return df
