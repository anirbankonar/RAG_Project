import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Tuple
import pickle
import os
import logging

logger = logging.getLogger(__name__)

class FAISSIndexer:
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2', dimension: int = 384):
        """
        Initialize FAISS indexer with a sentence transformer model.
        
        Args:
            model_name: Name of the sentence transformer model to use
            dimension: Dimension of the embeddings
        """
        self.model_name = model_name
        self.dimension = dimension
        self.model = None
        self.index = None
        self.chunks = []
        self.chunk_ids = []
        
        logger.info(f"Initializing FAISS indexer with model: {model_name}")
    
    def load_model(self):
        """Load the sentence transformer model."""
        try:
            self.model = SentenceTransformer(self.model_name)
            logger.info(f"Model {self.model_name} loaded successfully")
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise
    
    def create_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Create embeddings for a list of texts.
        
        Args:
            texts: List of text strings
            
        Returns:
            Numpy array of embeddings
        """
        if self.model is None:
            self.load_model()
        
        try:
            embeddings = self.model.encode(texts, show_progress_bar=True)
            logger.info(f"Created embeddings for {len(texts)} texts")
            return embeddings
        except Exception as e:
            logger.error(f"Error creating embeddings: {str(e)}")
            raise
    
    def build_index(self, chunks: List[Dict[str, str]], index_type: str = 'flat'):
        """
        Build FAISS index from text chunks.
        
        Args:
            chunks: List of text chunk dictionaries
            index_type: Type of FAISS index ('flat', 'ivf', 'hnsw')
        """
        if not chunks:
            raise ValueError("No chunks provided for indexing")
        
        self.chunks = chunks
        self.chunk_ids = [chunk['id'] for chunk in chunks]
        texts = [chunk['text'] for chunk in chunks]
        
        # Create embeddings
        embeddings = self.create_embeddings(texts)
        
        # Normalize embeddings for better search performance
        faiss.normalize_L2(embeddings)
        
        # Create FAISS index based on type
        if index_type == 'flat':
            self.index = faiss.IndexFlatIP(self.dimension)  # Inner product for normalized vectors
        elif index_type == 'ivf':
            nlist = min(100, len(chunks) // 10)  # Number of clusters
            quantizer = faiss.IndexFlatIP(self.dimension)
            self.index = faiss.IndexIVFFlat(quantizer, self.dimension, nlist, faiss.METRIC_INNER_PRODUCT)
            self.index.train(embeddings)
        elif index_type == 'hnsw':
            self.index = faiss.IndexHNSWFlat(self.dimension, 32)  # 32 neighbors
            self.index.hnsw.efConstruction = 200
            self.index.hnsw.efSearch = 100
        else:
            raise ValueError(f"Unsupported index type: {index_type}")
        
        # Add vectors to index
        self.index.add(embeddings)
        
        logger.info(f"FAISS index built successfully with {len(chunks)} vectors")
        logger.info(f"Index type: {index_type}, Dimension: {self.dimension}")
    
    def search(self, query: str, k: int = 5) -> List[Dict]:
        """
        Search for similar chunks given a query.
        
        Args:
            query: Search query string
            k: Number of top results to return
            
        Returns:
            List of dictionaries containing search results
        """
        if self.index is None:
            raise ValueError("Index not built. Call build_index() first.")
        
        if self.model is None:
            self.load_model()
        
        # Create query embedding
        query_embedding = self.model.encode([query])
        faiss.normalize_L2(query_embedding)
        
        # Search
        scores, indices = self.index.search(query_embedding, k)
        
        # Prepare results
        results = []
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx != -1:  # FAISS returns -1 for invalid indices
                results.append({
                    'rank': i + 1,
                    'chunk_id': self.chunk_ids[idx],
                    'text': self.chunks[idx]['text'],
                    'score': float(score),
                    'length': self.chunks[idx]['length']
                })
        
        logger.info(f"Search completed. Found {len(results)} results for query: '{query}'")
        return results
    
    def save_index(self, filepath: str):
        """
        Save the FAISS index and metadata to disk.
        
        Args:
            filepath: Base filepath for saving (without extension)
        """
        if self.index is None:
            raise ValueError("No index to save")
        
        try:
            # Save FAISS index
            faiss.write_index(self.index, f"{filepath}.faiss")
            
            # Save metadata
            metadata = {
                'chunks': self.chunks,
                'chunk_ids': self.chunk_ids,
                'model_name': self.model_name,
                'dimension': self.dimension
            }
            
            with open(f"{filepath}.pkl", 'wb') as f:
                pickle.dump(metadata, f)
            
            logger.info(f"Index saved to {filepath}.faiss and {filepath}.pkl")
            
        except Exception as e:
            logger.error(f"Error saving index: {str(e)}")
            raise
    
    def load_index(self, filepath: str):
        """
        Load a saved FAISS index and metadata from disk.
        
        Args:
            filepath: Base filepath for loading (without extension)
        """
        try:
            # Load FAISS index
            self.index = faiss.read_index(f"{filepath}.faiss")
            
            # Load metadata
            with open(f"{filepath}.pkl", 'rb') as f:
                metadata = pickle.load(f)
            
            self.chunks = metadata['chunks']
            self.chunk_ids = metadata['chunk_ids']
            self.model_name = metadata['model_name']
            self.dimension = metadata['dimension']
            
            logger.info(f"Index loaded from {filepath}.faiss and {filepath}.pkl")
            
        except Exception as e:
            logger.error(f"Error loading index: {str(e)}")
            raise
    
    def get_index_stats(self) -> Dict:
        """Get statistics about the current index."""
        if self.index is None:
            return {"error": "No index built"}
        
        stats = {
            "total_vectors": self.index.ntotal,
            "dimension": self.dimension,
            "total_chunks": len(self.chunks),
            "index_type": type(self.index).__name__
        }
        
        return stats
