from pdf_processor import PDFProcessor
from faiss_indexer import FAISSIndexer
from typing import List, Dict, Optional
import logging
import os

logger = logging.getLogger(__name__)

class RAGPipeline:
    def __init__(self, 
                 pdf_path: str,
                 chunk_size: int = 1000,
                 chunk_overlap: int = 200,
                 model_name: str = 'all-MiniLM-L6-v2',
                 index_type: str = 'flat'):
        """
        Initialize RAG pipeline.
        
        Args:
            pdf_path: Path to the PDF file
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks
            model_name: Sentence transformer model name
            index_type: Type of FAISS index
        """
        self.pdf_path = pdf_path
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.model_name = model_name
        self.index_type = index_type
        
        # Initialize components
        self.pdf_processor = PDFProcessor(chunk_size, chunk_overlap)
        self.faiss_indexer = FAISSIndexer(model_name)
        
        # State
        self.is_indexed = False
        self.chunks = []
        
        logger.info(f"RAG Pipeline initialized for PDF: {pdf_path}")
    
    def process_and_index(self, save_index: bool = True, index_filepath: str = "pdf_index") -> bool:
        """
        Process PDF and build FAISS index.
        
        Args:
            save_index: Whether to save the index to disk
            index_filepath: Filepath for saving index
            
        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info("Starting PDF processing and indexing...")
            
            # Process PDF
            self.chunks = self.pdf_processor.process_pdf(self.pdf_path)
            
            if not self.chunks:
                logger.error("No chunks extracted from PDF")
                return False
            
            # Build FAISS index
            self.faiss_indexer.build_index(self.chunks, self.index_type)
            self.is_indexed = True
            
            # Save index if requested
            if save_index:
                self.faiss_indexer.save_index(index_filepath)
                logger.info(f"Index saved to {index_filepath}")
            
            logger.info("PDF processing and indexing completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error in process_and_index: {str(e)}")
            return False
    
    def load_existing_index(self, index_filepath: str) -> bool:
        """
        Load an existing index from disk.
        
        Args:
            index_filepath: Filepath for loading index
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if not os.path.exists(f"{index_filepath}.faiss") or not os.path.exists(f"{index_filepath}.pkl"):
                logger.error(f"Index files not found at {index_filepath}")
                return False
            
            self.faiss_indexer.load_index(index_filepath)
            self.chunks = self.faiss_indexer.chunks
            self.is_indexed = True
            
            logger.info(f"Existing index loaded from {index_filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading existing index: {str(e)}")
            return False
    
    def query(self, query: str, k: int = 5) -> List[Dict]:
        """
        Query the indexed PDF.
        
        Args:
            query: Search query
            k: Number of top results
            
        Returns:
            List of search results
        """
        if not self.is_indexed:
            raise ValueError("Index not built. Call process_and_index() or load_existing_index() first.")
        
        return self.faiss_indexer.search(query, k)
    
    def get_chunk_by_id(self, chunk_id: int) -> Optional[Dict]:
        """
        Get a specific chunk by its ID.
        
        Args:
            chunk_id: ID of the chunk
            
        Returns:
            Chunk dictionary or None if not found
        """
        for chunk in self.chunks:
            if chunk['id'] == chunk_id:
                return chunk
        return None
    
    def get_statistics(self) -> Dict:
        """
        Get statistics about the pipeline.
        
        Returns:
            Dictionary with pipeline statistics
        """
        stats = {
            "pdf_path": self.pdf_path,
            "is_indexed": self.is_indexed,
            "total_chunks": len(self.chunks),
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "model_name": self.model_name,
            "index_type": self.index_type
        }
        
        if self.is_indexed:
            stats.update(self.faiss_indexer.get_index_stats())
        
        return stats
    
    def search_with_context(self, query: str, k: int = 5, context_window: int = 2) -> List[Dict]:
        """
        Search and return results with surrounding context.
        
        Args:
            query: Search query
            k: Number of top results
            context_window: Number of chunks to include before and after each result
            
        Returns:
            List of search results with context
        """
        results = self.query(query, k)
        
        for result in results:
            chunk_id = result['chunk_id']
            context_chunks = []
            
            # Add previous chunks
            for i in range(max(0, chunk_id - context_window), chunk_id):
                prev_chunk = self.get_chunk_by_id(i)
                if prev_chunk:
                    context_chunks.append({
                        'id': i,
                        'text': prev_chunk['text'][:200] + "..." if len(prev_chunk['text']) > 200 else prev_chunk['text'],
                        'position': 'before'
                    })
            
            # Add next chunks
            for i in range(chunk_id + 1, min(len(self.chunks), chunk_id + context_window + 1)):
                next_chunk = self.get_chunk_by_id(i)
                if next_chunk:
                    context_chunks.append({
                        'id': i,
                        'text': next_chunk['text'][:200] + "..." if len(next_chunk['text']) > 200 else next_chunk['text'],
                        'position': 'after'
                    })
            
            result['context'] = context_chunks
        
        return results
