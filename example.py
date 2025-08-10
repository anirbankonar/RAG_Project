#!/usr/bin/env python3
"""
Simple example script demonstrating the RAG pipeline usage.
"""

from pdf_processor import PDFProcessor
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def example_usage():
    """Example of how to use the PDF processor with FAISS."""
    
    # Initialize processor
    processor = PDFProcessor(chunk_size=1000, chunk_overlap=200)
    index_name = "test_index"
    
    # Check if index already exists, load it instead of recreating
    index_path = os.path.join(processor.faiss_index_path, index_name)
    
    if os.path.exists(index_path):
        logger.info("Loading existing FAISS index...")
        vector_store = processor.load_vector_store(index_name)
    else:
        logger.info("FAISS index not found, creating new one...")
        pdf_path = "IFB-CO-15079-IAS_Final-PART-2.pdf"
        vector_store = processor.process_pdf(pdf_path, save_index=True, index_name=index_name)
    
    # Example queries
    example_queries = [
        "What is the main topic of this document?",
        "What are the key requirements mentioned?",
        "What is the deadline for submission?",
        "What are the technical specifications?",
        "What is the budget or cost mentioned?"
    ]
    
    # Run example queries using RAG (vector search + LLM generation)
    for query in example_queries:
        logger.info(f"\nQuery: {query}")
        answer = processor.query_with_rag(vector_store, query, k=3)
        print(f"Answer: {answer}")
        print("-" * 80)
    
    # Example of direct search comparison
    logger.info("\n\nDirect search vs RAG comparison for 'technical requirements':")
    
    # Direct vector search
    print("Direct search results:")
    tech_results = vector_store.similarity_search("technical requirements", k=3)
    for i, doc in enumerate(tech_results, 1):
        print(f"  {i}. {doc.page_content[:150]}...")
    
    print("\nRAG-generated answer:")
    rag_answer = processor.query_with_rag(vector_store, "What are the technical requirements?", k=3)
    print(f"{rag_answer}")

if __name__ == "__main__":
    example_usage()
