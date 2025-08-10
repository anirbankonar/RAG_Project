#!/usr/bin/env python3
"""
Main script to run the RAG pipeline for PDF indexing and querying.
"""

import os
import sys
from rag_pipeline import RAGPipeline
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    # PDF file path
    pdf_path = "IFB-CO-15079-IAS_Final-PART-2.pdf"
    
    # Check if PDF exists
    if not os.path.exists(pdf_path):
        logger.error(f"PDF file not found: {pdf_path}")
        sys.exit(1)
    
    # Initialize RAG pipeline
    logger.info("Initializing RAG Pipeline...")
    rag = RAGPipeline(
        pdf_path=pdf_path,
        chunk_size=1000,
        chunk_overlap=200,
        model_name='all-MiniLM-L6-v2',
        index_type='flat'
    )
    
    # Check if index already exists
    index_filepath = "pdf_index"
    if os.path.exists(f"{index_filepath}.faiss") and os.path.exists(f"{index_filepath}.pkl"):
        logger.info("Loading existing index...")
        if rag.load_existing_index(index_filepath):
            logger.info("Existing index loaded successfully!")
        else:
            logger.warning("Failed to load existing index. Building new index...")
            if not rag.process_and_index(save_index=True, index_filepath=index_filepath):
                logger.error("Failed to build index!")
                sys.exit(1)
    else:
        logger.info("Building new index...")
        if not rag.process_and_index(save_index=True, index_filepath=index_filepath):
            logger.error("Failed to build index!")
            sys.exit(1)
    
    # Display statistics
    stats = rag.get_statistics()
    logger.info("Pipeline Statistics:")
    for key, value in stats.items():
        logger.info(f"  {key}: {value}")
    
    # Interactive query mode
    print("\n" + "="*60)
    print("RAG Pipeline Ready! You can now query your PDF.")
    print("="*60)
    print("Commands:")
    print("  - Type your query to search the PDF")
    print("  - Type 'stats' to see pipeline statistics")
    print("  - Type 'quit' or 'exit' to exit")
    print("="*60)
    
    while True:
        try:
            query = input("\nEnter your query: ").strip()
            
            if query.lower() in ['quit', 'exit', 'q']:
                logger.info("Exiting...")
                break
            
            if query.lower() == 'stats':
                stats = rag.get_statistics()
                print("\nPipeline Statistics:")
                for key, value in stats.items():
                    print(f"  {key}: {value}")
                continue
            
            if not query:
                continue
            
            # Perform search
            logger.info(f"Searching for: {query}")
            results = rag.search_with_context(query, k=3, context_window=1)
            
            if not results:
                print("No results found.")
                continue
            
            # Display results
            print(f"\nFound {len(results)} results for: '{query}'")
            print("-" * 60)
            
            for i, result in enumerate(results, 1):
                print(f"\nResult {i} (Score: {result['score']:.4f})")
                print(f"Chunk ID: {result['chunk_id']}")
                print(f"Text: {result['text'][:300]}{'...' if len(result['text']) > 300 else ''}")
                
                # Show context if available
                if 'context' in result and result['context']:
                    print("\nContext:")
                    for ctx in result['context']:
                        position = "before" if ctx['position'] == 'before' else "after"
                        print(f"  [{position}] {ctx['text']}")
                
                print("-" * 40)
        
        except KeyboardInterrupt:
            logger.info("Interrupted by user. Exiting...")
            break
        except Exception as e:
            logger.error(f"Error during query: {str(e)}")
            print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
