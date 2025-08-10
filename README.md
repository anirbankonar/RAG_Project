# RAG Pipeline with FAISS Index

A comprehensive Retrieval-Augmented Generation (RAG) pipeline that uses FAISS for efficient similarity search on PDF documents. This system extracts text from PDFs, chunks it into manageable pieces, creates embeddings using sentence transformers, and builds a searchable FAISS index.

## Features

- **PDF Processing**: Extract and clean text from PDF documents
- **Smart Chunking**: Intelligent text chunking with configurable overlap
- **Embedding Generation**: Uses sentence transformers for high-quality text embeddings
- **FAISS Indexing**: Multiple FAISS index types (Flat, IVF, HNSW) for different use cases
- **Efficient Search**: Fast similarity search with configurable result counts
- **Context-Aware Results**: Search results include surrounding context for better understanding
- **Persistence**: Save and load indexes to avoid reprocessing
- **Interactive Interface**: Command-line interface for easy querying

## Architecture

```
PDF Document → Text Extraction → Text Chunking → Embedding Generation → FAISS Index → Query Interface
```

## Components

1. **PDFProcessor** (`pdf_processor.py`): Handles PDF text extraction and chunking
2. **FAISSIndexer** (`faiss_indexer.py`): Manages embeddings and FAISS index operations
3. **RAGPipeline** (`rag_pipeline.py`): Main pipeline that orchestrates all components
4. **Main Script** (`main.py`): Interactive command-line interface
5. **Example Script** (`example.py`): Programmatic usage examples

## Installation

1. Clone or download the project files
2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Quick Start

1. **Run the interactive interface:**
```bash
python main.py
```

2. **Run the example script:**
```bash
python example.py
```

### Programmatic Usage

```python
from rag_pipeline import RAGPipeline

# Initialize pipeline
rag = RAGPipeline(
    pdf_path="your_document.pdf",
    chunk_size=1000,
    chunk_overlap=200
)

# Process and index PDF
rag.process_and_index(save_index=True, index_filepath="my_index")

# Query the document
results = rag.query("What is the main topic?", k=5)

# Search with context
context_results = rag.search_with_context("technical requirements", k=3, context_window=2)
```

## Configuration Options

### RAGPipeline Parameters

- `pdf_path`: Path to the PDF file
- `chunk_size`: Maximum size of each text chunk (default: 1000)
- `chunk_overlap`: Overlap between consecutive chunks (default: 200)
- `model_name`: Sentence transformer model (default: 'all-MiniLM-L6-v2')
- `index_type`: FAISS index type ('flat', 'ivf', 'hnsw') (default: 'flat')

### FAISS Index Types

- **Flat**: Exact search, fastest for small datasets
- **IVF**: Approximate search with clustering, good balance of speed/accuracy
- **HNSW**: Hierarchical navigable small world, best for large datasets

## File Structure

```
RAG_Project/
├── requirements.txt          # Python dependencies
├── pdf_processor.py         # PDF text extraction and chunking
├── faiss_indexer.py        # FAISS indexing and search
├── rag_pipeline.py         # Main RAG pipeline
├── main.py                 # Interactive command-line interface
├── example.py              # Programmatic usage examples
├── README.md               # This file
└── IFB-CO-15079-IAS_Final-PART-2.pdf  # Your PDF document
```

## API Reference

### RAGPipeline Class

#### Methods

- `process_and_index(save_index=True, index_filepath="pdf_index")`: Process PDF and build index
- `load_existing_index(index_filepath)`: Load saved index from disk
- `query(query, k=5)`: Search for similar chunks
- `search_with_context(query, k=5, context_window=2)`: Search with surrounding context
- `get_statistics()`: Get pipeline statistics
- `get_chunk_by_id(chunk_id)`: Retrieve specific chunk by ID

### FAISSIndexer Class

#### Methods

- `build_index(chunks, index_type='flat')`: Build FAISS index from text chunks
- `search(query, k=5)`: Perform similarity search
- `save_index(filepath)`: Save index to disk
- `load_index(filepath)`: Load index from disk
- `get_index_stats()`: Get index statistics

## Performance Considerations

- **First Run**: Initial indexing may take several minutes depending on PDF size
- **Memory Usage**: Large PDFs may require significant memory for processing
- **Index Size**: FAISS indexes are typically much smaller than the original text
- **Search Speed**: FAISS provides sub-second search times even for large documents

## Troubleshooting

### Common Issues

1. **Memory Errors**: Reduce chunk size or use a smaller embedding model
2. **Slow Indexing**: Use 'flat' index type for smaller documents, 'hnsw' for larger ones
3. **Poor Search Results**: Adjust chunk size and overlap parameters
4. **Model Download Issues**: Ensure internet connection for first-time model downloads

### Logging

The system provides detailed logging. Set log level to DEBUG for more verbose output:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Advanced Usage

### Custom Embedding Models

```python
# Use a different sentence transformer model
rag = RAGPipeline(
    pdf_path="document.pdf",
    model_name='paraphrase-multilingual-MiniLM-L12-v2'  # Multilingual support
)
```

### Batch Processing

```python
# Process multiple PDFs
pdfs = ["doc1.pdf", "doc2.pdf", "doc3.pdf"]
for pdf in pdfs:
    rag = RAGPipeline(pdf_path=pdf)
    rag.process_and_index(save_index=True, index_filepath=f"index_{pdf}")
```

### Custom Chunking Strategy

```python
# Modify chunking parameters for different document types
rag = RAGPipeline(
    pdf_path="technical_document.pdf",
    chunk_size=500,    # Smaller chunks for technical content
    chunk_overlap=100  # Less overlap for technical documents
)
```

## Contributing

Feel free to modify and extend this pipeline for your specific needs. Common enhancements include:

- Support for other document formats (DOCX, TXT, etc.)
- Integration with vector databases (Pinecone, Weaviate, etc.)
- Web interface for easier interaction
- Support for multiple languages
- Advanced text preprocessing and cleaning

## License

This project is provided as-is for educational and research purposes.

## Dependencies

- **faiss-cpu**: FAISS library for similarity search
- **PyPDF2**: PDF text extraction
- **sentence-transformers**: Text embedding generation
- **torch**: PyTorch backend for transformers
- **numpy**: Numerical operations
- **scikit-learn**: Machine learning utilities
