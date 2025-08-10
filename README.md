# RAG Document QA System

A comprehensive Retrieval-Augmented Generation (RAG) system that combines FAISS vector search with OpenAI language models for intelligent document question answering. Features a FastAPI REST API and web interface for easy interaction.

## Features

- **PDF Processing**: Extract and clean text from PDF documents using LangChain RecursiveCharacterTextSplitter
- **OpenAI Integration**: Uses OpenAI embeddings and chat models for high-quality responses
- **FAISS Vector Store**: Efficient similarity search with persistent local storage
- **REST API**: FastAPI server with CORS support for web integration
- **Web UI**: Clean HTML interface for document question answering
- **Smart Chunking**: Configurable chunk size (1000) and overlap (200)
- **Intelligent Responses**: Combines vector search with LLM generation for comprehensive answers

## Architecture

```
PDF Document → Text Extraction → Text Chunking → Embedding Generation → FAISS Index → Query Interface
```

## Components

1. **PDFProcessor** (`pdf_processor.py`): Handles PDF processing with LangChain and FAISS integration
2. **FastAPI Server** (`api.py`): REST API server with CORS support
3. **Web UI** (`index.html`): HTML interface for document question answering
4. **Example Script** (`example.py`): Demonstrates PDF processing and RAG queries

## Installation

1. Clone the repository:
```bash
git clone https://github.com/anirbankonar/RAG_Project.git
cd RAG_Project
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

3. Set your OpenAI API key:
```bash
export OPENAI_API_KEY=your_openai_api_key_here
```

## Usage

### Quick Start

1. **Process a PDF and create FAISS index:**
```bash
python example.py
```

2. **Start the FastAPI server:**
```bash
python api.py
```
The server will start on http://localhost:8000

3. **Use the Web UI:**
- Open `index.html` in your web browser
- Enter your question about the document
- Configure search parameters (number of sources, AI model)
- Click "Ask Question" to get AI-generated answers

### Alternative: Direct API Usage

You can also query the API directly:

```bash
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{"query": "What is this document about?", "k": 3}'
```

### API Endpoints

- `POST /query` - Submit a question and get an AI-generated answer
- `GET /health` - Check service health
- `GET /docs` - Interactive API documentation

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
