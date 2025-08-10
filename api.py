from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import logging
from pdf_processor import PDFProcessor
from langchain_community.vectorstores import FAISS

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="RAG API", description="Document Question Answering API using RAG")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
processor: PDFProcessor = None
vector_store: FAISS = None

class QueryRequest(BaseModel):
    query: str
    k: int = 3
    model: str = "gpt-3.5-turbo"

class QueryResponse(BaseModel):
    query: str
    answer: str
    sources_count: int

@app.on_event("startup")
async def startup_event():
    """Load FAISS index on startup"""
    global processor, vector_store
    
    try:
        logger.info("Initializing PDF processor...")
        processor = PDFProcessor(chunk_size=1000, chunk_overlap=200)
        
        logger.info("Loading FAISS index...")
        vector_store = processor.load_vector_store("test_index")
        
        logger.info("API ready! FAISS index loaded successfully.")
        
    except Exception as e:
        logger.error(f"Failed to initialize: {str(e)}")
        raise e

@app.post("/query", response_model=QueryResponse)
async def query_document(request: QueryRequest):
    """
    Query the document using RAG (Retrieval-Augmented Generation).
    
    Args:
        request: Query request containing the question and optional parameters
        
    Returns:
        Generated answer based on document context
    """
    global processor, vector_store
    
    if processor is None or vector_store is None:
        raise HTTPException(status_code=500, detail="Service not initialized")
    
    try:
        # Get RAG response
        answer = processor.query_with_rag(
            vector_store=vector_store,
            query=request.query,
            k=request.k,
            model=request.model
        )
        
        # Get source count for transparency
        results = vector_store.similarity_search(request.query, k=request.k)
        sources_count = len(results)
        
        return QueryResponse(
            query=request.query,
            answer=answer,
            sources_count=sources_count
        )
        
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    global processor, vector_store
    
    if processor is None or vector_store is None:
        return {"status": "unhealthy", "message": "Service not initialized"}
    
    return {"status": "healthy", "message": "Service is running"}

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "RAG Document QA API",
        "endpoints": {
            "POST /query": "Submit a question about the document",
            "GET /health": "Check service health",
            "GET /docs": "API documentation"
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)