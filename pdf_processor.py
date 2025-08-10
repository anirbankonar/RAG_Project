import logging
import os
import re
from typing import Dict, List

import pypdf
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from openai import OpenAI

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PDFProcessor:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200, openai_api_key: str = None):
        """
        Initialize PDF processor with chunking parameters.
        
        Args:
            chunk_size: Maximum size of each text chunk
            chunk_overlap: Overlap between consecutive chunks
            openai_api_key: OpenAI API key (optional, can be set via environment)
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        
        # Initialize OpenAI embeddings with optional API key
        if openai_api_key:
            self.embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
            self.openai_client = OpenAI(api_key=openai_api_key)
        else:
            self.embeddings = OpenAIEmbeddings()
            self.openai_client = OpenAI()
        
        self.faiss_index_path = "faiss_index"
    
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """
        Extract text content from a PDF file.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Extracted text as string
        """
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = pypdf.PdfReader(file)
                text = ""
                
                for page_num in range(len(pdf_reader.pages)):
                    page = pdf_reader.pages[page_num]
                    page_text = page.extract_text()
                    if page_text:
                        text += f"\n--- Page {page_num + 1} ---\n"
                        text += page_text
                        text += "\n"
                
                logger.info(f"Successfully extracted text from {len(pdf_reader.pages)} pages")
                return text
                
        except Exception as e:
            logger.error(f"Error extracting text from PDF: {str(e)}")
            raise
    
    def clean_text(self, text: str) -> str:
        """
        Clean and preprocess extracted text.
        
        Args:
            text: Raw extracted text
            
        Returns:
            Cleaned text
        """
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s\.\,\!\?\;\:\-\(\)]', '', text)
        
        # Remove page markers
        text = re.sub(r'--- Page \d+ ---', '', text)
        
        return text.strip()
    
    def create_chunks(self, text: str) -> List[Dict[str, str]]:
        """
        Split text into overlapping chunks using LangChain RecursiveCharacterTextSplitter.
        
        Args:
            text: Input text to chunk
            
        Returns:
            List of dictionaries containing chunked text with metadata
        """
        cleaned_text = self.clean_text(text)
        
        # Use LangChain text splitter
        chunks_text = self.text_splitter.split_text(cleaned_text)
        
        chunks = []
        for i, chunk_text in enumerate(chunks_text):
            chunks.append({
                'id': i,
                'text': chunk_text,
                'length': len(chunk_text)
            })
        
        logger.info(f"Created {len(chunks)} text chunks using RecursiveCharacterTextSplitter")
        return chunks
    
    def create_vector_store(self, chunks: List[Dict[str, str]]) -> FAISS:
        """
        Create FAISS vector store from text chunks.
        
        Args:
            chunks: List of text chunks with metadata
            
        Returns:
            FAISS vector store
        """
        texts = [chunk['text'] for chunk in chunks]
        metadatas = [{'id': chunk['id'], 'length': chunk['length']} for chunk in chunks]
        
        # Create FAISS vector store
        vector_store = FAISS.from_texts(
            texts=texts,
            embedding=self.embeddings,
            metadatas=metadatas
        )
        
        logger.info(f"Created FAISS vector store with {len(texts)} documents")
        return vector_store
    
    def save_vector_store(self, vector_store: FAISS, index_name: str = "pdf_index"):
        """
        Save FAISS vector store to local directory.
        
        Args:
            vector_store: FAISS vector store to save
            index_name: Name for the saved index
        """
        os.makedirs(self.faiss_index_path, exist_ok=True)
        save_path = os.path.join(self.faiss_index_path, index_name)
        vector_store.save_local(save_path)
        logger.info(f"FAISS index saved to {save_path}")
    
    def load_vector_store(self, index_name: str = "pdf_index") -> FAISS:
        """
        Load FAISS vector store from local directory.
        
        Args:
            index_name: Name of the saved index
            
        Returns:
            Loaded FAISS vector store
        """
        load_path = os.path.join(self.faiss_index_path, index_name)
        if not os.path.exists(load_path):
            raise FileNotFoundError(f"FAISS index not found at {load_path}")
        
        vector_store = FAISS.load_local(load_path, self.embeddings, allow_dangerous_deserialization=True)
        logger.info(f"FAISS index loaded from {load_path}")
        return vector_store

    def process_pdf(self, pdf_path: str, save_index: bool = True, index_name: str = "pdf_index") -> FAISS:
        """
        Complete pipeline to process PDF: extract, clean, chunk text, and create FAISS index.
        
        Args:
            pdf_path: Path to the PDF file
            save_index: Whether to save the FAISS index locally
            index_name: Name for the FAISS index
            
        Returns:
            FAISS vector store
        """
        logger.info(f"Processing PDF: {pdf_path}")
        
        # Extract text
        raw_text = self.extract_text_from_pdf(pdf_path)
        
        # Create chunks
        chunks = self.create_chunks(raw_text)
        
        # Create vector store
        vector_store = self.create_vector_store(chunks)
        
        # Save index if requested
        if save_index:
            self.save_vector_store(vector_store, index_name)
        
        logger.info(f"PDF processing complete. Total chunks: {len(chunks)}")
        return vector_store
    
    def query_with_rag(self, vector_store: FAISS, query: str, k: int = 3, model: str = "gpt-3.5-turbo") -> str:
        """
        Perform RAG query: search vector store and generate answer using OpenAI.
        
        Args:
            vector_store: FAISS vector store to search
            query: User query
            k: Number of relevant chunks to retrieve
            model: OpenAI model to use for generation
            
        Returns:
            Generated answer based on retrieved context
        """
        # Retrieve relevant chunks
        results = vector_store.similarity_search(query, k=k)
        
        if not results:
            return "No relevant information found in the document."
        
        # Combine retrieved chunks as context
        context = "\n\n".join([doc.page_content for doc in results])
        
        # Create prompt for OpenAI
        system_prompt = """You are a helpful Business analyst assistant specialized on IEEE Standard SRS document. You specialize in analyzing technical documents, contracts, and official documentation. 

Your role is to:
- Provide accurate, detailed answers based solely on the provided context
- Maintain a professional and informative tone
- Cite specific information when possible
- If the context doesn't contain enough information to answer the question, clearly state this limitation
- Focus on being precise and factual"""

        user_prompt = f"""Based on the following context from the document, please answer the user's question.

Context:
{context}

Question: {query}

Please provide a comprehensive answer based on the context above. If the context doesn't contain sufficient information to fully answer the question, please indicate what information is missing."""

        try:
            response = self.openai_client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1,
                max_tokens=1000
            )
            
            answer = response.choices[0].message.content
            logger.info(f"Generated answer for query: {query[:50]}...")
            return answer
            
        except Exception as e:
            logger.error(f"Error generating answer: {str(e)}")
            return f"Error generating answer: {str(e)}"
