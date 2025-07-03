from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
from typing import Optional
import os
import logging
from datetime import datetime
from contextlib import asynccontextmanager

# LlamaIndex and related imports
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext, Settings
from llama_index.vector_stores.pinecone import PineconeVectorStore
from pinecone import Pinecone, PodSpec
from llama_index.embeddings.gemini import GeminiEmbedding
from llama_index.llms.gemini import Gemini

# Environment and configuration
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Global variables
query_engine = None

class ChatbotConfig:
    """Configuration class for chatbot settings"""
    def __init__(self):
        self.pinecone_api_key = os.getenv("PINECONE_API_KEY")
        self.pinecone_environment = os.getenv("PINECONE_ENVIRONMENT", "us-east-1-aws")
        self.pinecone_index_name = os.getenv("PINECONE_INDEX", "chatbot")
        self.gemini_api_key = os.getenv("GEMINI_API_KEY")
        self.embedding_dimension = int(os.getenv("EMBEDDING_DIMENSION", "768"))
        self.pdf_path = os.getenv("PDF_PATH", "/home/mohankalyan/Downloads/KosaraMohanKalyanResume-1.pdf")
        
        # Validate required environment variables
        if not self.pinecone_api_key:
            raise ValueError("PINECONE_API_KEY environment variable is required")
        if not self.gemini_api_key:
            raise ValueError("GEMINI_API_KEY environment variable is required")

config = ChatbotConfig()

# Pydantic models for request/response validation
class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=1000, description="The question to ask the chatbot")
    context: Optional[str] = Field(None, max_length=500, description="Additional context for the query")
    
    @validator('question')
    def validate_question(cls, v):
        if not v.strip():
            raise ValueError('Question cannot be empty or whitespace only')
        return v.strip()

class QueryResponse(BaseModel):
    answer: str = Field(..., description="The chatbot's response")
    timestamp: datetime = Field(default_factory=datetime.now, description="Response timestamp")
    query_id: str = Field(..., description="Unique identifier for this query")
    status: str = Field(default="success", description="Response status")

class ErrorResponse(BaseModel):
    error: str = Field(..., description="Error message")
    error_code: str = Field(..., description="Error code")
    timestamp: datetime = Field(default_factory=datetime.now, description="Error timestamp")
    status: str = Field(default="error", description="Response status")

async def initialize_chatbot():
    """Initialize the chatbot components"""
    global query_engine
    
    try:
        logger.info("Initializing chatbot components...")
        
        # Initialize Pinecone
        pc = Pinecone(api_key=config.pinecone_api_key)
        
        # Setup LLM and embeddings
        llm = Gemini(
            model="models/gemini-1.5-flash",
            api_key=config.gemini_api_key
        )
        
        embed_model = GeminiEmbedding(
            model_name="models/embedding-001",
            api_key=config.gemini_api_key
        )
        
        # Configure global settings
        Settings.llm = llm
        Settings.embed_model = embed_model
        Settings.chunk_size = 500
        Settings.chunk_overlap = 200
        
        # Create or get Pinecone index
        if config.pinecone_index_name not in pc.list_indexes().names():
            logger.info(f"Creating Pinecone index: {config.pinecone_index_name}")
            pc.create_index(
                name=config.pinecone_index_name,
                dimension=config.embedding_dimension,
                metric="cosine",
                spec=PodSpec(environment=config.pinecone_environment)
            )
            
            # Load and index the document
            if os.path.exists(config.pdf_path):
                logger.info(f"Loading document: {config.pdf_path}")
                documents = SimpleDirectoryReader(input_files=[config.pdf_path]).load_data()
                
                # Initialize vector store and create index
                pinecone_index = pc.Index(config.pinecone_index_name)
                vector_store = PineconeVectorStore(pinecone_index=pinecone_index)
                storage_context = StorageContext.from_defaults(vector_store=vector_store)
                
                # Create index from documents
                index = VectorStoreIndex.from_documents(
                    documents,
                    storage_context=storage_context,
                )
                logger.info("Document indexed successfully")
            else:
                logger.warning(f"PDF file not found at: {config.pdf_path}")
                raise FileNotFoundError(f"PDF file not found at: {config.pdf_path}")
        else:
            logger.info(f"Using existing Pinecone index: {config.pinecone_index_name}")
            # Connect to existing index
            pinecone_index = pc.Index(config.pinecone_index_name)
            vector_store = PineconeVectorStore(pinecone_index=pinecone_index)
            storage_context = StorageContext.from_defaults(vector_store=vector_store)
            
            # Create index from existing vector store
            index = VectorStoreIndex.from_vector_store(
                vector_store=vector_store,
                storage_context=storage_context
            )
        
        # Create query engine
        query_engine = index.as_query_engine()
        
        logger.info("Chatbot initialization completed successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize chatbot: {str(e)}")
        raise

async def cleanup_chatbot():
    """Cleanup chatbot resources"""
    global query_engine
    try:
        logger.info("Cleaning up chatbot resources...")
        query_engine = None
        logger.info("Cleanup completed successfully")
    except Exception as e:
        logger.error(f"Error during cleanup: {str(e)}")

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan"""
    # Startup
    await initialize_chatbot()
    yield
    # Shutdown
    await cleanup_chatbot()

# Initialize FastAPI app
app = FastAPI(
    title="RAG Chatbot API",
    description="A FastAPI backend for querying a RAG chatbot using LlamaIndex, Pinecone, and Gemini AI",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["POST"],
    allow_headers=["*"],
)

# Exception handlers
@app.exception_handler(ValueError)
async def value_error_handler(request, exc):
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content=ErrorResponse(
            error=str(exc),
            error_code="VALIDATION_ERROR"
        ).dict()
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {str(exc)}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=ErrorResponse(
            error="An internal server error occurred",
            error_code="INTERNAL_ERROR"
        ).dict()
    )

# Main API Route - Single POST Method
@app.post("/query", response_model=QueryResponse)
async def query_chatbot(request: QueryRequest):
    """
    Query the RAG chatbot with a question
    
    - **question**: The question to ask the chatbot (required)
    - **context**: Optional additional context for the query
    
    Returns the chatbot's response based on the indexed document.
    """
    try:
        # Check if chatbot is initialized
        if query_engine is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Chatbot is not initialized. Please try again later."
            )
        
        # Generate unique query ID
        query_id = f"query_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        
        # Prepare the query
        query_text = request.question
        if request.context:
            query_text = f"Context: {request.context}\n\nQuestion: {request.question}"
        
        logger.info(f"Processing query {query_id}: {request.question}")
        
        # Query the engine
        response = query_engine.query(query_text)
        
        logger.info(f"Query {query_id} completed successfully")
        
        return QueryResponse(
            answer=str(response),
            query_id=query_id
        )
        
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process query: {str(e)}"
        )

if __name__ == "__main__":
    import uvicorn
    
    # Configuration for uvicorn
    uvicorn.run(
        "main:app",  # Replace "main" with your actual filename
        host="0.0.0.0",
        port=8000,
        reload=True,  # Set to False in production
        log_level="info"
    )