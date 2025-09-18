from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks, Depends, Query
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional, Union
import uvicorn
import os
import tempfile
from datetime import datetime
import asyncio
import logging
from contextlib import asynccontextmanager

# Import our custom modules
from document_processor import DocumentProcessor
from milvus_client import MilvusClient
from embedding_service import EmbeddingService, DocumentEmbeddingManager
from qna_agent import QNAAgent
from dotenv import load_dotenv

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables for services
milvus_client = None
embedding_service = None
embedding_manager = None
qna_agent = None
document_processor = None

# Pydantic models for request/response
class TextVectorizeRequest(BaseModel):
    text: str = Field(..., description="Text to vectorize")
    document_id: Optional[str] = Field(None, description="Optional document ID")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Optional metadata")

class QueryRequest(BaseModel):
    query: str = Field(..., description="Query text")
    limit: int = Field(default=5, ge=1, le=50, description="Number of results to return")
    active_documents_only: bool = Field(default=True, description="Only search active documents")
    document_ids: Optional[List[str]] = Field(None, description="Filter by specific document IDs")

class QARequest(BaseModel):
    question: str = Field(..., description="Question to ask")
    limit: int = Field(default=5, ge=1, le=50, description="Number of chunks to retrieve")
    active_documents_only: bool = Field(default=True, description="Only use active documents")
    document_ids: Optional[List[str]] = Field(None, description="Filter by specific document IDs")

class DocumentStatusUpdate(BaseModel):
    active: bool = Field(..., description="Active status")

class ProcessingStatus(BaseModel):
    task_id: str
    status: str  # pending, processing, completed, failed
    message: str
    progress: int  # 0-100
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    created_at: datetime
    updated_at: datetime

# In-memory task storage (in production, use Redis or database)
task_storage: Dict[str, ProcessingStatus] = {}

async def initialize_services():
    """Initialize all services"""
    global milvus_client, embedding_service, embedding_manager, qna_agent, document_processor

    try:
        logger.info("Initializing services...")

        # Initialize document processor
        document_processor = DocumentProcessor()

        # Initialize Milvus client
        milvus_client = MilvusClient(
            host=os.getenv("MILVUS_HOST", "localhost"),
            port=os.getenv("MILVUS_PORT", "19530"),
            collection_name=os.getenv("MILVUS_COLLECTION", "document_embeddings")
        )
        milvus_client.connect()

        # Initialize embedding service
        embedding_service = EmbeddingService(
            model_type="sentence_transformer",
            model_name="all-MiniLM-L6-v2"
        )

        # Initialize embedding manager
        embedding_manager = DocumentEmbeddingManager(embedding_service)

        # Initialize QNA agent if Google API key is available
        try:
            if os.getenv("GOOGLE_API_KEY"):
                qna_agent = QNAAgent(
                    milvus_client=milvus_client,
                    embedding_manager=embedding_manager,
                    model_type="gemini"
                )
                logger.info("QNA agent initialized successfully")
            else:
                logger.warning("Google API key not found. QNA features will be disabled.")
        except Exception as e:
            logger.warning(f"QNA agent initialization failed: {e}")

        logger.info("All services initialized successfully")

    except Exception as e:
        logger.error(f"Service initialization failed: {e}")
        raise

async def cleanup_services():
    """Cleanup services on shutdown"""
    global milvus_client
    if milvus_client:
        milvus_client.disconnect()
    logger.info("Services cleaned up")

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    await initialize_services()
    yield
    # Shutdown
    await cleanup_services()

# Create FastAPI app
app = FastAPI(
    title="Vector Database API",
    description="API for document vectorization, retrieval, and Q&A generation",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Dependency to get services
def get_services():
    if not all([milvus_client, embedding_service, embedding_manager, document_processor]):
        raise HTTPException(status_code=503, detail="Services not initialized")
    return {
        "milvus_client": milvus_client,
        "embedding_service": embedding_service,
        "embedding_manager": embedding_manager,
        "document_processor": document_processor,
        "qna_agent": qna_agent
    }

# Background task functions
async def process_document_task(task_id: str, filename: str, file_content: bytes):
    """Background task to process a document"""
    try:
        # Update task status
        task_storage[task_id].status = "processing"
        task_storage[task_id].progress = 10
        task_storage[task_id].message = "Extracting text..."
        task_storage[task_id].updated_at = datetime.now()

        # Process document
        document_data = document_processor.process_document(filename, file_content)

        task_storage[task_id].progress = 40
        task_storage[task_id].message = "Generating embeddings..."
        task_storage[task_id].updated_at = datetime.now()

        # Generate embeddings
        document_data = embedding_manager.process_document_chunks(document_data)

        task_storage[task_id].progress = 70
        task_storage[task_id].message = "Storing in vector database..."
        task_storage[task_id].updated_at = datetime.now()

        # Store in Milvus
        success = milvus_client.insert_embeddings(document_data, document_data['embeddings'])

        if success:
            task_storage[task_id].status = "completed"
            task_storage[task_id].progress = 100
            task_storage[task_id].message = "Document processed successfully"
            task_storage[task_id].result = {
                "document_id": document_data['id'],
                "filename": document_data['filename'],
                "chunk_count": document_data['chunk_count'],
                "embeddings_count": len(document_data['embeddings'])
            }
        else:
            raise Exception("Failed to store document in vector database")

    except Exception as e:
        task_storage[task_id].status = "failed"
        task_storage[task_id].error = str(e)
        task_storage[task_id].message = f"Failed: {str(e)}"
        logger.error(f"Task {task_id} failed: {e}")
    finally:
        task_storage[task_id].updated_at = datetime.now()

async def process_text_vectorization_task(task_id: str, text: str, document_id: Optional[str], metadata: Dict[str, Any]):
    """Background task to vectorize text"""
    try:
        task_storage[task_id].status = "processing"
        task_storage[task_id].progress = 20
        task_storage[task_id].message = "Chunking text..."
        task_storage[task_id].updated_at = datetime.now()

        # Create chunks from text
        chunks = document_processor.chunk_text(text)

        task_storage[task_id].progress = 50
        task_storage[task_id].message = "Generating embeddings..."
        task_storage[task_id].updated_at = datetime.now()

        # Generate embeddings
        embeddings = embedding_service.generate_embeddings(chunks)

        task_storage[task_id].progress = 80
        task_storage[task_id].message = "Storing in vector database..."
        task_storage[task_id].updated_at = datetime.now()

        # Create document data structure
        doc_id = document_id or f"text_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        document_data = {
            'id': doc_id,
            'filename': f"{doc_id}.txt",
            'file_type': '.txt',
            'text': text,
            'chunks': chunks,
            'chunk_count': len(chunks),
            'processed_at': datetime.now().isoformat(),
            'file_size': len(text.encode('utf-8')),
            'active': True,
            'embeddings': embeddings,
            **metadata
        }

        # Store in Milvus
        success = milvus_client.insert_embeddings(document_data, embeddings)

        if success:
            task_storage[task_id].status = "completed"
            task_storage[task_id].progress = 100
            task_storage[task_id].message = "Text vectorized successfully"
            task_storage[task_id].result = {
                "document_id": doc_id,
                "chunk_count": len(chunks),
                "embeddings_count": len(embeddings)
            }
        else:
            raise Exception("Failed to store text embeddings in vector database")

    except Exception as e:
        task_storage[task_id].status = "failed"
        task_storage[task_id].error = str(e)
        task_storage[task_id].message = f"Failed: {str(e)}"
        logger.error(f"Task {task_id} failed: {e}")
    finally:
        task_storage[task_id].updated_at = datetime.now()

# API Endpoints

@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Vector Database API",
        "version": "1.0.0",
        "status": "running",
        "timestamp": datetime.now().isoformat()
    }

@app.get("/health", response_model=Dict[str, Any])
async def health_check():
    """Health check endpoint"""
    try:
        services = get_services()

        # Check Milvus connection
        milvus_stats = services["milvus_client"].get_collection_stats()

        return {
            "status": "healthy",
            "services": {
                "milvus": "connected",
                "embedding_service": "loaded",
                "qna_agent": "available" if services["qna_agent"] else "unavailable"
            },
            "database_stats": milvus_stats,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Service unavailable: {e}")

@app.post("/documents/upload", response_model=Dict[str, Any])
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    services: Dict = Depends(get_services)
):
    """Upload and process a document (PDF, DOCX, TXT, MD)"""
    try:
        # Validate file type
        if not services["document_processor"].is_supported_file(file.filename):
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type. Supported: {services['document_processor'].supported_types}"
            )

        # Read file content
        file_content = await file.read()

        # Create task
        task_id = f"doc_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{file.filename}"
        task_storage[task_id] = ProcessingStatus(
            task_id=task_id,
            status="pending",
            message="Document upload received",
            progress=0,
            created_at=datetime.now(),
            updated_at=datetime.now()
        )

        # Start background processing
        background_tasks.add_task(process_document_task, task_id, file.filename, file_content)

        return {
            "task_id": task_id,
            "message": "Document upload successful. Processing started.",
            "filename": file.filename,
            "file_size": len(file_content),
            "status_url": f"/tasks/{task_id}"
        }

    except Exception as e:
        logger.error(f"Document upload failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/text/vectorize", response_model=Dict[str, Any])
async def vectorize_text(
    background_tasks: BackgroundTasks,
    request: TextVectorizeRequest,
    services: Dict = Depends(get_services)
):
    """Vectorize a text string directly"""
    try:
        # Create task
        task_id = f"text_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        task_storage[task_id] = ProcessingStatus(
            task_id=task_id,
            status="pending",
            message="Text vectorization request received",
            progress=0,
            created_at=datetime.now(),
            updated_at=datetime.now()
        )

        # Start background processing
        background_tasks.add_task(
            process_text_vectorization_task,
            task_id,
            request.text,
            request.document_id,
            request.metadata
        )

        return {
            "task_id": task_id,
            "message": "Text vectorization started.",
            "text_length": len(request.text),
            "status_url": f"/tasks/{task_id}"
        }

    except Exception as e:
        logger.error(f"Text vectorization failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/search", response_model=Dict[str, Any])
async def search_similar(
    request: QueryRequest,
    services: Dict = Depends(get_services)
):
    """Search for similar document chunks"""
    try:
        # Generate query embedding
        query_embedding = services["embedding_manager"].generate_query_embedding(request.query)

        # Search similar chunks
        results = services["milvus_client"].search_similar(
            query_embedding=query_embedding,
            limit=request.limit,
            active_only=request.active_documents_only,
            document_ids=request.document_ids
        )

        return {
            "query": request.query,
            "results": results,
            "count": len(results),
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Search failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/qa/ask", response_model=Dict[str, Any])
async def ask_question(
    request: QARequest,
    services: Dict = Depends(get_services)
):
    """Ask a question and get an AI-generated answer with sources"""
    try:
        if not services["qna_agent"]:
            raise HTTPException(
                status_code=503,
                detail="QNA service unavailable. Google API key required."
            )

        # Get answer with sources
        response = services["qna_agent"].get_answer_with_sources(
            query=request.question,
            limit=request.limit,
            active_documents_only=request.active_documents_only,
            document_ids=request.document_ids
        )

        return response

    except Exception as e:
        logger.error(f"QA failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/documents", response_model=Dict[str, Any])
async def list_documents(
    active_only: bool = Query(False, description="Filter by active documents only"),
    services: Dict = Depends(get_services)
):
    """List all documents in the vector database"""
    try:
        documents = services["milvus_client"].get_all_documents()

        if active_only:
            documents = [doc for doc in documents if doc.get('active', True)]

        return {
            "documents": documents,
            "count": len(documents),
            "total_available": len(services["milvus_client"].get_all_documents()),
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Document listing failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.patch("/documents/{document_id}/status", response_model=Dict[str, Any])
async def update_document_status(
    document_id: str,
    status_update: DocumentStatusUpdate,
    services: Dict = Depends(get_services)
):
    """Update document active status"""
    try:
        success = services["milvus_client"].update_document_status(document_id, status_update.active)

        if success:
            return {
                "document_id": document_id,
                "active": status_update.active,
                "message": f"Document {'activated' if status_update.active else 'deactivated'} successfully",
                "timestamp": datetime.now().isoformat()
            }
        else:
            raise HTTPException(status_code=404, detail="Document not found or update failed")

    except Exception as e:
        logger.error(f"Document status update failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/documents/{document_id}", response_model=Dict[str, Any])
async def delete_document(
    document_id: str,
    services: Dict = Depends(get_services)
):
    """Delete a document and all its chunks"""
    try:
        success = services["milvus_client"].delete_document(document_id)

        if success:
            return {
                "document_id": document_id,
                "message": "Document deleted successfully",
                "timestamp": datetime.now().isoformat()
            }
        else:
            raise HTTPException(status_code=404, detail="Document not found or deletion failed")

    except Exception as e:
        logger.error(f"Document deletion failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/tasks/{task_id}", response_model=ProcessingStatus)
async def get_task_status(task_id: str):
    """Get the status of a background task"""
    if task_id not in task_storage:
        raise HTTPException(status_code=404, detail="Task not found")

    return task_storage[task_id]

@app.get("/tasks", response_model=List[ProcessingStatus])
async def list_tasks():
    """List all background tasks"""
    return list(task_storage.values())

@app.delete("/tasks/{task_id}", response_model=Dict[str, str])
async def delete_task(task_id: str):
    """Delete a task from storage"""
    if task_id not in task_storage:
        raise HTTPException(status_code=404, detail="Task not found")

    del task_storage[task_id]
    return {"message": "Task deleted successfully", "task_id": task_id}

@app.get("/stats", response_model=Dict[str, Any])
async def get_database_stats(services: Dict = Depends(get_services)):
    """Get database statistics"""
    try:
        stats = services["milvus_client"].get_collection_stats()
        documents = services["milvus_client"].get_all_documents()

        active_docs = len([doc for doc in documents if doc.get('active', True)])

        return {
            "database_stats": stats,
            "document_stats": {
                "total_documents": len(documents),
                "active_documents": active_docs,
                "inactive_documents": len(documents) - active_docs
            },
            "task_stats": {
                "total_tasks": len(task_storage),
                "pending": len([t for t in task_storage.values() if t.status == "pending"]),
                "processing": len([t for t in task_storage.values() if t.status == "processing"]),
                "completed": len([t for t in task_storage.values() if t.status == "completed"]),
                "failed": len([t for t in task_storage.values() if t.status == "failed"])
            },
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Stats retrieval failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(
        "fastapi_app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )