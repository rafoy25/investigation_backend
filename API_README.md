# Vector Database FastAPI

A comprehensive FastAPI application for document vectorization, semantic search, and AI-powered question answering.

## Features

✅ **Document Processing**: Support for PDF, DOCX, TXT, MD files
✅ **Text Vectorization**: Direct text string vectorization
✅ **Background Processing**: Non-blocking operations using FastAPI BackgroundTasks
✅ **Semantic Search**: Vector similarity search with configurable filters
✅ **AI Q&A**: Question answering with source citations using Google Gemini
✅ **Document Management**: CRUD operations for documents
✅ **Task Monitoring**: Real-time status tracking for background operations

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set up environment variables:
```bash
# Required for Q&A functionality
export GOOGLE_API_KEY="your_google_api_key"

# Optional Milvus configuration
export MILVUS_HOST="localhost"
export MILVUS_PORT="19530"
export MILVUS_COLLECTION="document_embeddings"
```

3. Start Milvus (if not already running):
```bash
# Using Docker
docker run -d --name milvus_cpu \
  -p 19530:19530 \
  -v $(pwd)/milvus:/var/lib/milvus \
  milvusdb/milvus:v2.3.4-cpu
```

## Usage

### Start the API Server

```bash
python fastapi_app.py
```

The API will be available at `http://localhost:8000`

Interactive documentation: `http://localhost:8000/docs`

### API Endpoints

#### Health & Status
- `GET /` - API information
- `GET /health` - Health check with service status
- `GET /stats` - Database and system statistics

#### Document Operations
- `POST /documents/upload` - Upload document (PDF, DOCX, TXT, MD)
- `GET /documents` - List all documents
- `PATCH /documents/{document_id}/status` - Update document status
- `DELETE /documents/{document_id}` - Delete document

#### Text Processing
- `POST /text/vectorize` - Vectorize raw text string
- `POST /search` - Semantic search for similar content
- `POST /qa/ask` - Ask questions with AI-generated answers

#### Task Management
- `GET /tasks/{task_id}` - Get task status
- `GET /tasks` - List all tasks
- `DELETE /tasks/{task_id}` - Delete task

## API Examples

### 1. Upload Document

```bash
curl -X POST "http://localhost:8000/documents/upload" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@document.pdf"
```

**Response:**
```json
{
  "task_id": "doc_20231201_140530_document.pdf",
  "message": "Document upload successful. Processing started.",
  "filename": "document.pdf",
  "file_size": 1024000,
  "status_url": "/tasks/doc_20231201_140530_document.pdf"
}
```

### 2. Vectorize Text

```bash
curl -X POST "http://localhost:8000/text/vectorize" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Machine learning is transforming how we process information...",
    "document_id": "ml_article_001",
    "metadata": {"author": "John Doe", "category": "AI"}
  }'
```

### 3. Semantic Search

```bash
curl -X POST "http://localhost:8000/search" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "machine learning applications",
    "limit": 5,
    "active_documents_only": true
  }'
```

**Response:**
```json
{
  "query": "machine learning applications",
  "results": [
    {
      "score": 0.892,
      "document_id": "ml_article_001",
      "filename": "ml_document.pdf",
      "chunk_text": "Machine learning applications span across...",
      "metadata": {"author": "John Doe"}
    }
  ],
  "count": 1,
  "timestamp": "2023-12-01T14:05:30"
}
```

### 4. Ask Question

```bash
curl -X POST "http://localhost:8000/qa/ask" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What are the main benefits of machine learning?",
    "limit": 3,
    "active_documents_only": true
  }'
```

**Response:**
```json
{
  "question": "What are the main benefits of machine learning?",
  "answer": "Based on the provided context, the main benefits of machine learning include [1] automated pattern recognition, [2] predictive analytics capabilities, and [3] improved decision-making processes...",
  "sources": [
    {
      "citation_id": 1,
      "filename": "ml_benefits.pdf",
      "relevance_score": 0.945,
      "preview_text": "Machine learning enables automated pattern recognition...",
      "document_id": "ml_article_001"
    }
  ],
  "metadata": {
    "timestamp": "2023-12-01T14:10:00",
    "model_used": "gemini-1.5-flash",
    "tokens_used": 350,
    "chunks_retrieved": 3
  }
}
```

### 5. Monitor Task Progress

```bash
curl "http://localhost:8000/tasks/doc_20231201_140530_document.pdf"
```

**Response:**
```json
{
  "task_id": "doc_20231201_140530_document.pdf",
  "status": "processing",
  "message": "Generating embeddings...",
  "progress": 60,
  "result": null,
  "error": null,
  "created_at": "2023-12-01T14:05:30",
  "updated_at": "2023-12-01T14:06:15"
}
```

### 6. List Documents

```bash
curl "http://localhost:8000/documents?active_only=true"
```

## Background Processing

All document uploads and text vectorization operations are processed in the background to prevent API blocking:

1. **Submit Request** → Get task ID immediately
2. **Monitor Progress** → Use task ID to check status
3. **Get Results** → Retrieve final results when completed

Task statuses:
- `pending`: Task queued for processing
- `processing`: Currently being processed
- `completed`: Successfully completed
- `failed`: Failed with error details

## Testing

Run the test script to verify all endpoints:

```bash
python test_api.py
```

This will test:
- Health check
- Text vectorization
- Document upload
- Semantic search
- Q&A functionality
- Document management
- Statistics

## Configuration

### Environment Variables

- `GOOGLE_API_KEY`: Required for Q&A functionality
- `OPENAI_API_KEY`: Optional, for OpenAI embeddings
- `MILVUS_HOST`: Milvus server host (default: localhost)
- `MILVUS_PORT`: Milvus server port (default: 19530)
- `MILVUS_COLLECTION`: Collection name (default: document_embeddings)

### Embedding Models

The API uses `all-MiniLM-L6-v2` by default for generating embeddings. This model provides:
- **Dimension**: 384
- **Performance**: Fast inference
- **Quality**: Good for semantic search
- **Size**: Lightweight (~90MB)

### Production Considerations

For production deployment:

1. **Use Redis for task storage** instead of in-memory storage
2. **Implement Celery** for distributed task processing
3. **Add authentication** and rate limiting
4. **Configure CORS** properly for your frontend
5. **Use environment-specific configuration**
6. **Implement proper logging** and monitoring
7. **Add database connection pooling**

## Error Handling

The API includes comprehensive error handling:

- **400**: Bad Request (invalid file type, malformed request)
- **404**: Resource not found (document, task)
- **500**: Internal server error
- **503**: Service unavailable (missing dependencies)

## Performance

- **Concurrent Processing**: Multiple documents can be processed simultaneously
- **Chunked Processing**: Large documents are split into manageable chunks
- **Vector Search**: Optimized using Milvus IVF_FLAT index
- **Memory Efficient**: Streaming file processing

## Support

For issues or questions:
1. Check the logs for detailed error messages
2. Verify all dependencies are installed
3. Ensure Milvus is running and accessible
4. Test with the provided test script