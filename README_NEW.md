# Document QNA System with Milvus Vector Database

A comprehensive end-to-end system for document ingestion, embedding generation, and question-answering using Milvus vector database.

## 🚀 Features

- **Document Upload & Processing**: Support for PDF, Word documents, text files, and markdown
- **Vector Storage**: Uses Milvus vector database for efficient similarity search
- **Embedding Generation**: Supports both SentenceTransformers and OpenAI embeddings
- **Question & Answer**: AI-powered QNA with citation and source tracking
- **Document Management**: Activate/deactivate documents, view statistics
- **Web Interface**: Clean, intuitive Streamlit interface
- **Citation Display**: Shows source documents and relevance scores

## 📋 Prerequisites

- Python 3.8+
- Docker and Docker Compose
- OpenAI API key (for QNA functionality)

## 🛠️ Quick Start

### 1. Clone and Setup

```bash
# Navigate to your project directory
cd "C:\Datamicron\MyDMFolder\PERSONAL PROJECTS\VECTOR DATABASE"

# Run the automated setup
python setup.py
```

### 2. Configure Environment

```bash
# Copy the environment template
cp .env.example .env

# Edit .env and add your OpenAI API key
# OPENAI_API_KEY=your_key_here
```

### 3. Start the Application

```bash
# Start the Streamlit app
streamlit run main_app.py
```

### 4. Access the Interface

Open your browser to: `http://localhost:8501`

## 📁 Project Structure

```
├── main_app.py                    # Main Streamlit application
├── document_processor.py          # Document text extraction and chunking
├── milvus_client.py               # Milvus vector database client
├── embedding_service.py           # Embedding generation service
├── qna_agent.py                   # Question-answering agent
├── setup.py                       # Automated setup script
├── requirements.txt               # Python dependencies
├── milvus-standalone-docker-compose-gpu.yml  # Milvus Docker setup
└── README_NEW.md                  # This file
```

## 🔧 Manual Setup

If the automated setup fails, follow these manual steps:

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Start Milvus

```bash
# Start Milvus using Docker Compose
docker-compose -f milvus-standalone-docker-compose-gpu.yml up -d

# Verify containers are running
docker ps
```

### 3. Set Environment Variables

```bash
# Windows
set OPENAI_API_KEY=your_key_here

# Linux/Mac
export OPENAI_API_KEY=your_key_here
```

### 4. Run the Application

```bash
streamlit run main_app.py
```

## 📖 How to Use

### Upload Documents

1. Go to the "Upload Documents" tab
2. Select a file (PDF, DOCX, TXT, or MD)
3. Click "Process Document"
4. Wait for processing to complete

### Manage Documents

1. Go to the "Manage Documents" tab
2. View all uploaded documents
3. Activate/deactivate documents as needed
4. Delete documents if necessary

### Ask Questions

1. Go to the "Ask Questions" tab
2. Type your question in the input field
3. Optionally filter by specific documents
4. Click "Ask Question"
5. Review the answer with citations and sources

## 🔍 System Components

### Document Processor
- Extracts text from various file formats
- Splits text into chunks for embedding
- Generates unique document IDs

### Milvus Client
- Manages vector database connections
- Stores and retrieves embeddings
- Handles document metadata

### Embedding Service
- Generates embeddings using SentenceTransformers or OpenAI
- Supports different embedding models
- Batch processing capabilities

### QNA Agent
- Retrieval-augmented generation (RAG)
- Citation tracking and source attribution
- OpenAI GPT integration

## ⚙️ Configuration

### Embedding Models

The system supports two types of embedding models:

1. **SentenceTransformers** (default, free):
   - Model: `all-MiniLM-L6-v2`
   - Dimensions: 384
   - No API key required

2. **OpenAI** (requires API key):
   - Model: `text-embedding-ada-002`
   - Dimensions: 1536
   - Requires OPENAI_API_KEY

### Milvus Configuration

Default connection settings:
- Host: `localhost`
- Port: `19530`
- Collection: `document_embeddings`

### Text Chunking

Default settings:
- Chunk size: 1000 characters
- Chunk overlap: 200 characters
- Separator priority: paragraph → line → space → character

## 🐳 Docker Services

The system uses the following Docker containers:

- **milvus-standalone**: Main Milvus service (port 19530)
- **milvus-etcd**: etcd for Milvus metadata
- **milvus-minio**: MinIO for Milvus storage (port 9000, 9001)

## 📊 Monitoring

### System Status
- Check the sidebar for system initialization status
- View total embeddings and document counts
- Monitor API usage and token consumption

### Logs
- Application logs appear in the Streamlit interface
- Docker logs: `docker-compose logs milvus-standalone`

## 🔧 Troubleshooting

### Common Issues

1. **Milvus Connection Failed**
   ```bash
   # Check if containers are running
   docker ps
   
   # Restart Milvus
   docker-compose -f milvus-standalone-docker-compose-gpu.yml restart
   ```

2. **OpenAI API Errors**
   - Verify your API key is correct
   - Check your OpenAI account has sufficient credits
   - Ensure the API key has proper permissions

3. **Document Processing Fails**
   - Check file format is supported
   - Verify file isn't corrupted
   - Check file size limits

4. **Memory Issues**
   - Large documents may require more memory
   - Consider reducing chunk size
   - Process documents one at a time

### Performance Optimization

1. **Embedding Generation**
   - Use SentenceTransformers for faster processing
   - Batch process multiple documents
   - Consider GPU acceleration

2. **Vector Search**
   - Adjust search parameters in Milvus
   - Use appropriate index types
   - Monitor query performance

## 🤝 Contributing

Feel free to contribute by:
- Reporting bugs
- Suggesting features
- Submitting pull requests
- Improving documentation

## 📄 License

This project is licensed under the MIT License.

## 🆘 Support

For issues and questions:
1. Check the troubleshooting section
2. Review Docker and application logs
3. Ensure all dependencies are installed correctly

## 🔮 Future Enhancements

- Support for more document formats
- Advanced search filters
- Document versioning
- Multi-user support
- API endpoints
- Advanced analytics dashboard