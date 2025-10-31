# Vector Database Comparison Tool

A comprehensive application for comparing different vector databases by uploading files, generating embeddings, and evaluating performance metrics.

## Features

### Supported File Formats
- **Text files** (.txt)
- **PDF documents** (.pdf)
- **Word documents** (.docx)
- **JSON files** (.json)
- **CSV files** (.csv)

### Supported Vector Databases
- **ChromaDB** - Open-source embedding database
- **FAISS** - Facebook AI Similarity Search
- **Pinecone** - Managed vector database service
- **Weaviate** - Open-source vector search engine
- **Qdrant** - Vector similarity search engine

### Supported Embedding Models
- **Sentence Transformers** - Local transformer models
- **OpenAI** - GPT-based embeddings (requires API key)
- **Cohere** - Cohere's embedding models (requires API key)

## Installation

1. Clone or download this repository
2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

### Optional Dependencies

Install only the vector databases and embedding models you plan to use:

**Vector Databases:**
```bash
# ChromaDB
pip install chromadb

# FAISS
pip install faiss-cpu

# Pinecone
pip install pinecone-client

# Weaviate
pip install weaviate-client

# Qdrant
pip install qdrant-client
```

**Embedding Models:**
```bash
# Sentence Transformers (recommended for local usage)
pip install sentence-transformers

# OpenAI (requires API key)
pip install openai

# Cohere (requires API key)
pip install cohere
```

## Usage

1. Start the Streamlit app:
```bash
streamlit run app.py
```

2. Open your browser and navigate to the displayed URL (usually `http://localhost:8501`)

3. **Configure the application:**
   - Upload a file using the sidebar file uploader
   - Select an embedding model
   - Choose vector databases to compare
   - Provide API keys if using OpenAI or Cohere

4. **Generate embeddings and compare:**
   - Click "Generate Embeddings & Compare Databases"
   - View performance metrics and search results
   - Analyze the comparison charts

## Performance Metrics

The tool evaluates vector databases on:

- **Insert Time** - Time to store vectors in the database
- **Search Time** - Time to perform similarity search
- **Success Rate** - Whether operations completed successfully
- **Search Quality** - Sample results from similarity searches

## Configuration

### API Keys
- **OpenAI**: Get your API key from [OpenAI Platform](https://platform.openai.com/)
- **Cohere**: Get your API key from [Cohere Dashboard](https://dashboard.cohere.ai/)

### Database Configuration
- Most databases use default local configurations
- For production use, configure connection parameters in the code
- Pinecone and Weaviate may require additional setup for cloud instances

## File Processing

The application automatically:
1. Processes different file formats
2. Chunks large documents (1000 characters per chunk)
3. Generates embeddings for each chunk
4. Stores vectors with document metadata
5. Performs similarity searches for comparison

## Troubleshooting

### Common Issues

1. **"No embedding models available"**
   - Install at least one embedding model: `pip install sentence-transformers`

2. **"No vector databases available"**
   - Install at least one vector database: `pip install chromadb faiss-cpu`

3. **API Key errors**
   - Ensure valid API keys for OpenAI/Cohere
   - Check API key permissions and usage limits

4. **File processing errors**
   - Ensure file format is supported
   - Check file is not corrupted
   - Try with smaller files first

### Performance Tips

- **For large files**: Use Sentence Transformers for faster local processing
- **For accuracy**: Use OpenAI embeddings with larger models
- **For speed**: Use FAISS or ChromaDB for local vector storage
- **For production**: Consider Pinecone or Qdrant for managed solutions

## Architecture

```
app.py
├── FileProcessor - Handles different file formats
├── EmbeddingManager - Manages different embedding models
├── VectorDatabase - Abstracts vector database operations
└── Streamlit UI - Provides interactive interface
```

## License

This project is open source and available under the MIT License.

## Contributing

Feel free to submit issues, feature requests, or pull requests to improve the tool.
