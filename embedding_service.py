from typing import List, Dict, Any, Optional, Union
import numpy as np
from sentence_transformers import SentenceTransformer
import openai
import os
from datetime import datetime

class EmbeddingService:
    """Service for generating embeddings from text"""
    
    def __init__(self, model_type: str = "sentence_transformer", model_name: Optional[str] = None):
        self.model_type = model_type
        self.model = None
        self.embedding_dim = None
        
        if model_type == "sentence_transformer":
            self.model_name = model_name or "all-MiniLM-L6-v2"
            self._load_sentence_transformer()
        elif model_type == "openai":
            self.model_name = model_name or "text-embedding-ada-002"
            self._setup_openai()
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
    
    def _load_sentence_transformer(self):
        """Load Sentence Transformer model"""
        try:
            self.model = SentenceTransformer(self.model_name)
            # Get embedding dimension
            test_embedding = self.model.encode(["test"])
            self.embedding_dim = len(test_embedding[0])
            print(f"Loaded SentenceTransformer model: {self.model_name} (dim: {self.embedding_dim})")
        except Exception as e:
            raise Exception(f"Failed to load SentenceTransformer model: {str(e)}")
    
    def _setup_openai(self):
        """Setup OpenAI client"""
        try:
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY environment variable not set")
            
            openai.api_key = api_key
            self.embedding_dim = 1536  # Default for text-embedding-ada-002
            print(f"Setup OpenAI embeddings: {self.model_name} (dim: {self.embedding_dim})")
        except Exception as e:
            raise Exception(f"Failed to setup OpenAI: {str(e)}")
    
    def generate_embeddings(self, texts: Union[str, List[str]]) -> Union[List[float], List[List[float]]]:
        """Generate embeddings for text(s)"""
        if isinstance(texts, str):
            texts = [texts]
            return_single = True
        else:
            return_single = False
        
        try:
            if self.model_type == "sentence_transformer":
                embeddings = self._generate_sentence_transformer_embeddings(texts)
            elif self.model_type == "openai":
                embeddings = self._generate_openai_embeddings(texts)
            else:
                raise ValueError(f"Unsupported model type: {self.model_type}")
            
            if return_single:
                return embeddings[0]
            return embeddings
            
        except Exception as e:
            raise Exception(f"Failed to generate embeddings: {str(e)}")
    
    def _generate_sentence_transformer_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings using Sentence Transformer"""
        try:
            embeddings = self.model.encode(texts, convert_to_tensor=False)
            return [embedding.tolist() for embedding in embeddings]
        except Exception as e:
            raise Exception(f"SentenceTransformer embedding failed: {str(e)}")
    
    def _generate_openai_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings using OpenAI API"""
        try:
            embeddings = []
            # Process in batches to avoid rate limits
            batch_size = 100
            
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                
                response = openai.Embedding.create(
                    input=batch,
                    model=self.model_name
                )
                
                batch_embeddings = [item['embedding'] for item in response['data']]
                embeddings.extend(batch_embeddings)
            
            return embeddings
        except Exception as e:
            raise Exception(f"OpenAI embedding failed: {str(e)}")
    
    def get_embedding_dim(self) -> int:
        """Get embedding dimension"""
        return self.embedding_dim
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        return {
            'model_type': self.model_type,
            'model_name': self.model_name,
            'embedding_dim': self.embedding_dim,
            'loaded_at': datetime.now().isoformat()
        }

class DocumentEmbeddingManager:
    """Manager for document embedding operations"""
    
    def __init__(self, embedding_service: EmbeddingService):
        self.embedding_service = embedding_service
    
    def process_document_chunks(self, document_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process document chunks and generate embeddings"""
        try:
            chunks = document_data['chunks']
            if not chunks:
                raise ValueError("No chunks found in document data")
            
            print(f"Generating embeddings for {len(chunks)} chunks...")
            
            # Generate embeddings for all chunks
            embeddings = self.embedding_service.generate_embeddings(chunks)
            
            # Add embeddings to document data
            document_data['embeddings'] = embeddings
            document_data['embedding_model'] = self.embedding_service.get_model_info()
            
            print(f"Generated {len(embeddings)} embeddings")
            
            return document_data
            
        except Exception as e:
            raise Exception(f"Failed to process document chunks: {str(e)}")
    
    def generate_query_embedding(self, query: str) -> List[float]:
        """Generate embedding for a query"""
        try:
            return self.embedding_service.generate_embeddings(query)
        except Exception as e:
            raise Exception(f"Failed to generate query embedding: {str(e)}")
    
    def batch_process_documents(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process multiple documents in batch"""
        processed_documents = []
        
        for i, document_data in enumerate(documents):
            try:
                print(f"Processing document {i+1}/{len(documents)}: {document_data['filename']}")
                processed_doc = self.process_document_chunks(document_data)
                processed_documents.append(processed_doc)
            except Exception as e:
                print(f"Failed to process document {document_data['filename']}: {str(e)}")
                continue
        
        return processed_documents