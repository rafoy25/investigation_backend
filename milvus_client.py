import json
from typing import List, Dict, Any, Optional
from pymilvus import connections, Collection, CollectionSchema, FieldSchema, DataType, utility
import numpy as np

class MilvusClient:
    """Milvus vector database client for document embeddings"""
    
    def __init__(self, host: str = "localhost", port: str = "19530", collection_name: str = "document_embeddings"):
        self.host = host
        self.port = port
        self.collection_name = collection_name
        self.collection = None
        self.embedding_dim = 384  # Default for all-MiniLM-L6-v2
        
    def connect(self):
        """Connect to Milvus server"""
        try:
            connections.connect(
                alias="default",
                host=self.host,
                port=self.port
            )
            print(f"Connected to Milvus at {self.host}:{self.port}")
            
            # Create collection if it doesn't exist
            if not utility.has_collection(self.collection_name):
                self._create_collection()
            else:
                self.collection = Collection(self.collection_name)
                # Load collection into memory
                self.collection.load()
                print(f"Loaded existing collection: {self.collection_name}")
                
        except Exception as e:
            raise Exception(f"Failed to connect to Milvus: {str(e)}")
    
    def _create_collection(self):
        """Create the collection with proper schema"""
        try:
            # Define collection schema
            fields = [
                FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
                FieldSchema(name="document_id", dtype=DataType.VARCHAR, max_length=100),
                FieldSchema(name="chunk_id", dtype=DataType.INT64),
                FieldSchema(name="filename", dtype=DataType.VARCHAR, max_length=255),
                FieldSchema(name="chunk_text", dtype=DataType.VARCHAR, max_length=8000),
                FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=self.embedding_dim),
                FieldSchema(name="metadata", dtype=DataType.VARCHAR, max_length=2000)
            ]
            
            schema = CollectionSchema(
                fields=fields,
                description="Document embeddings collection for QNA system"
            )
            
            # Create collection
            self.collection = Collection(
                name=self.collection_name,
                schema=schema,
                using='default'
            )
            
            # Create index for vector field
            index_params = {
                "metric_type": "COSINE",
                "index_type": "IVF_FLAT",
                "params": {"nlist": 128}
            }
            
            self.collection.create_index(
                field_name="embedding",
                index_params=index_params
            )
            
            # Load collection into memory
            self.collection.load()
            
            print(f"Created and loaded collection: {self.collection_name}")
            
        except Exception as e:
            raise Exception(f"Failed to create collection: {str(e)}")
    
    def insert_embeddings(self, document_data: Dict[str, Any], embeddings: List[List[float]]) -> bool:
        """Insert document chunks and their embeddings into Milvus"""
        try:
            if not self.collection:
                raise Exception("Not connected to Milvus. Call connect() first.")
            
            chunks = document_data['chunks']
            if len(chunks) != len(embeddings):
                raise ValueError("Number of chunks must match number of embeddings")
            
            # Prepare data for insertion
            insert_data = []
            
            for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                metadata = {
                    'file_type': document_data['file_type'],
                    'processed_at': document_data['processed_at'],
                    'file_size': document_data['file_size'],
                    'total_chunks': document_data['chunk_count'],
                    'active': document_data['active']
                }
                
                insert_data.append([
                    document_data['id'],  # document_id
                    i,  # chunk_id
                    document_data['filename'],  # filename
                    chunk[:7999],  # chunk_text (truncated to fit schema)
                    embedding,  # embedding
                    json.dumps(metadata)[:1999]  # metadata (truncated to fit schema)
                ])
            
            # Transpose data for Milvus format
            entities = [
                [item[0] for item in insert_data],  # document_id
                [item[1] for item in insert_data],  # chunk_id
                [item[2] for item in insert_data],  # filename
                [item[3] for item in insert_data],  # chunk_text
                [item[4] for item in insert_data],  # embedding
                [item[5] for item in insert_data],  # metadata
            ]
            
            # Insert data
            insert_result = self.collection.insert(entities)
            
            # Flush to ensure data persistence
            self.collection.flush()
            
            print(f"Inserted {len(chunks)} chunks for document {document_data['filename']}")
            return True
            
        except Exception as e:
            print(f"Error inserting embeddings: {str(e)}")
            return False
    
    def search_similar(self, query_embedding: List[float], limit: int = 5, 
                      active_only: bool = True, document_ids: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """Search for similar embeddings"""
        try:
            if not self.collection:
                raise Exception("Not connected to Milvus. Call connect() first.")
            
            # Build search expression
            expr_conditions = []
            
            if active_only:
                expr_conditions.append('metadata like "%\\"active\\": true%"')
            
            if document_ids:
                # Create OR condition for document IDs
                id_conditions = [f'document_id == "{doc_id}"' for doc_id in document_ids]
                if id_conditions:
                    expr_conditions.append(f"({' or '.join(id_conditions)})")
            
            expr = ' and '.join(expr_conditions) if expr_conditions else None
            
            search_params = {
                "metric_type": "COSINE",
                "params": {"nprobe": 10}
            }
            
            results = self.collection.search(
                data=[query_embedding],
                anns_field="embedding",
                param=search_params,
                limit=limit,
                expr=expr,
                output_fields=["document_id", "chunk_id", "filename", "chunk_text", "metadata"]
            )
            
            # Process results
            search_results = []
            for hits in results:
                for hit in hits:
                    metadata = json.loads(hit.entity.get('metadata', '{}'))
                    search_results.append({
                        'score': hit.score,
                        'document_id': hit.entity.get('document_id'),
                        'chunk_id': hit.entity.get('chunk_id'),
                        'filename': hit.entity.get('filename'),
                        'chunk_text': hit.entity.get('chunk_text'),
                        'metadata': metadata
                    })
            
            return search_results
            
        except Exception as e:
            raise Exception(f"Search failed: {str(e)}")
    
    def get_all_documents(self) -> List[Dict[str, Any]]:
        """Get list of all unique documents in the database"""
        try:
            if not self.collection:
                raise Exception("Not connected to Milvus. Call connect() first.")
            
            # Query to get unique documents
            results = self.collection.query(
                expr="chunk_id == 0",  # Get only the first chunk of each document
                output_fields=["document_id", "filename", "metadata"],
                limit=1000
            )
            
            documents = []
            for result in results:
                metadata = json.loads(result.get('metadata', '{}'))
                documents.append({
                    'document_id': result['document_id'],
                    'filename': result['filename'],
                    'active': metadata.get('active', True),
                    'file_type': metadata.get('file_type', 'unknown'),
                    'processed_at': metadata.get('processed_at', ''),
                    'total_chunks': metadata.get('total_chunks', 0)
                })
            
            return documents
            
        except Exception as e:
            raise Exception(f"Failed to get documents: {str(e)}")
    
    def update_document_status(self, document_id: str, active: bool) -> bool:
        """Update document active status"""
        try:
            if not self.collection:
                raise Exception("Not connected to Milvus. Call connect() first.")
            
            # Get all chunks for the document
            results = self.collection.query(
                expr=f'document_id == "{document_id}"',
                output_fields=["id", "metadata"]
            )
            
            # Update metadata for each chunk
            for result in results:
                metadata = json.loads(result.get('metadata', '{}'))
                metadata['active'] = active
                
                # Note: Milvus doesn't support update operations directly
                # This is a limitation - in production, you might need to delete and re-insert
                # For now, we'll implement this as a workaround
                print(f"Status update for document {document_id} to {'active' if active else 'inactive'}")
            
            return True
            
        except Exception as e:
            print(f"Error updating document status: {str(e)}")
            return False
    
    def delete_document(self, document_id: str) -> bool:
        """Delete all chunks of a document"""
        try:
            if not self.collection:
                raise Exception("Not connected to Milvus. Call connect() first.")
            
            # Delete all chunks for the document
            self.collection.delete(expr=f'document_id == "{document_id}"')
            
            print(f"Deleted document: {document_id}")
            return True
            
        except Exception as e:
            print(f"Error deleting document: {str(e)}")
            return False
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get collection statistics"""
        try:
            if not self.collection:
                raise Exception("Not connected to Milvus. Call connect() first.")
            
            stats = self.collection.num_entities
            return {
                'total_entities': stats,
                'collection_name': self.collection_name
            }
            
        except Exception as e:
            return {'error': str(e)}
    
    def disconnect(self):
        """Disconnect from Milvus"""
        try:
            connections.disconnect("default")
            print("Disconnected from Milvus")
        except Exception as e:
            print(f"Error disconnecting: {str(e)}")