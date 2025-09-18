from typing import List, Dict, Any, Optional, Tuple
import google.generativeai as genai
import os
from datetime import datetime

class QNAAgent:
    """Question and Answer agent with retrieval and citation"""
    
    def __init__(self, milvus_client, embedding_manager, model_type: str = "gemini", model_name: Optional[str] = None):
        self.milvus_client = milvus_client
        self.embedding_manager = embedding_manager
        self.model_type = model_type
        self.model_name = model_name or "gemini-1.5-flash"
        
        if model_type == "gemini":
            self._setup_gemini()
    
    def _setup_gemini(self):
        """Setup Google Gemini client for chat completion"""
        try:
            api_key = os.getenv("GOOGLE_API_KEY")
            if not api_key:
                raise ValueError("GOOGLE_API_KEY environment variable not set")
            
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel(self.model_name)
            print(f"Setup Google Gemini model: {self.model_name}")
        except Exception as e:
            raise Exception(f"Failed to setup Google Gemini: {str(e)}")
    
    def retrieve_relevant_chunks(self, query: str, limit: int = 5, 
                               active_documents_only: bool = True, 
                               document_ids: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """Retrieve relevant document chunks for a query"""
        try:
            # Generate query embedding
            query_embedding = self.embedding_manager.generate_query_embedding(query)
            
            # Search for similar chunks
            search_results = self.milvus_client.search_similar(
                query_embedding=query_embedding,
                limit=limit,
                active_only=active_documents_only,
                document_ids=document_ids
            )
            
            return search_results
            
        except Exception as e:
            raise Exception(f"Failed to retrieve relevant chunks: {str(e)}")
    
    def format_context_with_citations(self, search_results: List[Dict[str, Any]]) -> Tuple[str, List[Dict[str, Any]]]:
        """Format search results into context string with citation tracking"""
        if not search_results:
            return "No relevant information found.", []
        
        context_parts = []
        citations = []
        
        for i, result in enumerate(search_results):
            citation_id = i + 1
            
            # Add citation to context
            context_parts.append(f"[{citation_id}] {result['chunk_text']}")
            
            # Track citation information
            citations.append({
                'id': citation_id,
                'filename': result['filename'],
                'document_id': result['document_id'],
                'chunk_id': result['chunk_id'],
                'score': result['score'],
                'chunk_text': result['chunk_text'][:200] + "..." if len(result['chunk_text']) > 200 else result['chunk_text']
            })
        
        context = "\n\n".join(context_parts)
        return context, citations
    
    def generate_answer(self, query: str, context: str, citations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate answer using the language model"""
        try:
            if self.model_type == "gemini":
                return self._generate_gemini_answer(query, context, citations)
            else:
                raise ValueError(f"Unsupported model type: {self.model_type}")
                
        except Exception as e:
            raise Exception(f"Failed to generate answer: {str(e)}")
    
    def _generate_gemini_answer(self, query: str, context: str, citations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate answer using Google Gemini"""
        system_prompt = """You are a helpful assistant that answers questions based on provided context. 
        
        Instructions:
        1. Answer the question using ONLY the information provided in the context.
        2. If the context doesn't contain enough information to answer the question, say so clearly.
        3. Include citation numbers [1], [2], etc. in your answer to reference specific parts of the context.
        4. Be detailed and comprehensive in your answer.
        5. If multiple sources support the same point, reference all relevant citations.
        """
        
        user_prompt = f"""{system_prompt}

Context:
{context}

Question: {query}

Please provide an answer based on the context above, including appropriate citations."""
        
        try:
            response = self.model.generate_content(
                user_prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.3,
                    max_output_tokens=3000,
                )
            )
            
            answer = response.text
            
            # Count tokens (approximate)
            tokens_used = len(user_prompt.split()) + len(answer.split())
            
            return {
                'answer': answer,
                'model_used': self.model_name,
                'tokens_used': tokens_used,
                'generated_at': datetime.now().isoformat()
            }
            
        except Exception as e:
            raise Exception(f"Google Gemini API call failed: {str(e)}")
    
    def ask_question(self, query: str, limit: int = 5, 
                    active_documents_only: bool = True, 
                    document_ids: Optional[List[str]] = None) -> Dict[str, Any]:
        """Complete QNA pipeline: retrieve, format, and generate answer"""
        try:
            print(f"Processing question: {query}")
            
            # Step 1: Retrieve relevant chunks
            print("Retrieving relevant chunks...")
            search_results = self.retrieve_relevant_chunks(
                query=query,
                limit=limit,
                active_documents_only=active_documents_only,
                document_ids=document_ids
            )
            
            if not search_results:
                return {
                    'answer': "I couldn't find any relevant information in the knowledge base to answer your question.",
                    'citations': [],
                    'context_used': "",
                    'retrieval_results': [],
                    'query': query,
                    'timestamp': datetime.now().isoformat()
                }
            
            # Step 2: Format context and prepare citations
            print(f"Found {len(search_results)} relevant chunks")
            context, citations = self.format_context_with_citations(search_results)
            
            # Step 3: Generate answer
            print("Generating answer...")
            answer_data = self.generate_answer(query, context, citations)
            
            # Step 4: Compile complete response
            response = {
                'answer': answer_data['answer'],
                'citations': citations,
                'context_used': context,
                'retrieval_results': search_results,
                'query': query,
                'timestamp': datetime.now().isoformat(),
                'model_info': {
                    'model_used': answer_data.get('model_used', self.model_name),
                    'tokens_used': answer_data.get('tokens_used', 0),
                },
                'retrieval_info': {
                    'chunks_found': len(search_results),
                    'active_documents_only': active_documents_only,
                    'document_ids_filter': document_ids
                }
            }
            
            print("Answer generated successfully")
            return response
            
        except Exception as e:
            return {
                'answer': f"An error occurred while processing your question: {str(e)}",
                'citations': [],
                'context_used': "",
                'retrieval_results': [],
                'query': query,
                'timestamp': datetime.now().isoformat(),
                'error': str(e)
            }
    
    def get_answer_with_sources(self, query: str, limit: int = 5, 
                               active_documents_only: bool = True, 
                               document_ids: Optional[List[str]] = None) -> Dict[str, Any]:
        """Get answer with detailed source information for UI display"""
        response = self.ask_question(query, limit, active_documents_only, document_ids)
        
        # Format sources for UI display
        sources = []
        for citation in response.get('citations', []):
            sources.append({
                'citation_id': citation['id'],
                'filename': citation['filename'],
                'relevance_score': round(citation['score'], 3),
                'preview_text': citation['chunk_text'],
                'document_id': citation['document_id']
            })
        
        return {
            'question': response['query'],
            'answer': response['answer'],
            'sources': sources,
            'metadata': {
                'timestamp': response['timestamp'],
                'model_used': response.get('model_info', {}).get('model_used', ''),
                'tokens_used': response.get('model_info', {}).get('tokens_used', 0),
                'chunks_retrieved': response.get('retrieval_info', {}).get('chunks_found', 0)
            }
        }
    
    def batch_questions(self, questions: List[str], **kwargs) -> List[Dict[str, Any]]:
        """Process multiple questions in batch"""
        results = []
        
        for i, question in enumerate(questions):
            print(f"Processing question {i+1}/{len(questions)}")
            try:
                result = self.ask_question(question, **kwargs)
                results.append(result)
            except Exception as e:
                results.append({
                    'answer': f"Error processing question: {str(e)}",
                    'citations': [],
                    'query': question,
                    'timestamp': datetime.now().isoformat(),
                    'error': str(e)
                })
        
        return results