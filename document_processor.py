import os
import io
import tempfile
from typing import List, Dict, Any, Optional
from datetime import datetime
import hashlib

import PyPDF2
import docx
from langchain_text_splitters import RecursiveCharacterTextSplitter

class DocumentProcessor:
    """Handles document processing for various file types"""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        self.supported_types = ['.pdf', '.docx', '.txt', '.md']
    
    def extract_text_from_pdf(self, file_content: bytes) -> str:
        """Extract text from PDF file"""
        try:
            pdf_reader = PyPDF2.PdfReader(io.BytesIO(file_content))
            text = ""
            for page_num, page in enumerate(pdf_reader.pages):
                text += f"\n[Page {page_num + 1}]\n"
                text += page.extract_text()
            return text.strip()
        except Exception as e:
            raise Exception(f"Error extracting text from PDF: {str(e)}")
    
    def extract_text_from_docx(self, file_content: bytes) -> str:
        """Extract text from DOCX file"""
        try:
            doc = docx.Document(io.BytesIO(file_content))
            text = ""
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
            return text.strip()
        except Exception as e:
            raise Exception(f"Error extracting text from DOCX: {str(e)}")
    
    def extract_text_from_txt(self, file_content: bytes) -> str:
        """Extract text from TXT file"""
        try:
            return file_content.decode('utf-8').strip()
        except UnicodeDecodeError:
            # Try with different encoding
            try:
                return file_content.decode('latin-1').strip()
            except Exception as e:
                raise Exception(f"Error extracting text from TXT: {str(e)}")
    
    def get_file_extension(self, filename: str) -> str:
        """Get file extension"""
        return os.path.splitext(filename.lower())[1]
    
    def is_supported_file(self, filename: str) -> bool:
        """Check if file type is supported"""
        return self.get_file_extension(filename) in self.supported_types
    
    def extract_text(self, filename: str, file_content: bytes) -> str:
        """Extract text from file based on its extension"""
        if not self.is_supported_file(filename):
            raise ValueError(f"Unsupported file type: {self.get_file_extension(filename)}")
        
        ext = self.get_file_extension(filename)
        
        if ext == '.pdf':
            return self.extract_text_from_pdf(file_content)
        elif ext == '.docx':
            return self.extract_text_from_docx(file_content)
        elif ext in ['.txt', '.md']:
            return self.extract_text_from_txt(file_content)
        else:
            raise ValueError(f"Unsupported file type: {ext}")
    
    def chunk_text(self, text: str) -> List[str]:
        """Split text into chunks"""
        if not text.strip():
            return []
        
        chunks = self.text_splitter.split_text(text)
        return [chunk.strip() for chunk in chunks if chunk.strip()]
    
    def generate_document_id(self, filename: str, file_content: bytes) -> str:
        """Generate unique document ID based on filename and content"""
        content_hash = hashlib.md5(file_content).hexdigest()
        return f"{filename}_{content_hash[:8]}"
    
    def process_document(self, filename: str, file_content: bytes) -> Dict[str, Any]:
        """Process a document and return structured data"""
        try:
            # Extract text
            text = self.extract_text(filename, file_content)
            
            # Generate chunks
            chunks = self.chunk_text(text)
            
            # Generate document ID
            doc_id = self.generate_document_id(filename, file_content)
            
            # Create document metadata
            document_data = {
                'id': doc_id,
                'filename': filename,
                'file_type': self.get_file_extension(filename),
                'text': text,
                'chunks': chunks,
                'chunk_count': len(chunks),
                'processed_at': datetime.now().isoformat(),
                'file_size': len(file_content),
                'active': True  # Default to active
            }
            
            return document_data
            
        except Exception as e:
            raise Exception(f"Error processing document {filename}: {str(e)}")