import streamlit as st
import pandas as pd
import os
from datetime import datetime
from typing import List, Dict, Any
import json

# Import our custom modules
from document_processor import DocumentProcessor
from milvus_client import MilvusClient
from embedding_service import EmbeddingService, DocumentEmbeddingManager
from qna_agent import QNAAgent
from dotenv import load_dotenv
load_dotenv()
# Configure page
st.set_page_config(
    page_title="Document QNA System",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# # Custom CS        # Settings
# st.markdown("## ⚙️ Settings")

# # Batch Processing Settings
# with st.expander("📦 Batch Processing"):
#     st.session_state.batch_parallel = st.checkbox(
#         "Process files in parallel", 
#         value=getattr(st.session_state, 'batch_parallel', False),
#         help="Process multiple files simultaneously (faster but uses more resources)"
#     )
#     st.session_state.batch_chunk_size = st.slider(
#         "Max files per batch", 
#         min_value=1, 
#         max_value=10, 
#         value=getattr(st.session_state, 'batch_chunk_size', 5),
#         help="Number of files to process in each batch"
#     )
        
        # API Key infor better UI
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #2E86AB;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #A23B72;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .citation-box {
        background-color: #f0f2f6;
        border-left: 4px solid #2E86AB;
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 0 5px 5px 0;
    }
    .document-item {
        background-color: #ffffff;
        border: 1px solid #e0e0e0;
        border-radius: 5px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    .success-message {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .error-message {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        color: #721c24;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
def initialize_session_state():
    if 'milvus_client' not in st.session_state:
        st.session_state.milvus_client = None
    if 'embedding_service' not in st.session_state:
        st.session_state.embedding_service = None
    if 'embedding_manager' not in st.session_state:
        st.session_state.embedding_manager = None
    if 'qna_agent' not in st.session_state:
        st.session_state.qna_agent = None
    if 'document_processor' not in st.session_state:
        st.session_state.document_processor = DocumentProcessor()
    if 'documents' not in st.session_state:
        st.session_state.documents = []
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'system_initialized' not in st.session_state:
        st.session_state.system_initialized = False

def initialize_system():
    """Initialize the complete system"""
    try:
        with st.spinner("Initializing system..."):
            # Initialize Milvus client
            st.session_state.milvus_client = MilvusClient(
                host="localhost",
                port="19530",
                collection_name="document_embeddings"
            )
            st.session_state.milvus_client.connect()
            
            # Initialize embedding service
            st.session_state.embedding_service = EmbeddingService(
                model_type="sentence_transformer",
                model_name="all-MiniLM-L6-v2"
            )
            
            # Initialize embedding manager
            st.session_state.embedding_manager = DocumentEmbeddingManager(
                st.session_state.embedding_service
            )
            
            # Initialize QNA agent (requires Google API key for full functionality)
            try:
                st.session_state.qna_agent = QNAAgent(
                    milvus_client=st.session_state.milvus_client,
                    embedding_manager=st.session_state.embedding_manager,
                    model_type="gemini"
                )
            except Exception as e:
                st.warning(f"QNA agent initialization failed (Google API key required): {str(e)}")
                st.session_state.qna_agent = None
            
            # Load existing documents
            try:
                st.session_state.documents = st.session_state.milvus_client.get_all_documents()
            except Exception as e:
                st.warning(f"Could not load existing documents: {str(e)}")
                st.session_state.documents = []
            
            st.session_state.system_initialized = True
            st.success("System initialized successfully!")
            
    except Exception as e:
        st.error(f"System initialization failed: {str(e)}")
        return False
    
    return True

def upload_and_process_document():
    """Handle multiple document upload and processing"""
    st.markdown('<div class="section-header">📁 Document Upload</div>', unsafe_allow_html=True)
    
    # Add drag and drop styling
    st.markdown("""
    <style>
    .uploadedFile {
        border: 2px dashed #cccccc;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        margin: 10px 0;
    }
    .uploadedFile:hover {
        border-color: #2E86AB;
        background-color: #f8f9fa;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Instructions
    st.info("💡 **Tip:** You can select multiple files at once or drag and drop them into the file uploader.")
    
    # Multiple file uploader
    uploaded_files = st.file_uploader(
        "Choose files to upload",
        type=['pdf', 'docx', 'txt', 'md'],
        help="Supported formats: PDF, Word Documents, Text files, Markdown files",
        accept_multiple_files=True,
        label_visibility="collapsed"
    )
    
    if uploaded_files:
        # Summary information
        total_size = sum(len(f.getvalue()) for f in uploaded_files)
        st.markdown(f"""
        **📊 Upload Summary:**
        - **Files selected:** {len(uploaded_files)}
        - **Total size:** {total_size:,} bytes ({total_size / (1024*1024):.2f} MB)
        """)
        
        # Quick actions row
        col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
        
        with col1:
            if st.button("🚀 Process All Files", type="primary", key="process_all_main"):
                process_multiple_files(uploaded_files)
        
        with col2:
            if st.button("📋 Show File Details", key="show_details"):
                st.session_state.show_file_details = not getattr(st.session_state, 'show_file_details', False)
        
        with col3:
            if st.button("🔄 Clear Selection", key="clear_files"):
                st.rerun()
        
        with col4:
            # File type filter
            file_types = list(set(f.type for f in uploaded_files))
            if len(file_types) > 1:
                st.selectbox("Filter by type", ["All"] + file_types, key="file_type_filter")
        
        # Display file information (expandable)
        if getattr(st.session_state, 'show_file_details', False):
            st.markdown("---")
            st.markdown("### 📄 File Details")
            
            for i, uploaded_file in enumerate(uploaded_files):
                with st.expander(f"📄 {uploaded_file.name}", expanded=(i < 2)):  # Show first 2 expanded
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        st.write(f"**Filename:** {uploaded_file.name}")
                        st.write(f"**File size:** {len(uploaded_file.getvalue()):,} bytes")
                        st.write(f"**File type:** {uploaded_file.type}")
                        
                        # Show file extension info
                        file_ext = uploaded_file.name.split('.')[-1].lower()
                        ext_info = {
                            'pdf': '📕 PDF Document',
                            'docx': '📘 Word Document', 
                            'txt': '📄 Text File',
                            'md': '📝 Markdown File'
                        }
                        st.write(f"**Format:** {ext_info.get(file_ext, '📄 Unknown')}")
                    
                    with col2:
                        if st.button(f"🚀 Process This File", key=f"process_single_{i}", type="secondary"):
                            with st.spinner(f"Processing {uploaded_file.name}..."):
                                process_uploaded_file(uploaded_file)
                        
                        # Preview button for text files
                        if uploaded_file.type in ['text/plain', 'text/markdown']:
                            if st.button(f"👁️ Preview", key=f"preview_{i}"):
                                try:
                                    content = uploaded_file.getvalue().decode('utf-8')
                                    st.text_area(
                                        f"Preview of {uploaded_file.name}:", 
                                        content[:500] + ("..." if len(content) > 500 else ""),
                                        height=150,
                                        disabled=True
                                    )
                                except:
                                    st.error("Could not preview file")
        
        # Progress tracking for batch processing
        if 'batch_processing' in st.session_state and st.session_state.batch_processing:
            st.markdown("---")
            st.markdown("### � Batch Processing Status")
            st.progress(st.session_state.get('batch_progress', 0))
            st.text(st.session_state.get('batch_status', 'Processing...'))
    
    else:
        # Empty state with helpful instructions
        st.markdown("""
        <div style="text-align: center; padding: 40px; border: 2px dashed #cccccc; border-radius: 10px; margin: 20px 0;">
            <h3>📁 No files selected</h3>
            <p>Click "Browse files" above or drag and drop your documents here</p>
            <p><strong>Supported formats:</strong> PDF, DOCX, TXT, MD</p>
            <p><strong>Multiple files:</strong> You can select and process multiple documents at once</p>
        </div>
        """, unsafe_allow_html=True)

def process_uploaded_file(uploaded_file):
    """Process the uploaded file"""
    try:
        with st.spinner(f"Processing {uploaded_file.name}..."):
            # Read file content
            file_content = uploaded_file.getvalue()
            
            # Process document
            progress_bar = st.progress(0)
            
            # Step 1: Extract text and create chunks
            st.text("Extracting text and creating chunks...")
            progress_bar.progress(25)
            document_data = st.session_state.document_processor.process_document(
                uploaded_file.name, file_content
            )
            
            # Step 2: Generate embeddings
            st.text("Generating embeddings...")
            progress_bar.progress(50)
            document_data = st.session_state.embedding_manager.process_document_chunks(document_data)
            
            # Step 3: Store in Milvus
            st.text("Storing in vector database...")
            progress_bar.progress(75)
            success = st.session_state.milvus_client.insert_embeddings(
                document_data, document_data['embeddings']
            )
            
            progress_bar.progress(100)
            
            if success:
                st.markdown(f"""
                <div class="success-message">
                    <strong>✅ Document processed successfully!</strong><br>
                    • File: {document_data['filename']}<br>
                    • Chunks created: {document_data['chunk_count']}<br>
                    • Embeddings generated: {len(document_data['embeddings'])}<br>
                    • Document ID: {document_data['id']}
                </div>
                """, unsafe_allow_html=True)
                
                # Refresh document list
                st.session_state.documents = st.session_state.milvus_client.get_all_documents()
                st.rerun()
            else:
                st.error("Failed to store document in vector database")
                
    except Exception as e:
        st.error(f"Error processing document: {str(e)}")

def process_multiple_files(uploaded_files):
    """Process multiple uploaded files with batch processing support"""
    try:
        total_files = len(uploaded_files)
        batch_size = getattr(st.session_state, 'batch_chunk_size', 5)
        parallel_processing = getattr(st.session_state, 'batch_parallel', False)
        
        st.markdown(f"### 🚀 Processing {total_files} files...")
        
        # Processing options display
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Files to Process", total_files)
        with col2:
            st.metric("Batch Size", min(batch_size, total_files))
        with col3:
            st.write(f"**Mode:** {'Parallel' if parallel_processing else 'Sequential'}")
        
        # Create overall progress tracking
        overall_progress = st.progress(0)
        status_text = st.empty()
        
        # Results tracking
        successful_files = []
        failed_files = []
        processing_stats = {
            'start_time': datetime.now(),
            'total_chunks': 0,
            'total_embeddings': 0
        }
        
        # Set batch processing state
        st.session_state.batch_processing = True
        st.session_state.batch_progress = 0
        
        # Process files in batches
        for batch_start in range(0, total_files, batch_size):
            batch_end = min(batch_start + batch_size, total_files)
            batch_files = uploaded_files[batch_start:batch_end]
            batch_num = (batch_start // batch_size) + 1
            total_batches = (total_files + batch_size - 1) // batch_size
            
            st.markdown(f"#### 📦 Batch {batch_num}/{total_batches} ({len(batch_files)} files)")
            
            # Process current batch
            for i, uploaded_file in enumerate(batch_files):
                file_index = batch_start + i
                overall_file_progress = file_index / total_files
                
                try:
                    # Update overall progress
                    overall_progress.progress(overall_file_progress)
                    status_text.text(f"Batch {batch_num}/{total_batches} - File {i+1}/{len(batch_files)}: {uploaded_file.name}")
                    st.session_state.batch_progress = overall_file_progress
                    st.session_state.batch_status = f"Processing {uploaded_file.name}"
                    
                    # Create expandable section for each file processing
                    with st.expander(f"🔄 Processing: {uploaded_file.name}", expanded=True):
                        file_progress = st.progress(0)
                        file_status = st.empty()
                        file_timer = st.empty()
                        
                        file_start_time = datetime.now()
                        
                        # Read file content
                        file_status.text("📖 Reading file content...")
                        file_progress.progress(10)
                        file_content = uploaded_file.getvalue()
                        
                        # Step 1: Extract text and create chunks
                        file_status.text("✂️ Extracting text and creating chunks...")
                        file_progress.progress(25)
                        document_data = st.session_state.document_processor.process_document(
                            uploaded_file.name, file_content
                        )
                        
                        # Step 2: Generate embeddings
                        file_status.text("🧠 Generating embeddings...")
                        file_progress.progress(50)
                        document_data = st.session_state.embedding_manager.process_document_chunks(document_data)
                        
                        # Step 3: Store in Milvus
                        file_status.text("💾 Storing in vector database...")
                        file_progress.progress(75)
                        success = st.session_state.milvus_client.insert_embeddings(
                            document_data, document_data['embeddings']
                        )
                        
                        # Final step
                        file_progress.progress(100)
                        file_end_time = datetime.now()
                        processing_time = (file_end_time - file_start_time).total_seconds()
                        
                        if success:
                            file_status.text(f"✅ Successfully processed in {processing_time:.1f}s!")
                            file_timer.text(f"⏱️ Processing time: {processing_time:.1f} seconds")
                            
                            file_info = {
                                'filename': document_data['filename'],
                                'id': document_data['id'],
                                'chunks': document_data['chunk_count'],
                                'embeddings': len(document_data['embeddings']),
                                'processing_time': processing_time,
                                'file_size': len(file_content)
                            }
                            successful_files.append(file_info)
                            
                            # Update stats
                            processing_stats['total_chunks'] += document_data['chunk_count']
                            processing_stats['total_embeddings'] += len(document_data['embeddings'])
                        else:
                            file_status.text("❌ Failed to store in database")
                            file_timer.text(f"⏱️ Failed after {processing_time:.1f} seconds")
                            failed_files.append({
                                'filename': uploaded_file.name, 
                                'error': 'Database storage failed',
                                'processing_time': processing_time
                            })
                            
                except Exception as e:
                    file_end_time = datetime.now()
                    processing_time = (file_end_time - file_start_time).total_seconds()
                    
                    failed_files.append({
                        'filename': uploaded_file.name, 
                        'error': str(e),
                        'processing_time': processing_time
                    })
                    
                    with st.expander(f"❌ Failed: {uploaded_file.name}", expanded=False):
                        st.error(f"Error: {str(e)}")
                        st.text(f"Processing time: {processing_time:.1f} seconds")
            
            # Small delay between batches if not parallel
            if not parallel_processing and batch_num < total_batches:
                st.info(f"Batch {batch_num} complete. Starting next batch...")
        
        # Final progress update
        overall_progress.progress(1.0)
        processing_stats['end_time'] = datetime.now()
        total_processing_time = (processing_stats['end_time'] - processing_stats['start_time']).total_seconds()
        
        status_text.text("🎉 Processing complete!")
        st.session_state.batch_processing = False
        
        # Display comprehensive results summary
        st.markdown("---")
        st.markdown("### 📊 Processing Results")
        
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("✅ Successful", len(successful_files))
        with col2:
            st.metric("❌ Failed", len(failed_files))
        with col3:
            st.metric("Total Chunks", processing_stats['total_chunks'])
        with col4:
            st.metric("Total Time", f"{total_processing_time:.1f}s")
        
        # Detailed results
        col1, col2 = st.columns(2)
        
        with col1:
            if successful_files:
                with st.expander("✅ Successful Files", expanded=True):
                    for file_info in successful_files:
                        st.markdown(f"""
                        **📄 {file_info['filename']}**
                        - Document ID: `{file_info['id']}`
                        - Chunks: {file_info['chunks']}
                        - Embeddings: {file_info['embeddings']}
                        - Size: {file_info['file_size']:,} bytes
                        - Time: {file_info['processing_time']:.1f}s
                        """)
        
        with col2:
            if failed_files:
                with st.expander("❌ Failed Files", expanded=True):
                    for file_info in failed_files:
                        st.markdown(f"""
                        **📄 {file_info['filename']}**
                        - Error: {file_info['error']}
                        - Time: {file_info['processing_time']:.1f}s
                        """)
        
        # Performance statistics
        if successful_files:
            with st.expander("📈 Performance Statistics"):
                avg_processing_time = sum(f['processing_time'] for f in successful_files) / len(successful_files)
                avg_chunks_per_file = processing_stats['total_chunks'] / len(successful_files)
                avg_embeddings_per_file = processing_stats['total_embeddings'] / len(successful_files)
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Avg Time/File", f"{avg_processing_time:.1f}s")
                with col2:
                    st.metric("Avg Chunks/File", f"{avg_chunks_per_file:.1f}")
                with col3:
                    st.metric("Avg Embeddings/File", f"{avg_embeddings_per_file:.0f}")
        
        # Refresh document list if any files were successful
        if successful_files:
            with st.spinner("Refreshing document list..."):
                st.session_state.documents = st.session_state.milvus_client.get_all_documents()
            
            st.success(f"🎉 Successfully processed {len(successful_files)} out of {total_files} files!")
            
            # Auto-refresh and celebration
            if len(successful_files) == total_files:
                st.balloons()
                st.markdown("### 🏆 All files processed successfully!")
                
        if failed_files:
            st.warning(f"⚠️ {len(failed_files)} files failed to process. Check the details above.")
            
    except Exception as e:
        st.error(f"Error in batch processing: {str(e)}")
        st.session_state.batch_processing = False
    finally:
        # Clean up batch processing state
        if hasattr(st.session_state, 'batch_processing'):
            st.session_state.batch_processing = False

def display_document_management():
    """Display document management interface"""
    st.markdown('<div class="section-header">📋 Document Management</div>', unsafe_allow_html=True)
    
    if not st.session_state.documents:
        st.info("No documents in the knowledge base yet. Upload a document to get started!")
        return
    
    # Create DataFrame for better display
    doc_df = pd.DataFrame(st.session_state.documents)
    
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        st.write(f"**Total Documents:** {len(st.session_state.documents)}")
    
    with col2:
        active_docs = sum(1 for doc in st.session_state.documents if doc.get('active', True))
        st.write(f"**Active:** {active_docs}")
    
    with col3:
        inactive_docs = len(st.session_state.documents) - active_docs
        st.write(f"**Inactive:** {inactive_docs}")
    
    # Display documents with management options
    for i, doc in enumerate(st.session_state.documents):
        with st.container():
            st.markdown(f"""
            <div class="document-item">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div>
                        <strong>{doc['filename']}</strong><br>
                        <small>ID: {doc['document_id']} | Type: {doc.get('file_type', 'unknown')} | 
                        Chunks: {doc.get('total_chunks', 0)}</small>
                    </div>
                    <div>
                        <span style="color: {'green' if doc.get('active', True) else 'red'};">
                            {'🟢 Active' if doc.get('active', True) else '🔴 Inactive'}
                        </span>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            col1, col2, col3 = st.columns([1, 1, 1])
            
            with col1:
                if doc.get('active', True):
                    if st.button(f"Deactivate", key=f"deactivate_{doc['document_id']}"):
                        toggle_document_status(doc['document_id'], False)
                else:
                    if st.button(f"Activate", key=f"activate_{doc['document_id']}"):
                        toggle_document_status(doc['document_id'], True)
            
            with col2:
                if st.button(f"Delete", key=f"delete_{doc['document_id']}", type="secondary"):
                    delete_document(doc['document_id'])

def toggle_document_status(document_id: str, active: bool):
    """Toggle document active status"""
    try:
        success = st.session_state.milvus_client.update_document_status(document_id, active)
        if success:
            # Update local state
            for doc in st.session_state.documents:
                if doc['document_id'] == document_id:
                    doc['active'] = active
            
            status_text = "activated" if active else "deactivated"
            st.success(f"Document {status_text} successfully!")
            st.rerun()
        else:
            st.error("Failed to update document status")
    except Exception as e:
        st.error(f"Error updating document status: {str(e)}")

def delete_document(document_id: str):
    """Delete a document"""
    try:
        success = st.session_state.milvus_client.delete_document(document_id)
        if success:
            # Update local state
            st.session_state.documents = [
                doc for doc in st.session_state.documents 
                if doc['document_id'] != document_id
            ]
            st.success("Document deleted successfully!")
            st.rerun()
        else:
            st.error("Failed to delete document")
    except Exception as e:
        st.error(f"Error deleting document: {str(e)}")

def display_qna_interface():
    """Display QNA interface"""
    st.markdown('<div class="section-header">💬 Ask Questions</div>', unsafe_allow_html=True)
    
    if not st.session_state.qna_agent:
        st.error("QNA agent not available. Please set your Google API key in the environment variables.")
        st.info("Set GOOGLE_API_KEY environment variable and restart the application.")
        return
    
    if not st.session_state.documents:
        st.warning("No documents in knowledge base. Please upload documents first.")
        return
    
    # Document filter options
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Active documents filter
        active_only = st.checkbox("Include only active documents", value=True)
    
    with col2:
        # Specific documents filter
        doc_options = ["All documents"] + [doc['filename'] for doc in st.session_state.documents]
        selected_docs = st.selectbox("Filter by document", doc_options)
    
    # Question input
    question = st.text_input("Ask a question about your documents:", placeholder="What is the main topic discussed in the documents?")
    
    col1, col2 = st.columns([1, 4])
    with col1:
        ask_button = st.button("🔍 Ask Question", type="primary", disabled=not question.strip())
    
    if ask_button and question.strip():
        # Prepare document filter
        document_ids = None
        if selected_docs != "All documents":
            # Find document ID for selected document
            for doc in st.session_state.documents:
                if doc['filename'] == selected_docs:
                    document_ids = [doc['document_id']]
                    break
        
        # Ask question
        with st.spinner("Searching knowledge base and generating answer..."):
            try:
                response = st.session_state.qna_agent.get_answer_with_sources(
                    query=question,
                    limit=5,
                    active_documents_only=active_only,
                    document_ids=document_ids
                )
                
                # Add to chat history
                st.session_state.chat_history.append(response)
                
                # Display response
                display_qna_response(response)
                
            except Exception as e:
                st.error(f"Error processing question: {str(e)}")
    
    # Display chat history
    if st.session_state.chat_history:
        st.markdown('<div class="section-header">📝 Chat History</div>', unsafe_allow_html=True)
        
        for i, response in enumerate(reversed(st.session_state.chat_history[-10:])):  # Show last 10
            with st.expander(f"Q: {response['question'][:100]}...", expanded=(i==0)):
                display_qna_response(response)

def display_qna_response(response: Dict[str, Any]):
    """Display QNA response with sources"""
    # Answer
    st.markdown("**Answer:**")
    st.write(response['answer'])
    
    # Sources
    if response.get('sources'):
        st.markdown("**Sources:**")
        for source in response['sources']:
            st.markdown(f"""
            <div class="citation-box">
                <strong>[{source['citation_id']}] {source['filename']}</strong><br>
                <small>Relevance Score: {source['relevance_score']}</small><br>
                <em>{source['preview_text']}</em>
            </div>
            """, unsafe_allow_html=True)
    
    # Metadata
    if response.get('metadata'):
        with st.expander("Query Details"):
            metadata = response['metadata']
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Chunks Retrieved", metadata.get('chunks_retrieved', 0))
            with col2:
                st.metric("Tokens Used", metadata.get('tokens_used', 0))
            with col3:
                st.write(f"**Model:** {metadata.get('model_used', 'N/A')}")

def main():
    """Main application"""
    initialize_session_state()
    
    # Header
    st.markdown('<div class="main-header">📚 Document QNA System with Milvus</div>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("## 🛠️ System Status")
        
        if not st.session_state.system_initialized:
            if st.button("🚀 Initialize System", type="primary"):
                initialize_system()
        else:
            st.success("✅ System Ready")
            
            # System info
            if st.session_state.milvus_client:
                stats = st.session_state.milvus_client.get_collection_stats()
                st.metric("Total Embeddings", stats.get('total_entities', 0))
            
            if st.button("🔄 Refresh Data"):
                try:
                    st.session_state.documents = st.session_state.milvus_client.get_all_documents()
                    st.success("Data refreshed!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Refresh failed: {str(e)}")
        
        # Settings
        st.markdown("## ⚙️ Settings")
        
        # Batch Processing Settings
        with st.expander("📦 Batch Processing"):
            st.session_state.batch_parallel = st.checkbox(
                "Process files in parallel", 
                value=getattr(st.session_state, 'batch_parallel', False),
                help="Process multiple files simultaneously (faster but uses more resources)"
            )
            st.session_state.batch_chunk_size = st.slider(
                "Max files per batch", 
                min_value=1, 
                max_value=10, 
                value=getattr(st.session_state, 'batch_chunk_size', 5),
                help="Number of files to process in each batch"
            )
        
        # API Key info
        google_key = os.getenv("GOOGLE_API_KEY")
        if google_key and google_key.strip():
            st.success("✅ Google API Key Set")
            # Optionally show partial key for verification
            masked_key = f"{google_key[:10]}...{google_key[-4:]}" if len(google_key) > 14 else "Set"
            st.text(f"Key: {masked_key}")
        else:
            st.warning("⚠️ Google API Key Not Set")
            st.info("QNA functionality requires Google API key")
            
            # Add manual API key input option
            with st.expander("Set API Key Manually"):
                manual_key = st.text_input("Enter Google API Key:", type="password")
                if st.button("Set API Key") and manual_key.strip():
                    os.environ["GOOGLE_API_KEY"] = manual_key.strip()
                    st.success("API Key set for this session!")
                    st.rerun()
    
    # Main content
    if not st.session_state.system_initialized:
        st.info("Please initialize the system using the sidebar to get started.")
        return
    
    # Tabs
    tab1, tab2, tab3 = st.tabs(["📁 Upload Documents", "📋 Manage Documents", "💬 Ask Questions"])
    
    with tab1:
        upload_and_process_document()
    
    with tab2:
        display_document_management()
    
    with tab3:
        display_qna_interface()

if __name__ == "__main__":
    main()