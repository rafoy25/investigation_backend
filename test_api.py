"""
Test script for the FastAPI Vector Database API
Run this script to test all endpoints
"""

import requests
import time
import json
from pathlib import Path

# API base URL
BASE_URL = "http://localhost:8000"

def test_health_check():
    """Test health check endpoint"""
    print("🔍 Testing health check...")
    response = requests.get(f"{BASE_URL}/health")
    if response.status_code == 200:
        print("✅ Health check passed")
        print(json.dumps(response.json(), indent=2))
    else:
        print(f"❌ Health check failed: {response.status_code}")
    print("-" * 50)

def test_text_vectorization():
    """Test text vectorization endpoint"""
    print("📝 Testing text vectorization...")

    sample_text = """
    This is a sample text for testing the vectorization endpoint.
    It contains multiple sentences and should be chunked appropriately.
    The system will generate embeddings for each chunk and store them in the vector database.
    This allows for semantic search and retrieval of relevant information.
    """

    data = {
        "text": sample_text,
        "document_id": "test_text_doc_001",
        "metadata": {
            "source": "test_script",
            "category": "sample"
        }
    }

    response = requests.post(f"{BASE_URL}/text/vectorize", json=data)
    if response.status_code == 200:
        result = response.json()
        print("✅ Text vectorization started")
        print(f"Task ID: {result['task_id']}")

        # Monitor task status
        task_id = result['task_id']
        return monitor_task(task_id)
    else:
        print(f"❌ Text vectorization failed: {response.status_code}")
        print(response.text)
    print("-" * 50)

def test_document_upload():
    """Test document upload endpoint"""
    print("📄 Testing document upload...")

    # Create a sample text file
    sample_content = """
    Machine Learning and Vector Databases

    Vector databases are specialized databases designed to store and query vector embeddings.
    These embeddings are numerical representations of unstructured data like text, images, or audio.

    Key benefits of vector databases:
    1. Semantic search capabilities
    2. Similarity matching
    3. Scalable vector operations
    4. Support for high-dimensional data

    Applications include:
    - Recommendation systems
    - Natural language processing
    - Computer vision
    - Anomaly detection
    """

    # Write to temporary file
    temp_file = Path("temp_test_doc.txt")
    temp_file.write_text(sample_content)

    try:
        with open(temp_file, 'rb') as f:
            files = {'file': ('test_document.txt', f, 'text/plain')}
            response = requests.post(f"{BASE_URL}/documents/upload", files=files)

        if response.status_code == 200:
            result = response.json()
            print("✅ Document upload started")
            print(f"Task ID: {result['task_id']}")

            # Monitor task status
            task_id = result['task_id']
            return monitor_task(task_id)
        else:
            print(f"❌ Document upload failed: {response.status_code}")
            print(response.text)
    finally:
        # Clean up temp file
        if temp_file.exists():
            temp_file.unlink()

    print("-" * 50)

def monitor_task(task_id, max_wait=60):
    """Monitor a background task until completion"""
    print(f"⏳ Monitoring task {task_id}...")

    start_time = time.time()
    while time.time() - start_time < max_wait:
        response = requests.get(f"{BASE_URL}/tasks/{task_id}")
        if response.status_code == 200:
            task_status = response.json()
            print(f"Status: {task_status['status']} - {task_status['message']} ({task_status['progress']}%)")

            if task_status['status'] in ['completed', 'failed']:
                if task_status['status'] == 'completed':
                    print("✅ Task completed successfully")
                    if task_status.get('result'):
                        print("Result:", json.dumps(task_status['result'], indent=2))
                    return task_status['result']
                else:
                    print(f"❌ Task failed: {task_status.get('error', 'Unknown error')}")
                    return None

        time.sleep(2)

    print("⏰ Task monitoring timed out")
    return None

def test_search():
    """Test search endpoint"""
    print("🔍 Testing search...")

    data = {
        "query": "What are the benefits of vector databases?",
        "limit": 3,
        "active_documents_only": True
    }

    response = requests.post(f"{BASE_URL}/search", json=data)
    if response.status_code == 200:
        result = response.json()
        print("✅ Search completed")
        print(f"Found {result['count']} results")
        for i, res in enumerate(result['results'][:2]):  # Show first 2 results
            print(f"Result {i+1}: Score {res['score']:.3f} - {res['filename']}")
            print(f"Text preview: {res['chunk_text'][:100]}...")
    else:
        print(f"❌ Search failed: {response.status_code}")
        print(response.text)
    print("-" * 50)

def test_qa():
    """Test Q&A endpoint"""
    print("🤖 Testing Q&A...")

    data = {
        "question": "What are the main applications of vector databases?",
        "limit": 3,
        "active_documents_only": True
    }

    response = requests.post(f"{BASE_URL}/qa/ask", json=data)
    if response.status_code == 200:
        result = response.json()
        print("✅ Q&A completed")
        print(f"Question: {result['question']}")
        print(f"Answer: {result['answer']}")
        print(f"Sources: {len(result.get('sources', []))}")
    else:
        print(f"❌ Q&A failed: {response.status_code}")
        if response.status_code == 503:
            print("Note: Q&A requires Google API key to be configured")
        else:
            print(response.text)
    print("-" * 50)

def test_list_documents():
    """Test document listing endpoint"""
    print("📋 Testing document listing...")

    response = requests.get(f"{BASE_URL}/documents")
    if response.status_code == 200:
        result = response.json()
        print("✅ Document listing completed")
        print(f"Total documents: {result['count']}")
        for doc in result['documents'][:3]:  # Show first 3 documents
            print(f"- {doc['filename']} (ID: {doc['document_id']}, Active: {doc['active']})")
    else:
        print(f"❌ Document listing failed: {response.status_code}")
        print(response.text)
    print("-" * 50)

def test_stats():
    """Test stats endpoint"""
    print("📊 Testing stats...")

    response = requests.get(f"{BASE_URL}/stats")
    if response.status_code == 200:
        result = response.json()
        print("✅ Stats retrieved")
        print(json.dumps(result, indent=2))
    else:
        print(f"❌ Stats failed: {response.status_code}")
        print(response.text)
    print("-" * 50)

def main():
    """Run all tests"""
    print("🚀 Starting FastAPI Vector Database API Tests")
    print("=" * 60)

    try:
        # Basic tests
        test_health_check()

        # Upload and processing tests
        test_text_vectorization()
        test_document_upload()

        # Wait a bit for processing to complete
        time.sleep(5)

        # Query tests
        test_search()
        test_qa()

        # Management tests
        test_list_documents()
        test_stats()

        print("🎉 All tests completed!")

    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to API. Make sure the FastAPI server is running on http://localhost:8000")
        print("Run: python fastapi_app.py")
    except Exception as e:
        print(f"❌ Test failed with error: {e}")

if __name__ == "__main__":
    main()