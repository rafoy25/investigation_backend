#!/usr/bin/env python3
"""
Setup script for the Document QNA System
"""

import subprocess
import sys
import os
from pathlib import Path

def run_command(command, description):
    """Run a shell command and handle errors"""
    print(f"\n{'='*50}")
    print(f"🔧 {description}")
    print(f"{'='*50}")
    
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        if result.stdout:
            print(result.stdout)
        print(f"✅ {description} completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error: {description} failed!")
        print(f"Command: {command}")
        print(f"Error output: {e.stderr}")
        return False

def check_python_version():
    """Check if Python version is compatible"""
    print("🔍 Checking Python version...")
    
    version = sys.version_info
    if version.major == 3 and version.minor >= 8:
        print(f"✅ Python {version.major}.{version.minor}.{version.micro} is compatible")
        return True
    else:
        print(f"❌ Python {version.major}.{version.minor}.{version.micro} is not compatible")
        print("⚠️  Python 3.8 or higher is required")
        return False

def install_requirements():
    """Install required packages"""
    if not run_command(f"{sys.executable} -m pip install --upgrade pip", "Upgrading pip"):
        return False
    
    if not run_command(f"{sys.executable} -m pip install -r requirements.txt", "Installing requirements"):
        return False
    
    return True

def check_docker():
    """Check if Docker is running"""
    print("\n🐳 Checking Docker...")
    
    if not run_command("docker --version", "Checking Docker version"):
        print("⚠️  Docker is not installed or not in PATH")
        return False
    
    if not run_command("docker ps", "Checking Docker daemon"):
        print("⚠️  Docker daemon is not running")
        return False
    
    return True

def start_milvus():
    """Start Milvus using docker-compose"""
    print("\n🚀 Starting Milvus...")
    
    compose_file = "milvus-standalone-docker-compose-gpu.yml"
    if not os.path.exists(compose_file):
        print(f"❌ Docker compose file {compose_file} not found!")
        return False
    
    # Start Milvus
    if not run_command(f"docker-compose -f {compose_file} up -d", "Starting Milvus containers"):
        return False
    
    # Wait a bit for services to start
    print("⏳ Waiting for services to start...")
    import time
    time.sleep(10)
    
    # Check if services are running
    if not run_command(f"docker-compose -f {compose_file} ps", "Checking service status"):
        return False
    
    return True

def test_milvus_connection():
    """Test connection to Milvus"""
    print("\n🔗 Testing Milvus connection...")
    
    test_script = """
import sys
sys.path.append('.')
from milvus_client import MilvusClient

try:
    client = MilvusClient()
    client.connect()
    print("✅ Successfully connected to Milvus!")
    stats = client.get_collection_stats()
    print(f"📊 Collection stats: {stats}")
    client.disconnect()
except Exception as e:
    print(f"❌ Connection failed: {str(e)}")
    sys.exit(1)
"""
    
    with open("test_connection.py", "w") as f:
        f.write(test_script)
    
    success = run_command(f"{sys.executable} test_connection.py", "Testing Milvus connection")
    
    # Cleanup
    if os.path.exists("test_connection.py"):
        os.remove("test_connection.py")
    
    return success

def create_env_file():
    """Create environment file template"""
    env_content = """# Environment variables for Document QNA System
# Copy this to .env and fill in your values

# OpenAI API Key (required for QNA functionality)
OPENAI_API_KEY=your_openai_api_key_here

# Milvus connection (default values for local docker setup)
MILVUS_HOST=localhost
MILVUS_PORT=19530

# Optional: Other embedding service API keys
COHERE_API_KEY=your_cohere_api_key_here
"""
    
    with open(".env.example", "w") as f:
        f.write(env_content)
    
    print("📝 Created .env.example file")
    print("💡 Copy .env.example to .env and add your API keys")

def main():
    """Main setup function"""
    print("🚀 Document QNA System Setup")
    print("="*50)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Install requirements
    if not install_requirements():
        print("❌ Failed to install requirements")
        sys.exit(1)
    
    # Check Docker
    if not check_docker():
        print("⚠️  Docker check failed - you'll need to start Milvus manually")
        print("💡 Run: docker-compose -f milvus-standalone-docker-compose-gpu.yml up -d")
    else:
        # Start Milvus
        if not start_milvus():
            print("❌ Failed to start Milvus")
            sys.exit(1)
        
        # Test connection
        if not test_milvus_connection():
            print("❌ Milvus connection test failed")
            print("💡 Check if Milvus containers are running: docker ps")
    
    # Create environment file
    create_env_file()
    
    print("\n" + "="*50)
    print("🎉 Setup completed!")
    print("="*50)
    print("\n📋 Next steps:")
    print("1. Copy .env.example to .env and add your OpenAI API key")
    print("2. Run the application: streamlit run main_app.py")
    print("3. Open your browser to http://localhost:8501")
    print("\n💡 For help, see README.md")

if __name__ == "__main__":
    main()