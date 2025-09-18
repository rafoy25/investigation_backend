#!/usr/bin/env python3
"""
Quick start script for the Document QNA System
"""

import subprocess
import sys
import os
import time
from pathlib import Path

def check_requirements():
    """Check if requirements are installed"""
    try:
        import streamlit
        import pymilvus
        import sentence_transformers
        return True
    except ImportError:
        return False

def install_requirements():
    """Install requirements"""
    print("📦 Installing requirements...")
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], check=True)
        return True
    except subprocess.CalledProcessError:
        return False

def check_milvus():
    """Check if Milvus is running"""
    try:
        from pymilvus import connections
        connections.connect(host="localhost", port="19530")
        connections.disconnect("default")
        return True
    except Exception:
        return False

def start_milvus():
    """Start Milvus containers"""
    compose_file = "milvus-standalone-docker-compose-gpu.yml"
    if not os.path.exists(compose_file):
        print(f"❌ Docker compose file not found: {compose_file}")
        return False
    
    print("🐳 Starting Milvus containers...")
    try:
        subprocess.run(["docker-compose", "-f", compose_file, "up", "-d"], check=True)
        print("⏳ Waiting for Milvus to start...")
        time.sleep(15)
        return True
    except subprocess.CalledProcessError:
        print("❌ Failed to start Milvus containers")
        return False

def run_streamlit():
    """Run the Streamlit application"""
    print("🚀 Starting Streamlit application...")
    try:
        subprocess.run([sys.executable, "-m", "streamlit", "run", "main_app.py"], check=True)
    except KeyboardInterrupt:
        print("\n👋 Application stopped by user")
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to run Streamlit: {e}")

def main():
    """Main function"""
    print("🚀 Document QNA System - Quick Start")
    print("=" * 40)
    
    # Check requirements
    if not check_requirements():
        print("📦 Installing missing requirements...")
        if not install_requirements():
            print("❌ Failed to install requirements")
            sys.exit(1)
    
    # Check Milvus
    if not check_milvus():
        print("⚠️  Milvus not running, attempting to start...")
        if not start_milvus():
            print("❌ Could not start Milvus")
            print("💡 Please ensure Docker is installed and running")
            print("💡 Manual start: docker-compose -f milvus-standalone-docker-compose-gpu.yml up -d")
            sys.exit(1)
        
        # Wait and check again
        time.sleep(5)
        if not check_milvus():
            print("⚠️  Milvus may still be starting up...")
            print("💡 The application will attempt to connect when you initialize the system")
    
    print("✅ All systems ready!")
    print("\n📋 Quick start tips:")
    print("1. Initialize the system using the sidebar")
    print("2. Set your OpenAI API key in environment variables for QNA")
    print("3. Upload documents and start asking questions!")
    
    # Run the application
    run_streamlit()

if __name__ == "__main__":
    main()