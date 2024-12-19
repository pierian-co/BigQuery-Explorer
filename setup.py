#!/usr/bin/env python3
import os
import sys
import subprocess
import json
from pathlib import Path
from typing import List, Dict

def check_python_version() -> bool:
    """Check if Python version is 3.7 or higher"""
    required_version = (3, 7)
    current_version = sys.version_info[:2]
    
    if current_version < required_version:
        print(f"❌ Python {required_version[0]}.{required_version[1]} or higher is required")
        print(f"Current version: {current_version[0]}.{current_version[1]}")
        return False
    
    print(f"✅ Python version {current_version[0]}.{current_version[1]} meets requirements")
    return True

def install_requirements() -> bool:
    """Install required packages from requirements.txt"""
    try:
        print("\nInstalling requirements...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Requirements installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install requirements: {str(e)}")
        return False

def check_gcloud_credentials() -> bool:
    """Check if Google Cloud credentials are properly configured"""
    # First check environment variable
    creds_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    
    if creds_path:
        try:
            with open(creds_path, 'r') as f:
                creds_data = json.load(f)
                
            required_keys = ['type', 'project_id', 'private_key_id', 'private_key']
            if all(key in creds_data for key in required_keys):
                print(f"✅ Found valid service account credentials for project: {creds_data['project_id']}")
                return True
                
        except FileNotFoundError:
            print(f"❌ Credentials file not found at: {creds_path}")
        except json.JSONDecodeError:
            print(f"❌ Credentials file is not valid JSON: {creds_path}")
    
    # Check for gcloud authentication
    try:
        result = subprocess.run(['gcloud', 'auth', 'list'], 
                              capture_output=True, 
                              text=True)
        
        if "No credentialed accounts." in result.stdout:
            print("❌ No gcloud authentication found")
            print("\nTo authenticate, you can either:")
            print("1. Run 'gcloud auth login' to authenticate with your Google account")
            print("2. Set GOOGLE_APPLICATION_CREDENTIALS with a service account key:")
            print('   export GOOGLE_APPLICATION_CREDENTIALS="path/to/your/credentials.json"')
            return False
            
        # Check if active account exists
        result = subprocess.run(['gcloud', 'config', 'get-value', 'account'], 
                              capture_output=True, 
                              text=True)
        
        if result.stdout.strip():
            print(f"✅ Found gcloud authentication for account: {result.stdout.strip()}")
            
            # Get project ID
            project = subprocess.run(['gcloud', 'config', 'get-value', 'project'], 
                                   capture_output=True, 
                                   text=True)
            if project.stdout.strip():
                print(f"✅ Using GCP project: {project.stdout.strip()}")
            return True
            
    except FileNotFoundError:
        print("❌ gcloud CLI not found. Please install Google Cloud SDK:")
        print("https://cloud.google.com/sdk/docs/install")
        return False
    
    print("❌ No valid authentication method found")
    return False

def create_directories() -> bool:
    """Create necessary project directories"""
    try:
        directories = [
            '.streamlit',
            'cache',
            'credentials'
        ]
        
        for directory in directories:
            Path(directory).mkdir(exist_ok=True)
            
        # Create empty streamlit secrets file if it doesn't exist
        secrets_file = Path('.streamlit/secrets.toml')
        if not secrets_file.exists():
            secrets_file.write_text("")
            
        print("✅ Project directories created")
        return True
        
    except Exception as e:
        print(f"❌ Failed to create directories: {str(e)}")
        return False

def main():
    print("BigQuery Explorer Setup\n")
    
    # Track setup status
    setup_successful = True
    
    # Run setup steps
    if not check_python_version():
        setup_successful = False
    
    if not install_requirements():
        setup_successful = False
    
    if not check_gcloud_credentials():
        setup_successful = False
    
    if not create_directories():
        setup_successful = False
    
    # Final status
    print("\nSetup Summary:")
    if setup_successful:
        print("✅ Setup completed successfully!")
        print("\nTo start the application:")
        print("1. cd frontend")
        print("2. streamlit run frontend/app.py")
    else:
        print("❌ Setup completed with errors. Please resolve the issues above and try again.")

if __name__ == "__main__":
    main() 