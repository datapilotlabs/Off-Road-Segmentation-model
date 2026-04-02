#!/usr/bin/env python3
"""
Startup script for the Offroad Segmentation Predictor Flask app
"""

import os
import sys
import subprocess

def main():
    """Start the Flask application."""
    print("🚀 Starting Offroad Segmentation Predictor...")
    print("=" * 50)

    # Check if virtual environment exists
    venv_path = "myenvs"
    if not os.path.exists(venv_path):
        print("❌ Virtual environment not found. Please run training first to set it up.")
        return 1

    # Check if model exists
    model_exists = os.path.exists("segmentation_head_best.pth") or os.path.exists("segmentation_head.pth")
    if not model_exists:
        print("❌ No trained model found. Please run training first.")
        return 1

    # Activate virtual environment and run Flask app
    activate_script = os.path.join(venv_path, "Scripts", "activate.bat")
    flask_command = "python app.py"

    print("📦 Activating virtual environment...")
    print("🤖 Loading models and starting server...")
    print("🌐 Once started, open http://localhost:5000 in your browser")
    print("=" * 50)

    try:
        # Run the Flask app
        subprocess.run(f'cmd /c "{activate_script} && {flask_command}"', shell=True)
    except KeyboardInterrupt:
        print("\n👋 Server stopped by user")
    except Exception as e:
        print(f"❌ Error starting server: {e}")
        return 1

    return 0

if __name__ == "__main__":
    sys.exit(main())