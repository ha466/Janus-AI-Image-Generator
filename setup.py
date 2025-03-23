import subprocess
import sys
import os

def install_dependencies():
    print("Installing Janus Image Generator dependencies...")
    
    # List of required packages
    packages = [
        "torch",
        "transformers",
        "flask",
        "pillow",
        "numpy",
        "opencv-python",
        "opencv-contrib-python"
    ]
    
    # Try to install Janus from GitHub if it's not already installed
    try:
        import janus
        print("Janus already installed.")
    except ImportError:
        print("Installing Janus from GitHub...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "git+https://github.com/deepseek-ai/Janus.git"])
    
    # Install other dependencies
    for package in packages:
        print(f"Installing {package}...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        except Exception as e:
            print(f"Error installing {package}: {str(e)}")
    
    # Create logo if needed
    if not os.path.exists('static/janus-logo.png'):
        try:
            from create_logo import create_logo
            create_logo()
        except Exception as e:
            print(f"Could not create logo: {str(e)}. Will use fallback in the app.")
    
    print("\nSetup completed! You can now run the application with 'python gpu.py'")

if __name__ == "__main__":
    install_dependencies() 