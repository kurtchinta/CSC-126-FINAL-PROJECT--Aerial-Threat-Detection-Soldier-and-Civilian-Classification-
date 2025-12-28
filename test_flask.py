"""Quick test to verify Flask works"""
import sys
print("Python executable:", sys.executable)
print("Python version:", sys.version)

try:
    import flask
    print("✓ Flask imported successfully, version:", flask.__version__)
except Exception as e:
    print("✗ Flask import failed:", e)

try:
    import flask_cors
    print("✓ flask_cors imported successfully")
except Exception as e:
    print("✗ flask_cors import failed:", e)

try:
    from flask import Flask
    app = Flask(__name__)
    print("✓ Flask app created successfully")
except Exception as e:
    print("✗ Flask app creation failed:", e)

print("\n✓ All imports successful!")
