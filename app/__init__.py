from flask import Flask

def create_app():
    app = Flask(__name__)
    
    # Import routes to avoid circular imports
    from . import server
    
    return app
