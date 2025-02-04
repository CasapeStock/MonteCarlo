from flask import Flask, render_template
import os

# Determine the base directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(__name__, 
            template_folder=os.path.join(BASE_DIR, 'app', 'templates'),
            static_folder=os.path.join(BASE_DIR, 'app', 'static'))

@app.route('/')
def index():
    print("Index route accessed!")  # Console logging
    return "Hello, World! Server is running."

@app.route('/test')
def test():
    print("Test route accessed!")  # Console logging
    return "Test route is working!"

if __name__ == '__main__':
    print(f"Template folder: {app.template_folder}")
    print(f"Static folder: {app.static_folder}")
    app.run(
        host='0.0.0.0', 
        port=5000, 
        debug=True
    )
