print("Loading flask_app.py")
from app import app

def main(event, context):
    print("flask_app.main function called")
    return app
