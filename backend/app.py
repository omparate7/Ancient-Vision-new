from flask import Flask
from flask_cors import CORS
from pymongo import MongoClient
from dotenv import load_dotenv
import os

load_dotenv()

app = Flask(__name__)
CORS(app)

# MongoDB configuration
mongo_uri = os.getenv("MONGO_URI")
client = MongoClient(mongo_uri)

# Route to test server
@app.route('/')
def home():
    return "Welcome to Ancient Vision Backend!"