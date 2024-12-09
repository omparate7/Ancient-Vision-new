from pymongo import MongoClient

# Connect to the MongoDB server
client = MongoClient("mongodb://localhost:27017/")

db = client["ancient_vision"]

print("Database initialized successfully!")