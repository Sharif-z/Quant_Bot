import os
from dotenv import load_dotenv
from pymongo import MongoClient

load_dotenv()
uri = os.getenv("MONGO_URI")
print(f"Connecting to: {uri.split('@')[1] if '@' in uri else 'Invalid URI'}")

try:
    client = MongoClient(uri, serverSelectionTimeoutMS=5000)
    info = client.server_info()
    print("SUCCESS: Connected to MongoDB Cluster!")
except Exception as e:
    print(f"ERROR: {e}")
