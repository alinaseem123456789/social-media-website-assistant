# main_optimized.py
import uuid
import random
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from groq import Groq 
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct, Filter, 
    FieldCondition, MatchValue, PayloadSchemaType,
    ScalarQuantization, ScalarType, OptimizersConfigDiff,
    QuantizationConfig  # Add this import
)
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from spellchecker import SpellChecker
import os
import re
import json
import time
from dotenv import load_dotenv
from collections import deque
from functools import lru_cache
import hashlib
import requests  # Add for API-based embeddings

# Import the new intent manager
from intents import IntentManager

# --- LOAD CONFIG ---
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173","https://social-media-project-one.vercel.app"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================
# FIX 1: Initialize Qdrant with compatibility check disabled
# ============================================================

q_client = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY,
    timeout=60,
    prefer_grpc=False,  # Use HTTP instead of gRPC for better compatibility
)

# Check and create collection with proper syntax
COLLECTION_NAME = "my_collection_optimized"

try:
    # First check if collection exists
    collections = q_client.get_collections()
    existing_names = [c.name for c in collections.collections]
    
    if COLLECTION_NAME not in existing_names:
        # Create collection with compatible parameters
        q_client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(
                size=384, 
                distance=Distance.COSINE,
            ),
            # Use simpler configuration without quantization (for compatibility)
        )
        print(f"✅ Created collection: {COLLECTION_NAME}")
    else:
        print(f"✅ Collection already exists: {COLLECTION_NAME}")
        
except Exception as e:
    print(f"Collection creation warning: {e}")
    # Continue anyway - collection might exist with different config

# Create indices if they don't exist
try:
    q_client.create_payload_index(COLLECTION_NAME, "user_id", PayloadSchemaType.INTEGER)
    q_client.create_payload_index(COLLECTION_NAME, "entity", PayloadSchemaType.KEYWORD)
    q_client.create_payload_index(COLLECTION_NAME, "timestamp", PayloadSchemaType.FLOAT)
    print("✅ Cloud indexes verified.")
except Exception as e:
    print(f"Index Note: {e}")

# ============================================================
# FIX 2: Use a proper free embedding service (no local model!)
# ============================================================

class RemoteEmbedder:
    """
    Generate embeddings using a free API service
    This avoids loading any local model!
    """
    def __init__(self):
        self.cache = {}
        # Option 1: Hugging Face Inference API (free tier)
        self.hf_api_key = os.getenv("HF_API_KEY")  # Optional, get from huggingface.co
        self.hf_headers = {"Authorization": f"Bearer {self.hf_api_key}"} if self.hf_api_key else {}
        
        # Option 2: Jina AI free API (5 free embeddings per day, no key needed for test)
        # Option 3: Local fallback (simple hash-based)
    
    def encode(self, text: str) -> list:
        """Get embedding using preferred method"""
        
        # Check cache
        cache_key = hashlib.md5(text.encode()).hexdigest()
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        # Try Hugging Face Inference API first (if key available)
        vector = self._get_huggingface_embedding(text)
        
        # Fallback to simple hash-based embedding
        if not vector:
            vector = self._simple_hash_embedding(text)
        
        # Cache with size limit
        self.cache[cache_key] = vector
        if len(self.cache) > 500:
            # Remove oldest 100 items
            keys_to_remove = list(self.cache.keys())[:100]
            for k in keys_to_remove:
                del self.cache[k]
        
        return vector
    
    def _get_huggingface_embedding(self, text: str) -> list:
        """Use Hugging Face's free embedding API"""
        if not self.hf_api_key:
            return None
        
        try:
            API_URL = "https://api-inference.huggingface.co/pipeline/feature-extraction/sentence-transformers/all-MiniLM-L6-v2"
            response = requests.post(
                API_URL,
                headers=self.hf_headers,
                json={"inputs": text},
                timeout=10
            )
            if response.status_code == 200:
                return response.json()
            else:
                print(f"HF API error: {response.status_code}")
                return None
        except Exception as e:
            print(f"HF embedding error: {e}")
            return None
    
    def _simple_hash_embedding(self, text: str) -> list:
        """Simple hash-based embedding (0 memory, works anywhere)"""
        vector = []
        text_hash = hashlib.sha256(text.encode()).digest()
        
        # Generate 384-dim vector from hash
        for i in range(384):
            hash_val = int.from_bytes(text_hash[i % 32:i % 32 + 1], 'little')
            # Normalize to range about [-1, 1]
            vector.append((hash_val / 255) * 2 - 1)
        
        return vector

# Initialize components
analyzer = SentimentIntensityAnalyzer()
spell = SpellChecker()
groq_client = Groq(api_key=GROQ_API_KEY)

# Initialize remote embedder (NO local model!)
embedder = RemoteEmbedder()

print("✅ All components initialized without local models!")

# ============================================================
# Chat history with memory limits
# ============================================================

chat_histories = {}
MAX_HISTORY = 10
MAX_USERS = 100
user_last_active = {}

class ChatRequest(BaseModel):
    message: str
    user_id: str

print("Initializing Intent Manager...")
intent_manager = IntentManager()
print("Intent Manager ready!")

def normalize_text(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text

def correct_spelling(text: str) -> str:
    words = text.split()
    corrected_words = []
    for w in words:
        if w.isalpha() and len(w) > 1:
            correction = spell.correction(w)
            corrected_words.append(correction if correction else w)
        else:
            corrected_words.append(w)
    return " ".join(corrected_words)

def extract_entity_and_value(text: str):
    prompt = f"""Extract structured personal info from this sentence: "{text}"
    
    Examples:
    "My name is John" -> {{"type": "profile", "entity": "name", "value": "John"}}
    "I live in London" -> {{"type": "profile", "entity": "city", "value": "London"}}
    "I like pizza" -> {{"type": "preference", "entity": "likes", "value": "pizza"}}
    "I am a developer" -> {{"type": "profile", "entity": "job", "value": "developer"}}
    
    Output ONLY the JSON, no other text."""
    
    try:
        response = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=100
        )
        result = response.choices[0].message.content
        result = result.strip()
        # Clean up markdown
        if result.startswith('```json'):
            result = result.replace('```json', '').replace('```', '')
        if result.startswith('```'):
            result = result.replace('```', '')
        return json.loads(result)
    except Exception as e:
        print(f"Entity extraction failed: {e}")
        return {"type": "none"}

def smart_save_to_memory(text: str, user_id: int):
    print(f"\n📝 Attempting to save: {text}")
    
    clean = normalize_text(text)
    clean = correct_spelling(clean)
    print(f"Cleaned: {clean}")
    
    entity_info = extract_entity_and_value(clean)
    print(f"Extracted: {entity_info}")

    if entity_info.get("type") == "none":
        print("Not memory-worthy")
        return

    mem_type = entity_info.get("type")
    entity = entity_info.get("entity")
    value = entity_info.get("value")
    
    if not entity or not value:
        print(f"Missing entity or value")
        return
        
    canonical_text = f"The user's {entity} is {value}."
    print(f"Canonical: {canonical_text}")
    
    # Generate embedding using remote service (NO local model!)
    vector = embedder.encode(canonical_text)

    try:
        user_filter = Filter(must=[
            FieldCondition(key="user_id", match=MatchValue(value=user_id)),
            FieldCondition(key="entity", match=MatchValue(value=entity))
        ])

        existing_points, _ = q_client.scroll(
            collection_name=COLLECTION_NAME,
            scroll_filter=user_filter,
            with_payload=True,
            limit=1
        )

        q_client.upsert(
            collection_name=COLLECTION_NAME,
            points=[PointStruct(
                id=str(uuid.uuid4()),
                vector=vector,
                payload={
                    "text": canonical_text, "user_id": user_id, "type": mem_type,
                    "entity": entity, "value": value, "timestamp": time.time()
                }
            )]
        )
        print(f"✅ Saved to memory!")

        if existing_points:
            q_client.delete(collection_name=COLLECTION_NAME, points_selector=[existing_points[0].id])
            print(f"🔄 Updated existing memory")
    except Exception as e:
        print(f"Qdrant error: {e}")

def delete_memory(search_query: str, user_id: int):
    try:
        query_vector = embedder.encode(search_query)
        user_filter = Filter(must=[FieldCondition(key="user_id", match=MatchValue(value=user_id))])
        search_result = q_client.search(
            collection_name=COLLECTION_NAME, 
            query_vector=query_vector, 
            query_filter=user_filter, 
            limit=1
        )
        
        if search_result:
            point_id = search_result[0].id
            deleted_text = search_result[0].payload.get("text")
            q_client.delete(collection_name=COLLECTION_NAME, points_selector=[point_id])
            return f"I have forgotten: '{deleted_text}'"
        return "I couldn't find that memory."
    except Exception as e:
        print(f"Delete error: {e}")
        return "I couldn't find that memory."

def extract_fact(message: str):
    prompt = f"Convert to 3rd person fact: '{message}'. Core Fact:"
    try:
        response = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=50
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Extract fact error: {e}")
        return message

def update_history(user_id: int, role: str, content: str):
    # Track user activity
    user_last_active[user_id] = time.time()
    
    # Clean up inactive users if too many
    if len(chat_histories) > MAX_USERS:
        inactive_users = sorted(user_last_active.items(), key=lambda x: x[1])[:MAX_USERS//2]
        for uid, _ in inactive_users:
            chat_histories.pop(uid, None)
            user_last_active.pop(uid, None)
    
    # Use deque with max length
    if user_id not in chat_histories:
        chat_histories[user_id] = deque(maxlen=MAX_HISTORY)
    
    chat_histories[user_id].append({"role": role, "content": content})

def generate_multi_queries(query: str, groq_client):
    prompt = f"Generate 3 short search queries for: '{query}'. One per line."
    try:
        response = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant", 
            messages=[{"role": "user", "content": prompt}], 
            temperature=0.2
        )
        return response.choices[0].message.content.strip().split("\n")
    except Exception as e:
        print(f"Multi-query error: {e}")
        return [query]

# ============================================================
# ROUTES 
# ============================================================

@app.post("/api/chat")
async def chat(request: ChatRequest): 
    try:
        numeric_user_id = int(request.user_id)
    except:
        numeric_user_id = 0

    sentiment = analyzer.polarity_scores(request.message)
    mood_context = "Cheerful" if sentiment['compound'] >= 0.05 else "Empathetic" if sentiment['compound'] <= -0.05 else "Helpful"

    # Use the intent manager
    intent_result = intent_manager.classify_intent(request.message)
    intent = intent_result['intent']
    
    print(f"\n👤 User: {request.message}")
    print(f"🎯 Intent: {intent} (confidence: {intent_result['confidence']:.2f})")
    
    context = ""
    status_update = ""

    if intent == "DELETE":
        status_update = delete_memory(request.message, numeric_user_id)
        context = f"SYSTEM: {status_update}"
    elif intent == "STORE":
        extracted_fact = extract_fact(request.message)
        smart_save_to_memory(extracted_fact, numeric_user_id)
        status_update = f"Saved: {extracted_fact}"
        context = f"SYSTEM: Just stored {extracted_fact}"
    elif intent == "RETRIEVE":
        expanded_queries = generate_multi_queries(request.message, groq_client)
        expanded_queries.append(request.message) 
        all_results = []
        for q in expanded_queries:
            vector = embedder.encode(q)
            try:
                search_result = q_client.search(
                    collection_name=COLLECTION_NAME,
                    query_vector=vector,
                    query_filter=Filter(must=[FieldCondition(key="user_id", match=MatchValue(value=numeric_user_id))]),
                    limit=2
                )
                all_results.extend([hit.payload["text"] for hit in search_result])
            except Exception as e:
                print(f"Search error for '{q}': {e}")
        context = "User Facts: " + " | ".join(list(set(all_results)))
        print(f"📚 Retrieved {len(set(all_results))} facts")
    
    posting_rule = "If the user mentions an achievement or interesting update, suggest a social media post using the format: [CREATE_POST: Your draft here]."
    
    history = list(chat_histories.get(numeric_user_id, []))
    messages = [{"role": "system", "content": f"You are a supportive AI friend. {mood_context}. Context: {context}. {posting_rule}"}]
    messages.extend(history)
    messages.append({"role": "user", "content": request.message})

    try:
        completion = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=messages,
            temperature=0.7,
            max_tokens=500
        )
        final_reply = completion.choices[0].message.content
        
        # Extract post suggestion
        post_suggestion = None
        match = re.search(r"\[CREATE_POST:\s*(.*?)\]", final_reply, re.DOTALL | re.IGNORECASE)
        if match:
            post_suggestion = match.group(1).strip()
            final_reply = re.sub(r"\[CREATE_POST:.*?\]", "", final_reply, flags=re.DOTALL | re.IGNORECASE).strip()

        update_history(numeric_user_id, "user", request.message)
        update_history(numeric_user_id, "assistant", final_reply)
        return {
            "response": str(final_reply),  
            "intent": str(intent),  
            "intent_confidence": float(intent_result['confidence']),  
            "action_taken": str(status_update) if status_update else None,
            "post_suggestion": str(post_suggestion) if post_suggestion else None
        }
    except Exception as e:
        print(f"Chat completion error: {e}")
        return {
            "response": "I hit a snag!", 
            "intent": intent,
            "intent_confidence": 0.0,
            "action_taken": None,
            "post_suggestion": None
        }

@app.get("/api/user-data/{user_id}")
async def get_user_data(user_id: str):
    try:
        points, _ = q_client.scroll(
            collection_name=COLLECTION_NAME,
            scroll_filter=Filter(must=[FieldCondition(key="user_id", match=MatchValue(value=int(user_id)))]),
            with_payload=True,
            limit=100
        )
        return {"memories": [p.payload.get("text") for p in points]}
    except Exception as e:
        print(f"Get user data error: {e}")
        return {"memories": []}

@app.get("/api/intent-test/{message}")
async def test_intent(message: str):
    """Test endpoint to check intent classification"""
    result = intent_manager.classify_intent(message)
    return {
        "message": message,
        "intent": result['intent'],
        "confidence": result['confidence'],
        "details": result['details']
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)