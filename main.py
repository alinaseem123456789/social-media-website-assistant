#This one Uses local sentence Transformer for embeddings
import uuid
import random
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from groq import Groq 
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue, PayloadSchemaType
from sentence_transformers import SentenceTransformer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from spellchecker import SpellChecker
import os
import re
import json
import time
from dotenv import load_dotenv

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

# --- INITIALIZE MODELS & CLOUD CLIENT ---
embed_model = SentenceTransformer('all-MiniLM-L6-v2')
analyzer = SentimentIntensityAnalyzer()
spell = SpellChecker()
groq_client = Groq(api_key=GROQ_API_KEY)

q_client = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY,
)

COLLECTION_NAME = "my_collection"

# --- CLOUD INITIALIZATION ---
if not q_client.collection_exists(COLLECTION_NAME):
    q_client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=384, distance=Distance.COSINE),
    )

try:
    q_client.create_payload_index(COLLECTION_NAME, "user_id", PayloadSchemaType.INTEGER)
    q_client.create_payload_index(COLLECTION_NAME, "entity", PayloadSchemaType.KEYWORD)
    q_client.create_payload_index(COLLECTION_NAME, "timestamp", PayloadSchemaType.FLOAT)
    print("Cloud indexes verified.")
except Exception as e:
    print(f"Index Note: {e}")

chat_histories = {}
MAX_HISTORY = 10  

class ChatRequest(BaseModel):
    message: str
    user_id: str

print("Initializing Intent Manager...")
intent_manager = IntentManager(embed_model)
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
        if w.isalpha():
            corrected_words.append(spell.correction(w) or w)
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
    
    response = groq_client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=100
    )
    try:
        result = response.choices[0].message.content
        result = result.strip()
        if result.startswith('```json'):
            result = result.replace('```json', '').replace('```', '')
        if result.startswith('```'):
            result = result.replace('```', '')
        return json.loads(result)
    except Exception as e:
        print(f" Entity extraction failed: {e}")
        return {"type": "none"}

def smart_save_to_memory(text: str, user_id: int):
    print(f"\n Attempting to save: {text}")
    
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
        print(f" Missing entity or value")
        return
        
    canonical_text = f"The user's {entity} is {value}."
    print(f" Canonical: {canonical_text}")
    
    vector = embed_model.encode(canonical_text).tolist()

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
    print(f" Saved to memory!")

    if existing_points:
        q_client.delete(collection_name=COLLECTION_NAME, points_selector=[existing_points[0].id])
        print(f"Updated existing memory")

def delete_memory(search_query: str, user_id: int):
    query_vector = embed_model.encode(search_query).tolist()
    user_filter = Filter(must=[FieldCondition(key="user_id", match=MatchValue(value=user_id))])
    search_result = q_client.query_points(  # Changed from search
        collection_name=COLLECTION_NAME,
        query=query_vector,  # Changed from query_vector
        query_filter=user_filter,
        limit=1
    ).points  # Added .points
    
    if search_result:
        point_id = search_result[0].id
        deleted_text = search_result[0].payload.get("text")
        q_client.delete(collection_name=COLLECTION_NAME, points_selector=[point_id])
        return f"I have forgotten: '{deleted_text}'"
    return "I couldn't find that memory."

def extract_fact(message: str):
    prompt = f"Convert to 3rd person fact: '{message}'. Core Fact:"
    response = groq_client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=50
    )
    return response.choices[0].message.content.strip()

def update_history(user_id: int, role: str, content: str):
    if user_id not in chat_histories: chat_histories[user_id] = []
    chat_histories[user_id].append({"role": role, "content": content})
    if len(chat_histories[user_id]) > MAX_HISTORY:
        chat_histories[user_id] = chat_histories[user_id][-MAX_HISTORY:]

def generate_multi_queries(query: str, groq_client):
    prompt = f"Generate 3 short search queries for: '{query}'. One per line."
    response = groq_client.chat.completions.create(
        model="llama-3.1-8b-instant", messages=[{"role": "user", "content": prompt}], temperature=0.2
    )
    return response.choices[0].message.content.strip().split("\n")

# ROUTES 
@app.post("/api/chat")
async def chat(request: ChatRequest): 
    try:
        numeric_user_id = int(request.user_id)
    except:
        numeric_user_id = 0

    sentiment = analyzer.polarity_scores(request.message)
    mood_context = "Cheerful" if sentiment['compound'] >= 0.05 else "Empathetic" if sentiment['compound'] <= -0.05 else "Helpful"

    # USE THE NEW INTENT MANAGER
    intent_result = intent_manager.classify_intent(request.message)
    intent = intent_result['intent']
    
    print(f"\n User: {request.message}")
    print(f"Intent: {intent} (confidence: {intent_result['confidence']:.2f})")
    
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
            vector = embed_model.encode(q).tolist()
            search_result = q_client.query_points(  # Changed from search
                collection_name=COLLECTION_NAME,
                query=vector,  # Changed from query_vector
                query_filter=Filter(must=[FieldCondition(key="user_id", match=MatchValue(value=numeric_user_id))]),
                limit=2
            ).points  # Added .points
            all_results.extend([hit.payload["text"] for hit in search_result])
        context = "User Facts: " + " | ".join(list(set(all_results)))
        print(f"Retrieved {len(set(all_results))} facts")
    posting_rule = "If the user mentions an achievement or interesting update, suggest a social media post using the format: [CREATE_POST: Your draft here]."
    
    history = chat_histories.get(numeric_user_id, [])
    messages = [{"role": "system", "content": f"You are a supportive AI friend. {mood_context}. Context: {context}. {posting_rule}"}]
    messages.extend(history)
    messages.append({"role": "user", "content": request.message})

    try:
        completion = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=messages,
            temperature=0.7
        )
        final_reply = completion.choices[0].message.content
        
        # EXTRACT POST SUGGESTION
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
        print(f"Error: {e}")
        return {
            "response": "I hit a snag!", 
            "intent": intent,
            "intent_confidence": 0.0,
            "action_taken": None,
            "post_suggestion": None
        }
@app.get("/api/user-data/{user_id}")
async def get_user_data(user_id: str):
    points, _ = q_client.scroll(  # This one stays as 'scroll' - no change
        collection_name=COLLECTION_NAME,
        scroll_filter=Filter(must=[FieldCondition(key="user_id", match=MatchValue(value=int(user_id)))]),
        with_payload=True
    )
    return {"memories": [p.payload.get("text") for p in points]}
#changes
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