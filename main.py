from agents.posts_agent import PostAgent
from agents.social_api import SocialAPI
from agents.friends_agents import FriendSuggestionAgent
from agents.engagement_agent import EngagementAgent
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
from RAG_Pipeline.NER import get_ner
ner = get_ner()
from supabase_client import get_conversation_store
from RAG_Pipeline.intent_manager import IntentManager
from RAG_Pipeline.context_compressor import ContextCompressor

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

embed_model = SentenceTransformer('all-MiniLM-L6-v2')
analyzer = SentimentIntensityAnalyzer()
spell = SpellChecker()
groq_client = Groq(api_key=GROQ_API_KEY)

q_client = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY,
)
COLLECTION_NAME = "my_collection"
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

context_compressor = ContextCompressor(groq_client, embed_model)
chat_histories = {}
MAX_HISTORY = 5 

social_api = SocialAPI()
post_agent = PostAgent(q_client, embed_model, groq_client)
pending_posts = {}

friend_agent = FriendSuggestionAgent(q_client, embed_model, groq_client)
pending_friends = {}

engagement_agent = EngagementAgent(q_client, embed_model, groq_client)
pending_engagement = {} 

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
        if w.isalpha():
            corrected_words.append(spell.correction(w) or w)
        else:
            corrected_words.append(w)
    return " ".join(corrected_words)


from RAG_Pipeline.final_extractor import FinalEntityExtractor
final_extractor = FinalEntityExtractor(ner, groq_client)

def extract_entity_and_value(text: str):
    """
    NEW: Complete extraction with generic fallback
    Returns ALL entities found
    """
    all_entities = final_extractor.extract_all_entities(text)
    if not all_entities:
        return {"type": "none"}
    primary = all_entities[0]
    
    return {
        "type": "profile",
        "entity": primary["entity"],
        "value": primary["value"],
        "all_entities": all_entities,  # Store all for multi-entity saving
        "generic_entities": [e for e in all_entities if e["entity"] == "generic"]
    }

def smart_save_to_memory(text: str, user_id: int, original_message: str = None):
    """Save ONLY valuable entities - rejects garbage"""
    print(f"\n Processing for storage: {text}")
    
    # Quick check for trivial messages
    trivial_phrases = ["hi", "hello", "ok", "good", "bad", "fine", "thanks"]
    if text.lower().strip() in trivial_phrases:
        print("Trivial message - nothing to store")
        return
    extraction_result = extract_entity_and_value(text)
    
    if extraction_result.get("type") == "none":
        print("No valuable entities found")
        return
    
    all_entities = extraction_result.get("all_entities", [])
    
    if not all_entities:
        print(" No valid entities after filtering")
        return
    print(f" Found {len(all_entities)} valuable entities")
    saved_count = 0
    for entity_info in all_entities:
        entity = entity_info["entity"]
        value = entity_info["value"]
        confidence = entity_info.get("confidence", 0.85)        
        if not value or len(value) < 2:
            continue
        garbage = ["her", "him", "alias", "name", "core fact", "it", "them", "they"]
        if value.lower() in garbage:
            print(f"   Skipping garbage: {entity} = {value}")
            continue
        
        print(f" Saving: {entity} = {value} (conf: {confidence})")        
        if entity == "likes":
            canonical_text = f"The user likes {value}."
        elif entity == "interest":
            canonical_text = f"The user is interested in {value}."
        elif entity == "age":
            canonical_text = f"The user is {value} years old."
        else:
            canonical_text = f"The user's {entity} is {value}."
        
        vector = embed_model.encode(canonical_text).tolist()        
        q_client.upsert(
            collection_name=COLLECTION_NAME,
            points=[PointStruct(
                id=str(uuid.uuid4()),
                vector=vector,
                payload={
                    "text": canonical_text,
                    "user_id": user_id,
                    "type": "profile",
                    "entity": entity,
                    "value": value,
                    "confidence": confidence,
                    "source": entity_info.get("source", "unknown"),
                    "original_text": original_message or text,
                    "timestamp": time.time()
                }
            )]
        )
        saved_count += 1
    print(f" Saved {saved_count} valuable memories")

def delete_memory(search_query: str, user_id: int):
    query_vector = embed_model.encode(search_query).tolist()
    user_filter = Filter(must=[FieldCondition(key="user_id", match=MatchValue(value=user_id))])
    search_result = q_client.query_points(  
        collection_name=COLLECTION_NAME,
        query=query_vector,  
        query_filter=user_filter,
        limit=1
    ).points  
    
    if search_result:
        point_id = search_result[0].id
        deleted_text = search_result[0].payload.get("text")
        q_client.delete(collection_name=COLLECTION_NAME, points_selector=[point_id])
        return f"I have forgotten: '{deleted_text}'"
    return "I couldn't find that memory."

def update_memory_by_entity(user_id: int, entity_type: str, new_value: str, original_message: str = None) -> str:
    """Update a specific entity type with new value"""
    search_result = q_client.scroll(
        collection_name=COLLECTION_NAME,
        scroll_filter=Filter(
            must=[
                FieldCondition(key="user_id", match=MatchValue(value=user_id)),
                FieldCondition(key="entity", match=MatchValue(value=entity_type.lower()))
            ]
        ),
        limit=1
    )
    
    if search_result[0]:
        old_point = search_result[0][0]
        old_value = old_point.payload.get("value")
        
        q_client.delete(collection_name=COLLECTION_NAME, points_selector=[old_point.id])
        
        if entity_type == "age":
            canonical_text = f"The user is {new_value} years old."
        elif entity_type == "likes":
            canonical_text = f"The user likes {new_value}."
        else:
            canonical_text = f"The user's {entity_type} is {new_value}."
        
        vector = embed_model.encode(canonical_text).tolist()
        
        q_client.upsert(
            collection_name=COLLECTION_NAME,
            points=[PointStruct(
                id=str(uuid.uuid4()),
                vector=vector,
                payload={
                    "text": canonical_text,
                    "user_id": user_id,
                    "type": "profile",
                    "entity": entity_type,
                    "value": new_value,
                    "confidence": 0.95,
                    "source": "user_update",
                    "original_text": original_message,
                    "previous_value": old_value,
                    "timestamp": time.time()
                }
            )]
        )
        return f"Updated {entity_type} from '{old_value}' to '{new_value}'"
    
    return None

def find_and_update_memory(user_id: int, search_text: str, new_value: str) -> str:
    """Find memory by text content and update it"""
    query_vector = embed_model.encode(search_text).tolist()
    user_filter = Filter(must=[FieldCondition(key="user_id", match=MatchValue(value=user_id))])
    
    search_result = q_client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        query_filter=user_filter,
        limit=1
    ).points
    
    if search_result:
        old_point = search_result[0]
        old_text = old_point.payload.get("text")
        
        q_client.delete(collection_name=COLLECTION_NAME, points_selector=[old_point.id])
        
        vector = embed_model.encode(new_value).tolist()
        
        q_client.upsert(
            collection_name=COLLECTION_NAME,
            points=[PointStruct(
                id=str(uuid.uuid4()),
                vector=vector,
                payload={
                    "text": new_value,
                    "user_id": user_id,
                    "type": "profile",
                    "entity": "updated_memory",
                    "value": new_value,
                    "confidence": 0.95,
                    "source": "user_update",
                    "original_text": new_value,
                    "previous_text": old_text,
                    "timestamp": time.time()
                }
            )]
        )
        return f"Updated memory from '{old_text}' to '{new_value}'"
    
    return None

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

@app.post("/api/chat")
async def chat(request: ChatRequest): 
    user_id = int(request.user_id)
    message = request.message    
    
    # ============================================
    # PART 1: PENDING WORKFLOWS (Highest Priority)
    # ============================================
    if user_id in pending_friends:
        result = await friend_agent.process(
            user_id=user_id,
            message=message,
            user_confirmation=message
        )
        
        if result.get('requests_sent') or result.get('cleared'):
            del pending_friends[user_id]
        elif result.get('awaiting_selection'):
            pending_friends[user_id] = result
        elif result.get('awaiting_preferences'):
            pending_friends[user_id] = result
        else:
            pending_friends[user_id] = result
        
        return {
            "response": result.get('final_response', ''),
            "friend_suggestions": result.get('friend_suggestions'),
            "intent": "FRIEND_SUGGESTION",
            "intent_confidence": 0.95,
            "action_taken": None,
            "response_time_ms": 0
        }
    
    if user_id in pending_posts:
        result = await post_agent.process(
            user_id=user_id,
            message=message,
            user_confirmation=message
        )
        
        if result.get('requests_sent') or result.get('cleared'):
            print(f"✅ Friend workflow completed, clearing pending_friends for user {user_id}")
            del pending_friends[user_id]
        elif result.get('awaiting_selection'):
            pending_friends[user_id] = result
        elif result.get('awaiting_preferences'):
            pending_friends[user_id] = result
        else:
            pending_friends[user_id] = result
        
        return {
            "response": result.get('final_response', ''),
            "post_suggestion": result.get('post_suggestion'),
            "intent": "CREATE_POST",
            "intent_confidence": 0.95,
            "action_taken": None,
            "response_time_ms": 0
        }
    
    intent_result = intent_manager.classify_intent(message, user_id=user_id)
    intent = intent_result['intent']
    print(f"🎯 Intent: {intent} (conf: {intent_result['confidence']:.2f}) - Message: {message[:50]}")
    if intent == "FRIEND_SUGGESTION":
        result = await friend_agent.process(user_id=user_id, message=message)
        if result.get('awaiting_selection') or result.get('awaiting_preferences'):
            pending_friends[user_id] = result
        return {
            "response": result.get('final_response', ''),
            "friend_suggestions": result.get('friend_suggestions'),
            "intent": "FRIEND_SUGGESTION",
            "intent_confidence": intent_result['confidence'],
            "action_taken": None,
            "response_time_ms": 0
        }
    
    if intent == "CREATE_POST":
        result = await post_agent.process(user_id=user_id, message=message)
        if result.get('awaiting_approval') or result.get('awaiting_edit'):
            pending_posts[user_id] = result
        return {
            "response": result.get('final_response', ''),
            "post_suggestion": result.get('post_suggestion'),
            "intent": "CREATE_POST",
            "intent_confidence": intent_result['confidence'],
            "action_taken": None,
            "response_time_ms": 0
        }
    
    if intent == "ENGAGEMENT":
        result = await engagement_agent.process(user_id=user_id, message=message)
        return {
            "response": result.get('final_response', ''),
            "engagement_suggestions": result.get('engagement_suggestions'),
            "birthday_suggestions": result.get('birthday_suggestions'),
            "inactive_friends": result.get('inactive_friends'),
            "intent": "ENGAGEMENT",
            "intent_confidence": intent_result['confidence'],
            "action_taken": None,
            "response_time_ms": 0
        }
    
    # ============================================
    # PART 4: RAG PROCESSING (DELETE, UPDATE, STORE, RETRIEVE, CHAT)
    # ============================================
    start_time = time.time()
    
    try:
        numeric_user_id = int(request.user_id)
    except:
        numeric_user_id = 0

    sentiment = analyzer.polarity_scores(request.message)
    mood_context = "Cheerful" if sentiment['compound'] >= 0.05 else "Empathetic" if sentiment['compound'] <= -0.05 else "Helpful"
    
    print(f" Sentiment: {sentiment['compound']:.2f}")
    
    context = ""
    status_update = ""
    
    if intent == "DELETE":
        status_update = delete_memory(request.message, numeric_user_id)
        context = f"SYSTEM: {status_update}"
        
    elif intent == "UPDATE":
        entity_type = intent_result.get('details', {}).get('entity')
        new_value = intent_result.get('details', {}).get('new_value')
        
        if entity_type and new_value:
            search_result = q_client.scroll(
                collection_name=COLLECTION_NAME,
                scroll_filter=Filter(
                    must=[
                        FieldCondition(key="user_id", match=MatchValue(value=numeric_user_id)),
                        FieldCondition(key="entity", match=MatchValue(value=entity_type.lower()))
                    ]
                ),
                limit=1
            )
            
            if search_result[0]:
                old_point = search_result[0][0]
                old_value = old_point.payload.get("value")
                
                q_client.delete(collection_name=COLLECTION_NAME, points_selector=[old_point.id])
                
                if entity_type == "age":
                    canonical_text = f"The user is {new_value} years old."
                elif entity_type == "likes":
                    canonical_text = f"The user likes {new_value}."
                else:
                    canonical_text = f"The user's {entity_type} is {new_value}."
                
                vector = embed_model.encode(canonical_text).tolist()
                
                q_client.upsert(
                    collection_name=COLLECTION_NAME,
                    points=[PointStruct(
                        id=str(uuid.uuid4()),
                        vector=vector,
                        payload={
                            "text": canonical_text,
                            "user_id": numeric_user_id,
                            "type": "profile",
                            "entity": entity_type,
                            "value": new_value,
                            "confidence": 0.95,
                            "source": "user_update",
                            "original_text": request.message,
                            "previous_value": old_value,
                            "timestamp": time.time()
                        }
                    )]
                )
                
                status_update = f"Updated {entity_type} from '{old_value}' to '{new_value}'"
                context = f"SYSTEM: {status_update}"
            else:
                extracted_fact = extract_fact(request.message)
                smart_save_to_memory(extracted_fact, numeric_user_id, request.message)
                status_update = f"Saved new fact: {entity_type} = {new_value}"
                context = f"SYSTEM: {status_update}"
        else:
            extracted_fact = extract_fact(request.message)
            if extracted_fact and extracted_fact != "NO_VALUABLE_INFO":
                smart_save_to_memory(extracted_fact, numeric_user_id, request.message)
                status_update = f"Updated based on: {extracted_fact}"
                context = f"SYSTEM: {status_update}"
            else:
                status_update = "Couldn't determine what to update"
                context = f"SYSTEM: {status_update}"
                
    elif intent == "STORE":
        extracted_fact = extract_fact(request.message)
        smart_save_to_memory(extracted_fact, numeric_user_id)
        status_update = f"Saved: {extracted_fact}"
        context = f"SYSTEM: Just stored {extracted_fact}"
        
    elif intent == "RETRIEVE":
        expanded_queries = generate_multi_queries(request.message, groq_client)
        expanded_queries.append(request.message)        
        expanded_queries = list(dict.fromkeys(expanded_queries))
        
        all_results = []
        seen_texts = set()
        
        for q in expanded_queries:
            vector = embed_model.encode(q).tolist()
            vector_results = q_client.query_points(
                collection_name=COLLECTION_NAME,
                query=vector,
                query_filter=Filter(must=[FieldCondition(key="user_id", match=MatchValue(value=numeric_user_id))]),
                limit=3
            ).points
            
            for hit in vector_results:
                text = hit.payload.get("text", "")
                if text and text not in seen_texts:
                    all_results.append(hit.payload)
                    seen_texts.add(text)
            
            if not vector_results:
                entities = extract_entity_and_value(q)
                if entities.get("type") != "none":
                    for entity_info in entities.get("all_entities", []):
                        entity_value = entity_info.get("value")
                        if entity_value and len(entity_value) > 2:
                            entity_results = q_client.scroll(
                                collection_name=COLLECTION_NAME,
                                scroll_filter=Filter(
                                    must=[
                                        FieldCondition(key="user_id", match=MatchValue(value=numeric_user_id)),
                                        FieldCondition(key="value", match=MatchValue(value=entity_value.lower()))
                                    ]
                                ),
                                limit=2
                            )
                            for point in entity_results[0]:
                                text = point.payload.get("text", "")
                                if text and text not in seen_texts:
                                    all_results.append(point.payload)
                                    seen_texts.add(text)
        
        print(f" Retrieved {len(all_results)} unique facts")
        
        if not all_results:
            context = "SYSTEM: I don't have any stored information about that yet."
            status_update = "No relevant memories found."
        else:
            all_results.sort(key=lambda x: x.get("confidence", 0), reverse=True)            
            current_time = time.time()
            for memory in all_results:
                memory_age = current_time - memory.get("timestamp", current_time)
                age_days = memory_age / (24 * 3600)
                time_penalty = max(0, min(0.3, age_days / 100))
                memory["adjusted_confidence"] = memory.get("confidence", 0.8) * (1 - time_penalty)
            
            all_results.sort(key=lambda x: x.get("adjusted_confidence", 0), reverse=True)
            top_memories = all_results[:10]
            
            try:
                compressed_context = context_compressor.compress_context(
                    query=request.message,
                    retrieved_memories=top_memories,
                    max_tokens=800,
                    strategy="adaptive"
                )
                
                if compressed_context and len(compressed_context.strip()) > 0:
                    context = f"Based on what you've told me: {compressed_context}"
                else:
                    simple_facts = [m.get("text", "") for m in top_memories[:3]]
                    context = f"Based on what you've told me: {' | '.join(simple_facts)}"
                    
            except Exception as e:
                print(f" Compression failed: {e}")
                simple_facts = [m.get("text", "") for m in top_memories[:3]]
                context = f"Based on what you've told me: {' | '.join(simple_facts)}"
            
            status_update = f"Found {len(all_results)} relevant facts (using top {len(top_memories)})"
    
    posting_rule = "If the user mentions an achievement or interesting update, suggest a social media post using the format: [CREATE_POST: Your draft here]."
    
    try:
        conv_store = get_conversation_store()
        history = conv_store.get_recent_history(numeric_user_id, limit=MAX_HISTORY)
        print(f"Loaded {len(history)} messages from Supabase")
    except Exception as e:
        print(f" Failed to load history from Supabase: {e}")
        history = []
    
    messages = [{"role": "system", "content": f"You are a supportive AI friend. {mood_context}. Context: {context}. {posting_rule}"}]
    
    for msg in history:
        if msg.get('role') != 'system':
            messages.append({"role": msg['role'], "content": msg['content']})
    
    messages.append({"role": "user", "content": request.message})

    try:
        completion = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=messages,
            temperature=0.7
        )
        final_reply = completion.choices[0].message.content        
        post_suggestion = None
        match = re.search(r"\[CREATE_POST:\s*(.*?)\]", final_reply, re.DOTALL | re.IGNORECASE)
        if match:
            post_suggestion = match.group(1).strip()
            final_reply = re.sub(r"\[CREATE_POST:.*?\]", "", final_reply, flags=re.DOTALL | re.IGNORECASE).strip()

        response_time = (time.time() - start_time) * 1000        
        try:
            conv_store.add_exchange(
                user_id=numeric_user_id,
                user_message=request.message,
                assistant_message=final_reply,
                intent=intent,
                intent_confidence=intent_result['confidence'],
                sentiment_score=sentiment['compound'],
                post_suggestion=post_suggestion
            )
            print(f" Saved conversation to Supabase")
        except Exception as e:
            print(f"Failed to save to Supabase: {e}")
        
        return {
            "response": str(final_reply),  
            "intent": str(intent),  
            "intent_confidence": float(intent_result['confidence']),  
            "action_taken": str(status_update) if status_update else None,
            "post_suggestion": str(post_suggestion) if post_suggestion else None,
            "response_time_ms": round(response_time, 2)
        }
    except Exception as e:
        print(f" Error: {e}")
        return {
            "response": "I hit a snag!", 
            "intent": intent,
            "intent_confidence": 0.0,
            "action_taken": None,
            "post_suggestion": None
        }

@app.get("/api/user-data/{user_id}")
async def get_user_data(user_id: str):
    points, _ = q_client.scroll(
        collection_name=COLLECTION_NAME,
        scroll_filter=Filter(must=[FieldCondition(key="user_id", match=MatchValue(value=int(user_id)))]),
        with_payload=True
    )
    return {"memories": [p.payload.get("text") for p in points]}

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