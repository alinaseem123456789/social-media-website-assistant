import os
from typing import List, Dict, Optional
from supabase import create_client, Client
from dotenv import load_dotenv
from datetime import datetime, timedelta

load_dotenv()

class SupabaseConversationStore:
    def __init__(self):
        self.url = os.getenv("SUPABASE_URL")
        self.key = os.getenv("SUPABASE_KEY")
        
        if not self.url or not self.key:
            raise ValueError("SUPABASE_URL and SUPABASE_KEY must be set")
        
        self.supabase: Client = create_client(self.url, self.key)
    
    def add_message(
        self, 
        user_id: int, 
        role: str, 
        content: str, 
        intent: Optional[str] = None,
        intent_confidence: Optional[float] = None,
        sentiment_score: Optional[float] = None,
        has_post_suggestion: bool = False
    ) -> Dict:
        """Add a single message to conversation history"""
        data = {
            "user_id": user_id,
            "role": role,
            "content": content,
            "intent": intent,
            "intent_confidence": intent_confidence,
            "sentiment_score": sentiment_score,
            "has_post_suggestion": has_post_suggestion
        }
        
        result = self.supabase.table("conversations").insert(data).execute()
        return result.data[0] if result.data else None
    
    def add_exchange(
        self, 
        user_id: int, 
        user_message: str, 
        assistant_message: str,
        intent: Optional[str] = None,
        intent_confidence: Optional[float] = None,
        sentiment_score: Optional[float] = None,
        post_suggestion: Optional[str] = None
    ) -> tuple:
        """Add both user and assistant messages in one call"""
        # Add user message
        user_msg = self.add_message(
            user_id=user_id,
            role="user",
            content=user_message,
            intent=intent,
            intent_confidence=intent_confidence,
            sentiment_score=sentiment_score
        )
        
        # Add assistant message
        assistant_msg = self.add_message(
            user_id=user_id,
            role="assistant",
            content=assistant_message,
            intent=intent,
            has_post_suggestion=post_suggestion is not None
        )
        
        return user_msg, assistant_msg
    
    def get_recent_history(
        self, 
        user_id: int, 
        limit: int = 10,
        include_system: bool = False
    ) -> List[Dict]:
        """Get recent conversation history"""
        query = self.supabase.table("conversations")\
            .select("role, content, intent, created_at")\
            .eq("user_id", user_id)\
            .order("created_at", desc=True)\
            .limit(limit)
        
        if not include_system:
            query = query.neq("role", "system")
        
        result = query.execute()
        
        # Reverse to chronological order (oldest first)
        messages = result.data[::-1] if result.data else []
        return messages
    
    def get_history_by_time_range(
        self, 
        user_id: int, 
        hours_back: int = 24
    ) -> List[Dict]:
        """Get conversations from last X hours"""
        cutoff = datetime.now() - timedelta(hours=hours_back)
        
        result = self.supabase.table("conversations")\
            .select("*")\
            .eq("user_id", user_id)\
            .gte("created_at", cutoff.isoformat())\
            .order("created_at", asc=True)\
            .execute()
        
        return result.data if result.data else []
    
    def search_conversations(
        self, 
        user_id: int, 
        search_term: str,
        limit: int = 5
    ) -> List[Dict]:
        """Search conversation content (simple text search)"""
        # Supabase full-text search
        result = self.supabase.table("conversations")\
            .select("role, content, created_at")\
            .eq("user_id", user_id)\
            .text_search("content", search_term)\
            .order("created_at", desc=True)\
            .limit(limit)\
            .execute()
        
        return result.data if result.data else []
    
    def get_conversation_stats(self, user_id: int) -> Dict:
        """Get statistics for a user"""
        # Total messages
        total = self.supabase.table("conversations")\
            .select("id", count="exact")\
            .eq("user_id", user_id)\
            .execute()
        
        # Messages by intent
        intents = self.supabase.table("conversations")\
            .select("intent, count")\
            .eq("user_id", user_id)\
            .group_by("intent")\
            .execute()
        
        return {
            "total_messages": total.count if total.count else 0,
            "intent_distribution": intents.data if intents.data else []
        }
    
    def delete_old_conversations(self, user_id: int, days_old: int = 90):
        """Delete conversations older than X days (GDPR/compliance)"""
        cutoff = datetime.now() - timedelta(days=days_old)
        
        result = self.supabase.table("conversations")\
            .delete()\
            .eq("user_id", user_id)\
            .lt("created_at", cutoff.isoformat())\
            .execute()
        
        return len(result.data) if result.data else 0
    
    def get_conversation_thread(
        self, 
        user_id: int, 
        message_id: int, 
        context_before: int = 5,
        context_after: int = 5
    ) -> List[Dict]:
        """Get conversation around a specific message"""
        # Get the target message first
        target = self.supabase.table("conversations")\
            .select("created_at")\
            .eq("id", message_id)\
            .eq("user_id", user_id)\
            .execute()
        
        if not target.data:
            return []
        
        timestamp = target.data[0]["created_at"]
        
        # Get messages around it
        result = self.supabase.table("conversations")\
            .select("*")\
            .eq("user_id", user_id)\
            .filter("created_at", "gte", timestamp - timedelta(minutes=30))\
            .filter("created_at", "lte", timestamp + timedelta(minutes=30))\
            .order("created_at", asc=True)\
            .limit(context_before + context_after + 1)\
            .execute()
        
        return result.data if result.data else []

_supabase_client = None

def get_supabase_client():
    """Get Supabase client instance (for posts and other operations)"""
    global _supabase_client
    if _supabase_client is None:
        url = os.getenv("SUPABASE_URL")
        key = os.getenv("SUPABASE_KEY")
        
        if not url or not key:
            raise ValueError("SUPABASE_URL and SUPABASE_KEY must be set")
        
        _supabase_client = create_client(url, key)
    
    return _supabase_client
# Singleton instance
_conversation_store = None

def get_conversation_store() -> SupabaseConversationStore:
    global _conversation_store
    if _conversation_store is None:
        _conversation_store = SupabaseConversationStore()
    return _conversation_store