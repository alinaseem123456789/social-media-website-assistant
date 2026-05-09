# intent_manager.py - Simplified, no context storage
import re
import json
from typing import Dict, Optional, Tuple
from groq import Groq
import os

class IntentManager:
    def __init__(self, embed_model=None):
        print("Initializing Groq-based Intent Manager...")
        
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            print("⚠️ GROQ_API_KEY not found, using keyword fallback")
            self.groq_client = None
        else:
            self.groq_client = Groq(api_key=api_key)
            print("✅ Groq client initialized")
        
        # Keyword patterns (fallback only)
        self.keyword_patterns = {
            "FRIEND_SUGGESTION": [
                r"suggest(?: me)? friends?", r"find(?: me)? friends?",
                r"recommend(?: me)? friends?", r"friend suggestions?",
                r"suggest me some friends", r"suggest me friends",
                r"recommend people", r"meet new people",
                r"connect with people", r"people to connect",
            ],
            "CREATE_POST": [
                r"create (?:a )?post", r"make (?:a )?post", r"write (?:a )?post",
                r"share (?:a )?post", r"post about", r"help me write",
                r"share my (?:achievement|milestone)", r"compose a post",
            ],
            "ENGAGEMENT": [
                r"what (?:should|can) I like", r"upcoming birthdays?",
                r"who hasn't posted", r"engage with", r"posts? to like",
            ],
            "DELETE": [r"delete (?:my )?memory", r"forget", r"remove"],
            "UPDATE": [r"update my", r"change my", r"actually"],
            "STORE": [r"remember", r"save that", r"my name is", r"i (?:love|like)"],
            "RETRIEVE": [r"tell me about", r"what do you know", r"recall", r"tell me about myself"],
        }
        
        # Compile patterns for faster matching
        self.compiled_patterns = {}
        for intent, patterns in self.keyword_patterns.items():
            self.compiled_patterns[intent] = re.compile("|".join(patterns), re.IGNORECASE)
        
        print("✅ Intent Manager ready (stateless)")
    
    def classify_intent(self, message: str, user_id: Optional[int] = None) -> Dict:
        """
        Classify intent - PURELY STATELESS
        No context storage, no pending workflow tracking
        """
        # Try Groq API first (if available)
        if self.groq_client:
            try:
                intent, confidence = self._classify_with_groq(message)
                if confidence > 0.6:
                    print(f"🤖 Groq: {intent} (confidence: {confidence:.2f})")
                    details = self._extract_details(message, intent)
                    return {"intent": intent, "confidence": confidence, "details": details}
            except Exception as e:
                print(f"⚠️ Groq API error: {e}, falling back to keywords")
        
        # Fallback to keyword matching
        intent, confidence = self._classify_with_keywords(message)
        details = self._extract_details(message, intent)
        return {"intent": intent, "confidence": confidence, "details": details}
    
    def _classify_with_groq(self, message: str) -> Tuple[str, float]:
        """Use Groq for intent classification"""
        prompt = f"""Classify this user message into ONE of these intents:
- CREATE_POST: User wants to create/write/share a post
- FRIEND_SUGGESTION: User wants friend suggestions or to connect with people  
- ENGAGEMENT: User wants to like/comment/interact with content
- DELETE: User wants to delete/forget stored information
- UPDATE: User wants to update/change stored information
- STORE: User wants to save/remember new information
- RETRIEVE: User wants to recall what they've told me
- CHAT: Normal conversation

Message: "{message}"

Return ONLY the intent name, nothing else."""

        response = self.groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",  
            messages=[{"role": "user", "content": prompt}],
            temperature=0, 
            max_tokens=20   
        )
        
        intent = response.choices[0].message.content.strip().upper()
        
        valid_intents = ["CREATE_POST", "FRIEND_SUGGESTION", "ENGAGEMENT", "DELETE", "UPDATE", "STORE", "RETRIEVE", "CHAT"]
        if intent not in valid_intents:
            intent = "CHAT"
        
        return intent, 0.9
    
    def _classify_with_keywords(self, message: str) -> Tuple[str, float]:
        """Fallback keyword-based classification"""
        message_lower = message.lower()
        
        # Direct matches for common phrases
        if "suggest me friends" in message_lower or "suggest me some friends" in message_lower:
            return "FRIEND_SUGGESTION", 0.95
        
        if "help me write" in message_lower or "create a post" in message_lower:
            return "CREATE_POST", 0.85
        
        # Check all patterns
        for intent, pattern in self.compiled_patterns.items():
            if pattern.search(message_lower):
                return intent, 0.7
        
        return "CHAT", 0.5
    
    def _extract_details(self, message: str, intent: str) -> Dict:
        """Extract additional details from message"""
        details = {}
        message_lower = message.lower()
        
        if intent == "FRIEND_SUGGESTION":
            interests = re.findall(r'(?:like|enjoy|love) (\w+)', message_lower)
            if interests:
                details["interests"] = interests
        
        elif intent == "CREATE_POST":
            match = re.search(r'about (.+?)(?:\.|\?|$)', message)
            if match:
                details["topic"] = match.group(1).strip()
        
        return details