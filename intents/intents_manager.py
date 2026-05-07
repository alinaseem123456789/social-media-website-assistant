"""
Improved Intent Manager - Optimized based on test failures
"""

import re
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# EXPANDED STORE INTENTS - More comprehensive
STORAGE_INTENTS = [
    "remember that", "save this", "my name is", "I live in", "I like",
    "I love", "I hate", "I am", "I work", "my favorite", "I enjoy",
    "don't forget", "keep in mind", "remember i", "fact about me",
    "save my", "store my", "remember my", "note my", "record that",
    "my favourite", "i prefer", "i want to remember", "keep this",
    "add this", "learn this", "memorize this", "i have a",
    "my hobby is", "i am learning", "i currently",
    # Added based on test failures
    "cricket is my", "sport is my", "is my favourite",
    "always loved", "been a fan", "my go-to",
]

RETRIEVAL_INTENTS = [
    "tell me about", "what do you know", "recall", "remember when",
    "what did I say", "do you remember", "can you recall", "my history",
    "what have I told you", "tell me my", "show me", "what do I like",
    "what is my", "what's my", "remind me", "what do you remember",
    "what did I tell", "show me my", "tell me something",
]

DELETION_INTENTS = [
    "forget", "delete memory", "remove", "erase", "clear my memory",
    "forget that", "delete that memory", "remove memory", "forget i said",
    "please forget", "i want you to forget", "you can forget",
]

# Improved regex patterns - More specific to avoid false positives
STORAGE_PATTERNS = [
    r'^save\s+my\s+',           # "save my favorite sport"
    r'^store\s+my\s+',          # "store my name"
    r'^remember\s+my\s+',       # "remember my favorite"
    r'^my\s+\w+\s+is\s+',       # "my name is", "my favorite sport is"
    r'^i\s+like\s+',            # "i like pizza"
    r'^i\s+love\s+',            # "i love cricket"
    r'^i\s+am\s+',              # "i am john"
    r'^i\s+live\s+in\s+',       # "i live in london"
    r'^i\s+prefer\s+',          # "i prefer coffee"
    r'^my\s+favourite\s+\w+\s+is\s+',  # "my favourite sport is cricket"
    r'^\w+\s+is\s+my\s+favourite\s+\w+$',  # "cricket is my favourite sport"
    r'^i(\'ve|\s+have)\s+always\s+loved\s+',  # "I've always loved cricket"
    r'^i(\'m|\s+am)\s+really\s+into\s+',  # "I'm really into photography"
]

# CHAT patterns - New! Explicitly catch greetings and small talk
CHAT_PATTERNS = [
    r'^(hi|hello|hey|howdy|greetings)\s*$',  # Simple greetings
    r'^how\s+(are|is)\s+(you|it\s+going)$',  # "how are you", "how's it going"
    r'^what\'?s\s+up\s*$',                   # "what's up"
    r'^what\'?s\s+new\s*$',                  # "what's new"
    r'^good\s+(morning|afternoon|evening)\s*$',  # "good morning"
    r'^nice\s+to\s+(meet|see)\s+you\s*$',   # "nice to meet you"
    r'^thanks?\s+(you\s+)?(for\s+)?(the\s+)?help?\s*$',  # "thanks for the help"
    r'^thank\s+you\s*$',                    # "thank you"
    r'^you(\'re|\s+are)\s+(helpful|cool|great|awesome)\s*$',  # "you're helpful"
    r'^(that\'?s|thats)\s+(cool|nice|interesting|great)\s*$',  # "that's cool"
    r'^(okay|ok|k|alright|sure)\s*$',       # "okay", "alright"
    r'^(i\s+see|got\s+it|understood)\s*$',  # "i see"
    r'^(awesome|cool|great|nice|wow)\s*$',   # Single positive reactions
]

RETRIEVAL_PATTERNS = [
    r'^what\s+(is|are|was|were)\s+my\s+',   # "what is my name"
    r'^do\s+you\s+remember\s+',            # "do you remember"
    r'^can\s+you\s+recall\s+',             # "can you recall"
    r'^tell\s+me\s+about\s+',              # "tell me about"
    r'^remind\s+me\s+',                    # "remind me"
    r'^what\s+do\s+i\s+like\s*$',          # "what do i like"
    r'^what\s+did\s+i\s+say\s+',           # "what did I say"
    r'^show\s+me\s+',                      # "show me"
    r'^what\'?s\s+in\s+your\s+memory\s*$', # "what's in your memory"
]

DELETION_PATTERNS = [
    r'^forget\s+my\s+',      # "forget my name"
    r'^delete\s+my\s+',      # "delete my memory"
    r'^remove\s+my\s+',      # "remove my information"
    r'^erase\s+',            # "erase that"
    r'^clear\s+my\s+',       # "clear my data"
    r'^please\s+forget\s+',  # "please forget that"
]

class IntentManager:
    def __init__(self, embed_model):
        self.embed_model = embed_model
        self.storage_patterns = STORAGE_PATTERNS
        self.retrieval_patterns = RETRIEVAL_PATTERNS
        self.deletion_patterns = DELETION_PATTERNS
        self.chat_patterns = CHAT_PATTERNS
        
        # Pre-compute vectors for all intents
        print("🔄 Pre-computing intent vectors...")
        self.storage_vectors = embed_model.encode(STORAGE_INTENTS)
        self.retrieval_vectors = embed_model.encode(RETRIEVAL_INTENTS)
        self.deletion_vectors = embed_model.encode(DELETION_INTENTS)
        print(f"✅ Loaded: {len(STORAGE_INTENTS)} storage, {len(RETRIEVAL_INTENTS)} retrieval, {len(DELETION_INTENTS)} deletion intents")
    
    def classify_intent(self, message: str, confidence_threshold: float = 0.35) -> dict:
        msg_lower = message.lower().strip()
        
        # LEVEL 0: Explicit CHAT patterns (HIGHEST priority)
        for pattern in self.chat_patterns:
            if re.match(pattern, msg_lower, re.IGNORECASE):
                return {
                    'intent': 'CHAT',
                    'confidence': 0.98,
                    'matched_pattern': pattern,
                    'details': {'type': 'chat_pattern_match'}
                }
        
        # LEVEL 1: Check storage patterns
        for pattern in self.storage_patterns:
            if re.match(pattern, msg_lower, re.IGNORECASE):
                return {
                    'intent': 'STORE',
                    'confidence': 0.95,
                    'matched_pattern': pattern,
                    'details': {'type': 'regex_match'}
                }
        
        # LEVEL 2: Check retrieval patterns
        for pattern in self.retrieval_patterns:
            if re.match(pattern, msg_lower, re.IGNORECASE):
                return {
                    'intent': 'RETRIEVE',
                    'confidence': 0.95,
                    'matched_pattern': pattern,
                    'details': {'type': 'regex_match'}
                }
        
        # LEVEL 3: Check deletion patterns
        for pattern in self.deletion_patterns:
            if re.match(pattern, msg_lower, re.IGNORECASE):
                return {
                    'intent': 'DELETE',
                    'confidence': 0.95,
                    'matched_pattern': pattern,
                    'details': {'type': 'regex_match'}
                }
        
        # LEVEL 4: Semantic similarity
        similarity_result = self._check_semantic_similarity(message)
        
        # Boost CHAT confidence for short messages
        if len(msg_lower.split()) <= 2 and similarity_result['intent'] == 'CHAT':
            similarity_result['confidence'] = min(0.85, similarity_result['confidence'] + 0.2)
            return similarity_result
        
        # Boost STORE for preference statements
        preference_words = ['love', 'like', 'enjoy', 'prefer', 'favourite', 'favorite']
        if any(word in msg_lower for word in preference_words):
            if similarity_result['intent'] == 'CHAT' or similarity_result['confidence'] < 0.5:
                return {
                    'intent': 'STORE',
                    'confidence': 0.65,
                    'matched_pattern': None,
                    'details': {'type': 'preference_boost'}
                }
        
        if similarity_result['confidence'] >= confidence_threshold:
            return similarity_result
        
        # Default fallback - check if it's likely CHAT
        if len(msg_lower.split()) <= 2:
            return {
                'intent': 'CHAT',
                'confidence': 0.6,
                'matched_pattern': None,
                'details': {'type': 'short_message_fallback'}
            }
        
        return {
            'intent': 'CHAT',
            'confidence': 0.4,
            'matched_pattern': None,
            'details': {'reason': 'No intent matched'}
        }
    
    def _check_patterns(self, msg_lower: str) -> dict or None:
        """Check regex patterns for explicit intent matches"""
        
        for pattern in self.storage_patterns:
            if re.match(pattern, msg_lower):
                return {
                    'intent': 'STORE',
                    'confidence': 0.95,
                    'matched_pattern': pattern,
                    'details': {'type': 'regex_match'}
                }
        
        for pattern in self.retrieval_patterns:
            if re.match(pattern, msg_lower):
                return {
                    'intent': 'RETRIEVE',
                    'confidence': 0.95,
                    'matched_pattern': pattern,
                    'details': {'type': 'regex_match'}
                }
        
        for pattern in self.deletion_patterns:
            if re.match(pattern, msg_lower):
                return {
                    'intent': 'DELETE',
                    'confidence': 0.95,
                    'matched_pattern': pattern,
                    'details': {'type': 'regex_match'}
                }
        
        return None
    
    def _check_semantic_similarity(self, message: str) -> dict:
        msg_vector = self.embed_model.encode([message])
        
        storage_sim = float(max(cosine_similarity(msg_vector, self.storage_vectors)[0]))
        retrieval_sim = float(max(cosine_similarity(msg_vector, self.retrieval_vectors)[0]))
        deletion_sim = float(max(cosine_similarity(msg_vector, self.deletion_vectors)[0]))
        
        # Add bias to reduce false STORE for greetings
        msg_lower = message.lower()
        if any(word in msg_lower for word in ['hello', 'hi', 'hey', 'how are', 'what\'s up']):
            storage_sim *= 0.3  # Reduce STORE similarity for greetings
        
        similarities = {
            'STORE': storage_sim,
            'RETRIEVE': retrieval_sim,
            'DELETE': deletion_sim,
            'CHAT': 0.3  # Baseline for CHAT
        }
        
        best_intent = max(similarities, key=similarities.get)
        best_confidence = similarities[best_intent]
        
        return {
            'intent': best_intent,
            'confidence': best_confidence,
            'matched_pattern': None,
            'details': {'type': 'semantic', 'similarities': similarities}
        } 