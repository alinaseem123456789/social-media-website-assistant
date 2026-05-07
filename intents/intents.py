# intents.py - COMPLETE VERSION (NO embeddings needed!)
import re
from typing import Dict, Any

# ============================================================
# CHAT patterns - Highest priority greetings and small talk
# ============================================================
CHAT_PATTERNS = [
    r'^(hi|hello|hey|howdy|greetings)\s*$',
    r'^how\s+(are|is)\s+(you|it\s+going)$',
    r'^what\'?s\s+up\s*$',
    r'^what\'?s\s+new\s*$',
    r'^good\s+(morning|afternoon|evening)\s*$',
    r'^nice\s+to\s+(meet|see)\s+you\s*$',
    r'^thanks?\s*(you\s+)?(for\s+)?(the\s+)?help?\s*$',
    r'^thank\s+you\s*$',
    r'^you(\'re|\s+are)\s+(helpful|cool|great|awesome)\s*$',
    r'^(that\'?s|thats)\s+(cool|nice|interesting|great)\s*$',
    r'^(okay|ok|k|alright|sure)\s*$',
    r'^(i\s+see|got\s+it|understood)\s*$',
    r'^(awesome|cool|great|nice|wow)\s*$',
]

# ============================================================
# STORE patterns - Personal information and achievements
# ============================================================
STORAGE_PATTERNS = [
    # Basic personal info
    r'^save\s+my\s+',
    r'^store\s+my\s+',
    r'^remember\s+my\s+',
    r'^my\s+\w+\s+is\s+',           # "my name is", "my favorite sport is"
    r'^i\s+like\s+',                 # "i like pizza"
    r'^i\s+love\s+',                 # "i love cricket"
    r'^i\s+am\s+',                   # "i am john"
    r'^i\s+live\s+in\s+',            # "i live in london"
    r'^i\s+prefer\s+',               # "i prefer coffee"
    r'^my\s+favourite\s+\w+\s+is\s+', # "my favourite sport is cricket"
    
    # Achievements and milestones (CRITICAL for post generation!)
    r'^i\s+(just|finally|successfully)\s+(finished|completed|graduated|passed)',
    r'^i\s+(got|received|earned|achieved)\s+(a|my|the)',
    r'^i\s+(won|won\s+the|came\s+first|placed)',
    r'^i\s+(started|began|launched)\s+(a|my|the)',
    r'^i\s+(learned|mastered|studied)\s+',
    r'^i\s+(ran|walked|cycled|swam)\s+',
    r'^i\s+(lost|gained)\s+\d+\s+(pounds|kgs|lbs)',
    r'^i\s+got\s+a\s+(promotion|raise|new\s+job)',
    r'^i\s+just\s+(bought|purchased|got)\s+',
    r'^i\'?ve\s+always\s+loved\s+',
    r'^i\'?ve\s+been\s+(a|an)\s+(fan|enthusiast)\s+of',
    r'^cricket\s+is\s+my\s+favourite',
    r'^\w+\s+is\s+my\s+favourite\s+\w+$',  # "cricket is my favourite sport"
    r'^i(\'m|\s+am)\s+really\s+into\s+',    # "I'm really into photography"
]

# ============================================================
# RETRIEVE patterns - Asking for stored information
# ============================================================
RETRIEVAL_PATTERNS = [
    r'^what\s+(is|are|was|were)\s+my\s+',   # "what is my name"
    r'^do\s+you\s+remember\s+',             # "do you remember"
    r'^can\s+you\s+recall\s+',              # "can you recall"
    r'^tell\s+me\s+about\s+',               # "tell me about"
    r'^remind\s+me\s+',                     # "remind me"
    r'^what\s+do\s+i\s+like\s*$',           # "what do i like"
    r'^what\s+did\s+i\s+say\s+',            # "what did I say"
    r'^show\s+me\s+',                       # "show me"
    r'^what\'?s\s+in\s+your\s+memory\s*$',  # "what's in your memory"
    r'^what\s+do\s+you\s+know\s+about\s+',  # "what do you know about"
    r'^tell\s+me\s+something\s+(about|i)',  # "tell me something about"
]

# ============================================================
# DELETE patterns - Removing stored information
# ============================================================
DELETION_PATTERNS = [
    r'^forget\s+my\s+',      # "forget my name"
    r'^delete\s+my\s+',      # "delete my memory"
    r'^remove\s+my\s+',      # "remove my information"
    r'^erase\s+',            # "erase that"
    r'^clear\s+my\s+',       # "clear my data"
    r'^please\s+forget\s+',  # "please forget that"
]


class IntentManager:
    """
    Intent classifier using regex patterns ONLY (no embeddings needed)
    Works within 512MB memory limit!
    """
    
    def __init__(self, embed_model=None):
        # embed_model is optional - we don't use it
        self.chat_patterns = CHAT_PATTERNS
        self.storage_patterns = STORAGE_PATTERNS
        self.retrieval_patterns = RETRIEVAL_PATTERNS
        self.deletion_patterns = DELETION_PATTERNS
        self.default_intent = "CHAT"
        
        print("✅ IntentManager initialized with regex patterns (no embeddings)")
    
    def classify_intent(self, message: str) -> Dict[str, Any]:
        """Classify intent using regex pattern matching"""
        msg_lower = message.lower().strip()
        
        # LEVEL 0: Explicit CHAT patterns (HIGHEST priority)
        for pattern in self.chat_patterns:
            if re.match(pattern, msg_lower, re.IGNORECASE):
                return {
                    'intent': 'CHAT',
                    'confidence': 0.98,
                    'details': {'matched_pattern': pattern, 'type': 'chat'}
                }
        
        # LEVEL 1: Check STORE patterns (including achievements!)
        for pattern in self.storage_patterns:
            if re.match(pattern, msg_lower, re.IGNORECASE):
                return {
                    'intent': 'STORE',
                    'confidence': 0.95,
                    'details': {'matched_pattern': pattern, 'type': 'storage'}
                }
        
        # LEVEL 2: Check RETRIEVAL patterns
        for pattern in self.retrieval_patterns:
            if re.match(pattern, msg_lower, re.IGNORECASE):
                return {
                    'intent': 'RETRIEVE',
                    'confidence': 0.95,
                    'details': {'matched_pattern': pattern, 'type': 'retrieval'}
                }
        
        # LEVEL 3: Check DELETION patterns
        for pattern in self.deletion_patterns:
            if re.match(pattern, msg_lower, re.IGNORECASE):
                return {
                    'intent': 'DELETE',
                    'confidence': 0.95,
                    'details': {'matched_pattern': pattern, 'type': 'deletion'}
                }
        
        # LEVEL 4: Check for achievement keywords (even without "I" prefix)
        achievement_keywords = [
            'graduated', 'completed', 'finished', 'passed', 'won', 'got promoted',
            'new job', 'started', 'learned', 'mastered', 'achieved', 'earned',
            'ran', 'marathon', 'lost weight', 'fitness goal', 'milestone'
        ]
        for keyword in achievement_keywords:
            if keyword in msg_lower:
                return {
                    'intent': 'STORE',
                    'confidence': 0.85,
                    'details': {'matched_keyword': keyword, 'type': 'achievement'}
                }
        
        # LEVEL 5: Check for preference keywords
        preference_keywords = ['love', 'like', 'enjoy', 'prefer', 'favourite', 'favorite']
        for keyword in preference_keywords:
            if keyword in msg_lower:
                return {
                    'intent': 'STORE',
                    'confidence': 0.80,
                    'details': {'matched_keyword': keyword, 'type': 'preference'}
                }
        
        # LEVEL 6: Check for question patterns (RETRIEVE)
        if msg_lower.startswith(('what', 'where', 'who', 'when', 'how', 'why', 'can you', 'do you')):
            return {
                'intent': 'RETRIEVE',
                'confidence': 0.70,
                'details': {'type': 'question_pattern'}
            }
        
        # Default fallback for very short messages
        if len(msg_lower.split()) <= 3:
            return {
                'intent': 'CHAT',
                'confidence': 0.60,
                'details': {'type': 'short_message'}
            }
        
        # Final fallback
        return {
            'intent': self.default_intent,
            'confidence': 0.50,
            'details': {'type': 'default'}
        }

if __name__ == "__main__":
    # Test the intent manager
    intent_manager = IntentManager()
    
    test_phrases = [
        ("I just graduated from university!", "STORE"),
        ("I love cricket", "STORE"),
        ("I got promoted at work", "STORE"),
        ("I ran my first 5k marathon", "STORE"),
        ("What's my name?", "RETRIEVE"),
        ("Do you remember my favorite sport?", "RETRIEVE"),
        ("Forget my name", "DELETE"),
        ("Hello", "CHAT"),
        ("How are you?", "CHAT"),
        ("Cricket is my favourite sport", "STORE"),
        ("I finished learning React", "STORE"),
    ]
    
    print("\n🧪 Testing IntentManager:\n")
    for phrase, expected in test_phrases:
        result = intent_manager.classify_intent(phrase)
        status = "✅" if result['intent'] == expected else "❌"
        print(f"{status} '{phrase}' → {result['intent']} (expected: {expected})")
        print(f"   confidence: {result['confidence']}, details: {result['details']}\n")