"""
Intent Management System for Chatbot
This package handles intent classification for STORE, RETRIEVE, DELETE, and CHAT operations
"""

# from .storage_intents import STORAGE_INTENTS, STORAGE_PATTERNS, STORAGE_KEYWORDS
# from .retrieval_intents import RETRIEVAL_INTENTS, RETRIEVAL_PATTERNS, RETRIEVAL_KEYWORDS
# from .deletion_intents import DELETION_INTENTS, DELETION_PATTERNS, DELETION_KEYWORDS
from .intents import IntentManager

__all__ = [
    'STORAGE_INTENTS',
    'STORAGE_PATTERNS', 
    'STORAGE_KEYWORDS',
    'RETRIEVAL_INTENTS',
    'RETRIEVAL_PATTERNS',
    'RETRIEVAL_KEYWORDS',
    'DELETION_INTENTS',
    'DELETION_PATTERNS',
    'DELETION_KEYWORDS',
    'IntentManager'
]

print("🎯 Intent package loaded successfully!")