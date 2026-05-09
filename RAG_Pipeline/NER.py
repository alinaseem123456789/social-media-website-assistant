import spacy
from typing import Dict, Optional

# First time: python -m spacy download en_core_web_sm

class SimpleNER:
    def __init__(self):
        """Initialize spaCy NER model"""
        self.nlp = spacy.load("en_core_web_sm")
        
    def extract_entities(self, text: str) -> Dict:
        """Extract all entities from text"""
        doc = self.nlp(text)
        
        entities = {
            "PERSON": [],      # Names
            "GPE": [],         # Cities/countries  
            "ORG": [],         # Organizations
            "MONEY": [],       # Prices
            "DATE": [],        # Dates
            "PRODUCT": [],     # Products
            "EVENT": [],       # Events
            "WORK_OF_ART": [], # Books, songs
            "LAW": [],         # Laws
            "LANGUAGE": [],    # Languages
            "LOC": [],         # Locations
            "NORP": [],        # Nationalities/religions
            "FAC": [],         # Facilities (airports, bridges)
            "TIME": [],        # Times
            "PERCENT": [],     # Percentages
            "CARDINAL": [],    # Numbers
            "ORDINAL": []      # First, second, third
        }
        
        for ent in doc.ents:
            if ent.label_ in entities:
                # Avoid duplicates
                if ent.text not in entities[ent.label_]:
                    entities[ent.label_].append(ent.text)
        
        return entities
    
    def extract_personal_info(self, text: str) -> Optional[Dict]:
        """Extract only personal profile information"""
        entities = self.extract_entities(text)
        
        # Priority mapping for personal info
        mapping = {
            "PERSON": "name",
            "GPE": "location",
            "LOC": "location", 
            "NORP": "nationality",
            "ORG": "workplace",
            "PRODUCT": "likes",
            "WORK_OF_ART": "likes"
        }
        
        # Return first found personal entity
        for ner_type, entity_name in mapping.items():
            if entities.get(ner_type):
                value = entities[ner_type][0]
                
                # Validate before returning
                if self.is_valid_value(entity_name, value):
                    return {
                        "entity": entity_name,
                        "value": value,
                        "confidence": 0.95
                    }
        
        return None
    
    def is_valid_value(self, entity_type: str, value: str) -> bool:
        """Validate extracted value"""
        value_lower = value.lower().strip()
        
        # Reject placeholders
        if value_lower in ["unknown", "none", "null", "not specified", "n/a"]:
            return False
        
        # Reject unrealistic names
        if entity_type == "name":
            fake_names = ["michael jackson", "john doe", "jane doe", "test", "anonymous"]
            if value_lower in fake_names:
                return False
        
        # Check length
        if len(value) < 2 or len(value) > 50:
            return False
        
        return True


# Singleton instance
_ner = None

def get_ner() -> SimpleNER:
    global _ner
    if _ner is None:
        _ner = SimpleNER()
    return _ner