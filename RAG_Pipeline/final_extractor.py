# final_extractor.py - COMPREHENSIVE SOCIAL MEDIA ENTITY EXTRACTOR
import json
import re
from typing import List, Dict, Optional, Tuple, Any
from datetime import datetime
from enum import Enum

class EntityType(Enum):
    """Comprehensive entity types for social media profiles"""
    # Basic Profile
    NAME = "name"
    AGE = "age"
    GENDER = "gender"
    BIRTHDAY = "birthday"
    
    # Location & Origin
    LOCATION = "location"
    HOMETOWN = "hometown"
    NATIONALITY = "nationality"
    LANGUAGE = "language"
    
    # Professional
    JOB = "job"
    WORKPLACE = "workplace"
    EDUCATION = "education"
    SKILLS = "skills"
    
    # Interests & Preferences
    LIKES = "likes"
    DISLIKES = "dislikes"
    INTEREST = "interest"
    HOBBIES = "hobbies"
    FAVORITE = "favorite"
    
    # Media & Entertainment
    MUSIC = "music"
    MOVIES = "movies"
    BOOKS = "books"
    GAMES = "games"
    FOOD = "food"
    SPORTS = "sports"
    
    # Social & Relationships
    RELATIONSHIP = "relationship_status"
    FAMILY = "family"
    PETS = "pets"
    
    # Personality & Values
    PERSONALITY = "personality_trait"
    VALUES = "values"
    GOALS = "goals"
    DREAMS = "dreams"
    
    # Technical
    DEVICE = "device"
    SOFTWARE = "software"
    PROGRAMMING = "programming_language"
    
    # Generic fallback with smart categorization
    CUSTOM = "custom"

class ExtractionSource(Enum):
    NER = "ner"
    PATTERN = "pattern"
    LLM = "llm"
    INFERRED = "inferred"

class FinalEntityExtractor:
    """
    Smart entity extractor that:
    - Categorizes entities intelligently
    - Learns from patterns
    - Handles out-of-schema entities gracefully
    """
    
    def __init__(self, ner_model, groq_client):
        self.ner = ner_model
        self.groq = groq_client
        
        # All allowed entity types
        self.allowed_entity_types = {e.value for e in EntityType}
        
        # NER label mappings
        self.ner_mappings = {
            "name": ["PERSON"],
            "location": ["GPE", "LOC"],
            "hometown": ["GPE", "LOC"],
            "nationality": ["NORP"],
            "workplace": ["ORG"],
            "language": ["LANGUAGE"],
            "age": ["CARDINAL", "DATE"],
        }
        
        # Keyword-based entity detection
        self.entity_keywords = {
            "music": ["music", "song", "band", "singer", "artist", "album", "genre"],
            "movies": ["movie", "film", "actor", "actress", "director", "cinema", "hollywood", "bollywood"],
            "books": ["book", "author", "novel", "reading", "writer", "literature"],
            "games": ["game", "gaming", "video game", "play", "console", "pc game", "mobile game"],
            "food": ["food", "cuisine", "dish", "recipe", "cooking", "eat", "restaurant"],
            "sports": ["sport", "game", "team", "player", "football", "cricket", "soccer", "basketball"],
            "hobbies": ["hobby", "hobbies", "passion", "free time", "leisure"],
            "skills": ["skill", "good at", "proficient", "expert", "know how to"],
            "education": ["study", "major", "degree", "university", "college", "school", "graduate"],
            "pets": ["pet", "dog", "cat", "bird", "fish", "animal"],
            "relationship": ["relationship", "married", "single", "engaged", "dating"],
            "goals": ["goal", "aim", "dream", "want to", "plan to", "aspire"],
            "values": ["believe", "value", "important to me", "care about"],
            "programming": ["code", "programming", "developer", "python", "javascript", "react", "node"],
        }
        
        # Context patterns for different entity types
        self.context_patterns = {
            "preference": [
                r'\b(?:like|love|enjoy|adore|prefer|into|dig)\b',
                r'\b(?:my favorite|favourite)\s+(?:is|are)\b',
                r'\b(?:i\s+am\s+obsessed\s+with)\b',
                r'\b(?:i\s+could\s+(?:listen|watch|eat)\s+)\b',
            ],
            "identity": [
                r'\b(?:my name is|i am|i\'m called|call me)\b',
                r'\b(?:i\s+live\s+in|i\s+am\s+from|my hometown)\b',
                r'\b(?:i\s+work\s+as|i\s+am\s+a\s+|my job is)\b',
                r'\b(?:i\s+study|my major is|i\s+go to school at)\b',
            ],
            "ownership": [
                r'\b(?:i have|i own|i use|my)\b',
                r'\b(?:i\s+play|i\s+read|i\s+watch|i\s+listen)\b',
            ],
            "future": [
                r'\b(?:i want to|i plan to|i hope to|i dream of|my goal is)\b',
            ],
            "negation": [
                r'\b(?:i don\'t like|i hate|i dislike|i can\'t stand)\b',
            ]
        }
        
        # Reject worthless values
        self.reject_values = {
            "her", "him", "it", "them", "they", "she", "he", "we", "you",
            "me", "us", "his", "hers", "theirs", "ours", "yours",
            "alias", "name", "thing", "something", "someone", "somebody",
            "nothing", "everything", "anything", "whatever", "whoever",
            "stuff", "things", "somewhere", "a", "an", "the", "this", "that",
        }
        
        # Skip trivial messages
        self.skip_patterns = [
            r'^(hi|hello|hey|thanks|thank you|ok|okay|yes|no)$',
            r'^(good|bad|fine|great|awesome|terrible)$',
            r'i (am|feel) (good|bad|happy|sad|ok|fine)',
        ]
        
    def extract_all_entities(self, text: str, user_id: int = None) -> List[Dict]:
        """
        Extract ALL valuable entities - smart categorization
        """
        
        # Skip trivial messages
        if self._is_trivial_message(text):
            return []
        
        # Analyze context
        context = self._analyze_context(text)
        print(f"📊 Context: {context['primary']} (confidence: {context['confidence']})")
        
        # Extract entities using multiple strategies
        extracted = []
        seen_values = set()
        
        # 1. Extract from patterns (highest confidence)
        pattern_entities = self._extract_pattern_entities(text, context)
        for entity in pattern_entities:
            key = f"{entity['entity']}:{entity['value'].lower()}"
            if key not in seen_values and self._is_valuable_entity(entity):
                extracted.append(entity)
                seen_values.add(key)
        
        # 2. Extract from NER
        raw_entities = self.ner.extract_entities(text)
        ner_entities = self._extract_ner_entities(text, raw_entities, context)
        for entity in ner_entities:
            key = f"{entity['entity']}:{entity['value'].lower()}"
            if key not in seen_values and self._is_valuable_entity(entity):
                extracted.append(entity)
                seen_values.add(key)
        
        # 3. Extract keyword-based entities
        keyword_entities = self._extract_keyword_entities(text, context)
        for entity in keyword_entities:
            key = f"{entity['entity']}:{entity['value'].lower()}"
            if key not in seen_values and self._is_valuable_entity(entity):
                extracted.append(entity)
                seen_values.add(key)
        
        # 4. Extract preferences
        preference_entities = self._extract_preference_entities(text, context)
        for entity in preference_entities:
            key = f"{entity['entity']}:{entity['value'].lower()}"
            if key not in seen_values and self._is_valuable_entity(entity):
                extracted.append(entity)
                seen_values.add(key)
        
        # 5. Extract future goals/dreams
        future_entities = self._extract_future_entities(text)
        for entity in future_entities:
            key = f"{entity['entity']}:{entity['value'].lower()}"
            if key not in seen_values and self._is_valuable_entity(entity):
                extracted.append(entity)
                seen_values.add(key)
        
        # 6. Extract from NER categories
        for entity_type, ner_labels in self.ner_mappings.items():
            for ner_label in ner_labels:
                if ner_label in raw_entities and raw_entities[ner_label]:
                    for value in raw_entities[ner_label]:
                        key = f"{entity_type}:{value.lower()}"
                        if key in seen_values:
                            continue
                        entity = {
                            "entity": entity_type,
                            "value": value,
                            "confidence": 0.85,
                            "source": ExtractionSource.NER.value,
                            "context": context["primary"],
                            "original_text": text
                        }
                        if self._is_valuable_entity(entity):
                            extracted.append(entity)
                            seen_values.add(key)
        
        # 7. LLM fallback for complex or new entity types
        if not extracted and not self._is_likely_chitchat(text):
            llm_entities = self._extract_with_llm(text)
            for entity in llm_entities:
                if self._is_valuable_entity(entity):
                    key = f"{entity['entity']}:{entity['value'].lower()}"
                    if key not in seen_values:
                        extracted.append(entity)
                        seen_values.add(key)
        
        # 8. Smart categorization for remaining valuable info
        if extracted:
            extracted = self._smart_categorize(extracted, text)
        
        # Remove duplicates and sort
        extracted = self._merge_related_entities(extracted)
        extracted.sort(key=lambda x: x["confidence"], reverse=True)
        
        return extracted
    
    def _extract_pattern_entities(self, text: str, context: Dict) -> List[Dict]:
        """Extract using regex patterns for common patterns"""
        entities = []
        
        # Job patterns
        job_patterns = [
            r'i am a\s+([A-Za-z\s]+?)(?:\s+and|\s+\.|\s*$)',
            r'i\'m a\s+([A-Za-z\s]+?)(?:\s+and|\s+\.|\s*$)',
            r'i work as\s+([A-Za-z\s]+?)(?:\s+and|\s+\.|\s*$)',
            r'my job is\s+([A-Za-z\s]+?)(?:\s+and|\s+\.|\s*$)',
        ]
        
        for pattern in job_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                value = match.strip()
                if len(value) > 2 and value.lower() not in self.reject_values:
                    entities.append({
                        "entity": "job",
                        "value": value,
                        "confidence": 0.92,
                        "source": ExtractionSource.PATTERN.value,
                        "context": context["primary"],
                        "original_text": text
                    })
        
        # Name patterns
        name_patterns = [
            r'my name is\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
            r'call me\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
        ]
        
        for pattern in name_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                value = match.strip()
                if value and value.lower() not in self.reject_values:
                    entities.append({
                        "entity": "name",
                        "value": value,
                        "confidence": 0.95,
                        "source": ExtractionSource.PATTERN.value,
                        "context": context["primary"],
                        "original_text": text
                    })
        
        # Age patterns
        age_patterns = [
            r'i am\s+(\d+)\s+years? old',
            r'i\'m\s+(\d+)\s+years? old',
            r'age\s+(\d+)',
        ]
        
        for pattern in age_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                age = match if isinstance(match, str) else str(match[0])
                if age.isdigit() and 1 <= int(age) <= 120:
                    entities.append({
                        "entity": "age",
                        "value": age,
                        "confidence": 0.95,
                        "source": ExtractionSource.PATTERN.value,
                        "context": context["primary"],
                        "original_text": text
                    })
        
        # Location patterns
        location_patterns = [
            r'i live in\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
            r'i am from\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
            r'my hometown is\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
        ]
        
        for pattern in location_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                value = match.strip()
                if value and value.lower() not in self.reject_values:
                    entities.append({
                        "entity": "location",
                        "value": value,
                        "confidence": 0.92,
                        "source": ExtractionSource.PATTERN.value,
                        "context": context["primary"],
                        "original_text": text
                    })
        
        return entities
    
    def _extract_keyword_entities(self, text: str, context: Dict) -> List[Dict]:
        """Extract using keyword mapping for different categories"""
        entities = []
        text_lower = text.lower()
        
        for entity_type, keywords in self.entity_keywords.items():
            for keyword in keywords:
                # Look for patterns like "I like [something]" or "My favorite [category] is [value]"
                patterns = [
                    rf'{keyword}\s+(?:is|are|:)\s+([A-Za-z0-9\s]+?)(?:\.|,|and|$)',
                    rf'(?:like|love|enjoy)\s+([A-Za-z0-9\s]+?)\s+{keyword}',
                    rf'my favorite\s+{keyword}\s+(?:is|are)\s+([A-Za-z0-9\s]+?)(?:\.|,|$)',
                ]
                
                for pattern in patterns:
                    matches = re.findall(pattern, text_lower, re.IGNORECASE)
                    for match in matches:
                        value = match.strip()
                        if len(value) > 2 and value not in self.reject_values:
                            # Determine if it's a like or dislike based on context
                            is_negative = any(neg in text_lower for neg in ["don't like", "hate", "dislike"])
                            actual_entity_type = "dislikes" if is_negative else entity_type
                            
                            entities.append({
                                "entity": actual_entity_type,
                                "value": value.title() if value[0].islower() else value,
                                "confidence": 0.85,
                                "source": ExtractionSource.PATTERN.value,
                                "context": context["primary"],
                                "original_text": text
                            })
                            break
                    if entities and entities[-1]["entity"] == entity_type:
                        break
        
        return entities
    
    def _extract_preference_entities(self, text: str, context: Dict) -> List[Dict]:
        """Extract explicit preferences"""
        entities = []
        text_lower = text.lower()
        
        # Detect if it's a dislike
        is_dislike = any(neg in text_lower for neg in ["don't like", "hate", "dislike", "can't stand"])
        entity_type = "dislikes" if is_dislike else "likes"
        
        # Extract the preference value
        patterns = [
            r'(?:like|love|enjoy|prefer|hate|dislike)\s+([A-Za-z\s]+?)(?:\.|,|and|$)',
            r'(?:my favorite|favourite)\s+(?:is|are)\s+([A-Za-z\s]+?)(?:\.|,|$)',
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text_lower, re.IGNORECASE)
            for match in matches:
                value = match.strip()
                if len(value) > 2 and value not in self.reject_values:
                    # Try to categorize the preference
                    category = self._categorize_preference(value)
                    final_entity = category if category else entity_type
                    
                    entities.append({
                        "entity": final_entity,
                        "value": value.title() if value[0].islower() else value,
                        "confidence": 0.88,
                        "source": ExtractionSource.PATTERN.value,
                        "context": context["primary"],
                        "original_text": text,
                        "preference_type": "dislike" if is_dislike else "like"
                    })
                    break
        
        return entities
    
    def _extract_future_entities(self, text: str) -> List[Dict]:
        """Extract future goals, dreams, aspirations"""
        entities = []
        text_lower = text.lower()
        
        future_patterns = [
            r'(?:i want to|i plan to|i hope to|i dream of)\s+([A-Za-z\s]+?)(?:\.|,|$)',
            r'(?:my goal is|my dream is)\s+([A-Za-z\s]+?)(?:\.|,|$)',
        ]
        
        for pattern in future_patterns:
            matches = re.findall(pattern, text_lower, re.IGNORECASE)
            for match in matches:
                value = match.strip()
                if len(value) > 5:  # Goals should be substantial
                    entities.append({
                        "entity": "goals",
                        "value": value,
                        "confidence": 0.85,
                        "source": ExtractionSource.PATTERN.value,
                        "context": "future",
                        "original_text": text
                    })
        
        return entities
    
    def _extract_ner_entities(self, text: str, raw_entities: Dict, context: Dict) -> List[Dict]:
        """Extract and map NER entities to our schema"""
        entities = []
        
        # Map NER types to our entity types
        ner_mapping = {
            "PERSON": "name",
            "GPE": "location",
            "LOC": "location",
            "ORG": "workplace",
            "NORP": "nationality",
            "LANGUAGE": "language",
            "PRODUCT": "likes",
            "EVENT": "interest",
        }
        
        for ner_type, values in raw_entities.items():
            if ner_type in ner_mapping:
                for value in values:
                    entity_type = ner_mapping[ner_type]
                    # Validate based on context
                    if self._should_extract(entity_type, value, text, context):
                        entities.append({
                            "entity": entity_type,
                            "value": value,
                            "confidence": 0.82,
                            "source": ExtractionSource.NER.value,
                            "context": context["primary"],
                            "original_text": text
                        })
        
        return entities
    
    def _categorize_preference(self, value: str) -> Optional[str]:
        """Intelligently categorize a preference into an entity type"""
        value_lower = value.lower()
        
        # Check against keyword categories
        for category, keywords in self.entity_keywords.items():
            for keyword in keywords:
                if keyword in value_lower or value_lower in keywords:
                    return category
        
        return None
    
    def _smart_categorize(self, entities: List[Dict], text: str) -> List[Dict]:
        """Smart post-processing to improve entity categorization"""
        
        for entity in entities:
            value_lower = entity["value"].lower()
            
            # Upgrade generic likes to specific categories
            if entity["entity"] == "likes":
                category = self._categorize_preference(value_lower)
                if category:
                    entity["entity"] = category
                    entity["original_entity"] = "likes"
                    entity["confidence"] += 0.05
            
            # Split compound entities (e.g., "software engineer at Google")
            if entity["entity"] == "job" and " at " in value_lower:
                parts = value_lower.split(" at ")
                entity["value"] = parts[0].strip()
                if len(parts) > 1:
                    entities.append({
                        "entity": "workplace",
                        "value": parts[1].strip().title(),
                        "confidence": entity["confidence"] - 0.05,
                        "source": ExtractionSource.INFERRED.value,
                        "context": entity["context"],
                        "original_text": text
                    })
        
        return entities
    
    def _analyze_context(self, text: str) -> Dict:
        """Analyze the context of the statement"""
        text_lower = text.lower()
        
        for pattern in self.context_patterns["preference"]:
            if re.search(pattern, text_lower):
                return {"primary": "preference", "confidence": 0.9}
        
        for pattern in self.context_patterns["identity"]:
            if re.search(pattern, text_lower):
                return {"primary": "identity", "confidence": 0.9}
        
        for pattern in self.context_patterns["future"]:
            if re.search(pattern, text_lower):
                return {"primary": "future", "confidence": 0.85}
        
        for pattern in self.context_patterns["negation"]:
            if re.search(pattern, text_lower):
                return {"primary": "negation", "confidence": 0.85}
        
        return {"primary": "generic", "confidence": 0.5}
    
    def _is_valuable_entity(self, entity: Dict) -> bool:
        """Validate if entity is worth storing"""
        entity_type = entity.get("entity")
        value = entity.get("value", "").strip()
        confidence = entity.get("confidence", 0)
        
        if entity_type not in self.allowed_entity_types:
            return False
        
        if not value or len(value) < 2:
            return False
        
        value_lower = value.lower()
        if value_lower in self.reject_values:
            return False
        
        # Check confidence threshold
        min_confidence = 0.65
        if confidence < min_confidence:
            return False
        
        return True
    
    def _should_extract(self, entity_type: str, value: str, text: str, context: Dict) -> bool:
        """Determine if entity should be extracted"""
        value_lower = value.lower()
        
        if value_lower in self.reject_values:
            return False
        
        # Context-specific validation
        if entity_type == "name" and context["primary"] != "identity":
            return False
        
        return True
    
    def _is_trivial_message(self, text: str) -> bool:
        """Check if message is trivial"""
        text_lower = text.lower().strip()
        
        for pattern in self.skip_patterns:
            if re.match(pattern, text_lower, re.IGNORECASE):
                return True
        
        return len(text.split()) < 2
    
    def _is_likely_chitchat(self, text: str) -> bool:
        """Detect casual conversation"""
        chitchat = ["how are you", "what's up", "good morning", "good evening", "bye"]
        return any(phrase in text.lower() for phrase in chitchat)
    
    def _extract_with_llm(self, text: str) -> List[Dict]:
        """LLM fallback for complex extraction"""
        prompt = f"""Extract valuable personal information from: "{text}"

Possible categories: name, age, location, job, education, music, movies, books, games, food, sports, hobbies, skills, goals, likes, dislikes, pets, relationship_status

Rules:
- Return [] if nothing valuable
- Be specific, not generic
- One entity per type

Output JSON array only: [{{"entity": "category", "value": "extracted value"}}]"""

        try:
            response = self.groq.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=200
            )
            
            result = response.choices[0].message.content.strip()
            result = result.replace('```json', '').replace('```', '')
            
            if not result or result == "[]":
                return []
            
            entities = json.loads(result)
            for e in entities:
                e["confidence"] = 0.80
                e["source"] = ExtractionSource.LLM.value
                e["original_text"] = text
            
            return entities
        except:
            return []
    
    def _merge_related_entities(self, entities: List[Dict]) -> List[Dict]:
        """Merge duplicates"""
        merged = {}
        for entity in entities:
            key = f"{entity['entity']}:{entity['value'].lower()}"
            if key not in merged or entity.get("confidence", 0) > merged[key].get("confidence", 0):
                merged[key] = entity
        return list(merged.values())