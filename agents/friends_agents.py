from typing import Dict, Optional, List
from agents.friends_social_api import FriendsSocialAPI
import re
import json

class FriendSuggestionAgent:
    def __init__(self, q_client, embed_model, groq_client):
        self.q_client = q_client
        self.embed_model = embed_model
        self.llm = groq_client
        self.social_api = FriendsSocialAPI()
        self.current_suggestions = None
        self.awaiting_response = False
        self.pending_action = None
        self.supabase = self.social_api.supabase          
    
    async def process(self, user_id: int, message: str = None, user_confirmation: str = None, selected_friends: List[int] = None):
        """Main entry point for friend suggestions - returns suggestions with buttons"""        
        if user_confirmation == "send_requests" and selected_friends:
            return await self.send_friend_requests(user_id, selected_friends)
        if user_confirmation == "send_to_all":
            if self.current_suggestions:
                all_ids = [f['id'] for f in self.current_suggestions]
                return await self.send_friend_requests(user_id, all_ids)        
        if user_confirmation in ["skip", "cancel", "no", "not now"]:
            self.awaiting_response = False
            self.current_suggestions = None
            return {
                "final_response": "No problem! I'll suggest friends another time.",
                "cleared": True
            }        
        if user_confirmation == "edit_preferences":
            return {
                "final_response": "What kind of friends are you looking for? (e.g., 'people who like hiking', 'developers in NYC', 'photography enthusiasts')",
                "awaiting_preferences": True
            }        
        if self.awaiting_response and message and len(message) > 5:
            return await self.find_friends_by_preferences(user_id, message)        
        if message and message.startswith("approve_friend_"):
            friend_id = int(message.split("_")[-1])
            return await self.send_friend_requests(user_id, [friend_id])        
        if message and message.startswith("send_to_all_"):
            try:
                friend_ids = json.loads(message.replace("send_to_all_", ""))
                return await self.send_friend_requests(user_id, friend_ids)
            except:
                pass        
        if message or user_confirmation:
            intent = await self._understand_intent(message or user_confirmation)
            
            if intent.get('type') == 'specific_preferences':
                return await self.find_friends_by_preferences(user_id, intent.get('preferences', message))
            else:
                return await self.suggest_friends(user_id, message or user_confirmation)
        
        return {"final_response": "I can help you find friends! What kind of people are you looking for?"}
    
    async def _understand_intent(self, message: str) -> Dict:
        """Extract user's friend preferences from natural language"""
        
        prompt = f"""
        Extract friend preferences from this user message: "{message}"
        
        Return JSON with:
        - type: "specific_preferences" or "general"
        - preferences: list of keywords (interests, location, profession, etc.)
        - count: number of suggestions wanted (default 5)
        
        Example output: {{"type": "specific_preferences", "preferences": ["hiking", "photography"], "count": 3}}
        """
        
        response = self.llm.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=150
        )
        
        try:
            result = json.loads(response.choices[0].message.content)
            return result
        except:
            return {"type": "general", "preferences": [], "count": 5}
    
    async def suggest_friends(self, user_id: int, query: str = None) -> Dict:
        """Generate friend suggestions based on user context and query"""
        
        user_interests = await self._get_user_interests(user_id)
        existing_friends = await self._get_existing_friend_ids(user_id)
        
        if query:
            preferences = await self._extract_preferences(query)
            suggestions = await self._find_potential_friends(user_id, preferences, existing_friends, limit=5)
        else:
            suggestions = await self._find_potential_friends(user_id, user_interests, existing_friends, limit=5)
        
        if not suggestions:
            return {
                "final_response": "I couldn't find any matching friends right now. Try different interests or check back later!",
                "friend_suggestions": []
            }
        
        for suggestion in suggestions:
            suggestion['match_reason'] = await self._generate_match_reason(suggestion, query or user_interests)
            mutual_count = self.social_api.get_mutual_friends_count(user_id, suggestion['id'])
            if mutual_count > 0:
                suggestion['match_reason'] += f" + {mutual_count} mutual friend(s)"
        
        self.current_suggestions = suggestions
        self.awaiting_response = True
        
        return self._format_suggestions_response(suggestions)
    
    async def find_friends_by_preferences(self, user_id: int, preferences_text: str) -> Dict:
        """Find friends based on specific user preferences"""
        
        preferences = await self._extract_preferences(preferences_text)
        existing_friends = await self._get_existing_friend_ids(user_id)
        
        suggestions = await self._find_potential_friends(user_id, preferences, existing_friends, limit=5)
        
        if not suggestions:
            return {
                "final_response": f"I couldn't find anyone matching '{preferences_text}'. Try different interests?",
                "friend_suggestions": []
            }
        
        for suggestion in suggestions:
            suggestion['match_reason'] = await self._generate_match_reason(suggestion, preferences_text)
            mutual_count = self.social_api.get_mutual_friends_count(user_id, suggestion['id'])
            if mutual_count > 0:
                suggestion['match_reason'] += f" + {mutual_count} mutual friend(s)"
        
        self.current_suggestions = suggestions
        self.awaiting_response = True
        
        return self._format_suggestions_response(suggestions)
    
    async def _find_potential_friends(self, user_id: int, interests: List[str], exclude_ids: List[int], limit: int = 5) -> List[Dict]:
        """Find potential friends with multiple fallback strategies"""
        suggestions = []        
        if interests:
            for interest in interests[:3]:
                result = self.supabase.table('user_profiles')\
                    .select('user_id, hobbies, bio, country')\
                    .ilike('hobbies', f'%{interest}%')\
                    .neq('user_id', user_id)\
                    .limit(limit)\
                    .execute()
                
                if result.data:
                    for profile in result.data:
                        if profile['user_id'] not in exclude_ids:
                            user = self.social_api.get_user_details(profile['user_id'])
                            if user:
                                suggestions.append({
                                    'id': user['id'],
                                    'username': user['username'],
                                    'email': user['email'],
                                    'hobbies': profile.get('hobbies', ''),
                                    'bio': profile.get('bio', ''),
                                    'location': profile.get('country', ''),
                                    'match_interests': [interest]
                                })
        
        if not suggestions:
            print(" No interest matches, looking for users with hobbies...")
            result = self.supabase.table('user_profiles')\
                .select('user_id, hobbies, bio, country')\
                .not_.is_('hobbies', 'null')\
                .neq('hobbies', '')\
                .neq('user_id', user_id)\
                .limit(limit)\
                .execute()
            
            if result.data:
                for profile in result.data:
                    if profile['user_id'] not in exclude_ids:
                        user = self.social_api.get_user_details(profile['user_id'])
                        if user:
                            suggestions.append({
                                'id': user['id'],
                                'username': user['username'],
                                'email': user['email'],
                                'hobbies': profile.get('hobbies', ''),
                                'bio': profile.get('bio', ''),
                                'location': profile.get('country', ''),
                                'match_interests': []
                            })        
        if not suggestions:
            print(" No users with hobbies found, getting recent users...")
            result = self.supabase.table('users')\
                .select('id, username, email')\
                .neq('id', user_id)\
                .order('created_at', desc=True)\
                .limit(limit + len(exclude_ids))\
                .execute()
            
            if result.data:
                for user in result.data:
                    if user['id'] not in exclude_ids:
                        profile_result = self.supabase.table('user_profiles')\
                            .select('hobbies, bio, country')\
                            .eq('user_id', user['id'])\
                            .execute()
                        
                        profile = profile_result.data[0] if profile_result.data else {}
                        
                        suggestions.append({
                            'id': user['id'],
                            'username': user['username'],
                            'email': user['email'],
                            'hobbies': profile.get('hobbies', ''),
                            'bio': profile.get('bio', ''),
                            'location': profile.get('country', ''),
                            'match_interests': [],
                            'match_reason': 'New user'
                        })        
        seen = set()
        unique_suggestions = []
        for s in suggestions:
            if s['id'] not in seen:
                seen.add(s['id'])
                unique_suggestions.append(s)
        
        return unique_suggestions[:limit]
    
    async def _get_user_interests(self, user_id: int) -> List[str]:
        """Get user interests from Qdrant memory"""
        try:
            from qdrant_client.models import Filter, FieldCondition, MatchValue
            
            search_result = self.q_client.scroll(
                collection_name="my_collection",
                scroll_filter=Filter(
                    must=[
                        FieldCondition(key="user_id", match=MatchValue(value=user_id)),
                        FieldCondition(key="entity", match=MatchValue(value="interest"))
                    ]
                ),
                limit=20
            )
            
            interests = []
            for point in search_result[0]:
                value = point.payload.get("value")
                if value:
                    interests.append(value)
            return interests
        except:
            return []
    
    async def _get_existing_friend_ids(self, user_id: int) -> List[int]:
        """Get IDs of existing friends using the social API"""
        friends = self.social_api.get_friends_list(user_id)
        return [f['id'] for f in friends]
    
    async def _extract_preferences(self, text: str) -> List[str]:
        """Extract interest keywords from text"""
        
        prompt = f"""
        Extract interest keywords from: "{text}"
        
        Return as JSON array of strings.
        Example: ["hiking", "photography", "coffee"]
        """
        
        response = self.llm.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=100
        )
        
        try:
            interests = json.loads(response.choices[0].message.content)
            return interests[:5]
        except:
            words = re.findall(r'\b\w+\b', text.lower())
            return [w for w in words if len(w) > 3][:5]
    
    async def _generate_match_reason(self, friend: Dict, preferences) -> str:
        """Generate a friendly match reason"""
        
        if friend.get('match_interests'):
            interests = ', '.join(friend['match_interests'][:2])
            return f"Shares interests: {interests}"
        elif friend.get('location'):
            return f"Located in {friend['location']}"
        elif friend.get('hobbies'):
            hobbies = friend['hobbies'].split(',')[0][:30]
            return f"Interests: {hobbies}"
        else:
            return "Potential connection"
    
    def _format_suggestions_response(self, suggestions: List[Dict]) -> Dict:
        """Format suggestions with button data for frontend"""
        
        if not suggestions:
            return {
                "final_response": "No suggestions found.",
                "friend_suggestions": []
            }
        
        response_text = f"Found {len(suggestions)} people you might know:**"
        
        for i, friend in enumerate(suggestions, 1):
            response_text += f"\n\n{i}. **{friend['username']}**"
            if friend.get('match_reason'):
                response_text += f"\n {friend['match_reason']}"
            if friend.get('location'):
                response_text += f"\n  {friend['location']}"
            if friend.get('hobbies'):
                hobbies_short = friend['hobbies'][:50] + ('...' if len(friend['hobbies']) > 50 else '')
                response_text += f"\n {hobbies_short}"
        
        return {
            "final_response": response_text,
            "friend_suggestions": suggestions,
            "suggestions_count": len(suggestions)
        }
    
    async def send_friend_requests(self, user_id: int, friend_ids: List[int]) -> Dict:
        """Send friend requests using the social API"""
        
        sent = []
        failed = []
        
        for friend_id in friend_ids:
            result = self.social_api.send_friend_request(user_id, friend_id)
            
            if result['status'] == 'sent':
                sent.append(friend_id)
            else:
                failed.append({"id": friend_id, "reason": result.get('message', 'Unknown error')})
        
        self.current_suggestions = None
        self.awaiting_response = False
        
        success_count = len(sent)
        fail_count = len(failed)
        
        if success_count > 0:
            response = f" Sent {success_count} friend request(s)!"
            if fail_count > 0:
                response += f" ({fail_count} failed)"
        else:
            response = "Could not send friend requests. Please try again."
        
        return {
            "final_response": response,
            "sent_requests": sent,
            "failed_requests": failed,
            "requests_sent": success_count > 0
        }