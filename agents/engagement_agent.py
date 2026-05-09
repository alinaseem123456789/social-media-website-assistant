from typing import Dict, Optional, List
from agents.engagement_social_api import EngagementSocialAPI
import re
import json

class EngagementAgent:
    def __init__(self, q_client, embed_model, groq_client):
        self.q_client = q_client
        self.embed_model = embed_model
        self.llm = groq_client
        self.social_api = EngagementSocialAPI()
        self.current_suggestions = None
        self.awaiting_response = False
        self.pending_action = None
        
    async def process(self, user_id: int, message: str = None, user_confirmation: str = None, 
                      post_id: int = None, comment_text: str = None):
        """Main entry point for engagement actions"""        
        if user_confirmation == "like" and post_id:
            return await self.like_post(user_id, post_id)        
        if user_confirmation == "comment" and post_id:
            if comment_text:
                return await self.add_comment(user_id, post_id, comment_text)
            else:
                return {
                    "final_response": " What would you like to comment?",
                    "awaiting_comment": True,
                    "post_id": post_id
                }        
        if user_confirmation == "wish_birthday" and post_id:
            return await self.send_birthday_wish(user_id, post_id)        
        if user_confirmation == "check_in":
            return await self.suggest_check_in(user_id)        
        if self.awaiting_response and message and len(message) > 2:
            if self.pending_action == 'comment':
                return await self.add_comment(user_id, self.pending_post_id, message)        
        if message or user_confirmation:
            intent = await self._understand_intent(message or user_confirmation)
            
            if intent.get('type') == 'birthdays':
                return await self.get_birthdays(user_id)
            elif intent.get('type') == 'inactive':
                return await self.get_inactive_friends(user_id)
            elif intent.get('type') == 'suggestions':
                return await self.get_engagement_suggestions(user_id)
            else:
                return await self.get_engagement_suggestions(user_id)
        
        return {"final_response": "I can help you engage with friends! What would you like to do?"}
    
    async def _understand_intent(self, message: str) -> Dict:
        """Extract user's engagement intent"""
        
        prompt = f"""
        Analyze this user message about social engagement: "{message}"
        
        Return JSON with:
        - type: "birthdays", "inactive", "suggestions", or "general"
        - details: any specific information
        
        Examples:
        - "Any birthdays coming up?" → {{"type": "birthdays"}}
        - "Who hasn't posted in a while?" → {{"type": "inactive"}}
        - "What should I like?" → {{"type": "suggestions"}}
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
            return {"type": "suggestions"}
    
    async def get_engagement_suggestions(self, user_id: int) -> Dict:
        """Get posts that need engagement"""
        
        suggestions = self.social_api.get_engagement_suggestions(user_id)
        
        if not suggestions:
            return {
                "final_response": "All caught up! No pending engagement suggestions right now.",
                "engagement_suggestions": []
            }
        
        self.current_suggestions = suggestions
        self.awaiting_response = True
        
        return self._format_suggestions_response(suggestions)
    
    async def get_birthdays(self, user_id: int) -> Dict:
        """Get upcoming birthdays"""
        
        birthdays = self.social_api.get_upcoming_birthdays(user_id)
        
        if not birthdays:
            return {
                "final_response": "No upcoming birthdays in the next 7 days!",
                "birthday_suggestions": []
            }
        
        return self._format_birthdays_response(birthdays)
    
    async def get_inactive_friends(self, user_id: int) -> Dict:
        """Find inactive friends to check in on"""
        
        inactive = self.social_api.get_inactive_friends(user_id)
        
        if not inactive:
            return {
                "final_response": "All your friends have been active recently!",
                "inactive_friends": []
            }
        
        return self._format_inactive_response(inactive)
    
    async def like_post(self, user_id: int, post_id: int) -> Dict:
        """Like a post"""
        
        result = self.social_api.like_post(user_id, post_id)
        
        if result['status'] == 'liked':
            return {
                "final_response": "Liked the post!",
                "liked": True
            }
        elif result['status'] == 'already_liked':
            return {
                "final_response": "You already liked this post!",
                "already_liked": True
            }
        else:
            return {
                "final_response": "Failed to like post. Please try again.",
                "liked": False
            }
    
    async def add_comment(self, user_id: int, post_id: int, comment_text: str) -> Dict:
        """Add a comment to a post"""        
        if not comment_text or comment_text.lower() == 'suggest':
            comment_text = await self._generate_comment(post_id)
        
        result = self.social_api.add_comment(user_id, post_id, comment_text)
        
        if result['status'] == 'commented':
            return {
                "final_response": f"Comment added: \"{comment_text}\"",
                "commented": True
            }
        else:
            return {
                "final_response": "Failed to add comment. Please try again.",
                "commented": False
            }
    
    async def _generate_comment(self, post_id: int) -> str:
        """Generate an appropriate comment using LLM"""        
        post = self.social_api.supabase.table('posts')\
            .select('content')\
            .eq('post_id', post_id)\
            .single()\
            .execute()
        
        if not post.data:
            return "Great post!"
        
        prompt = f"""
        Generate a short, friendly comment (1 sentence max) for this post:
        "{post.data['content']}"
        
        Keep it positive and natural. No hashtags.
        """
        
        response = self.llm.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=50
        )
        
        return response.choices[0].message.content.strip()
    
    async def send_birthday_wish(self, user_id: int, friend_id: int) -> Dict:
        """Send a birthday wish (would integrate with messaging system)"""
        
        friend = self.social_api.get_user_details(friend_id)
        
        if not friend:
            return {"final_response": "Could not find friend."}
        
        wish = f"Happy Birthday @{friend['username']}! Wishing you an amazing day!"
        return {
            "final_response": f" Birthday wish ready to send:\n\n\"{wish}\"\n\nWant me to send it?",
            "birthday_wish": wish,
            "friend_id": friend_id,
            "awaiting_confirmation": True
        }
    
    async def suggest_check_in(self, user_id: int) -> Dict:
        """Suggest checking in on inactive friends"""
        
        inactive = self.social_api.get_inactive_friends(user_id, days_inactive=14)
        
        if not inactive:
            return {
                "final_response": " All your friends have been active recently!"
            }
        
        suggestions = []
        for friend in inactive[:3]:
            suggestions.append({
                'friend_id': friend['friend_id'],
                'username': friend['username'],
                'days_inactive': friend['days_inactive'],
                'suggested_message': f"Hey! Haven't seen you in a while. Hope you're doing great! "
            })
        
        return {
            "final_response": f" Here are some friends you could check in on:",
            "check_in_suggestions": suggestions
        }
    
    def _format_suggestions_response(self, suggestions: List[Dict]) -> Dict:
        """Format engagement suggestions with buttons"""
        
        response_text = f" Found {len(suggestions)} posts to engage with:**\n"
        
        for i, post in enumerate(suggestions, 1):
            response_text += f"\n{i}. {post['username']}**: \"{post['content']}...\""
            response_text += f"\n    {post['reason']}"
        
        return {
            "final_response": response_text,
            "engagement_suggestions": suggestions,
            "suggestions_count": len(suggestions),
            "awaiting_selection": True
        }
    
    def _format_birthdays_response(self, birthdays: List[Dict]) -> Dict:
        """Format birthday suggestions"""
        
        response_text = f" Upcoming Birthdays ({len(birthdays)}):**\n"
        
        for birthday in birthdays:
            if birthday['days_until'] == 0:
                response_text += f"\n {birthday['username']}** - TODAY!"
            else:
                response_text += f"\n {birthday['username']}** - in {birthday['days_until']} days ({birthday['birth_date']})"
        
        return {
            "final_response": response_text,
            "birthday_suggestions": birthdays
        }
    
    def _format_inactive_response(self, inactive: List[Dict]) -> Dict:
        """Format inactive friends response"""
        
        response_text = f" Friends who've been quiet ({len(inactive)}):**\n"
        
        for friend in inactive:
            response_text += f"\n {friend['username']}** - {friend['days_inactive']} days since last post"
        
        return {
            "final_response": response_text,
            "inactive_friends": inactive
        }