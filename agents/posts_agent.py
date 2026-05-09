from typing import Dict, Optional
class PostAgent:
    def __init__(self, q_client, embed_model, groq_client):
        self.q_client = q_client
        self.embed_model = embed_model
        self.llm = groq_client
        self.current_draft = None
        self.awaiting_response = False
        self.pending_action = None  
        
    async def process(self, user_id: int, message: str = None, user_confirmation: str = None, image_url: str = None):
        """Main entry point for post creation - returns post_suggestion for button"""        
        if user_confirmation == "approve":
            if self.current_draft:
                return await self.publish_post(user_id)
            else:
                return {"final_response": "No draft found. Please create a post first."}        
        if user_confirmation == "edit":
            return {
                "final_response": " Send me the edited version of your post.",
                "awaiting_edit": True
            }        
        if user_confirmation == "cancel":
            self.awaiting_response = False
            self.current_draft = None
            self.pending_action = None
            return {
                "final_response": "Post cancelled. Let me know if you want to create another!",
                "cancelled": True
            }        
        if self.awaiting_response and message and len(message) > 3:
            return await self.regenerate_post(user_id, message)        
        if image_url:
            if self.current_draft:
                self.current_draft['image_url'] = image_url
                return await self.present_draft_with_button()        
        if message or user_confirmation:
            return await self.create_post_draft(user_id, message or user_confirmation)
        
        return {"final_response": "I can help you create a post! What would you like to post about?"}
    
    async def create_post_draft(self, user_id: int, topic: str):
        """Create initial post draft and return with button suggestion"""
        
        prompt = f"""
        Create a social media post about: "{topic}"
        
        Requirements:
        - Natural, conversational tone
        - 1-2 sentences only
        - Include 1 emoji max
        - Add 2-3 relevant hashtags at the end
        
        Return ONLY the post text with hashtags.
        """
        response = self.llm.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=200
        )
        full_post = response.choices[0].message.content.strip()        
        self.current_draft = {
            "content": full_post,
            "hashtags": self._extract_hashtags(full_post),
            "image_url": None
        }
        self.awaiting_response = True
        self.pending_action = 'approval'        
        return {
            "final_response": f"I've drafted a post about {topic}:",
            "post_suggestion": full_post,  
            "draft_content": full_post,
            "hashtags": self.current_draft['hashtags'],
            "awaiting_approval": True
        }
    
    async def present_draft_with_button(self):
        """Show draft with button options"""
        if not self.current_draft:
            return {"final_response": "No draft found."}
        
        return {
            "final_response": "Here's your post with the added photo:",
            "post_suggestion": self.current_draft['content'],
            "draft_content": self.current_draft['content'],
            "hashtags": self.current_draft['hashtags'],
            "has_media": self.current_draft['image_url'] is not None,
            "awaiting_approval": True
        }
    
    async def publish_post(self, user_id: int):
        """Publish the post using Supabase"""
        if not self.current_draft:
            return {"final_response": "No draft found. Please create a post first."}
        
        draft = self.current_draft
        full_content = draft['content']
        
        from supabase_client import get_supabase_client
        supabase = get_supabase_client()
        
        try:
            result = supabase.table('posts').insert({
                'user_id': user_id,
                'content': full_content,
                'image_url': draft.get('image_url'),
                'total_likes': 0,
                'total_comments': 0
            }).execute()
            
            if result.data:
                post_id = result.data[0]['post_id']
                self.current_draft = None
                self.awaiting_response = False
                self.pending_action = None
                return {
                    "final_response": f"✅ Post published successfully!",
                    "final_post_id": post_id,
                    "post_published": True,
                    "completed": True 
                }
            else:
                return {"final_response": "❌ Failed to publish post. Please try again."}
                
        except Exception as e:
            print(f"Error publishing: {e}")
            return {"final_response": f"❌ Database error: {str(e)}"}
    
    async def regenerate_post(self, user_id: int, edit_request: str):
        """Regenerate post based on edit request and return button"""
        
        if not self.current_draft:
            return {"final_response": "No draft found."}
        
        original = self.current_draft['content']
        
        prompt = f"""
        Original post: "{original}"
        
        Edit request: "{edit_request}"
        
        Rewrite the post according to the request. Keep the same tone.
        Include relevant hashtags at the end.
        Return ONLY the revised post text.
        """
        
        response = self.llm.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=200
        )
        
        new_content = response.choices[0].message.content.strip()
        
        self.current_draft = {
            "content": new_content,
            "hashtags": self._extract_hashtags(new_content),
            "image_url": None
        }
        
        return {
            "final_response": "Here's the edited version:",
            "post_suggestion": new_content,
            "draft_content": new_content,
            "hashtags": self.current_draft['hashtags'],
            "awaiting_approval": True
        }
    
    def _extract_hashtags(self, text: str) -> list:
        """Extract hashtags from post content"""
        import re
        hashtags = re.findall(r'#\w+', text)
        return hashtags