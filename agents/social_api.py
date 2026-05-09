from typing import List, Dict, Optional
from supabase_client import get_supabase_client

class SocialAPI:
    def __init__(self):
        self.supabase = get_supabase_client()
    
    def create_post(self, user_id: int, content: str, image_url: Optional[str] = None) -> Dict:
        """Create post using your existing posts table"""
        result = self.supabase.table('posts').insert({
            'user_id': user_id,
            'content': content,
            'image_url': image_url,
            'total_likes': 0,
            'total_comments': 0
        }).execute()
        
        if result.data:
            return {
                'post_id': result.data[0]['post_id'],
                'status': 'published',
                'url': f'/posts/{result.data[0]["post_id"]}'
            }
        raise Exception("Failed to create post")
    
    def get_user_interests(self, user_id: int) -> List[str]:
        """Get user interests from user_profiles table"""
        result = self.supabase.table('user_profiles')\
            .select('hobbies')\
            .eq('user_id', user_id)\
            .execute()
        
        if result.data and result.data[0].get('hobbies'):
            return [h.strip() for h in result.data[0]['hobbies'].split(',')]
        return []
    
    def get_friends(self, user_id: int) -> List[Dict]:
        """Get user's friends list"""
        result = self.supabase.table('friendships')\
            .select('friend_id')\
            .eq('user_id', user_id)\
            .eq('status', 'accepted')\
            .execute()
        
        if not result.data:
            return []
        friend_ids = [f['friend_id'] for f in result.data]        
        users_result = self.supabase.table('users')\
            .select('id, username')\
            .in_('id', friend_ids)\
            .execute()
        return users_result.data if users_result.data else []
    def send_friend_request(self, from_user_id: int, to_user_id: int) -> Dict:
        """Send friend request using your existing system"""
        existing = self.supabase.table('friendships')\
            .select('*')\
            .eq('user_id', from_user_id)\
            .eq('friend_id', to_user_id)\
            .execute()
        
        if existing.data:
            return {'status': 'already_sent', 'friendship_id': existing.data[0]['id']}        
        result = self.supabase.table('friendships').insert({
            'user_id': from_user_id,
            'friend_id': to_user_id,
            'status': 'pending'
        }).execute()
        
        return {'status': 'sent', 'friendship_id': result.data[0]['id'] if result.data else None}