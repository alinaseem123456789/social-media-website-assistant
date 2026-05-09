from typing import List, Dict, Optional
from supabase_client import get_supabase_client

class FriendsSocialAPI:
    def __init__(self):
        self.supabase = get_supabase_client()
    
    def get_user_profile(self, user_id: int) -> Dict:
        """Get user profile with interests and location"""
        result = self.supabase.table('user_profiles')\
            .select('user_id, hobbies, bio, country')\
            .eq('user_id', user_id)\
            .execute()
        
        if result.data:
            return result.data[0]
        return {}
    
    def get_user_interests(self, user_id: int) -> List[str]:
        """Get user interests from user_profiles table"""
        result = self.supabase.table('user_profiles')\
            .select('hobbies')\
            .eq('user_id', user_id)\
            .execute()
        
        if result.data and result.data[0].get('hobbies'):
            return [h.strip() for h in result.data[0]['hobbies'].split(',')]
        return []
    
    def get_user_details(self, user_id: int) -> Dict:
        """Get basic user details"""
        result = self.supabase.table('users')\
            .select('id, username, email')\
            .eq('id', user_id)\
            .single()\
            .execute()
        
        return result.data if result.data else {}
    
    def get_friends_list(self, user_id: int) -> List[Dict]:
        """Get user's existing friends (accepted)"""
        result = self.supabase.table('friends')\
            .select('requester_id, recipient_id')\
            .eq('status', 'accepted')\
            .or_(f'requester_id.eq.{user_id},recipient_id.eq.{user_id}')\
            .execute()
        
        if not result.data:
            return []
        
        friend_ids = []
        for row in result.data:
            if row['requester_id'] == user_id:
                friend_ids.append(row['recipient_id'])
            else:
                friend_ids.append(row['requester_id'])
        
        if not friend_ids:
            return []
        
        users_result = self.supabase.table('users')\
            .select('id, username, email')\
            .in_('id', friend_ids)\
            .execute()
        
        return users_result.data if users_result.data else []
    
    def check_friend_request_exists(self, requester_id: int, recipient_id: int) -> bool:
        """Check if a friend request already exists between users"""
        result = self.supabase.table('friends')\
            .select('friendship_id')\
            .or_(f'and(requester_id.eq.{requester_id},recipient_id.eq.{recipient_id}),and(requester_id.eq.{recipient_id},recipient_id.eq.{requester_id})')\
            .execute()
        
        return len(result.data) > 0 if result.data else False
    
    def send_friend_request(self, requester_id: int, recipient_id: int) -> Dict:
        """Send a friend request"""
        if self.check_friend_request_exists(requester_id, recipient_id):
            return {'status': 'already_exists', 'message': 'Friend request already exists'}
        
        result = self.supabase.table('friends').insert({
            'requester_id': requester_id,
            'recipient_id': recipient_id,
            'status': 'pending'
        }).execute()
        
        if result.data:
            return {
                'status': 'sent',
                'friendship_id': result.data[0]['friendship_id'],
                'message': 'Friend request sent'
            }
        
        return {'status': 'failed', 'message': 'Failed to send friend request'}
    
    def accept_friend_request(self, friendship_id: int) -> Dict:
        """Accept a friend request"""
        result = self.supabase.table('friends')\
            .update({'status': 'accepted'})\
            .eq('friendship_id', friendship_id)\
            .execute()
        
        if result.data:
            return {
                'status': 'accepted',
                'friendship_id': friendship_id,
                'message': 'Friend request accepted'
            }
        
        return {'status': 'failed', 'message': 'Failed to accept friend request'}
    
    def reject_friend_request(self, friendship_id: int) -> Dict:
        """Reject a friend request"""
        result = self.supabase.table('friends')\
            .update({'status': 'rejected'})\
            .eq('friendship_id', friendship_id)\
            .execute()
        
        if result.data:
            return {
                'status': 'rejected',
                'friendship_id': friendship_id,
                'message': 'Friend request rejected'
            }
        
        return {'status': 'failed', 'message': 'Failed to reject friend request'}
    
    def get_mutual_friends_count(self, user_id: int, other_user_id: int) -> int:
        """Get number of mutual friends between two users"""
        user_friends = self.get_friends_list(user_id)
        other_friends = self.get_friends_list(other_user_id)
        
        user_friend_ids = {f['id'] for f in user_friends}
        other_friend_ids = {f['id'] for f in other_friends}
        
        return len(user_friend_ids & other_friend_ids)
    
    def find_users_by_interests(self, interests: List[str], exclude_user_id: int, limit: int = 10) -> List[Dict]:
        """Find users who share similar interests"""
        suggestions = []
        
        for interest in interests[:3]:
            result = self.supabase.table('user_profiles')\
                .select('user_id, hobbies, bio, country')\
                .ilike('hobbies', f'%{interest}%')\
                .neq('user_id', exclude_user_id)\
                .limit(limit)\
                .execute()
            
            if result.data:
                for profile in result.data:
                    user = self.get_user_details(profile['user_id'])
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
        
        seen = set()
        unique_suggestions = []
        for s in suggestions:
            if s['id'] not in seen:
                seen.add(s['id'])
                unique_suggestions.append(s)
        
        return unique_suggestions[:limit]
    
    def get_recent_users(self, exclude_user_id: int, limit: int = 10) -> List[Dict]:
        """Get recent users as fallback suggestions"""
        result = self.supabase.table('users')\
            .select('id, username, email')\
            .neq('id', exclude_user_id)\
            .order('created_at', desc=True)\
            .limit(limit)\
            .execute()
        
        suggestions = []
        for user in result.data:
            profile = self.get_user_profile(user['id'])
            suggestions.append({
                'id': user['id'],
                'username': user['username'],
                'email': user['email'],
                'hobbies': profile.get('hobbies', ''),
                'bio': profile.get('bio', ''),
                'location': profile.get('country', ''),
                'match_reason': 'Recently joined'
            })
        
        return suggestions