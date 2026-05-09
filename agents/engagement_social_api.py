from typing import List, Dict, Optional
from supabase_client import get_supabase_client
from datetime import datetime, timedelta

class EngagementSocialAPI:
    def __init__(self):
        self.supabase = get_supabase_client()
    def get_friends_posts(self, user_id: int, limit: int = 10) -> List[Dict]:
        """Get recent posts from user's friends"""
        friends = self.get_friends_list(user_id)
        friend_ids = [f['id'] for f in friends]
        if not friend_ids:
            return []
        result = self.supabase.table('posts')\
            .select('post_id, user_id, content, created_at, total_likes')\
            .in_('user_id', friend_ids)\
            .order('created_at', desc=True)\
            .limit(limit)\
            .execute()
        posts = []
        if result.data:
            for post in result.data:
                user = self.get_user_details(post['user_id'])
                if user:
                    posts.append({
                        'post_id': post['post_id'],
                        'user_id': post['user_id'],
                        'username': user['username'],
                        'content': post['content'],
                        'created_at': post['created_at'],
                        'total_likes': post.get('total_likes', 0)
                    })
        
        return posts
    
    def get_friends_list(self, user_id: int) -> List[Dict]:
        """Get user's accepted friends from friends table"""
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
            .select('id, username')\
            .in_('id', friend_ids)\
            .execute()
        
        return users_result.data if users_result.data else []
    
    def get_user_details(self, user_id: int) -> Dict:
        """Get basic user details"""
        result = self.supabase.table('users')\
            .select('id, username')\
            .eq('id', user_id)\
            .execute()
        
        return result.data[0] if result.data else {}
    
    def check_if_liked(self, user_id: int, post_id: int) -> bool:
        """Check if user already liked a post using like_id"""
        result = self.supabase.table('likes')\
            .select('like_id')\
            .eq('user_id', user_id)\
            .eq('post_id', post_id)\
            .execute()
        
        return len(result.data) > 0 if result.data else False
    
    def like_post(self, user_id: int, post_id: int) -> Dict:
        """Like a post"""
        if self.check_if_liked(user_id, post_id):
            return {'status': 'already_liked', 'message': 'Already liked this post'}
        
        try:
            # Insert into likes table
            result = self.supabase.table('likes').insert({
                'user_id': user_id,
                'post_id': post_id
            }).execute()
            
            if result.data:
                current_post = self.supabase.table('posts')\
                    .select('total_likes')\
                    .eq('post_id', post_id)\
                    .execute()
                
                current_likes = current_post.data[0].get('total_likes', 0) if current_post.data else 0
                
                self.supabase.table('posts')\
                    .update({'total_likes': current_likes + 1})\
                    .eq('post_id', post_id)\
                    .execute()
                
                return {'status': 'liked', 'message': 'Post liked!', 'like_id': result.data[0]['like_id']}
        except Exception as e:
            print(f"Error liking post: {e}")
            return {'status': 'failed', 'message': f'Error: {str(e)}'}
        
        return {'status': 'failed', 'message': 'Failed to like post'}
    
    def unlike_post(self, user_id: int, post_id: int) -> Dict:
        """Unlike a post"""
        result = self.supabase.table('likes')\
            .delete()\
            .eq('user_id', user_id)\
            .eq('post_id', post_id)\
            .execute()
        
        if result.data:
            # Update post total_likes
            current_post = self.supabase.table('posts')\
                .select('total_likes')\
                .eq('post_id', post_id)\
                .execute()
            
            current_likes = current_post.data[0].get('total_likes', 0) if current_post.data else 0
            new_likes = max(0, current_likes - 1)
            
            self.supabase.table('posts')\
                .update({'total_likes': new_likes})\
                .eq('post_id', post_id)\
                .execute()
            
            return {'status': 'unliked', 'message': 'Post unliked!'}
        
        return {'status': 'failed', 'message': 'Failed to unlike post'}
    
    def add_comment(self, user_id: int, post_id: int, comment_text: str) -> Dict:
        """Add a comment to a post"""
        try:
            result = self.supabase.table('comments').insert({
                'user_id': user_id,
                'post_id': post_id,
                'comment_text': comment_text
            }).execute()
            
            if result.data:
                # Update post total_comments if column exists
                try:
                    current_post = self.supabase.table('posts')\
                        .select('total_comments')\
                        .eq('post_id', post_id)\
                        .execute()
                    
                    if current_post.data and 'total_comments' in current_post.data[0]:
                        current_comments = current_post.data[0].get('total_comments', 0)
                        self.supabase.table('posts')\
                            .update({'total_comments': current_comments + 1})\
                            .eq('post_id', post_id)\
                            .execute()
                except:
                    pass
                
                return {
                    'status': 'commented',
                    'comment_id': result.data[0].get('comment_id', 0),
                    'message': 'Comment added!'
                }
        except Exception as e:
            print(f"Error adding comment: {e}")
            return {'status': 'failed', 'message': f'Error: {str(e)}'}
        
        return {'status': 'failed', 'message': 'Failed to add comment'}
    
    def get_post_likes_count(self, post_id: int) -> int:
        """Get total likes for a post"""
        result = self.supabase.table('likes')\
            .select('like_id', count='exact')\
            .eq('post_id', post_id)\
            .execute()
        
        return result.count if result.count else 0
    
    def get_user_liked_posts(self, user_id: int, limit: int = 20) -> List[int]:
        """Get IDs of posts user has liked"""
        result = self.supabase.table('likes')\
            .select('post_id')\
            .eq('user_id', user_id)\
            .limit(limit)\
            .execute()
        
        return [row['post_id'] for row in result.data] if result.data else []
    
    def get_upcoming_birthdays(self, user_id: int, days_ahead: int = 7) -> List[Dict]:
        """Get friends with upcoming birthdays"""
        friends = self.get_friends_list(user_id)    
        if not friends:
            return []
    
        birthdays = []
        today = datetime.now().date()        
        for friend in friends:
            try:
                profile = self.supabase.table('user_profiles')\
                    .select('birth_date, age')\
                    .eq('user_id', friend['id'])\
                    .execute()                
                if profile.data and profile.data[0].get('birth_date'):
                    birth_date = datetime.strptime(profile.data[0]['birth_date'], '%Y-%m-%d').date()
                    birth_date_this_year = birth_date.replace(year=today.year)
                    
                    if birth_date_this_year < today:
                        birth_date_this_year = birth_date_this_year.replace(year=today.year + 1)
                    
                    days_until = (birth_date_this_year - today).days
                    
                    if 0 <= days_until <= days_ahead:
                        birthdays.append({
                            'friend_id': friend['id'],
                            'username': friend['username'],
                            'birth_date': birth_date.strftime('%B %d'),
                            'days_until': days_until
                        })
            except Exception as e:
                print(f"Error getting birthday for {friend['username']}: {e}")
        
        return birthdays
    
    def get_inactive_friends(self, user_id: int, days_inactive: int = 30) -> List[Dict]:
        """Find friends who haven't posted in a while"""
        friends = self.get_friends_list(user_id)
        
        if not friends:
            return []
        
        inactive = []
        cutoff_date = datetime.now() - timedelta(days=days_inactive)
        
        for friend in friends:
            try:
                # Check last post date
                last_post = self.supabase.table('posts')\
                    .select('created_at')\
                    .eq('user_id', friend['id'])\
                    .order('created_at', desc=True)\
                    .limit(1)\
                    .execute()
                
                if last_post.data:
                    last_post_date = datetime.strptime(last_post.data[0]['created_at'], '%Y-%m-%dT%H:%M:%S')
                    if last_post_date < cutoff_date:
                        inactive.append({
                            'friend_id': friend['id'],
                            'username': friend['username'],
                            'last_post': last_post_date.strftime('%Y-%m-%d'),
                            'days_inactive': (datetime.now() - last_post_date).days
                        })
                else:
                    # Never posted
                    inactive.append({
                        'friend_id': friend['id'],
                        'username': friend['username'],
                        'last_post': 'Never',
                        'days_inactive': days_inactive
                    })
            except Exception as e:
                print(f"Error checking posts for {friend['username']}: {e}")
        
        return inactive
    
    def get_engagement_suggestions(self, user_id: int) -> List[Dict]:
        """Get posts that need engagement (recent, not liked, etc.)"""
        posts = self.get_friends_posts(user_id, limit=10)
        
        suggestions = []
        for post in posts:
            liked = self.check_if_liked(user_id, post['post_id'])
            if not liked:
                if post['total_likes'] == 0:
                    reason = f"No one has liked this yet - be the first!"
                elif post['total_likes'] < 5:
                    reason = f"Only {post['total_likes']} likes so far"
                else:
                    reason = f"Your friend {post['username']} shared this recently"
                
                suggestions.append({
                    'type': 'like',
                    'post_id': post['post_id'],
                    'username': post['username'],
                    'content': post['content'][:100] if post['content'] else '',
                    'total_likes': post['total_likes'],
                    'reason': reason
                })
        
        return suggestions[:5]