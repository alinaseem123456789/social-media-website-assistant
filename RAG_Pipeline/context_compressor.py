import numpy as np
from typing import List, Dict
from sentence_transformers import CrossEncoder

class ContextCompressor:
    """Multi-stage context compression for RAG"""
    
    def __init__(self, groq_client, embed_model):
        self.groq = groq_client
        self.embed = embed_model
        self.reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        
    def compress_context(self, 
                        query: str, 
                        retrieved_memories: List[Dict], 
                        max_tokens: int = 1500,
                        strategy: str = "adaptive") -> str:
        """
        Compress retrieved memories into concise context
        
        Strategies:
        - 'relevance': Keep only top-k relevant
        - 'summary': LLM summarization
        - 'dedupe': Remove duplicates
        - 'adaptive': Choose best strategy
        """
        
        if not retrieved_memories:
            return ""
        
        # Extract text from memories
        memory_texts = [m['text'] for m in retrieved_memories]
        
        # Step 1: Deduplicate
        deduplicated = self._deduplicate_memories(memory_texts)
        
        # Step 2: Score relevance
        relevant = self._filter_by_relevance(query, deduplicated, top_k=10)
        
        # Step 3: Estimate tokens
        current_tokens = sum(len(t.split()) for t in relevant)
        
        # Step 4: Apply compression strategy
        if strategy == "relevance":
            compressed = self._relevance_compress(query, relevant, max_tokens)
        elif strategy == "summary":
            compressed = self._summary_compress(query, relevant, max_tokens)
        else:  # adaptive
            compressed = self._adaptive_compress(query, relevant, current_tokens, max_tokens)
        
        return compressed
    
    def _deduplicate_memories(self, memories: List[str]) -> List[str]:
        """Remove near-duplicates using embeddings"""
        
        if len(memories) <= 1:
            return memories
        
        # Encode all memories
        vectors = self.embed.encode(memories)
        
        # Find unique
        unique = []
        seen_embeddings = []
        
        for i, vec in enumerate(vectors):
            is_duplicate = False
            for seen_vec in seen_embeddings:
                similarity = np.dot(vec, seen_vec) / (np.linalg.norm(vec) * np.linalg.norm(seen_vec))
                if similarity > 0.95:  # Very similar
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique.append(memories[i])
                seen_embeddings.append(vec)
        
        return unique
    
    def _filter_by_relevance(self, query: str, memories: List[str], top_k: int = 10) -> List[str]:
        """Keep only most relevant memories"""
        
        if len(memories) <= top_k:
            return memories
        
        # Use cross-encoder for accurate scoring
        pairs = [[query, mem] for mem in memories]
        scores = self.reranker.predict(pairs)
        
        # Sort and keep top-k
        scored = list(zip(memories, scores))
        scored.sort(key=lambda x: x[1], reverse=True)
        
        return [mem for mem, score in scored[:top_k]]
    
    def _relevance_compress(self, query: str, memories: List[str], max_tokens: int) -> str:
        """Simple: keep only most relevant, cut off by token limit"""
        
        if not memories:
            return ""
        
        # Score all memories
        pairs = [[query, mem] for mem in memories]
        scores = self.reranker.predict(pairs)
        
        scored = list(zip(memories, scores))
        scored.sort(key=lambda x: x[1], reverse=True)
        
        # Add until token limit
        result = []
        current_tokens = 0
        
        for memory, score in scored:
            memory_tokens = len(memory.split())
            if current_tokens + memory_tokens <= max_tokens:
                result.append(memory)
                current_tokens += memory_tokens
            else:
                break
        
        return " | ".join(result)
    
    def _summary_compress(self, query: str, memories: List[str], max_tokens: int) -> str:
        """Use LLM to create intelligent summary"""
        
        if not memories:
            return ""
        
        # If already small enough, don't summarize
        total_tokens = sum(len(m.split()) for m in memories)
        if total_tokens <= max_tokens:
            return " | ".join(memories)
        
        prompt = f"""User query: "{query}"

Relevant facts about the user:
{chr(10).join(f'- {mem}' for mem in memories[:20])}  # Limit to 20 for LLM

Task: Create a COMPACT, DENSE summary (max {max_tokens} tokens) that preserves ALL unique information.
Remove duplicates and obvious statements.
Keep specific facts (names, ages, locations, preferences).
Output as bullet points without markdown.

Summary:"""
        
        response = self.groq.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=max_tokens
        )
        
        return response.choices[0].message.content.strip()
    
    def _adaptive_compress(self, query: str, memories: List[str], current_tokens: int, max_tokens: int) -> str:
        """Choose best strategy based on compression ratio"""
        
        compression_ratio = current_tokens / max_tokens
        
        if compression_ratio <= 1.2:
            # Close to limit, just use relevance filtering
            return self._relevance_compress(query, memories, max_tokens)
        
        elif compression_ratio <= 3:
            # Moderate compression, use smarter relevance
            return self._relevance_compress(query, memories, max_tokens)
        
        else:
            # High compression needed, use summarization
            return self._summary_compress(query, memories, max_tokens)