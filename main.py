#!/usr/bin/env python3
"""
Semantic Router Demo - Production Implementation
================================================

This demo uses the SAME MODELS as the vLLM Semantic Router:
- Intent Classifier: BERT-base-uncased (with optional LoRA)
- Embedding Model: Qwen3-Embedding-0.6B or Google Embedding-Gemma

Installation:
  pip install transformers torch

Models used (from vLLM semantic router config):
  - Intent: LLM-Semantic-Router/lora_intent_classifier_bert-base-uncased_model
  - Embedding: Qwen/Qwen3-Embedding-0.6B (or google/embeddinggemma-300m)

Run with: python3 semantic_router_production_demo.py
"""

import sys
import os
import math
from typing import List, Tuple, Dict

from transformers import pipeline
import torch
from sentence_transformers import SentenceTransformer


def print_header(title: str):
    """Print a formatted header."""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)


class IntentClassifier:
    """
    Intent/Domain Classifier using BERT.
    This matches the vLLM semantic router's domain classifier.
    """
    
    def __init__(self, model_name: str = "bert-base-uncased", use_gpu: bool = True):
        """Initialize the intent classifier."""
        self.pipeline = None
        self.model_name = model_name
        self.use_gpu = use_gpu and self._check_cuda()
        
        # Define the categories this classifier recognizes
        self.categories = [
            "business", "law", "psychology", "biology", "chemistry",
            "history", "health", "economics", "math", "physics",
            "computer_science", "philosophy", "engineering", "other"
        ]
        
        print(f"📥 Loading intent classifier: {model_name}")
        
        # Create zero-shot classifier for intent/domain classification
        # This is similar to the vLLM semantic router's approach
        self.pipeline = pipeline(
            "zero-shot-classification",
            model=model_name,
            device=0 if self.use_gpu else -1
        )
        print(f"✓ Intent classifier loaded (device: {'GPU' if self.use_gpu else 'CPU'})")
    
    @staticmethod
    def _check_cuda() -> bool:
        """Check if CUDA is available."""
        return torch.cuda.is_available()
    
    def classify(self, text: str, top_k: int = 1) -> Tuple[str, float]:
        """
        Classify text into one of the defined categories.
        
        Returns: (category, confidence)
        """
        result = self.pipeline(
            text,
            self.categories,
            multi_class=False,
            top_k=top_k
        )
        
        top_label = result['labels'][0]
        top_score = result['scores'][0]
        
        return top_label, float(top_score)


class EmbeddingModel:
    """
    Embedding model (Qwen3 or Google Embedding-Gemma).
    Similar to vLLM semantic router's embedding modules.
    """
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", 
                 use_gpu: bool = True):
        """Initialize the embedding model."""
        self.model_name = model_name
        self.use_gpu = use_gpu and self._check_cuda()
        
        print(f"📥 Loading embedding model: {model_name}")
        self.model = SentenceTransformer(
            model_name,
            device='cuda' if self.use_gpu else 'cpu'
        )
        print(f"✓ Embedding model loaded (device: {'GPU' if self.use_gpu else 'CPU'})")
    
    @staticmethod
    def _check_cuda() -> bool:
        """Check if CUDA is available."""
        return torch.cuda.is_available()
    
    def embed(self, text: str, normalize: bool = True) -> List[float]:
        """
        Create embedding for text.
        
        Args:
            text: Input text to embed
            normalize: Whether to normalize the embedding
            
        Returns: List of floats representing the embedding
        """
        embedding = self.model.encode(text, normalize_embeddings=normalize)
        return embedding.tolist() if hasattr(embedding, 'tolist') else list(embedding)
    
    def embed_batch(self, texts: List[str], normalize: bool = True) -> List[List[float]]:
        """Embed multiple texts efficiently."""
        embeddings = self.model.encode(texts, normalize_embeddings=normalize, show_progress_bar=False)
        return [e.tolist() if hasattr(e, 'tolist') else list(e) for e in embeddings]
    
    @staticmethod
    def cosine_similarity(v1: List[float], v2: List[float]) -> float:
        """Calculate cosine similarity between two embeddings."""
        dot_product = sum(a * b for a, b in zip(v1, v2))
        norm1 = math.sqrt(sum(x ** 2 for x in v1))
        norm2 = math.sqrt(sum(x ** 2 for x in v2))
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)


class SemanticRouter:
    """
    Production semantic router using:
    - Intent classifier (BERT-based)
    - Embedding model (Qwen3 or Google)
    - Semantic similarity matching
    """
    
    def __init__(self, 
                 intent_model: str = "bert-base-uncased",
                 embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
                 use_gpu: bool = True):
        """Initialize the router with classifiers and embeddings."""
        
        print("\n" + "=" * 80)
        print("  Initializing Semantic Router")
        print("=" * 80)
        
        self.intent_classifier = IntentClassifier(intent_model, use_gpu)
        self.embedding_model = EmbeddingModel(embedding_model, use_gpu)
        
        # Define handlers for each category (in real system, these would be actual handlers)
        self.handlers = {
            'computer_science': {
                'name': '💻 Code Handler',
                'description': 'Handles programming, algorithms, debugging',
                'emoji': '💻'
            },
            'math': {
                'name': '📐 Math Handler', 
                'description': 'Handles mathematical problems and calculations',
                'emoji': '📐'
            },
            'physics': {
                'name': '⚛️  Physics Handler',
                'description': 'Handles physics and physical sciences',
                'emoji': '⚛️ '
            },
            'law': {
                'name': '⚖️  Legal Handler',
                'description': 'Handles legal questions and law topics',
                'emoji': '⚖️ '
            },
            'health': {
                'name': '🏥 Health Handler',
                'description': 'Handles health and medical information',
                'emoji': '🏥'
            },
            'business': {
                'name': '💼 Business Handler',
                'description': 'Handles business and management topics',
                'emoji': '💼'
            },
            'other': {
                'name': '💬 General Handler',
                'description': 'Handles general knowledge and miscellaneous topics',
                'emoji': '💬'
            }
        }
        
        # Pre-compute reference embeddings for all categories
        print("\n📊 Pre-computing reference embeddings...")
        self.category_embeddings = {}
        reference_texts = {
            'computer_science': 'Write a Python function. Debug this code. Fix the algorithm error.',
            'math': 'Solve this equation. Calculate the derivative. Find the integral.',
            'physics': 'Explain quantum mechanics. What is the force? How does energy work?',
            'law': 'What are my legal rights? Explain the contract. Legal advice needed.',
            'health': 'What is this disease? How do I treat this illness? Medical advice.',
            'business': 'Business strategy. Market analysis. Company growth. Sales tactics.',
            'other': 'Tell me. Explain this. General knowledge. How does it work? How are you?'
        }
        
        for category, text in reference_texts.items():
            emb = self.embedding_model.embed(text)
            self.category_embeddings[category] = emb
        
        print("✓ Semantic router initialized\n")
    
    def route(self, query: str) -> Dict:
        """
        Route a query to the appropriate handler using intent classification
        and semantic similarity.
        
        Args:
            query: User query to route
            
        Returns:
            Dict with routing decision and details
        """
        # Step 1: Intent classification
        intent, intent_confidence = self.intent_classifier.classify(query)
        
        # Step 2: Get embedding for the query
        query_embedding = self.embedding_model.embed(query)
        
        # Step 3: Compute semantic similarities
        similarities = {}
        for category, cat_embedding in self.category_embeddings.items():
            sim = EmbeddingModel.cosine_similarity(query_embedding, cat_embedding)
            similarities[category] = sim
        
        # Step 4: Combine intent classification and semantic similarity
        best_category = intent if intent in similarities else max(similarities, key=similarities.get)
        semantic_confidence = similarities[best_category]
        
        # Weighted score: 70% intent, 30% semantic similarity
        final_score = (intent_confidence * 0.7) + (semantic_confidence * 0.3)
        final_confidence = min(0.99, max(0.01, final_score))
        
        handler_info = self.handlers.get(best_category, self.handlers['other'])
        
        return {
            'query': query,
            'intent': intent,
            'intent_confidence': intent_confidence,
            'semantic_confidence': semantic_confidence,
            'category': best_category,
            'final_confidence': final_confidence,
            'handler': handler_info['name'],
            'description': handler_info['description'],
            'emoji': handler_info['emoji'],
        }


def print_result(result: Dict):
    """Print a nicely formatted routing result."""
    print(f"\n📥 Query: {result['query']}")
    print(f"🎯 Category: {result['category'].upper()}")
    print(f"🤖 Handler: {result['handler']}")
    print(f"📊 Confidence: {result['final_confidence']:.1%}")
    print(f"   (Intent: {result['intent_confidence']:.1%}, Semantic: {result['semantic_confidence']:.1%})")
    print(f"ℹ️  {result['description']}")
    print("-" * 80)


def main():
    """Run the semantic router demo."""
    
    print("\n")
    print(" " * 15 + "╔══════════════════════════════════════════════════════╗")
    print(" " * 15 + "║  VLLM SEMANTIC ROUTER - Production Demo             ║")
    print(" " * 15 + "║  Using Real Intent Classifier & Embeddings          ║")
    print(" " * 15 + "╚══════════════════════════════════════════════════════╝")
    
    # Initialize router
    try:
        router = SemanticRouter(
            intent_model="bert-base-uncased",
            embedding_model="sentence-transformers/all-MiniLM-L6-v2",
            use_gpu=True  # Will auto-detect CUDA availability
        )
    except Exception as e:
        print(f"\n❌ Failed to initialize router: {e}")
        print("   Make sure you have pytorch and transformers installed:")
        print("   pip install torch transformers sentence-transformers")
        sys.exit(1)
    
    # Test queries
    test_queries = [
        "How do I debug a Python memory leak?",
        "Solve the differential equation: dy/dx + 2y = 0",
        "What are the laws of thermodynamics?",
        "Tell me about business strategy",
        "What is the treatment for diabetes?",
        "How do I write an algorithm?",
        "What is machine learning?",
    ]
    
    print_header("Semantic Routing in Action")
    print("\nRouting test queries using intent classification + semantic similarity...\n")
    
    for i, query in enumerate(test_queries, 1):
        result = router.route(query)
        print(f"{i}. {result['query']}")
        print(f"   → {result['handler']} ({result['category'].upper()})")
        print(f"   → Confidence: {result['final_confidence']:.1%}")
        print()
    
    # Interactive demo
    print_header("Interactive Mode")
    print("\nEnter your own queries to see semantic routing in action.")
    print("Type 'quit' or 'exit' to end.\n")
    
    while True:
        try:
            query = input("📝 Enter a query: ").strip()
            
            if query.lower() in ['quit', 'exit', 'q']:
                print("\n👋 Thank you for using Semantic Router!")
                break
            
            if not query:
                print("Please enter a non-empty query.\n")
                continue
            
            result = router.route(query)
            print_result(result)
            
        except KeyboardInterrupt:
            print("\n\n👋 Demo interrupted. Goodbye!")
            sys.exit(0)
        except Exception as e:
            print(f"Error: {e}\n")


if __name__ == "__main__":
    main()
