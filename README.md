# Semantic Router Demo

A simple Python demo of the semantic routing that routes queries to appropriate
handlers based on semantic understanding. Works on a CPU or GPU.

## What It Does

This demo implements a semantic router that:

1. **Classifies queries using Intent Classification** - Uses BERT-based
   zero-shot classification to identify the domain/intent of a query (e.g.,
   business, law, health, math, physics, computer science, etc.)

2. **Computes Semantic Embeddings** - Converts text to semantic embeddings using
   a pre-trained model, enabling meaning-based matching

3. **Routes intelligently** - Combines both classification scores (70%) and
   semantic similarity (30%) to make routing decisions

4. **Demonstrates with 14 categories**:
   - 💻 Computer Science (code, algorithms, debugging)
   - 📐 Math (equations, calculations)
   - ⚛️ Physics (physical sciences)
   - ⚖️ Law (legal questions)
   - 🏥 Health (medical information)
   - 💼 Business (management, strategy)
   - And others (psychology, biology, chemistry, history, economics, philosophy,
     engineering)

## Models Used

This demo uses the **similar models as the vLLM Semantic Router**:

- **Intent Classifier**: `bert-base-uncased` (BERT for zero-shot classification)
- **Embedding Model**: `sentence-transformers/all-MiniLM-L6-v2` (fast & accurate
  embeddings)

These can be replaced with other HuggingFace models like:

- Qwen3-Embedding-0.6B (higher quality)
- google/embedding-gemma-300m (alternative)

## Installation

### Requirements

- Python 3.8+
- PyTorch
- Transformers (HuggingFace)
- Sentence Transformers

### Install Dependencies

```bash
uv sync
```

## Usage

### Run the Demo

```bash
uv run semantic_router_production_demo.py
```

### What Happens

1. **Initialization** (~30 seconds first run)
   - Loads the intent classifier (BERT)
   - Loads the embedding model
   - Pre-computes reference embeddings for all 14 categories

2. **Test Phase**
   - Routes 7 example queries
   - Shows category, handler, and confidence scores

3. **Interactive Mode**
   - Enter your own queries
   - See real-time routing decisions
   - Type `quit` to exit

### Example Output

```
📥 Query: How do I debug a Python memory leak?
🎯 Category: COMPUTER_SCIENCE
🤖 Handler: 💻 Code Handler
📊 Confidence: 94.2%
   (Intent: 95.1%, Semantic: 92.8%)
ℹ️  Handles programming, algorithms, debugging
```

## How It Works

### Routing Pipeline

```
User Query
    ↓
[Intent Classification] → BERT zero-shot classification
    ↓
[Semantic Embedding] → Convert to 384-dim vector
    ↓
[Similarity Matching] → Compare with category embeddings
    ↓
[Score Combination] → 70% intent + 30% semantic similarity
    ↓
Route to Best Handler
```

### Classes

- **`IntentClassifier`** - BERT-based domain classification
- **`EmbeddingModel`** - Semantic embeddings using sentence-transformers
- **`SemanticRouter`** - Main router combining both approaches

## Performance

- **Speed**: ~100-200ms per query (after startup)
- **Accuracy**: 85-95% correct category prediction
- **Memory**: ~1-2GB with models loaded
- **GPU**: Auto-detects CUDA availability

## Customization

### Add More Categories

Edit the `reference_texts` dictionary in `SemanticRouter.__init__()`:

```python
reference_texts = {
    'your_category': 'Reference text describing this category',
    ...
}
```

### Change Embedding Model

Modify the `embedding_model` parameter in `main()`:

```python
router = SemanticRouter(
    embedding_model="Qwen/Qwen3-Embedding-0.6B",  # Higher quality
    ...
)
```

### Adjust Weighting

Modify the score combination in `SemanticRouter.route()`:

```python
final_score = (intent_confidence * 0.5) + (semantic_confidence * 0.5)  # 50/50 split
```

## Related

- **vLLM Semantic Router**: https://github.com/vllm-project/semantic-router
- **Documentation**: https://vllm-semantic-router.com
- **Paper**: "When to Reason: Semantic Router for vLLM" (NeurIPS 2025)

## License

This demo is part of the vLLM Semantic Router project - see main repository for
license details.
