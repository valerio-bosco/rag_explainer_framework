# rag_evaluation_v2.py
import spacy
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
import itertools
import os # For API keys
import warnings # To suppress warnings

# --- Embedding Model Configuration ---
# Choose your embedding provider and model here
EMBEDDING_CONFIG = {
    # Options: "spacy", "sentence-transformers", "huggingface", "openai"
    "provider": "sentence-transformers",

    # --- Provider Specific Settings ---
    # spacy: Model name (e.g., "en_core_web_sm", "en_core_web_md", "en_core_web_lg")
    "spacy_model_name": "en_core_web_sm",

    # sentence-transformers: Model name from Hugging Face Hub (e.g., "all-MiniLM-L6-v2", "multi-qa-mpnet-base-dot-v1")
    "st_model_name": "all-MiniLM-L6-v2",

    # huggingface: Model name from Hugging Face Hub (e.g., "bert-base-uncased", "roberta-base")
    "hf_model_name": "bert-base-uncased",
    "hf_pooling": "mean", # Options: "mean", "cls"

    # openai: Model name (e.g., "text-embedding-ada-002", "text-embedding-3-small")
    "openai_model_name": "text-embedding-ada-002",
}

# --- Global variables to hold loaded models (lazy loading) ---
_nlp = None
_st_model = None
_hf_tokenizer = None
_hf_model = None
_openai_client = None

# --- Model Loading and Embedding Function ---

def initialize_embedding_model():
    """Loads the embedding model based on EMBEDDING_CONFIG."""
    global _nlp, _st_model, _hf_tokenizer, _hf_model, _openai_client
    provider = EMBEDDING_CONFIG["provider"]

    if provider == "spacy":
        if _nlp is None:
            model_name = EMBEDDING_CONFIG["spacy_model_name"]
            try:
                _nlp = spacy.load(model_name)
                print(f"Loaded spaCy model: {model_name}")
            except OSError:
                print(f"SpaCy model '{model_name}' not found. Please install it: python -m spacy download {model_name}")
                raise
    elif provider == "sentence-transformers":
        if _st_model is None:
            try:
                from sentence_transformers import SentenceTransformer
                model_name = EMBEDDING_CONFIG["st_model_name"]
                _st_model = SentenceTransformer(model_name)
                print(f"Loaded Sentence Transformer model: {model_name}")
            except ImportError:
                print("Sentence Transformers library not found. Please install it: pip install sentence-transformers")
                raise
    elif provider == "huggingface":
        if _hf_model is None or _hf_tokenizer is None:
            try:
                # Suppress warnings about weights not being used for sequence classification
                warnings.filterwarnings("ignore", category=UserWarning, module='transformers.modeling_utils')
                from transformers import AutoTokenizer, AutoModel
                import torch
                warnings.filterwarnings("default", category=UserWarning, module='transformers.modeling_utils') # Reset warning

                model_name = EMBEDDING_CONFIG["hf_model_name"]
                _hf_tokenizer = AutoTokenizer.from_pretrained(model_name)
                _hf_model = AutoModel.from_pretrained(model_name)
                # Use GPU if available
                if torch.cuda.is_available():
                   _hf_model.to('cuda')
                print(f"Loaded Hugging Face model and tokenizer: {model_name}")
            except ImportError:
                print("Hugging Face Transformers library or PyTorch not found. Please install them: pip install transformers torch")
                raise
    elif provider == "openai":
        if _openai_client is None:
            try:
                from openai import OpenAI
                # Requires OPENAI_API_KEY environment variable to be set
                if not os.getenv("OPENAI_API_KEY"):
                    print("Warning: OPENAI_API_KEY environment variable not set for OpenAI embeddings.")
                    # Optionally raise an error here if you require OpenAI
                    # raise ValueError("OPENAI_API_KEY environment variable not set.")
                _openai_client = OpenAI() # Client automatically picks up the key from env var
                print(f"Initialized OpenAI client for model: {EMBEDDING_CONFIG['openai_model_name']}")
            except ImportError:
                print("OpenAI library not found. Please install it: pip install openai")
                raise
    else:
        raise ValueError(f"Unsupported embedding provider: {provider}")

def get_embedding(text):
    """Generates an embedding for the given text using the configured model."""
    initialize_embedding_model() # Ensure model is loaded
    provider = EMBEDDING_CONFIG["provider"]

    if not text or not text.strip(): # Handle empty or whitespace-only strings
      warnings.warn("Attempting to get embedding for empty or whitespace-only text. Returning zero vector.")
      # Return a zero vector of the expected dimension. This is tricky without knowing the model first.
      # Let's try to get the dimension from the loaded model if possible, otherwise default.
      dim = 300 # Default fallback dimension
      if provider == "spacy" and _nlp and _nlp.vocab.vectors_length > 0:
          dim = _nlp.vocab.vectors_length
      elif provider == "sentence-transformers" and _st_model:
          dim = _st_model.get_sentence_embedding_dimension()
      elif provider == "huggingface" and _hf_model:
          dim = _hf_model.config.hidden_size
      elif provider == "openai":
           # OpenAI dims vary by model, ada-002 is 1536, text-embedding-3-small is 1536
           if EMBEDDING_CONFIG["openai_model_name"] in ["text-embedding-ada-002", "text-embedding-3-small"]:
               dim = 1536
           elif EMBEDDING_CONFIG["openai_model_name"] == "text-embedding-3-large":
               dim = 3072
           # Add other OpenAI models or a lookup mechanism if needed
      return np.zeros(dim)


    if provider == "spacy":
        doc = _nlp(text)
        return doc.vector
    elif provider == "sentence-transformers":
        # The encode function expects a list of sentences, but works fine with a single string too
        embedding = _st_model.encode(text)
        return embedding # Already a numpy array
    elif provider == "huggingface":
        import torch
        inputs = _hf_tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
        # Move inputs to GPU if model is on GPU
        if torch.cuda.is_available():
            inputs = {k: v.to('cuda') for k, v in inputs.items()}

        with torch.no_grad():
            outputs = _hf_model(**inputs)

        # Pooling strategy
        if EMBEDDING_CONFIG["hf_pooling"] == "mean":
            # Mean pooling: Average token embeddings, ignore padding
            attention_mask = inputs['attention_mask']
            mask_expanded = attention_mask.unsqueeze(-1).expand(outputs.last_hidden_state.size()).float()
            sum_embeddings = torch.sum(outputs.last_hidden_state * mask_expanded, 1)
            sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
            embedding = sum_embeddings / sum_mask
        elif EMBEDDING_CONFIG["hf_pooling"] == "cls":
            # CLS pooling: Use the embedding of the [CLS] token
            embedding = outputs.last_hidden_state[:, 0, :]
        else:
            raise ValueError(f"Unsupported Hugging Face pooling strategy: {EMBEDDING_CONFIG['hf_pooling']}")

        # Move embedding to CPU and convert to numpy
        return embedding.cpu().numpy().squeeze()

    elif provider == "openai":
        if not _openai_client:
             # Handle case where API key might be missing and client wasn't initialized
             print("Error: OpenAI client not available. Cannot get embedding.")
             # Return zero vector or raise error, similar to empty text handling
             # Using the dimension logic from above
             dim = 1536 # Default for common models
             if EMBEDDING_CONFIG["openai_model_name"] == "text-embedding-3-large":
                 dim = 3072
             return np.zeros(dim)

        response = _openai_client.embeddings.create(
            input=text,
            model=EMBEDDING_CONFIG["openai_model_name"]
        )
        return np.array(response.data[0].embedding)

    else:
        # Should have been caught by initialize_embedding_model, but just in case
        raise ValueError(f"Unsupported embedding provider: {provider}")

# --- Rest of the code (mostly unchanged, but uses the new get_embedding) ---

# Load spaCy *only* for tokenization if not used for embeddings
_tokenizer_nlp = None
def get_tokenizer_nlp():
    """Loads a small spaCy model specifically for tokenization if needed."""
    global _tokenizer_nlp
    if _tokenizer_nlp is None:
        try:
            # Using small model just for tokenization is efficient
            _tokenizer_nlp = spacy.load("en_core_web_sm")
        except OSError:
            print("Default spaCy model 'en_core_web_sm' not found for tokenization. Please install it: python -m spacy download en_core_web_sm")
            # Fallback to basic split if spacy is unavailable
            print("Warning: Falling back to basic whitespace tokenization.")
            _tokenizer_nlp = 'fallback'
    return _tokenizer_nlp

# Default threshold configuration with multiple levels
THRESHOLD_CONFIG = {
    "context": {
        "method": "gaussian_average",  # Options: "gaussian_average", "custom", None
        "custom_value": 0.1,           # Value to use if method is "custom"
        "levels": [0.5, 1.0, 1.5],     # Standard deviations above the mean for levels
        "colors": ['lightcoral', 'salmon', 'red'] # Corresponding colors for levels
    },
    "query": {
        "method": "gaussian_average",  # Options: "gaussian_average", "custom", None
        "custom_value": 0.1,           # Value to use if method is "custom"
        "levels": [0.5, 1.0, 1.5],     # Standard deviations above the mean for levels
        "colors": ['lightblue', 'cornflowerblue', 'blue'] # Corresponding colors for levels
    },
}

def calculate_distance(embedding1, embedding2):
    """Calculates the cosine distance between two embeddings."""
    # Ensure inputs are numpy arrays
    embedding1 = np.asarray(embedding1)
    embedding2 = np.asarray(embedding2)

    # Check for zero vectors
    norm1 = np.linalg.norm(embedding1)
    norm2 = np.linalg.norm(embedding2)
    if norm1 == 0 or norm2 == 0:
        if norm1 == 0 and norm2 == 0:
            return 0.0 # Distance between two zero vectors is 0
        else:
            return 1.0 # Distance between zero vector and non-zero vector is 1 (max cosine distance)

    # Reshape for cosine_similarity function (expects 2D arrays)
    if embedding1.ndim == 1:
        embedding1 = embedding1.reshape(1, -1)
    if embedding2.ndim == 1:
        embedding2 = embedding2.reshape(1, -1)

    # Calculate cosine similarity and convert to distance
    similarity = cosine_similarity(embedding1, embedding2)[0][0]
    # Clip similarity to [-1, 1] range due to potential floating point inaccuracies
    similarity = np.clip(similarity, -1.0, 1.0)
    return 1 - similarity

def tokenize_and_filter(text):
    """Tokenizes text and filters out stop words and punctuation."""
    tokenizer_nlp = get_tokenizer_nlp()
    if tokenizer_nlp == 'fallback':
         # Basic fallback if spaCy isn't available
         tokens = text.lower().split()
         # Very basic stop word list (expand if needed)
         stop_words = {"a", "an", "the", "is", "in", "it", "of", "and", "to"}
         punct = set(".,!?;:'\"()[]{}")
         filtered_tokens = [t for t in tokens if t not in stop_words and not all(c in punct for c in t)]
         return filtered_tokens

    # Use spaCy for robust tokenization
    doc = tokenizer_nlp(text)
    filtered_tokens = [token.text for token in doc if not token.is_stop and not token.is_punct and not token.is_space]
    return filtered_tokens

def rag_simulate(context, query):
    """
    A very basic simulation of a RAG system.
    In a real system, this would involve retrieving relevant parts of the context
    based on the query and then generating an answer.
    For this explainability task, we'll simply concatenate them (with a separator).
    This allows us to observe how changes in context/query affect the embedding
    of this "simulated" response.
    """
    return f"Context: {context} Query: {query}"

def remove_and_compute(sentence_tokens, original_response_embedding, category, token_impact_dict, original_query):
    """
    Computes the impact of removing each token.
    Note: This implementation measures the change in the embedding of the *remaining text*
    compared to the *original full response embedding*. This is one way to interpret impact.
    An alternative would be to re-simulate the RAG response with the modified text.
    """
    for i, token_to_remove in enumerate(sentence_tokens):
        # Create the text with the token removed
        temp_text_list = [t for j, t in enumerate(sentence_tokens) if i != j]
        temp_text = " ".join(temp_text_list)

        if temp_text.strip():
             # Get embedding of the text *after* removing the token
            temp_embedding = get_embedding(temp_text)

            # Calculate distance change compared to the *original* response embedding
            distance_change = calculate_distance(temp_embedding, original_response_embedding)
            token_impact_dict[f"{category}_{token_to_remove}"] = distance_change
        else:
            # If removing the token leaves the text empty
             token_impact_dict[f"{category}_{token_to_remove}"] = calculate_distance(get_embedding(""), original_response_embedding) # Distance from empty embedding


def explain_by_token_removal(context, query, response):
    """
    Explains the influence of each token in the context and query
    by measuring the change in embedding distance of the response
    when each token is removed.
    """
    response_embedding = get_embedding(response) # Embedding of the actual RAG response
    context_tokens = tokenize_and_filter(context)
    query_tokens = tokenize_and_filter(query)
    token_impact = defaultdict(float)

    print("Analyzing context tokens...")
    # Pass the original response embedding for comparison
    remove_and_compute(context_tokens, response_embedding, "context", token_impact, query)

    print("Analyzing query tokens...")
    # Pass the original response embedding for comparison
    remove_and_compute(query_tokens, response_embedding, "query", token_impact, context) # Pass context if needed by alternative method

    return token_impact

def explain_by_context_permutation(context, query, response, num_permutations=5):
    """
    Explains the influence of context syntactic formulation by measuring
    the change in embedding distance of the response for different
    permutations of the context sentences.
    """
    response_embedding = get_embedding(response) # Original response embedding

    # Use spaCy for sentence splitting
    tokenizer_nlp = get_tokenizer_nlp()
    if tokenizer_nlp == 'fallback':
        print("Warning: Cannot perform sentence permutation analysis without spaCy. Skipping.")
        return {}
    sentences = [sent.text.strip() for sent in tokenizer_nlp(context).sents if sent.text.strip()]

    if len(sentences) < 2:
        print("Context has less than 2 sentences. Skipping permutation analysis.")
        return {}

    permutation_impact = defaultdict(float)
    unique_permutations = set() # To avoid redundant permutations if num_permutations is large
    original_context_order = " ".join(sentences)

    # Ensure the original order is considered if possible within the limit
    unique_permutations.add(original_context_order)

    all_perms = list(itertools.permutations(sentences))
    np.random.shuffle(all_perms) # Shuffle to get varied permutations if num_permutations is smaller than total

    count = 0
    for perm_tuple in all_perms:
        permuted_context = " ".join(perm_tuple)
        if permuted_context not in unique_permutations:
            unique_permutations.add(permuted_context)
            count += 1
            if count >= num_permutations: # Stop after reaching the desired number *excluding* the original
                 break

    # If original wasn't added due to limit, remove one random to make space if needed
    if original_context_order not in unique_permutations and len(unique_permutations) >= num_permutations:
         unique_permutations.pop() # Remove a random element
         unique_permutations.add(original_context_order)


    print(f"Analyzing {len(unique_permutations)} context permutations (including original)...")
    for perm_context in unique_permutations:
        # Simulate RAG with the permuted context
        temp_simulated_response = rag_simulate(perm_context, query)
        temp_response_embedding = get_embedding(temp_simulated_response)

        # Compare the embedding of the *simulated response from permuted context*
        # to the embedding of the *original actual response*
        distance_change = calculate_distance(temp_response_embedding, response_embedding)
        permutation_impact[perm_context] = distance_change # Store distance, not change from original perm

    return permutation_impact


def calculate_token_thresholds_with_levels(token_impact, threshold_config):
    """Calculates separate thresholds with multiple levels for context and query tokens."""
    context_impacts = [v for k, v in token_impact.items() if k.startswith("context_")]
    query_impacts = [v for k, v in token_impact.items() if k.startswith("query_")]
    thresholds = {}

    def get_thresholds_and_colors(impacts, config):
        method = config.get("method")
        levels = config.get("levels", [])
        colors = config.get("colors", [])
        threshold_colors = []
        if not impacts: # Handle cases with no tokens
            return [] # Return empty list if no impacts to calculate from

        # Ensure impacts are numbers
        impacts = [float(i) for i in impacts if isinstance(i, (int, float, np.number))]
        if not impacts: # Handle case where impacts were not numbers
             return []

        if method == "gaussian_average":
            avg = np.mean(impacts)
            std = np.std(impacts)
            # Ensure std is not zero or extremely small to avoid issues
            if std < 1e-9:
                std = 1e-9 # Use a small epsilon if std dev is effectively zero

            for i, level in enumerate(levels):
                threshold = avg + level * std
                color = colors[i] if i < len(colors) else 'gray'
                threshold_colors.append((threshold, color))
            return sorted(threshold_colors) # Sort by threshold value ascending
        elif method == "custom":
            custom_value = config.get("custom_value", 0.0) # Default custom base to 0
            for i, level in enumerate(levels):
                # Assuming levels for custom are additive offsets from custom_value
                threshold = custom_value + level
                color = colors[i] if i < len(colors) else 'gray'
                threshold_colors.append((threshold, color))
            return sorted(threshold_colors) # Sort by threshold value ascending
        elif method is None:
             return [] # No thresholds calculated
        else:
            print(f"Warning: Unknown threshold method '{method}'. No thresholds calculated.")
            return []

    thresholds["context"] = get_thresholds_and_colors(context_impacts, threshold_config.get("context", {}))
    thresholds["query"] = get_thresholds_and_colors(query_impacts, threshold_config.get("query", {}))
    return thresholds

# Main analysis function
def analyze_rag(context, query, response, analyze_permutations=True, num_permutations=5):
    """
    Performs a complete RAG analysis and returns the results.

    Args:
        context: The context text for RAG
        query: The query text
        response: The response generated by the RAG system
        analyze_permutations: Whether to analyze context permutations
        num_permutations: Number of permutations to analyze if enabled

    Returns:
        Dictionary containing token_impact, permutation_impact, and thresholds
    """
    print(f"--- Starting RAG Analysis ---")
    print(f"Embedding Provider: {EMBEDDING_CONFIG['provider']}")
    # Print specific model name based on provider
    provider = EMBEDDING_CONFIG['provider']
    model_key = f"{provider.split('-')[0]}_model_name" # e.g. spacy_model_name, st_model_name
    if model_key in EMBEDDING_CONFIG:
      print(f"Embedding Model: {EMBEDDING_CONFIG[model_key]}")
    elif provider == 'huggingface':
        print(f"Embedding Model: {EMBEDDING_CONFIG['hf_model_name']} (Pooling: {EMBEDDING_CONFIG['hf_pooling']})")


    # Initialize the model before analysis starts
    try:
        initialize_embedding_model()
    except Exception as e:
        print(f"Error initializing embedding model: {e}")
        # Return partial results or raise error
        return {
            "error": f"Failed to initialize embedding model: {e}",
            "token_impact": {},
            "permutation_impact": {},
            "thresholds": {},
            "context": context,
            "query": query,
            "response": response,
            "threshold_config": THRESHOLD_CONFIG,
            "embedding_config": EMBEDDING_CONFIG
        }


    # Analyze token impact
    token_impact = explain_by_token_removal(context, query, response)

    # Analyze permutations if requested
    permutation_impact = {}
    if analyze_permutations:
        permutation_impact = explain_by_context_permutation(
            context, query, response, num_permutations=num_permutations
        )

    # Calculate thresholds
    thresholds = calculate_token_thresholds_with_levels(token_impact, THRESHOLD_CONFIG)

    print(f"--- Analysis Complete ---")

    return {
        "token_impact": token_impact,
        "permutation_impact": permutation_impact,
        "thresholds": thresholds,
        "context": context,
        "query": query,
        "response": response,
        "threshold_config": THRESHOLD_CONFIG,
        "embedding_config": EMBEDDING_CONFIG # Include embedding config in results
    }

# --- Example Usage ---
if __name__ == "__main__":
    # Sample Data
    sample_context = (
        "The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris, France. "
        "It is named after the engineer Gustave Eiffel, whose company designed and built the tower. "
        "Constructed from 1887 to 1889 as the entrance to the 1889 World's Fair, it was initially criticized by some of France's leading artists and intellectuals for its design, "
        "but it has become a global cultural icon of France and one of the most recognizable structures in the world."
    )
    sample_query = "Who designed the Eiffel Tower?"
    # This response *should* ideally be generated by your actual RAG system
    # For this example, we'll use a plausible response.
    sample_response = "The Eiffel Tower was designed and built by the company of the engineer Gustave Eiffel."

    # --- Select and Configure Embedding Model ---
    # Option 1: Sentence Transformers (Default in EMBEDDING_CONFIG above)
    EMBEDDING_CONFIG["provider"] = "sentence-transformers"
    EMBEDDING_CONFIG["st_model_name"] = "all-MiniLM-L6-v2"

    # Option 2: Hugging Face with Mean Pooling
    # EMBEDDING_CONFIG["provider"] = "huggingface"
    # EMBEDDING_CONFIG["hf_model_name"] = "bert-base-uncased"
    # EMBEDDING_CONFIG["hf_pooling"] = "mean"

    # Option 3: SpaCy (Medium model)
    # EMBEDDING_CONFIG["provider"] = "spacy"
    # EMBEDDING_CONFIG["spacy_model_name"] = "en_core_web_md" # Make sure to download: python -m spacy download en_core_web_md

    # Option 4: OpenAI (Requires API Key in environment)
    # Set OPENAI_API_KEY environment variable first
    # EMBEDDING_CONFIG["provider"] = "openai"
    # EMBEDDING_CONFIG["openai_model_name"] = "text-embedding-3-small" # Or "text-embedding-ada-002"


    # Perform analysis
    analysis_results = analyze_rag(sample_context, sample_query, sample_response, analyze_permutations=True, num_permutations=5)

    # Print results (or use them for visualization)
    print("\n--- Analysis Results ---")
    print("\nToken Impact:")
    # Sort token impact for better readability
    sorted_token_impact = sorted(analysis_results["token_impact"].items(), key=lambda item: item[1], reverse=True)
    for token, impact in sorted_token_impact:
        print(f"  {token}: {impact:.6f}")

    print("\nPermutation Impact:")
    # Sort permutation impact for better readability
    sorted_perm_impact = sorted(analysis_results["permutation_impact"].items(), key=lambda item: item[1], reverse=True)
    for context_perm, impact in sorted_perm_impact:
         # Only print the first 80 chars of the context for brevity
        print(f"  '{context_perm[:80]}...': {impact:.6f}")

    print("\nCalculated Thresholds:")
    print(f"  Context Thresholds: {analysis_results['thresholds']['context']}")
    print(f"  Query Thresholds: {analysis_results['thresholds']['query']}")

    print("\nEmbedding Configuration Used:")
    print(f"  {analysis_results['embedding_config']}")