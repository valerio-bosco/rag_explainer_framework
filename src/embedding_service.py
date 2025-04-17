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