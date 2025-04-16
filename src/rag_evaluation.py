# rag_evaluation.py
import spacy
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
import itertools

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Default threshold configuration with multiple levels
THRESHOLD_CONFIG = {
    "context": {
        "method": "gaussian_average",  # Options: "gaussian_average", "custom", None
        "custom_value": 0.1,          # Value to use if method is "custom"
        "levels": [0.5, 1.0, 1.5],     # Standard deviations above the mean for levels
        "colors": ['lightcoral', 'salmon', 'red'] # Corresponding colors for levels
    },
    "query": {
        "method": "gaussian_average",  # Options: "gaussian_average", "custom", None
        "custom_value": 0.1,          # Value to use if method is "custom"
        "levels": [0.5, 1.0, 1.5],     # Standard deviations above the mean for levels
        "colors": ['lightblue', 'cornflowerblue', 'blue'] # Corresponding colors for levels
    },
}

def get_embedding(text):
    """Generates an embedding for the given text using spaCy."""
    doc = nlp(text)
    return doc.vector

def calculate_distance(embedding1, embedding2):
    """Calculates the cosine distance between two embeddings."""
    if np.linalg.norm(embedding1) == 0 or np.linalg.norm(embedding2) == 0:
        return 1.0  # Return max distance if either embedding is zero
    return 1 - cosine_similarity([embedding1], [embedding2])[0][0]

def tokenize_and_filter(text):
    """Tokenizes text and filters out stop words and punctuation."""
    doc = nlp(text)
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

def remove_and_compute(sentence_tokens, response_embedding, category, token_impact_dict):
    for i, token in enumerate(sentence_tokens):
        temp_context = " ".join([t for j, t in enumerate(sentence_tokens) if i != j])
        if temp_context.strip():
            temp_response_embedding = get_embedding(temp_context)#rag_simulate(temp_context, query)) # Simulate RAG
            distance_change = calculate_distance(temp_response_embedding, response_embedding)
            token_impact_dict[f"{category}_{token}"] = distance_change
        else:
            token_impact_dict[f"{category}_{token}"] = 0.0 # If removing the only token, no context
    
def explain_by_token_removal(context, query, response):
    """
    Explains the influence of each token in the context and query
    by measuring the change in embedding distance of the response
    when each token is removed.
    """
    response_embedding = get_embedding(response)
    context_tokens = tokenize_and_filter(context)
    query_tokens = tokenize_and_filter(query)
    token_impact = defaultdict(float)

    print("Analyzing context tokens...")
    remove_and_compute(context_tokens, response_embedding, "context", token_impact)
    print("Analyzing query tokens...")
    remove_and_compute(query_tokens, response_embedding, "query", token_impact)

    return token_impact

def explain_by_context_permutation(context, query, response, num_permutations=5):
    """
    Explains the influence of context syntactic formulation by measuring
    the change in embedding distance of the response for different
    permutations of the context sentences.
    """
    response_embedding = get_embedding(response)
    sentences = [sent.text.strip() for sent in nlp(context).sents if sent.text.strip()]
    if len(sentences) < 2:
        return {}

    permutation_impact = defaultdict(float)
    permutations = set() # To avoid redundant permutations

    for perm_tuple in itertools.permutations(sentences):
        permutations.add(" ".join(perm_tuple))
        if len(permutations) >= num_permutations:
            break

    print(f"Analyzing {len(permutations)} context permutations...")
    for permuted_context in permutations:
        temp_response_embedding = get_embedding(rag_simulate(permuted_context, query))
        distance_change = calculate_distance(temp_response_embedding, response_embedding)
        permutation_impact[permuted_context] = distance_change

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
        if not impacts:
            return []
        if method == "gaussian_average":
            avg = np.mean(impacts)
            std = np.std(impacts)
            for i, level in enumerate(levels):
                threshold = avg + level * std
                color = colors[i] if i < len(colors) else 'gray'
                threshold_colors.append((threshold, color))
            return sorted(threshold_colors)
        elif method == "custom":
            custom_value = config.get("custom_value", -float('inf'))
            for i, level in enumerate(levels):
                threshold = custom_value + level
                color = colors[i] if i < len(colors) else 'gray'
                threshold_colors.append((threshold, color))
            return sorted(threshold_colors)
        else:
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
    
    return {
        "token_impact": token_impact,
        "permutation_impact": permutation_impact,
        "thresholds": thresholds,
        "context": context,
        "query": query,
        "response": response,
        "threshold_config": THRESHOLD_CONFIG
    }