
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
import itertools
import embedding_service as es
import hashlib

from rag_thread import RagThread

# def tokenize_and_filter(text):
    # """Tokenizes text and filters out stop words and punctuation."""
    # tokenizer_nlp = es.get_tokenizer_nlp()
    # if tokenizer_nlp == 'fallback':
    #      # Basic fallback if spaCy isn't available
    #      tokens = text.lower().split()
    #      # Very basic stop word list (expand if needed)
    #      stop_words = {"a", "an", "the", "is", "in", "it", "of", "and", "to"}
    #      punct = set(".,!?;:'\"()[]{}")
    #      filtered_tokens = [t for t in tokens if t not in stop_words and not all(c in punct for c in t)]
    #      return filtered_tokens

    # # Use spaCy for robust tokenization
    # doc = tokenizer_nlp(text)
    # filtered_tokens = [token.text for token in doc if not token.is_stop and not token.is_punct and not token.is_space]
    # return filtered_tokens

def tokenize_and_filter(text):
    """Splits text into sentences instead of tokens."""
    tokenizer_nlp = es.get_tokenizer_nlp()
    
    if tokenizer_nlp == 'fallback':
        # Basic fallback if spaCy isn't available
        # Simple sentence splitting by punctuation and cleanup
        import re
        
        # Split on common sentence endings
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        # Remove empty sentences and strip whitespace
        sentences = [sentence.strip() for sentence in sentences if sentence.strip()]
        
        return sentences
    
    # Use spaCy for more robust sentence tokenization
    doc = tokenizer_nlp(text)
    sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
    
    return sentences
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

# def remove_and_compute(sentence_tokens, original_response_embedding, category, token_impact_dict, original_query):
    """
    Computes the impact of removing each token.
    Note: This implementation measures the change in the embedding of the *remaining text*
    compared to the *original full response embedding*. This is one way to interpret impact.
    An alternative would be to re-simulate the RAG response with the modified text.
    """
    # temp_dict = defaultdict(list)
    # for i, token_to_remove in enumerate(sentence_tokens):
    #     # Create the text with the token removed
    #     temp_text_list = [t for j, t in enumerate(sentence_tokens) if i != j]
    #     temp_text = " ".join(temp_text_list)

    #     impact_values = []
    #     if temp_text.strip():
    #          # Get embedding of the text *after* removing the token
    #         temp_embedding = es.get_embedding(temp_text)

    #         # Calculate distance change compared to the *original* response embedding
    #         distance_change = es.calculate_distance(temp_embedding, original_response_embedding)
    #         key = f"{category}_{token_to_remove}"
    #         if(temp_dict[key]):
    #             temp_dict[key].append(distance_change)
                
    #         else:
    #             temp_dict[key] = [distance_change]
    #         impact_values.append(distance_change)
                
    #     else:
    #         # If removing the token leaves the text empty
    #          temp_dict[f"{category}_{token_to_remove}"] = es.calculate_distance(es.get_embedding(""), original_response_embedding) # Distance from empty embedding

    # mean_impact = np.mean(impact_values)
     
    # print(mean_impact)
    # for key in temp_dict.keys():
    #     if(token_impact_dict[key]):
    #          token_impact_dict[key].append((np.mean(temp_dict[key]) - mean_impact))
    #     else:
    #         token_impact_dict[key]=[(np.mean(temp_dict[key]) - mean_impact)]

def remove_and_compute(sentences, original_response_embedding, category, sentence_impact_dict, original_query):
    """
    Computes the impact of removing each sentence.
    Uses hash of sentences as dictionary keys for consistent lookup.
    """
    impact_values = []
    new_sentence_impact_dict = defaultdict(list)
    
    # Process each sentence
    for i, sentence_to_remove in enumerate(sentences):
        # Create a hash of the sentence for consistent key generation
        sentence_hash = hashlib.md5(sentence_to_remove.encode()).hexdigest()
        
        # Create text with the current sentence removed
        remaining_sentences = [s for j, s in enumerate(sentences) if i != j]
        remaining_text = " ".join(remaining_sentences)

        text_to_evaluate = ""
        if remaining_text.strip():
            text_to_evaluate = remaining_text
    
        text_to_evaluate_embedding = es.get_embedding(text_to_evaluate)
        distance_change = es.calculate_distance(text_to_evaluate_embedding, original_response_embedding)
        
        key = f"{category}_{sentence_hash}"
        if key in new_sentence_impact_dict:
            new_sentence_impact_dict[key].append(distance_change)
        else:
            new_sentence_impact_dict[key] = [distance_change]
            
        impact_values.append(distance_change)
       
    # Calculate mean impact across all sentences
    if impact_values:
        mean_impact = np.mean(impact_values)
        std_impact = np.std(impact_values) if np.std(impact_values) > 0 else 1

        max_impact = 1
        # Normalize the impacts relative to the mean
        for key, values in new_sentence_impact_dict.items():
            normalized_impact = (np.mean(values) - mean_impact) / (std_impact)  
            if normalized_impact > max_impact:
                max_impact = normalized_impact          
            if(key in sentence_impact_dict):
                sentence_impact_dict[key].append(normalized_impact)
            else:
                sentence_impact_dict[key] = [normalized_impact]
        
        for key, values in sentence_impact_dict.items():
            sentence_impact_dict[key] = [v / max_impact for v in values]  # Normalize by max impact (to make the range in -1, 1)


    
def explain_by_token_removal(context, query, response):
    """
    Explains the influence of each token in the context and query
    by measuring the change in embedding distance of the response
    when each token is removed.
    """
    response_embedding = es.get_embedding(response) # Embedding of the actual RAG response
    query_embedding = es.get_embedding(query)
    
    context_sentences = tokenize_and_filter(context)
    query_sentences = tokenize_and_filter(query)
    
    token_impact = defaultdict(list)

    print("Analyzing context tokens wrt response...")
    remove_and_compute(context_sentences, response_embedding, "context", token_impact, query)
    
    print("Analyzing context tokens wrt query...")
    remove_and_compute(context_sentences, query_embedding, "context", token_impact, query)

    print("Analyzing query tokens wrt response...")
    remove_and_compute(query_sentences, response_embedding, "query", token_impact, context) 
   
    for key in  token_impact.keys():
        token_impact[key] = np.mean(token_impact[key])
    

    return token_impact

def explain_by_context_permutation(context, query, response, num_permutations=5):
    """
    Explains the influence of context syntactic formulation by measuring
    the change in embedding distance of the response for different
    permutations of the context sentences.
    """
    response_embedding = es.get_embedding(response) # Original response embedding
    context_hash = hashlib.md5(context.encode()).hexdigest() # Hash for consistent key generation

    # Use spaCy for sentence splitting
    tokenizer_nlp = es.get_tokenizer_nlp()
    if tokenizer_nlp == 'fallback':
        print("Warning: Cannot perform sentence permutation analysis without spaCy. Skipping.")
        return {}
    sentences = [sent.text.strip() for sent in tokenizer_nlp(context).sents if sent.text.strip()]

    if len(sentences) < 2:
        print("Context has less than 2 sentences. Skipping permutation analysis.")
        return {}

    permutation_impact = defaultdict(float)
    unique_permutations = set() # To avoid redundant permutations if num_permutations is large


    # Ensure the original order is considered if possible within the limit
    unique_permutations.add(context)

    all_perms = []
    for i in range(5):
        all_perms.append(np.random.permutation(sentences))
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
    if context not in unique_permutations and len(unique_permutations) >= num_permutations:
         unique_permutations.pop() # Remove a random element
         unique_permutations.add(context)


    print(f"Analyzing {len(unique_permutations)} context permutations (including original)...")
    for perm_context in unique_permutations:
        # Simulate RAG with the permuted context
        temp_simulated_response = rag_simulate(perm_context, query)
        temp_response_embedding = es.get_embedding(temp_simulated_response)

        # Compare the embedding of the *simulated response from permuted context*
        # to the embedding of the *original actual response*
        distance_change =es.calculate_distance(temp_response_embedding, response_embedding)
        permutation_hash = hashlib.md5(perm_context.encode()).hexdigest() # Hash for consistent key generation
        if permutation_hash == context_hash:
            permutation_impact["original"] = distance_change # Store distance, not change from original perm
        else:
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
                threshold = avg + level 
                print(f"Threshold {i}: {threshold:.8f} (Level: {level})")
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
    print(f"Embedding Provider: {es.EMBEDDING_CONFIG['provider']}")
    # Print specific model name based on provider
    provider = es.EMBEDDING_CONFIG['provider']
    model_key = f"{provider.split('-')[0]}_model_name" # e.g. spacy_model_name, st_model_name
    if model_key in es.EMBEDDING_CONFIG:
      print(f"Embedding Model: {es.EMBEDDING_CONFIG[model_key]}")
    elif provider == 'huggingface':
        print(f"Embedding Model: {es.EMBEDDING_CONFIG['hf_model_name']} (Pooling: {es.EMBEDDING_CONFIG['hf_pooling']})")


    # Initialize the model before analysis starts
    try:
        es.initialize_embedding_model()
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
            "threshold_config": es.THRESHOLD_CONFIG,
            "embedding_config": es.EMBEDDING_CONFIG
        }

    token_impact = defaultdict(float)
    #multi-threading for each chunk of context which is an array of chunks
    threads = []
    for i, chunk in enumerate(context):
        print(f"Processing chunk {i+1}/{len(context)}: {chunk[:50]}...")
        thread = RagThread(target=explain_by_token_removal, args=(chunk, query, response))
        thread.start()
        threads.append(thread)
        
    for i in range(len(context)):
        test = threads[i].join()
        token_impact.update(test)
        

    # Analyze permutations if requested
    permutation_impact = {}
    if analyze_permutations:
        permutation_impact = explain_by_context_permutation(
            context, query, response, num_permutations=num_permutations
        )

    # Calculate thresholds
    thresholds = calculate_token_thresholds_with_levels(token_impact, es.THRESHOLD_CONFIG)
    print(f"--- Analysis Complete ---")

    return {
        "token_impact": token_impact,
        "permutation_impact": permutation_impact,
        "thresholds": thresholds,
        "context": context,
        "query": query,
        "response": response,
        "threshold_config": es.THRESHOLD_CONFIG,
        "embedding_config": es.EMBEDDING_CONFIG # Include embedding config in results
    }