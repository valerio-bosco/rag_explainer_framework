import spacy
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
import itertools
import re

# Load a smaller spaCy model for efficiency
nlp = spacy.load("en_core_web_sm")

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
    for i, token in enumerate(context_tokens):
        temp_context = " ".join([t for j, t in enumerate(context_tokens) if i != j])
        if temp_context.strip():
            temp_response_embedding = get_embedding(rag_simulate(temp_context, query)) # Simulate RAG
            distance_change = calculate_distance(temp_response_embedding, response_embedding)
            token_impact[f"context_{token}"] = distance_change
        else:
            token_impact[f"context_{token}"] = 0.0 # If removing the only token, no context

    print("Analyzing query tokens...")
    for i, token in enumerate(query_tokens):
        temp_query = " ".join([t for j, t in enumerate(query_tokens) if i != j])
        if temp_query.strip():
            temp_response_embedding = get_embedding(rag_simulate(context, temp_query)) # Simulate RAG
            distance_change = calculate_distance(temp_response_embedding, response_embedding)
            token_impact[f"query_{token}"] = distance_change
        else:
            token_impact[f"query_{token}"] = 0.0 # If removing the only token, no query

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

def visualize_results_html(token_impact, context, query, permutation_impact=None):
    """Generates an HTML visualization of the token and permutation impact."""
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>RAG Explainability Analysis</title>
        <style>
            body { font-family: sans-serif; margin: 20px; }
            h1, h2 { color: #333; }
            .explanation-section { margin-bottom: 20px; border: 1px solid #eee; padding: 15px; border-radius: 5px; }
            .token-highlight { cursor: help; border-radius: 3px; padding: 0.1em 0.3em; color: black; }
            .impact-high { background-color: rgba(255, 0, 0, 0.7); }
            .impact-medium { background-color: rgba(255, 165, 0, 0.5); }
            .impact-low { background-color: rgba(255, 255, 0, 0.3); }
            .impact-min { background-color: rgba(0, 128, 0, 0.3); } /* Green for minimum */
            .legend { margin-top: 10px; padding: 10px; border: 1px solid #ddd; border-radius: 5px; background-color: #f9f9f9; }
            .legend-item { display: inline-block; margin-right: 15px; }
            .legend-color { width: 20px; height: 15px; display: inline-block; vertical-align: middle; margin-right: 5px; border-radius: 3px; }
            .permutation-item { margin-bottom: 5px; padding: 8px; border: 1px solid #eee; border-radius: 3px; }
            .permutation-item strong { font-weight: bold; color: #555; }
        </style>
    </head>
    <body>
        <h1>RAG Explainability Analysis</h1>

        <div class="explanation-section">
            <h2>Token Impact</h2>
            <p>The following highlights indicate the influence of each token on the response. Hover over a highlighted word to see its impact percentage.</p>
            <div>
                <h3>Context:</h3>
                <p>
    """
    context_tokens = nlp(context)
    context_impacts = {k.split("context_")[1]: v for k, v in token_impact.items() if k.startswith("context_")}
    if context_impacts:
        max_context_impact = max(context_impacts.values())
        min_context_impact = min(context_impacts.values())
        range_context_impact = max_context_impact - min_context_impact if max_context_impact > min_context_impact else 1
    else:
        max_context_impact = min_context_impact = range_context_impact = 1

    for token in context_tokens:
        if not token.is_stop and not token.is_punct and not token.is_space and token.text in context_impacts:
            impact = context_impacts[token.text]
            normalized_impact = (impact - min_context_impact) / range_context_impact if range_context_impact > 0 else 0.5
            percentage = f"{normalized_impact * 100:.2f}%"
            if normalized_impact > 0.7:
                highlight_class = "impact-high"
            elif normalized_impact > 0.3:
                highlight_class = "impact-medium"
            elif normalized_impact == 0 and len(context_impacts) > 1: # Mark minimum if there are multiple
                highlight_class = "impact-min"
            else:
                highlight_class = "impact-low"
            html += f"<span class='token-highlight {highlight_class}' title='Impact: {percentage}'>{token.text}</span> "
        else:
            html += f"{token.text} "
    html += "</p>"

    html += """
                <h3>Query:</h3>
                <p>
    """
    query_tokens = nlp(query)
    query_impacts = {k.split("query_")[1]: v for k, v in token_impact.items() if k.startswith("query_")}
    if query_impacts:
        max_query_impact = max(query_impacts.values())
        min_query_impact = min(query_impacts.values())
        range_query_impact = max_query_impact - min_query_impact if max_query_impact > min_query_impact else 1
    else:
        max_query_impact = min_query_impact = range_query_impact = 1

    print("min query impact", min_query_impact)
    for token in query_tokens:
        if not token.is_stop and not token.is_punct and not token.is_space and token.text in query_impacts:
            impact = query_impacts[token.text]
            print("impact of", token, ":" , impact)
            normalized_impact = (impact - min_query_impact) / range_query_impact if range_query_impact > 0 else 0.5
            percentage = f"{normalized_impact * 100:.2f}%"
            if normalized_impact > 0.7:
                highlight_class = "impact-high"
            elif normalized_impact > 0.3:
                highlight_class = "impact-medium"
            elif normalized_impact == 0 and len(query_impacts) > 1: # Mark minimum if there are multiple
                highlight_class = "impact-min"
            else:
                highlight_class = "impact-low"
            html += f"<span class='token-highlight {highlight_class}' title='Impact: {percentage}'>{token.text}</span> "
        else:
            html += f"{token.text} "
    html += "</p>"

    html += """
            </div>
            <div class="legend">
                <h3>Legend: Token Impact</h3>
                <div class="legend-item"><span class="legend-color impact-high"></span> High Influence</div>
                <div class="legend-item"><span class="legend-color impact-medium"></span> Medium Influence</div>
                <div class="legend-item"><span class="legend-color impact-low"></span> Low Influence</div>
                <div class="legend-item"><span class="legend-color impact-min"></span> Minimum Influence</div>
            </div>
        </div>

        <div class="explanation-section">
            <h2>Context Permutation Impact (if analyzed)</h2>
    """
    if permutation_impact:
        sorted_permutations = sorted(permutation_impact.items(), key=lambda item: item[1], reverse=True)
        for context_perm, distance in sorted_permutations:
            html += f"<div class='permutation-item'><strong>Distance: {distance:.4f}</strong> - Context: {context_perm}</div>"
    else:
        html += "<p>Context permutation analysis was not performed.</p>"

    html += """
        </div>

    </body>
    </html>
    """
    return html

if __name__ == "__main__":
    context = "Marie Curie was a physicist and chemist who conducted pioneering research on radioactivity. She was the first woman to win a Nobel Prize, and the only person to win Nobel Prizes in two different scientific fields: Physics and Chemistry. Her work led to the development of X-rays in surgery, and she discovered the elements polonium and radium."
    query = "Who discovered radium and what was their contribution to medicine?"
    response = "Radium was discovered by Marie Curie, whose research on radioactivity significantly advanced medical imaging, especially through the development of X-rays used in surgery."


    # Explain by token removal
    token_impact = explain_by_token_removal(context, query, response)
    print("\nToken Impact:")
    for token, impact in sorted(token_impact.items(), key=lambda item: item[1], reverse=True):
        print(f"{token}: {impact:.4f}")

    # Explain by context permutation (optional)
    analyze_permutations = True
    permutation_impact = {}
    if analyze_permutations:
        permutation_impact = explain_by_context_permutation(context, query, response, num_permutations=5)
        print("\nContext Permutation Impact:")
        for perm, impact in sorted(permutation_impact.items(), key=lambda item: item[1], reverse=True):
            print(f"{impact:.4f}: {perm[:50]}...") # Print first 50 chars

    # Visualize results in HTML
    html_output = visualize_results_html(token_impact, context, query, permutation_impact)
    with open("rag_explanation.html", "w") as f:
        f.write(html_output)
    print("\nExplanation saved to rag_explanation.html")