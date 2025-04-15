import spacy
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
import itertools
import re
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA

# Load a larger spaCy model for potentially better embeddings
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

def get_token_level_color(impact, thresholds):
    """Determines the color for a token based on its impact and defined levels."""
    for threshold, color in thresholds:
        if impact > threshold:
            return color
    return None

def plot_token_embeddings(context, query, response, token_impact, threshold_config):
    """Plots the embeddings of individual tokens from context, query, and response colored by their impact level."""
    context_doc = nlp(context)
    query_doc = nlp(query)
    response_doc = nlp(response)

    thresholds = calculate_token_thresholds_with_levels(token_impact, threshold_config)
    context_thresholds = thresholds.get("context", [])
    query_thresholds = thresholds.get("query", [])

    tokens_to_plot = []
    embeddings_to_plot = []
    colors_to_plot = []
    labels_to_plot = []

    processed_tokens = set() # To avoid duplicate tokens

    for token in context_doc:
        if not token.is_stop and not token.is_punct and not token.is_space and f"context_{token.text}" in token_impact and token.text not in processed_tokens:
            impact = token_impact[f"context_{token.text}"]
            color = get_token_level_color(impact, context_thresholds)
            if color:
                tokens_to_plot.append(token.text)
                embeddings_to_plot.append(token.vector)
                colors_to_plot.append(color)
                labels_to_plot.append('Context')
                processed_tokens.add(token.text)

    for token in query_doc:
        if not token.is_stop and not token.is_punct and not token.is_space and f"query_{token.text}" in token_impact :#and token.text not in processed_tokens:
            impact = token_impact[f"query_{token.text}"]
            color = get_token_level_color(impact, query_thresholds)
            if color:
                tokens_to_plot.append(token.text)
                embeddings_to_plot.append(token.vector)
                colors_to_plot.append(color)
                labels_to_plot.append('Query')
                processed_tokens.add(token.text)

    for token in response_doc:
        if not token.is_stop and not token.is_punct and not token.is_space :#and token.text not in processed_tokens:
            tokens_to_plot.append(token.text)
            embeddings_to_plot.append(token.vector)
            colors_to_plot.append('green')
            labels_to_plot.append('Response')
            processed_tokens.add(token.text)

    if not embeddings_to_plot:
        print(f"No tokens above the specified thresholds for plotting.")
        return

    embeddings_array = np.array(embeddings_to_plot)

    # Reduce dimensionality for 2D plotting
    pca_2d = PCA(n_components=2)
    embeddings_2d = pca_2d.fit_transform(embeddings_array)

    # Reduce dimensionality for 3D plotting
    pca_3d = PCA(n_components=3)
    embeddings_3d = pca_3d.fit_transform(embeddings_array)

    # 2D Plot
    plt.figure(figsize=(10, 10))
    scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=colors_to_plot, s=50, alpha=0.7)
    for i, token in enumerate(tokens_to_plot):
        plt.annotate(token, (embeddings_2d[i, 0], embeddings_2d[i, 1]), textcoords="offset points", xytext=(5, 5), ha='center', fontsize=8)

    legend_handles = []
    legend_labels = []
    context_colors = THRESHOLD_CONFIG.get("context", {}).get("colors", [])
    context_levels = THRESHOLD_CONFIG.get("context", {}).get("levels", [])
    avg_context = np.mean([v for k, v in token_impact.items() if k.startswith("context_")]) if [v for k, v in token_impact.items() if k.startswith("context_")] else 0
    std_context = np.std([v for k, v in token_impact.items() if k.startswith("context_")]) if [v for k, v in token_impact.items() if k.startswith("context_")] else 0
    for i, color in enumerate(context_colors):
        legend_handles.append(plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=8))
        legend_labels.append(f'Context (>{avg_context + context_levels[i]*std_context:.2f})')
    legend_handles.append(plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=8))
    legend_labels.append('Response')
    query_colors = THRESHOLD_CONFIG.get("query", {}).get("colors", [])
    query_levels = THRESHOLD_CONFIG.get("query", {}).get("levels", [])
    avg_query = np.mean([v for k, v in token_impact.items() if k.startswith("query_")]) if [v for k, v in token_impact.items() if k.startswith("query_")] else 0
    std_query = np.std([v for k, v in token_impact.items() if k.startswith("query_")]) if [v for k, v in token_impact.items() if k.startswith("query_")] else 0
    for i, color in enumerate(query_colors):
        legend_handles.append(plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=8))
        legend_labels.append(f'Query (>{avg_query + query_levels[i]*std_query:.2f})')

    plt.legend(legend_handles, legend_labels)

    plt.title("Token Embeddings in 2D Space (PCA)")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.grid(True)
    plt.show()

    # 3D Plot
    fig_3d = plt.figure(figsize=(12, 10))
    ax_3d = fig_3d.add_subplot(111, projection='3d')
    scatter_3d = ax_3d.scatter(embeddings_3d[:, 0], embeddings_3d[:, 1], embeddings_3d[:, 2], c=colors_to_plot, s=50, alpha=0.7)
    for i, token in enumerate(tokens_to_plot):
        ax_3d.text(embeddings_3d[i, 0], embeddings_3d[i, 1], embeddings_3d[i, 2], token, fontsize=8)

    ax_3d.legend(legend_handles, legend_labels)

    ax_3d.set_title("Token Embeddings in 3D Space (PCA)")
    ax_3d.set_xlabel("Principal Component 1")
    ax_3d.set_ylabel("Principal Component 2")
    ax_3d.set_zlabel("Principal Component 3")
    plt.show()


def visualize_results_html(token_impact, context, query, permutation_impact=None, threshold_config=None):
    """Generates an HTML visualization of the token and permutation impact, considering separate thresholds with levels."""
    if threshold_config is None:
        threshold_config = {}
    thresholds = calculate_token_thresholds_with_levels(token_impact, threshold_config)
    context_thresholds = thresholds.get("context", [])
    query_thresholds = thresholds.get("query", [])

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
            .legend { margin-top: 10px; padding: 10px; border: 1px solid #ddd; border-radius: 5px; background-color: #f9f9f9; }
            .legend-item { display: inline-block; margin-right: 15px; }
            .legend-color { width: 20px; height: 15px; display: inline-block; vertical-align: middle; margin-right: 5px; border-radius: 3px; }
            .permutation-item { margin-bottom: 5px; padding: 8px; border: 1px solid #eee; border-radius: 3px; }
            .permutation-item strong { font-weight: bold; color: #555; }
        </style>
    </head>
    """
    html += f"""
    <body>
        <h1>RAG Explainability Analysis</h1>
        <p>Context tokens and query tokens are highlighted based on their impact levels.</p>

        <div class="explanation-section">
            <h2>Token Impact</h2>
            <p>The following highlights indicate the influence of each token on the response. Hover over a highlighted word to see its impact percentage.</p>
            <div>
                <h3>Context:</h3>
                <p>
    """
    context_tokens = nlp(context)
    context_impacts = {k.split("context_")[1]: v for k, v in token_impact.items() if k.startswith("context_")}

    def get_highlight_class(impact, thresholds):
        for threshold, color in reversed(thresholds):
            if impact > threshold:
                if color == 'lightcoral':
                    return "impact-level-1"
                elif color == 'salmon':
                    return "impact-level-2"
                elif color == 'red':
                    return "impact-level-3"
        return None

    if context_impacts:
        context_thresholds_with_colors = thresholds.get("context", [])
        for token in context_tokens:
            if not token.is_stop and not token.is_punct and not token.is_space and token.text in context_impacts:
                impact = context_impacts[token.text]
                highlight_class = get_highlight_class(impact, context_thresholds_with_colors)
                percentage = f"{impact * 100:.2f}%"
                if highlight_class:
                    html += f"<span class='token-highlight {highlight_class}' title='Impact: {percentage}'>{token.text}</span> "
                else:
                    html += f"{token.text} "
            else:
                html += f"{token.text} "
    else:
        html += f"{context} "
    html += "</p>"

    html += """
                    <h3>Query:</h3>
                    <p>
    """
    query_tokens = nlp(query)
    query_impacts = {k.split("query_")[1]: v for k, v in token_impact.items() if k.startswith("query_")}

    def get_query_highlight_class(impact, thresholds):
        for threshold, color in reversed(thresholds):
            if impact > threshold:
                if color == 'lightblue':
                    return "impact-level-1-query"
                elif color == 'cornflowerblue':
                    return "impact-level-2-query"
                elif color == 'blue':
                    return "impact-level-3-query"
        return None

    if query_impacts:
        query_thresholds_with_colors = thresholds.get("query", [])
        for token in query_tokens:
            if not token.is_stop and not token.is_punct and not token.is_space and token.text in query_impacts:
                impact = query_impacts[token.text]
                highlight_class = get_query_highlight_class(impact, query_thresholds_with_colors)
                percentage = f"{impact * 100:.2f}%"
                if highlight_class:
                    html += f"<span class='token-highlight {highlight_class}' title='Impact: {percentage}'>{token.text}</span> "
                else:
                    html += f"{token.text} "
            else:
                html += f"{token.text} "
    else:
        html += f"{query} "
    html += "</p>"

    html += """
                </div>
                <div class="legend">
                    <h3>Legend: Token Impact</h3>
    """
    context_colors = THRESHOLD_CONFIG.get("context", {}).get("colors", [])
    context_levels = THRESHOLD_CONFIG.get("context", {}).get("levels", [])
    avg_context = np.mean([v for k, v in token_impact.items() if k.startswith("context_")]) if [v for k, v in token_impact.items() if k.startswith("context_")] else 0
    std_context = np.std([v for k, v in token_impact.items() if k.startswith("context_")]) if [v for k, v in token_impact.items() if k.startswith("context_")] else 0
    for i, color in enumerate(context_colors):
        html += f'<div class="legend-item"><span class="legend-color" style="background-color: {color};"></span> Context (>{avg_context + context_levels[i]*std_context:.2f})</div>'

    query_colors = THRESHOLD_CONFIG.get("query", {}).get("colors", [])
    query_levels = THRESHOLD_CONFIG.get("query", {}).get("levels", [])
    avg_query = np.mean([v for k, v in token_impact.items() if k.startswith("query_")]) if [v for k, v in token_impact.items() if k.startswith("query_")] else 0
    std_query = np.std([v for k, v in token_impact.items() if k.startswith("query_")]) if [v for k, v in token_impact.items() if k.startswith("query_")] else 0
    for i, color in enumerate(query_colors):
        html += f'<div class="legend-item"><span class="legend-color" style="background-color: {color};"></span> Query (>{avg_query + query_levels[i]*std_query:.2f})</div>'

    html += """
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

        <style>
            .impact-level-1 { background-color: lightcoral; }
            .impact-level-2 { background-color: salmon; }
            .impact-level-3 { background-color: red; }
            .impact-level-1-query { background-color: lightblue; }
            .impact-level-2-query { background-color: cornflowerblue; }
            .impact-level-3-query { background-color: blue; }
        </style>
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

    # Plot token embeddings
    plot_token_embeddings(context, query, response, token_impact, THRESHOLD_CONFIG)

    # Visualize results in HTML
    html_output = visualize_results_html(token_impact, context, query, permutation_impact, THRESHOLD_CONFIG)
    with open("rag_explanation.html", "w") as f:
        f.write(html_output)
    print("\nExplanation saved to rag_explanation.html")