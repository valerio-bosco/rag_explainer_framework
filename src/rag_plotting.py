# rag_plotting.py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import spacy

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

def plot_token_embeddings(analysis_results):
    """
    Plots the embeddings of individual tokens from context, query, and response colored by their impact level.
    
    Args:
        analysis_results: Dictionary with analysis results from analyze_rag()
    """
    context = analysis_results["context"]
    query = analysis_results["query"]
    response = analysis_results["response"]
    token_impact = analysis_results["token_impact"]
    threshold_config = analysis_results["threshold_config"]
    thresholds = analysis_results["thresholds"]
    
    context_doc = nlp(context)
    query_doc = nlp(query)
    response_doc = nlp(response)
    
    context_thresholds = thresholds.get("context", [])
    query_thresholds = thresholds.get("query", [])

    tokens_to_plot = []
    embeddings_to_plot = []
    colors_to_plot = []
    labels_to_plot = []

    processed_tokens = set()  # To avoid duplicate tokens

    def get_token_level_color(impact, thresholds):
        """Determines the color for a token based on its impact and defined levels."""
        for threshold, color in thresholds:
            if impact > threshold:
                return color
        return None

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
        if not token.is_stop and not token.is_punct and not token.is_space and f"query_{token.text}" in token_impact:
            impact = token_impact[f"query_{token.text}"]
            color = get_token_level_color(impact, query_thresholds)
            if color:
                tokens_to_plot.append(token.text)
                embeddings_to_plot.append(token.vector)
                colors_to_plot.append(color)
                labels_to_plot.append('Query')
                processed_tokens.add(token.text)

    for token in response_doc:
        if not token.is_stop and not token.is_punct and not token.is_space:
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

    # Plot 2D visualization
    plot_2d_embeddings(embeddings_2d, tokens_to_plot, colors_to_plot, token_impact, threshold_config)
    
    # Plot 3D visualization
    plot_3d_embeddings(embeddings_3d, tokens_to_plot, colors_to_plot, token_impact, threshold_config)

def plot_2d_embeddings(embeddings_2d, tokens_to_plot, colors_to_plot, token_impact, threshold_config):
    """Plot the 2D PCA visualization of token embeddings."""
    plt.figure(figsize=(10, 10))
    scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=colors_to_plot, s=50, alpha=0.7)
    
    for i, token in enumerate(tokens_to_plot):
        plt.annotate(token, (embeddings_2d[i, 0], embeddings_2d[i, 1]), textcoords="offset points", 
                    xytext=(5, 5), ha='center', fontsize=8)

    legend_handles = []
    legend_labels = []
    
    # Generate legend items for context
    context_colors = threshold_config.get("context", {}).get("colors", [])
    context_levels = threshold_config.get("context", {}).get("levels", [])
    avg_context = np.mean([v for k, v in token_impact.items() if k.startswith("context_")]) if [v for k, v in token_impact.items() if k.startswith("context_")] else 0
    std_context = np.std([v for k, v in token_impact.items() if k.startswith("context_")]) if [v for k, v in token_impact.items() if k.startswith("context_")] else 0
    for i, color in enumerate(context_colors):
        legend_handles.append(plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=8))
        legend_labels.append(f'Context (>{avg_context + context_levels[i]*std_context:.2f})')
    
    # Add response to legend
    legend_handles.append(plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=8))
    legend_labels.append('Response')
    
    # Generate legend items for query
    query_colors = threshold_config.get("query", {}).get("colors", [])
    query_levels = threshold_config.get("query", {}).get("levels", [])
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

def plot_3d_embeddings(embeddings_3d, tokens_to_plot, colors_to_plot, token_impact, threshold_config):
    """Plot the 3D PCA visualization of token embeddings."""
    fig_3d = plt.figure(figsize=(12, 10))
    ax_3d = fig_3d.add_subplot(111, projection='3d')
    
    scatter_3d = ax_3d.scatter(embeddings_3d[:, 0], embeddings_3d[:, 1], 
                              embeddings_3d[:, 2], c=colors_to_plot, s=50, alpha=0.7)
    
    for i, token in enumerate(tokens_to_plot):
        ax_3d.text(embeddings_3d[i, 0], embeddings_3d[i, 1], embeddings_3d[i, 2], token, fontsize=8)

    # Generate legend items for context
    legend_handles = []
    legend_labels = []
    context_colors = threshold_config.get("context", {}).get("colors", [])
    context_levels = threshold_config.get("context", {}).get("levels", [])
    avg_context = np.mean([v for k, v in token_impact.items() if k.startswith("context_")]) if [v for k, v in token_impact.items() if k.startswith("context_")] else 0
    std_context = np.std([v for k, v in token_impact.items() if k.startswith("context_")]) if [v for k, v in token_impact.items() if k.startswith("context_")] else 0
    for i, color in enumerate(context_colors):
        legend_handles.append(plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=8))
        legend_labels.append(f'Context (>{avg_context + context_levels[i]*std_context:.2f})')
    
    # Add response to legend
    legend_handles.append(plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=8))
    legend_labels.append('Response')
    
    # Generate legend items for query
    query_colors = threshold_config.get("query", {}).get("colors", [])
    query_levels = threshold_config.get("query", {}).get("levels", [])
    avg_query = np.mean([v for k, v in token_impact.items() if k.startswith("query_")]) if [v for k, v in token_impact.items() if k.startswith("query_")] else 0
    std_query = np.std([v for k, v in token_impact.items() if k.startswith("query_")]) if [v for k, v in token_impact.items() if k.startswith("query_")] else 0
    for i, color in enumerate(query_colors):
        legend_handles.append(plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=8))
        legend_labels.append(f'Query (>{avg_query + query_levels[i]*std_query:.2f})')

    ax_3d.legend(legend_handles, legend_labels)
    ax_3d.set_title("Token Embeddings in 3D Space (PCA)")
    ax_3d.set_xlabel("Principal Component 1")
    ax_3d.set_ylabel("Principal Component 2")
    ax_3d.set_zlabel("Principal Component 3")
    plt.show()