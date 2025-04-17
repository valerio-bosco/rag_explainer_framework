# rag_plotting.py
import warnings
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import spacy
from scipy.stats import norm
# Load spaCy model
nlp = spacy.load("en_core_web_sm")



def plot_token_impact_distribution(analysis_results):
    """
    Plots the distribution of token impact scores as a histogram
    and overlays a fitted Gaussian curve.

    Args:
        analysis_results: Dictionary with analysis results from analyze_rag()
                          containing the 'token_impact' dictionary.
    """
    token_impact = analysis_results.get("token_impact", {})

    if not token_impact:
        print("No token impact scores found in analysis results.")
        return

    # Extract all impact scores (values from the dictionary)
    impact_scores = list(token_impact.values())
    impact_array = np.array(impact_scores)

    if impact_array.size == 0:
        print("Token impact dictionary is empty.")
        return

    # Calculate mean (mu) and standard deviation (sigma)
    mu = np.mean(impact_array)
    sigma = np.std(impact_array)

    print(f"Token Impact Distribution Stats: Mean = {mu:.4f}, Std Dev = {sigma:.4f}")

    # Prevent issues if standard deviation is zero (e.g., all impacts are the same)
    if sigma == 0:
        print("Standard deviation is zero. Cannot plot a meaningful Gaussian curve.")
        # Optionally plot just the histogram or a single point
        plt.figure(figsize=(10, 6))
        plt.hist(impact_array, bins=1, density=False, alpha=0.6, color='g')
        plt.title('Distribution of Token Impact Scores (Std Dev = 0)')
        plt.xlabel('Token Impact Score')
        plt.ylabel('Count')
        plt.grid(True)
        plt.show()
        return

    plt.figure(figsize=(10, 6))

    # Plot the histogram of the actual impact scores
    # 'density=True' normalizes the histogram so the area sums to 1,
    # making it comparable to the probability density function (PDF)
    plt.hist(impact_array, bins='auto', density=True, alpha=0.7, color='skyblue', label='Actual Impact Distribution')

    # Generate x-values for the Gaussian curve plot
    # Cover range from mean - 4*std_dev to mean + 4*std_dev
    xmin, xmax = plt.xlim() # Get limits from histogram for better range
    x_values = np.linspace(xmin, xmax, 200)

    # Calculate the y-values (PDF) for the Gaussian curve
    # using the calculated mean (mu) and standard deviation (sigma)
    pdf_values = norm.pdf(x_values, mu, sigma)

    # Plot the Gaussian curve
    plt.plot(x_values, pdf_values, 'r-', linewidth=2, label=f'Gaussian Fit ($\mu={mu:.2f}, \sigma={sigma:.2f}$)')

    # Add plot details
    plt.title('Distribution of Token Impact Scores and Gaussian Fit')
    plt.xlabel('Token Impact Score')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True)
    plt.show()
    

def plot_token_embeddings(analysis_results, embedding_function):
    """
    Plots the embeddings of individual tokens from context, query, and response
    colored by their impact level, using a provided embedding function for vectors.

    Args:
        analysis_results: Dictionary with analysis results from analyze_rag().
                          Must contain 'context', 'query', 'response', 'token_impact',
                          'threshold_config', 'thresholds'.
        embedding_function: The function to call to get embeddings (e.g., get_embedding).
                            It should accept a string (token text) and return a numpy array
                            or None/zero vector on failure.
    """
    # --- 1. Extract data from analysis_results ---
    context = analysis_results.get("context", "")
    query = analysis_results.get("query", "")
    response = analysis_results.get("response", "")
    token_impact = analysis_results.get("token_impact", {})
    threshold_config = analysis_results.get("threshold_config", {})
    thresholds = analysis_results.get("thresholds", {})

    if not context and not query and not response:
        print("Warning: Context, Query, and Response are all empty.")
        # return # Or continue if response might still be generated

    if not token_impact:
        print("Warning: token_impact dictionary is empty. Plot might lack colored points.")

    # --- 2. Tokenize text using spaCy ---
    context_doc = nlp(context)
    query_doc = nlp(query)
    response_doc = nlp(response)

    # --- 3. Prepare lists for plotting ---
    context_thresholds = thresholds.get("context", [])
    query_thresholds = thresholds.get("query", [])

    tokens_to_plot = []
    embeddings_to_plot = []
    colors_to_plot = []
    labels_to_plot = []
    processed_tokens = set()  # Avoid plotting the exact same word string multiple times

    # --- Helper for threshold coloring ---
    def get_token_level_color(impact, thresholds_list):
        """Determines the color for a token based on its impact and defined levels."""
        # Sort thresholds from high to low to apply the color for the highest threshold met
        sorted_thresholds = sorted(thresholds_list, key=lambda x: x[0], reverse=True)
        for threshold_value, color in sorted_thresholds:
            if impact > threshold_value:
                return color
        return None # No threshold met

    # --- 4. Process Tokens and Get Embeddings ---
    print("Processing tokens and generating embeddings...")

    # Process Context Tokens
    for token in context_doc:
        token_key = f"context_{token.text}"
        # Filter out stop words, punctuation, spaces, and check impact exists
        if not token.is_stop and not token.is_punct and not token.is_space and token_key in token_impact:
            impact = token_impact[token_key]
            color = get_token_level_color(impact, context_thresholds)
            # Only plot if a color threshold is met and token hasn't been plotted yet
            if color and token.text not in processed_tokens:
                token_text = token.text
                embedding_vector = embedding_function(token_text) # *** USE EXTERNAL FUNCTION ***
                # Check if embedding is valid (not None and not all zeros)
                if embedding_vector is not None and embedding_vector.any():
                    tokens_to_plot.append(token_text)
                    embeddings_to_plot.append(embedding_vector)
                    colors_to_plot.append(color)
                    labels_to_plot.append('Context')
                    processed_tokens.add(token_text)
                else:
                    warnings.warn(f"Could not get valid embedding for context token: '{token_text}'. Skipping.")

    # Process Query Tokens
    for token in query_doc:
        token_key = f"query_{token.text}"
        if not token.is_stop and not token.is_punct and not token.is_space and token_key in token_impact:
            impact = token_impact[token_key]
            color = get_token_level_color(impact, query_thresholds)
            # Only plot if a color threshold is met and token hasn't been plotted yet
            if color and token.text not in processed_tokens:
                token_text = token.text
                embedding_vector = embedding_function(token_text) # *** USE EXTERNAL FUNCTION ***
                if embedding_vector is not None and embedding_vector.any():
                    tokens_to_plot.append(token_text)
                    embeddings_to_plot.append(embedding_vector)
                    colors_to_plot.append(color)
                    labels_to_plot.append('Query')
                    processed_tokens.add(token_text)
                else:
                    warnings.warn(f"Could not get valid embedding for query token: '{token_text}'. Skipping.")

    # Process Response Tokens (plotted regardless of impact, colored green)
    for token in response_doc:
        if not token.is_stop and not token.is_punct and not token.is_space:
             # Only plot if token hasn't been plotted yet
            if token.text not in processed_tokens:
                token_text = token.text
                embedding_vector = embedding_function(token_text) # *** USE EXTERNAL FUNCTION ***
                if embedding_vector is not None and embedding_vector.any():
                    tokens_to_plot.append(token_text)
                    embeddings_to_plot.append(embedding_vector)
                    colors_to_plot.append('green') # Default color for response tokens
                    labels_to_plot.append('Response')
                    processed_tokens.add(token_text)
                else:
                    warnings.warn(f"Could not get valid embedding for response token: '{token_text}'. Skipping.")

    # --- 5. Check if any embeddings were generated ---
    if not embeddings_to_plot:
        print("No valid token embeddings generated for plotting based on filters, thresholds, and embedding function.")
        return

    print(f"Generated {len(embeddings_to_plot)} embeddings for plotting.")

    # --- 6. Convert to NumPy array and check dimensions ---
    try:
        embeddings_array = np.array(embeddings_to_plot)
        if embeddings_array.ndim != 2:
             # This can happen if embeddings have inconsistent lengths or only one was generated
             raise ValueError(f"Embeddings array has unexpected shape: {embeddings_array.shape}. Expected (n_samples, n_features). Check embedding function consistency.")
        n_samples, n_features = embeddings_array.shape
        print(f"Embeddings array shape: ({n_samples}, {n_features})")

    except ValueError as e:
        print(f"Error creating NumPy array from embeddings: {e}")
        # Optional: Print lengths for debugging
        # for i, emb in enumerate(embeddings_to_plot): print(f"Token: {tokens_to_plot[i]}, Length: {len(emb) if emb is not None else 'None'}")
        return

    # --- 7. Perform PCA Dimensionality Reduction ---
    print("Reducing dimensionality with PCA...")

    # Check requirements for PCA
    if n_samples <= 1:
        print("PCA requires at least 2 samples. Skipping PCA and plotting.")
        return

    # Reduce dimensionality for 2D plotting
    pca_dim_2d = min(2, n_samples, n_features)
    embeddings_2d = None
    if pca_dim_2d == 2:
        try:
            pca_2d = PCA(n_components=pca_dim_2d)
            embeddings_2d = pca_2d.fit_transform(embeddings_array)
            print(f"PCA 2D reduction complete. Explained variance ratio: {pca_2d.explained_variance_ratio_}")
        except Exception as e:
            print(f"Error during 2D PCA: {e}")
    else:
        print(f"Cannot perform 2D PCA (need n_components=2, samples={n_samples}, features={n_features}).")

    # Reduce dimensionality for 3D plotting
    pca_dim_3d = min(3, n_samples, n_features)
    embeddings_3d = None
    if pca_dim_3d == 3:
         try:
            pca_3d = PCA(n_components=pca_dim_3d)
            embeddings_3d = pca_3d.fit_transform(embeddings_array)
            print(f"PCA 3D reduction complete. Explained variance ratio: {pca_3d.explained_variance_ratio_}")
         except Exception as e:
            print(f"Error during 3D PCA: {e}")
    else:
        print(f"Cannot perform 3D PCA (need n_components=3, samples={n_samples}, features={n_features}).")

    # --- 8. Plot Visualizations ---

    # Plot 2D visualization if possible
    if embeddings_2d is not None:
        print("Plotting 2D embeddings...")
        # plot_2d_embeddings function needs access to token_impact and threshold_config for the legend
        plot_2d_embeddings(embeddings_2d, tokens_to_plot, colors_to_plot, token_impact, threshold_config)
    else:
        print("Skipping 2D plot.")

    # Plot 3D visualization if possible
    if embeddings_3d is not None:
        print("Plotting 3D embeddings...")
         # plot_3d_embeddings function needs access to token_impact and threshold_config for the legend
        plot_3d_embeddings(embeddings_3d, tokens_to_plot, colors_to_plot, token_impact, threshold_config)
    else:
        print("Skipping 3D plot.")


# --- Keep your plot_2d_embeddings and plot_3d_embeddings functions as they were ---
# They don't need changes as they receive the already computed embeddings and colors.

def plot_2d_embeddings(embeddings_2d, tokens_to_plot, colors_to_plot, token_impact, threshold_config):
    """Plot the 2D PCA visualization of token embeddings."""
    # --- (Your existing plot_2d_embeddings code) ---
    plt.figure(figsize=(10, 10))
    scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=colors_to_plot, s=50, alpha=0.7)

    for i, token in enumerate(tokens_to_plot):
        plt.annotate(token, (embeddings_2d[i, 0], embeddings_2d[i, 1]), textcoords="offset points",
                     xytext=(5, 5), ha='center', fontsize=8)

    legend_handles = []
    legend_labels = []

    # Use provided token_impact and threshold_config to build legend dynamically
    context_impacts = [v for k, v in token_impact.items() if k.startswith("context_")]
    avg_context = np.mean(context_impacts) if context_impacts else 0
    std_context = np.std(context_impacts) if context_impacts else 0
    context_thresh_info = threshold_config.get("context", {})
    context_colors = context_thresh_info.get("colors", [])
    context_levels = context_thresh_info.get("levels", []) # These were threshold factors (like std devs) in original example

    # Assuming levels correspond to colors and represent factors of std dev above mean
    # Reconstruct the approximate threshold values for the legend
    context_legend_thresholds = sorted([avg_context + lvl * std_context for lvl in context_levels], reverse=True)

    for i, color in enumerate(context_colors):
        threshold_val = context_legend_thresholds[i] if i < len(context_legend_thresholds) else -np.inf
        legend_handles.append(plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=8))
        # Adjust label based on how thresholds were defined. Original example used std dev factors.
        # If your thresholds are absolute values, use those directly from analysis_results["thresholds"]
        # Example using original std dev factor logic:
        legend_labels.append(f'Context (>{threshold_val:.2f})')
        # Example using absolute thresholds if available in thresholds structure like [(value, color),...]:
        # legend_labels.append(f'Context (>{analysis_results["thresholds"]["context"][i][0]:.2f})')


    legend_handles.append(plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=8))
    legend_labels.append('Response')

    query_impacts = [v for k, v in token_impact.items() if k.startswith("query_")]
    avg_query = np.mean(query_impacts) if query_impacts else 0
    std_query = np.std(query_impacts) if query_impacts else 0
    query_thresh_info = threshold_config.get("query", {})
    query_colors = query_thresh_info.get("colors", [])
    query_levels = query_thresh_info.get("levels", [])
    query_legend_thresholds = sorted([avg_query + lvl * std_query for lvl in query_levels], reverse=True)

    for i, color in enumerate(query_colors):
        threshold_val = query_legend_thresholds[i] if i < len(query_legend_thresholds) else -np.inf
        legend_handles.append(plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=8))
        # Adjust label as needed (see context example)
        legend_labels.append(f'Query (>{threshold_val:.2f})')

    plt.legend(legend_handles, legend_labels, title="Token Source & Impact") # Added title
    plt.title("Token Embeddings in 2D Space (PCA)")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.grid(True)
    plt.show()


def plot_3d_embeddings(embeddings_3d, tokens_to_plot, colors_to_plot, token_impact, threshold_config):
    """Plot the 3D PCA visualization of token embeddings."""
    # --- (Your existing plot_3d_embeddings code, potentially with updated legend logic like 2D) ---
    fig_3d = plt.figure(figsize=(12, 10))
    ax_3d = fig_3d.add_subplot(111, projection='3d')

    scatter_3d = ax_3d.scatter(embeddings_3d[:, 0], embeddings_3d[:, 1],
                               embeddings_3d[:, 2], c=colors_to_plot, s=50, alpha=0.7)

    for i, token in enumerate(tokens_to_plot):
        ax_3d.text(embeddings_3d[i, 0], embeddings_3d[i, 1], embeddings_3d[i, 2], token, fontsize=8)

    # --- Rebuild legend similar to plot_2d_embeddings ---
    legend_handles = []
    legend_labels = []

    # Context Legend Items (copied logic from 2D)
    context_impacts = [v for k, v in token_impact.items() if k.startswith("context_")]
    avg_context = np.mean(context_impacts) if context_impacts else 0
    std_context = np.std(context_impacts) if context_impacts else 0
    context_thresh_info = threshold_config.get("context", {})
    context_colors = context_thresh_info.get("colors", [])
    context_levels = context_thresh_info.get("levels", [])
    context_legend_thresholds = sorted([avg_context + lvl * std_context for lvl in context_levels], reverse=True)
    for i, color in enumerate(context_colors):
        threshold_val = context_legend_thresholds[i] if i < len(context_legend_thresholds) else -np.inf
        legend_handles.append(plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=8))
        legend_labels.append(f'Context (>{threshold_val:.2f})')

    # Response Legend Item
    legend_handles.append(plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=8))
    legend_labels.append('Response')

    # Query Legend Items (copied logic from 2D)
    query_impacts = [v for k, v in token_impact.items() if k.startswith("query_")]
    avg_query = np.mean(query_impacts) if query_impacts else 0
    std_query = np.std(query_impacts) if query_impacts else 0
    query_thresh_info = threshold_config.get("query", {})
    query_colors = query_thresh_info.get("colors", [])
    query_levels = query_thresh_info.get("levels", [])
    query_legend_thresholds = sorted([avg_query + lvl * std_query for lvl in query_levels], reverse=True)
    for i, color in enumerate(query_colors):
        threshold_val = query_legend_thresholds[i] if i < len(query_legend_thresholds) else -np.inf
        legend_handles.append(plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=8))
        legend_labels.append(f'Query (>{threshold_val:.2f})')


    ax_3d.legend(legend_handles, legend_labels, title="Token Source & Impact") # Added title
    ax_3d.set_title("Token Embeddings in 3D Space (PCA)")
    ax_3d.set_xlabel("Principal Component 1")
    ax_3d.set_ylabel("Principal Component 2")
    ax_3d.set_zlabel("Principal Component 3")
    plt.show()
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