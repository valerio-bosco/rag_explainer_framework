# rag_plotting.py - IMPROVED VERSION
import hashlib
import warnings
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import spacy
from embedding_service import calculate_distance
from rag_evaluation import tokenize_and_filter
from scipy.stats import norm
from matplotlib import cm
import matplotlib.colors as mcolors
from collections import defaultdict

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Define a colorblind-friendly color palette
# Using a high-contrast palette based on ColorBrewer's colorblind-safe schemes
COLORBLIND_COLORS = {
    'context_high': '#d73027',    # Strong red
    'context_medium': '#fc8d59',  # Orange
    'context_low': '#fee090',     # Light yellow/orange
    'query_high': '#4575b4',      # Strong blue
    'query_medium': '#74add1',    # Medium blue
    'query_low': '#abd9e9',       # Light blue
    'response': '#006400'         # Dark green
}

def truncate_text(text, max_length=20):
    """Truncate text for display in plots."""
    if len(text) <= max_length:
        return text
    return text[:max_length-3] + "..."

def get_sentence_hash(sentence):
    """Generate a short hash for a sentence to use as identifier."""
    return hashlib.md5(sentence.encode()).hexdigest()[:6]

def plot_token_impact_distribution(analysis_results, highlight_range=None):
    """
    Plots the distribution of token impact scores as a histogram
    and overlays a fitted Gaussian curve. Optionally highlights a specific range.

    Args:
        analysis_results: Dictionary with analysis results from analyze_rag()
                          containing the 'token_impact' dictionary.
        highlight_range: Optional tuple (min_value, max_value) to highlight a specific range
                        of impact values in the distribution.
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
        plt.figure(figsize=(10, 6))
        plt.hist(impact_array, bins=1, density=False, alpha=0.6, color=COLORBLIND_COLORS['context_high'])
        plt.title('Distribution of Token Impact Scores (Std Dev = 0)')
        plt.xlabel('Token Impact Score')
        plt.ylabel('Count')
        plt.grid(True)
        plt.show()
        return

    plt.figure(figsize=(10, 6))

    # Use a colorblind-friendly color for the histogram
    plt.hist(impact_array, bins='auto', density=True, alpha=0.7, 
             color=COLORBLIND_COLORS['context_medium'], label='Actual Impact Distribution')

    # Generate x-values for the Gaussian curve plot
    xmin, xmax = plt.xlim() # Get limits from histogram for better range
    x_values = np.linspace(xmin, xmax, 200)

    # Calculate the y-values (PDF) for the Gaussian curve
    pdf_values = norm.pdf(x_values, mu, sigma)

    # Plot the Gaussian curve with high-contrast color
    plt.plot(x_values, pdf_values, color=COLORBLIND_COLORS['context_high'], 
             linewidth=2, label=f'Gaussian Fit ($\mu={mu:.2f}, \sigma={sigma:.2f}$)')

    # Add plot details
    plt.title('Distribution of Token Impact Scores and Gaussian Fit')
    plt.xlabel('Token Impact Score')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True)
    plt.show()

def extract_sentences(text):
    """
    Extract sentences from text using spaCy.
    """
    doc = nlp(text)
    return [sent.text.strip() for sent in doc.sents]

def plot_sentence_embeddings(analysis_results, embedding_function, impact_filter=None):
    """
    Plots the embeddings of sentences from context, query, and response
    colored by their impact level, using a provided embedding function for vectors.

    Args:
        analysis_results: Dictionary with analysis results from analyze_rag().
                          Must contain 'context', 'query', 'response', 'token_impact',
                          'threshold_config', 'thresholds'.
        embedding_function: The function to call to get embeddings.
                            It should accept a string (sentence text) and return a numpy array.
        impact_filter: Optional tuple (min_impact, max_impact) to filter sentences by impact score.
                      Only sentences with impact within this range will be displayed.
    """
    # --- 1. Extract data from analysis_results ---
    context = analysis_results.get("context", "")
    query = analysis_results.get("query", "")
    response = analysis_results.get("response", "")
    token_impact = analysis_results.get("token_impact", {})
    pseudo_response = analysis_results.get("pseudo_response", "")
    threshold_config = analysis_results.get("threshold_config", {})
    thresholds = analysis_results.get("thresholds", {})

    if not context and not query and not response:
        print("Warning: Context, Query, and Response are all empty.")
        return

    if not token_impact:
        print("Warning: token_impact dictionary is empty. Plot might lack colored points.")

    context_impact = {k : v for k,v in token_impact.items() if k.startswith("context_")}
    query_impact = {k : v for k,v in token_impact.items() if k.startswith("query_")}
    
    # --- 2. Extract sentences from text ---
    context_sentences = []
    for chunk in context:
        context_sentences.extend(extract_sentences(chunk))
    
    # If query is a single sentence, keep it as is; otherwise split into sentences
    query_sentences = extract_sentences(query) if len(query.split()) > 5 else [query]
    response_sentences = extract_sentences(response)

    print(f"Extracted {len(context_sentences)} context sentences, {len(query_sentences)} query sentences, "
          f"and {len(response_sentences)} response sentences.")

    # --- 3. Calculate sentence impact scores by averaging token impacts ---
    context_thresholds = thresholds.get("context", [])
    query_thresholds = thresholds.get("query", [])
    
    # Dictionary to store sentence impacts
    sentence_impacts = {}
    
    # Function to calculate sentence impact from token impacts
    def calculate_sentence_impact(sentence_text, source_prefix):
        sentence_tokens = tokenize_and_filter(sentence_text)
        token_impacts = []
        
        for token in sentence_tokens:
            token_key = f"{source_prefix}_{hashlib.md5(token.encode()).hexdigest()}"
            if token_key in token_impact:
                token_impacts.append(token_impact[token_key])
        
        if token_impacts:
            return np.mean(token_impacts)
        return 0.0  # Default impact if no token impacts found
    
    def calculate_percentage(value, impact_dict):
        if not impact_dict:
            return 0
        max_value = max(float(v) for v in impact_dict.values()) 
        max_value = max(max_value, 1)  # Avoid division by zero 
        min_value = min(float(v) for v in impact_dict.values())
        min_value = min(min_value, -1)  # Avoid division by zero
        
        if max_value == 0:
            return 0
        return (float(value)-min_value) / (max_value- min_value) 

    # Calculate impacts for context sentences
    for i, sentence in enumerate(context_sentences):
        impact = calculate_sentence_impact(sentence, "context")
        impact = calculate_percentage(impact, context_impact)
        key = f"context_sentence_{i}"
        sentence_impacts[key] = {
            "text": sentence,
            "impact": impact,
            "source": "context"
        }
    
    # Calculate impacts for query sentences
    for i, sentence in enumerate(query_sentences):
        impact = calculate_sentence_impact(sentence, "query")
        impact = calculate_percentage(impact, query_impact)

        key = f"query_sentence_{i}"
        sentence_impacts[key] = {
            "text": sentence,
            "impact": impact,
            "source": "query"
        }
    
    # Add response sentences (with default impact)
    for i, sentence in enumerate(response_sentences):
        key = f"response_sentence_{i}"
        sentence_impacts[key] = {
            "text": sentence,
            "impact": 0.0,  # Default impact for response
            "source": "response"
        }

    # --- 4. Determine colors based on impact thresholds ---
    def get_sentence_color(sentence_info):
        """Determine color based on sentence source and impact."""
        source = sentence_info["source"]
        impact = sentence_info["impact"]
        
        if source == "response":
            return COLORBLIND_COLORS["response"]
        
        if source == "context":
            thresholds_to_check = context_thresholds
        else:  # query
            thresholds_to_check = query_thresholds
        
        # Sort thresholds from high to low
        sorted_thresholds = sorted(thresholds_to_check, key=lambda x: x[0], reverse=True)
        
        for threshold_value, color in sorted_thresholds:
            if impact > threshold_value:
                # Map standard colors to our colorblind-friendly palette
                if source == "context":
                    if color == "red" or color == "yellow":  # Typically highest impact
                        return COLORBLIND_COLORS["context_high"]
                    elif color == "salmon":  # Medium impact
                        return COLORBLIND_COLORS["context_medium"]
                    else:  # Low impact
                        return COLORBLIND_COLORS["context_low"]
                else:  # query
                    if color == "blue":  # Highest impact
                        return COLORBLIND_COLORS["query_high"]
                    elif color == "cornflowerblue":  # Medium impact
                        return COLORBLIND_COLORS["query_medium"]
                    else:  # Low impact
                        return COLORBLIND_COLORS["query_low"]
        
        # Default colors if no threshold is met
        return COLORBLIND_COLORS["context_low"] if source == "context" else COLORBLIND_COLORS["query_low"]

    # --- 5. Generate sentence embeddings ---
    print("Generating sentence embeddings...")
    sentences_to_plot = []
    embeddings_to_plot = []
    colors_to_plot = []
    labels_to_plot = []
    sources_to_plot = []  # Track sentence sources for legend
    impact_values = []    # Track impact values for reference
    
    processed_sentences = set()  # Track processed sentences to avoid duplicates
    
    # Apply impact filter if specified
    if impact_filter is not None:
        min_impact, max_impact = impact_filter
        print(f"Filtering sentences with impact between {min_impact:.4f} and {max_impact:.4f}")
    
    for key, info in sentence_impacts.items():
        sentence_text = info["text"]
        impact = info["impact"]
        
        # Skip if empty or already processed
        if not sentence_text or sentence_text in processed_sentences:
            continue
            
        # Apply impact filter if specified
        if impact_filter is not None:
            min_impact, max_impact = impact_filter
            # Skip sentences outside the specified impact range
            # Special case for response (which has default impact 0)
            if info["source"] != "response" and (impact < min_impact or impact > max_impact):
                continue
        
        # Generate embedding for the sentence
        embedding_vector = embedding_function(sentence_text)
        
        # Skip if embedding is invalid
        if embedding_vector is None or not embedding_vector.any():
            warnings.warn(f"Could not get valid embedding for sentence: '{truncate_text(sentence_text)}'. Skipping.")
            continue
        
        # Add to plotting data
        sentences_to_plot.append(sentence_text)
        embeddings_to_plot.append(embedding_vector)
        color = get_sentence_color(info)
        colors_to_plot.append(color)
        impact_values.append(impact)
        
        # Create a short label that includes source and sentence hash
        source_prefix = info["source"][:1].upper()  # C for context, Q for query, R for response
        sentence_id = get_sentence_hash(sentence_text)
        label = f"{source_prefix}{sentence_id}"
        labels_to_plot.append(label)
        
        sources_to_plot.append(info["source"])
        processed_sentences.add(sentence_text)
    
    
    
    ####
    pseudo_response_embedding = embedding_function(pseudo_response)
    response_embedding = embedding_function(response)
    
    print( "THE FAKE AND REAL RESPONSEs ARE AT THIS DISTANCE",calculate_distance(pseudo_response_embedding, response_embedding))
    # Add to plotting data
    sentences_to_plot.append(pseudo_response)
    sentences_to_plot.append(response)
    
    embeddings_to_plot.append(pseudo_response_embedding)
    embeddings_to_plot.append(response_embedding)

    color = "grey"
    colors_to_plot.append(color)
    colors_to_plot.append(color)
        
    # Create a short label that includes source and sentence hash
    pseudo_response_id = get_sentence_hash(pseudo_response)
    response_id = get_sentence_hash(response)
    
    label_p = f"PW{pseudo_response_id}"
    label_r = f"W{response_id}"
    labels_to_plot.append(label_p)
    labels_to_plot.append(label_r)
    
    impact_values.append(0)
    impact_values.append(0)
    
    sources_to_plot.append(pseudo_response)
    sources_to_plot.append(response)
    ####
    
    # --- 6. Check if any embeddings were generated ---
    if not embeddings_to_plot:
        print("No valid sentence embeddings generated for plotting.")
        return

    print(f"Generated {len(embeddings_to_plot)} embeddings for plotting.")

    # --- 7. Convert to NumPy array and check dimensions ---
    try:
        embeddings_array = np.array(embeddings_to_plot)
        if embeddings_array.ndim != 2:
            raise ValueError(f"Embeddings array has unexpected shape: {embeddings_array.shape}. Expected (n_samples, n_features).")
        n_samples, n_features = embeddings_array.shape
        print(f"Embeddings array shape: ({n_samples}, {n_features})")

    except ValueError as e:
        print(f"Error creating NumPy array from embeddings: {e}")
        return

    # --- 8. Perform PCA Dimensionality Reduction ---
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

    # --- 9. Plot Visualizations ---
    # Create a mapping from sentence hash to full text for tooltips or lookups
    sentence_mapping = {get_sentence_hash(s): s for s in sentences_to_plot}

    # Plot 2D visualization if possible
    if embeddings_2d is not None:
        print("Plotting 2D embeddings...")
        plot_2d_sentence_embeddings(embeddings_2d, labels_to_plot, colors_to_plot, 
                                   sources_to_plot, sentence_mapping, impact_values)
    else:
        print("Skipping 2D plot.")

    # Plot 3D visualization if possible
    if embeddings_3d is not None:
        print("Plotting 3D embeddings...")
        plot_3d_sentence_embeddings(embeddings_3d, labels_to_plot, colors_to_plot, 
                                   sources_to_plot, sentence_mapping, impact_values)
    else:
        print("Skipping 3D plot.")

def plot_2d_sentence_embeddings(embeddings_2d, labels_to_plot, colors_to_plot, 
                              sources_to_plot, sentence_mapping, impact_values=None):
    """Plot the 2D PCA visualization of sentence embeddings with improved colors."""
    plt.figure(figsize=(12, 10))
    
    # Create scatter plot with improved clarity
    scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                         c=colors_to_plot, s=80, alpha=0.9, edgecolors='black', linewidth=0.5)

    # Add sentence labels with better positioning and improved readability
    for i, label in enumerate(labels_to_plot):
        plt.annotate(label, 
                    (embeddings_2d[i, 0], embeddings_2d[i, 1]), 
                    textcoords="offset points",
                    xytext=(5, 5), 
                    ha='center', 
                    fontsize=9,
                    fontweight='bold',
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", alpha=0.7))

    # Create a better legend with colorblind-friendly colors
    legend_handles = []
    legend_labels = []
    
    # Group colors by source for cleaner legend
    source_colors = defaultdict(set)
    for i, source in enumerate(sources_to_plot):
        source_colors[source].add(colors_to_plot[i])
    
    # Context entries
    if "context" in source_colors:
        if COLORBLIND_COLORS['context_high'] in source_colors["context"]:
            legend_handles.append(plt.Line2D([0], [0], marker='o', color='w', 
                                           markerfacecolor=COLORBLIND_COLORS['context_high'], 
                                           markersize=10, markeredgecolor='black', markeredgewidth=0.5))
            legend_labels.append('Context (High Impact)')
        
        if COLORBLIND_COLORS['context_medium'] in source_colors["context"]:
            legend_handles.append(plt.Line2D([0], [0], marker='o', color='w', 
                                           markerfacecolor=COLORBLIND_COLORS['context_medium'], 
                                           markersize=10, markeredgecolor='black', markeredgewidth=0.5))
            legend_labels.append('Context (Medium Impact)')
        
        if COLORBLIND_COLORS['context_low'] in source_colors["context"]:
            legend_handles.append(plt.Line2D([0], [0], marker='o', color='w', 
                                           markerfacecolor=COLORBLIND_COLORS['context_low'], 
                                           markersize=10, markeredgecolor='black', markeredgewidth=0.5))
            legend_labels.append('Context (Low Impact)')
    
    # Query entries
    if "query" in source_colors:
        if COLORBLIND_COLORS['query_high'] in source_colors["query"]:
            legend_handles.append(plt.Line2D([0], [0], marker='o', color='w', 
                                           markerfacecolor=COLORBLIND_COLORS['query_high'], 
                                           markersize=10, markeredgecolor='black', markeredgewidth=0.5))
            legend_labels.append('Query (High Impact)')
        
        if COLORBLIND_COLORS['query_medium'] in source_colors["query"]:
            legend_handles.append(plt.Line2D([0], [0], marker='o', color='w', 
                                           markerfacecolor=COLORBLIND_COLORS['query_medium'], 
                                           markersize=10, markeredgecolor='black', markeredgewidth=0.5))
            legend_labels.append('Query (Medium Impact)')
        
        if COLORBLIND_COLORS['query_low'] in source_colors["query"]:
            legend_handles.append(plt.Line2D([0], [0], marker='o', color='w', 
                                           markerfacecolor=COLORBLIND_COLORS['query_low'], 
                                           markersize=10, markeredgecolor='black', markeredgewidth=0.5))
            legend_labels.append('Query (Low Impact)')
    
    # Response entry
    if "response" in source_colors:
        legend_handles.append(plt.Line2D([0], [0], marker='o', color='w', 
                                       markerfacecolor=COLORBLIND_COLORS['response'], 
                                       markersize=10, markeredgecolor='black', markeredgewidth=0.5))
        legend_labels.append('Response')

    # Add legend with better formatting
    plt.legend(legend_handles, legend_labels, title="Sentence Source & Impact", 
              loc="upper right", frameon=True, fontsize=10)
    
    plt.title("Sentence Embeddings in 2D Space (PCA)", fontsize=14)
    plt.xlabel("Principal Component 1", fontsize=12)
    plt.ylabel("Principal Component 2", fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Add a note about sentence IDs and impact filter if applicable
    note_text = "Note: Labels show source (C=Context, Q=Query, R=Response) and sentence ID"
    if impact_values is not None:
        impact_range = f"Impact Range: {min(impact_values):.4f} to {max(impact_values):.4f}"
        note_text += f"\n{impact_range}"
    plt.figtext(0.5, 0.01, note_text, ha="center", fontsize=10, 
               bbox={"facecolor":"white", "alpha":0.8, "pad":5})
    
    plt.tight_layout()
    plt.show()

def plot_3d_sentence_embeddings(embeddings_3d, labels_to_plot, colors_to_plot, 
                              sources_to_plot, sentence_mapping, impact_values=None):
    """Plot the 3D PCA visualization of sentence embeddings with improved colors."""
    fig_3d = plt.figure(figsize=(14, 12))
    ax_3d = fig_3d.add_subplot(111, projection='3d')
    
    # Create scatter plot with improved clarity
    scatter_3d = ax_3d.scatter(embeddings_3d[:, 0], embeddings_3d[:, 1], embeddings_3d[:, 2],
                              c=colors_to_plot, s=100, alpha=0.9, edgecolors='black', linewidth=0.5)
    
    # Add sentence labels
    for i, label in enumerate(labels_to_plot):
        ax_3d.text(embeddings_3d[i, 0], embeddings_3d[i, 1], embeddings_3d[i, 2], 
                  label, fontsize=9, fontweight='bold')
    
    # Create a better legend with colorblind-friendly colors
    legend_handles = []
    legend_labels = []
    
    # Group colors by source for cleaner legend
    source_colors = defaultdict(set)
    for i, source in enumerate(sources_to_plot):
        source_colors[source].add(colors_to_plot[i])
    
    # Context entries
    if "context" in source_colors:
        if COLORBLIND_COLORS['context_high'] in source_colors["context"]:
            legend_handles.append(plt.Line2D([0], [0], marker='o', color='w', 
                                           markerfacecolor=COLORBLIND_COLORS['context_high'], 
                                           markersize=10, markeredgecolor='black', markeredgewidth=0.5))
            legend_labels.append('Context (High Impact)')
        
        if COLORBLIND_COLORS['context_medium'] in source_colors["context"]:
            legend_handles.append(plt.Line2D([0], [0], marker='o', color='w', 
                                           markerfacecolor=COLORBLIND_COLORS['context_medium'], 
                                           markersize=10, markeredgecolor='black', markeredgewidth=0.5))
            legend_labels.append('Context (Medium Impact)')
        
        if COLORBLIND_COLORS['context_low'] in source_colors["context"]:
            legend_handles.append(plt.Line2D([0], [0], marker='o', color='w', 
                                           markerfacecolor=COLORBLIND_COLORS['context_low'], 
                                           markersize=10, markeredgecolor='black', markeredgewidth=0.5))
            legend_labels.append('Context (Low Impact)')
    
    # Query entries
    if "query" in source_colors:
        if COLORBLIND_COLORS['query_high'] in source_colors["query"]:
            legend_handles.append(plt.Line2D([0], [0], marker='o', color='w', 
                                           markerfacecolor=COLORBLIND_COLORS['query_high'], 
                                           markersize=10, markeredgecolor='black', markeredgewidth=0.5))
            legend_labels.append('Query (High Impact)')
        
        if COLORBLIND_COLORS['query_medium'] in source_colors["query"]:
            legend_handles.append(plt.Line2D([0], [0], marker='o', color='w', 
                                           markerfacecolor=COLORBLIND_COLORS['query_medium'], 
                                           markersize=10, markeredgecolor='black', markeredgewidth=0.5))
            legend_labels.append('Query (Medium Impact)')
        
        if COLORBLIND_COLORS['query_low'] in source_colors["query"]:
            legend_handles.append(plt.Line2D([0], [0], marker='o', color='w', 
                                           markerfacecolor=COLORBLIND_COLORS['query_low'], 
                                           markersize=10, markeredgecolor='black', markeredgewidth=0.5))
            legend_labels.append('Query (Low Impact)')
    
    # Response entry
    if "response" in source_colors:
        legend_handles.append(plt.Line2D([0], [0], marker='o', color='w', 
                                       markerfacecolor=COLORBLIND_COLORS['response'], 
                                       markersize=10, markeredgecolor='black', markeredgewidth=0.5))
        legend_labels.append('Response')
    
    # Add legend with better formatting
    ax_3d.legend(legend_handles, legend_labels, title="Sentence Source & Impact", 
               loc="upper right", frameon=True, fontsize=10)
    
    ax_3d.set_title("Sentence Embeddings in 3D Space (PCA)", fontsize=14)
    ax_3d.set_xlabel("Principal Component 1", fontsize=12)
    ax_3d.set_ylabel("Principal Component 2", fontsize=12)
    ax_3d.set_zlabel("Principal Component 3", fontsize=12)
    
    # Add a note about sentence IDs as a text annotation below the plot
    note_text = "Note: Labels show source (C=Context, Q=Query, R=Response) and sentence ID"
    if impact_values is not None:
        impact_range = f"Impact Range: {min(impact_values):.4f} to {max(impact_values):.4f}"
        note_text += f"\n{impact_range}"
    fig_3d.text(0.5, 0.01, note_text, ha="center", fontsize=10, 
               bbox={"facecolor":"white", "alpha":0.8, "pad":5})
    
    # Add a help function to print full sentence text when exploring
    def generate_sentence_lookup_table():
        print("\nSentence ID Lookup Table:")
        print("-" * 80)
        print(f"{'ID':<8} | {'Source':<10} | {'Text':<60}")
        print("-" * 80)
        for i, label in enumerate(labels_to_plot):
            source = sources_to_plot[i]
            text = sentence_mapping.get(label[1:], "Unknown")  # Remove the first letter (source indicator)
            print(f"{label:<8} | {source:<10} | {truncate_text(text, 60):<60}")
    
    # Print the lookup table with impact values
    def generate_sentence_lookup_table_with_impact():
        print("\nSentence ID Lookup Table:")
        print("-" * 90)
        print(f"{'ID':<8} | {'Source':<10} | {'Impact':<10} | {'Text':<60}")
        print("-" * 90)
        for i, label in enumerate(labels_to_plot):
            source = sources_to_plot[i]
            impact = impact_values[i] if impact_values else "N/A"
            text = sentence_mapping.get(label[1:], "Unknown")  # Remove the first letter (source indicator)
            print(f"{label:<8} | {source:<10} | {impact:<10.4f} | {truncate_text(text, 60):<60}")
    
    # Print the improved lookup table
    generate_sentence_lookup_table_with_impact()
    
    plt.tight_layout()
    plt.show()

# Rename the function to reflect the change from token to sentence-based visualization
def plot_embeddings(analysis_results, embedding_function, impact_filter=None):
    """
    Main entry point - plots sentence embeddings instead of individual tokens.
    This replaces the original plot_token_embeddings function.
    
    Args:
        analysis_results: Dictionary with analysis results from analyze_rag().
        embedding_function: Function to generate embeddings for sentences.
        impact_filter: Optional tuple (min_impact, max_impact) to filter sentences by impact score.
                      Only sentences with impact within this range will be displayed.
    """
    return plot_sentence_embeddings(analysis_results, embedding_function, impact_filter)