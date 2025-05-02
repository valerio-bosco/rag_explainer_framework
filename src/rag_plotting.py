# improved_rag_plotting.py
import hashlib
import warnings
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import spacy
from embedding_service import calculate_distance
from rag_evaluation import tokenize_and_filter
from scipy.stats import norm
from collections import defaultdict

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Define a colorblind-friendly color palette
COLORBLIND_COLORS = {
    'context_high': '#d73027',    # Strong red
    'context_medium': '#fc8d59',  # Orange
    'context_low': '#fee090',     # Light yellow/orange
    'query_high': '#4575b4',      # Strong blue
    'query_medium': '#74add1',    # Medium blue
    'query_low': '#abd9e9',       # Light blue
    'response': '#006400',        # Dark green
    'pseudo_response': '#808080', # Grey for pseudo response
    'w_response': '#808080'    # Grey for w response
    
}

def truncate_text(text, max_length=20):
    """Truncate text for display in plots."""
    if len(text) <= max_length:
        return text
    return text[:max_length-3] + "..."

def get_sentence_hash(sentence):
    """Generate a short hash for a sentence to use as identifier."""
    return hashlib.md5(sentence.encode()).hexdigest()[:6]

def extract_sentences(text):
    """Extract sentences from text using spaCy."""
    return tokenize_and_filter(text)

def plot_token_impact_distribution(analysis_results, highlight_range=None):
    """
    Plots the distribution of token impact scores as a histogram
    and overlays a fitted Gaussian curve.

    Args:
        analysis_results: Dictionary with analysis results
        highlight_range: Optional tuple (min_value, max_value) to highlight
    """
    token_impact = analysis_results.get("token_impact", {})

    if not token_impact:
        print("No token impact scores found in analysis results.")
        return

    # Extract all impact scores
    impact_scores = list(token_impact.values())
    impact_array = np.array(impact_scores)

    if impact_array.size == 0:
        print("Token impact dictionary is empty.")
        return

    # Calculate statistics
    mu = np.mean(impact_array)
    sigma = np.std(impact_array)
    print(f"Token Impact Distribution Stats: Mean = {mu:.4f}, Std Dev = {sigma:.4f}")

    # Handle zero standard deviation case
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

    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.hist(impact_array, bins='auto', density=True, alpha=0.7, 
             color=COLORBLIND_COLORS['context_medium'], label='Actual Impact Distribution')

    # Generate Gaussian curve
    xmin, xmax = plt.xlim()
    x_values = np.linspace(xmin, xmax, 200)
    pdf_values = norm.pdf(x_values, mu, sigma)
    plt.plot(x_values, pdf_values, color=COLORBLIND_COLORS['context_high'], 
             linewidth=2, label=f'Gaussian Fit ($\mu={mu:.2f}, \sigma={sigma:.2f}$)')

    # Add plot details
    plt.title('Distribution of Token Impact Scores and Gaussian Fit')
    plt.xlabel('Token Impact Score')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True)
    plt.show()

def calculate_sentence_impact(sentence_text, source_prefix, token_impact):
    """Calculate impact score for a sentence by averaging token impacts."""
    sentence_tokens = tokenize_and_filter(sentence_text)
    token_impacts = []
    
    for token in sentence_tokens:
        token_key = f"{source_prefix}_{hashlib.md5(token.encode()).hexdigest()}"
        if token_key in token_impact:
            token_impacts.append(token_impact[token_key])
    
    return np.mean(token_impacts) if token_impacts else 0.0

def calculate_percentage(value, impact_dict):
    """Calculate normalized percentage of impact value within range."""
    if not impact_dict:
        return 0
        
    values = [float(v) for v in impact_dict.values()]
    max_value = max(max(values), 1)  # Avoid division by zero
    min_value = min(min(values), -1)  # Ensure we have a range
    
    if max_value == min_value:
        return 0
    return (float(value) - min_value) / (max_value - min_value)

def get_sentence_color(sentence_info, context_thresholds, query_thresholds):
    """Determine color based on sentence source and impact."""
    source = sentence_info["source"]
    impact = sentence_info["impact"]
    
    if source == "response":
        return COLORBLIND_COLORS["response"]
    
    if source == "pseudo_response":
        return COLORBLIND_COLORS["pseudo_response"]
        
    thresholds_to_check = context_thresholds if source == "context" else query_thresholds
    
    # Sort thresholds from high to low
    sorted_thresholds = sorted(thresholds_to_check, key=lambda x: x[0], reverse=True)
    
    for threshold_value, color in sorted_thresholds:
        if impact > threshold_value:
            # Map standard colors to our colorblind-friendly palette
            if source == "context":
                if color in ["red", "yellow"]:  # Typically highest impact
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
    
    # Default colors
    return COLORBLIND_COLORS["context_low"] if source == "context" else COLORBLIND_COLORS["query_low"]

def prepare_plot_data(sentence_impacts, embedding_function, impact_filter=None):
    """Prepare data for plotting by generating embeddings and collecting attributes."""
    sentences_to_plot = []
    embeddings_to_plot = []
    colors_to_plot = []
    labels_to_plot = []
    sources_to_plot = []
    impact_values = []
    
    processed_sentences = set()  # Track processed sentences to avoid duplicates
    
    # Apply impact filter if specified
    if impact_filter is not None:
        min_impact, max_impact = impact_filter
        print(f"Filtering sentences with impact between {min_impact:.4f} and {max_impact:.4f}")
    
    for key, info in sentence_impacts.items():
        sentence_text = info["text"]
        impact = info["impact"]
        source = info["source"]
        
        # Skip if empty or already processed
        if not sentence_text or sentence_text in processed_sentences:
            continue
            
        # Apply impact filter if specified
        if impact_filter is not None:
            min_impact, max_impact = impact_filter
            # Skip sentences outside the specified impact range 
            # (except responses which always display)
            if source not in ["response", "pseudo_response", "w_response"] and (impact < min_impact or impact > max_impact):
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
        colors_to_plot.append(info["color"])
        impact_values.append(impact)
        
        # Create a short label that includes source and sentence hash
        source_prefix = source[:1].upper()  # C for context, Q for query, R for response, P for pseudo
        sentence_id = get_sentence_hash(sentence_text)
        label = f"{source_prefix}{sentence_id}"
        labels_to_plot.append(label)
        
        sources_to_plot.append(source)
        processed_sentences.add(sentence_text)
    
    return {
        "sentences": sentences_to_plot,
        "embeddings": np.array(embeddings_to_plot) if embeddings_to_plot else None,
        "colors": colors_to_plot,
        "labels": labels_to_plot,
        "sources": sources_to_plot,
        "impacts": impact_values,
        "sentence_mapping": {get_sentence_hash(s): s for s in sentences_to_plot}
    }

def create_legend_handles(source_colors):
    """Create legend handles and labels based on sources and colors used."""
    legend_handles = []
    legend_labels = []
    
    color_label_map = {
        'context_high': 'Context (High Impact)',
        'context_medium': 'Context (Medium Impact)',
        'context_low': 'Context (Low Impact)',
        'query_high': 'Query (High Impact)',
        'query_medium': 'Query (Medium Impact)',
        'query_low': 'Query (Low Impact)',
        'response': 'Response',
        'pseudo_response': 'Pseudo Response',
        'w_response': 'w Response'
        
    }
    
    # Check which colors we actually used
    for color_key, color_value in COLORBLIND_COLORS.items():
        for source, colors in source_colors.items():
            if color_value in colors:
                legend_handles.append(plt.Line2D([0], [0], marker='o', color='w', 
                                     markerfacecolor=color_value, 
                                     markersize=10, markeredgecolor='black', markeredgewidth=0.5))
                legend_labels.append(color_label_map[color_key])
                break  # Only add this color once
    
    return legend_handles, legend_labels

def print_sentence_lookup_table(plot_data):
    """Print a lookup table mapping sentence IDs to their full text and impact values."""
    print("\nSentence ID Lookup Table:")
    print("-" * 90)
    print(f"{'ID':<8} | {'Source':<10} | {'Impact':<10} | {'Text':<60}")
    print("-" * 90)
    
    for i, label in enumerate(plot_data["labels"]):
        source = plot_data["sources"][i]
        impact = plot_data["impacts"][i] if plot_data["impacts"] else "N/A"
        text = plot_data["sentence_mapping"].get(label[1:], "Unknown")  # Remove source indicator
        print(f"{label:<8} | {source:<10} | {impact:<10.4f} | {truncate_text(text, 60):<60}")

def plot_2d_sentence_embeddings(plot_data):
    """Plot the 2D PCA visualization of sentence embeddings."""
    embeddings_2d = plot_data["embeddings"]
    
    plt.figure(figsize=(12, 10))
    
    # Create scatter plot
    plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
               c=plot_data["colors"], s=80, alpha=0.9, edgecolors='black', linewidth=0.5)

    # Add sentence labels
    for i, label in enumerate(plot_data["labels"]):
        plt.annotate(label, 
                    (embeddings_2d[i, 0], embeddings_2d[i, 1]), 
                    textcoords="offset points",
                    xytext=(5, 5), 
                    ha='center', 
                    fontsize=9,
                    fontweight='bold',
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", alpha=0.7))
    
    # Group colors by source for legend
    source_colors = defaultdict(set)
    for i, source in enumerate(plot_data["sources"]):
        source_colors[source].add(plot_data["colors"][i])
        
    # Create legend
    legend_handles, legend_labels = create_legend_handles(source_colors)
    plt.legend(legend_handles, legend_labels, title="Sentence Source & Impact", 
              loc="upper right", frameon=True, fontsize=10)
    
    # Set plot details
    plt.title("Sentence Embeddings in 2D Space (PCA)", fontsize=14)
    plt.xlabel("Principal Component 1", fontsize=12)
    plt.ylabel("Principal Component 2", fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Add note about sentence IDs and impact range
    impact_values = plot_data["impacts"]
    note_text = "Note: Labels show source (C=Context, Q=Query, R=Response, P=Pseudo) and sentence ID"
    if impact_values:
        impact_range = f"Impact Range: {min(impact_values):.4f} to {max(impact_values):.4f}"
        note_text += f"\n{impact_range}"
        
    plt.figtext(0.5, 0.01, note_text, ha="center", fontsize=10, 
               bbox={"facecolor":"white", "alpha":0.8, "pad":5})
    
    plt.tight_layout()
    plt.show()

def plot_3d_sentence_embeddings(plot_data):
    """Plot the 3D PCA visualization of sentence embeddings."""
    embeddings_3d = plot_data["embeddings"]
    
    fig_3d = plt.figure(figsize=(14, 12))
    ax_3d = fig_3d.add_subplot(111, projection='3d')
    
    # Create scatter plot
    ax_3d.scatter(embeddings_3d[:, 0], embeddings_3d[:, 1], embeddings_3d[:, 2],
                 c=plot_data["colors"], s=100, alpha=0.9, edgecolors='black', linewidth=0.5)
    
    # Add sentence labels
    for i, label in enumerate(plot_data["labels"]):
        ax_3d.text(embeddings_3d[i, 0], embeddings_3d[i, 1], embeddings_3d[i, 2], 
                  label, fontsize=9, fontweight='bold')
    
    # Group colors by source for legend
    source_colors = defaultdict(set)
    for i, source in enumerate(plot_data["sources"]):
        source_colors[source].add(plot_data["colors"][i])
        
    # Create legend
    legend_handles, legend_labels = create_legend_handles(source_colors)
    ax_3d.legend(legend_handles, legend_labels, title="Sentence Source & Impact", 
               loc="upper right", frameon=True, fontsize=10)
    
    # Set plot details
    ax_3d.set_title("Sentence Embeddings in 3D Space (PCA)", fontsize=14)
    ax_3d.set_xlabel("Principal Component 1", fontsize=12)
    ax_3d.set_ylabel("Principal Component 2", fontsize=12)
    ax_3d.set_zlabel("Principal Component 3", fontsize=12)
    
    # Add note about sentence IDs and impact range
    impact_values = plot_data["impacts"]
    note_text = "Note: Labels show source (C=Context, Q=Query, R=Response, P=Pseudo) and sentence ID"
    if impact_values:
        impact_range = f"Impact Range: {min(impact_values):.4f} to {max(impact_values):.4f}"
        note_text += f"\n{impact_range}"
        
    fig_3d.text(0.5, 0.01, note_text, ha="center", fontsize=10, 
               bbox={"facecolor":"white", "alpha":0.8, "pad":5})
    
    plt.tight_layout()
    plt.show()

def process_embeddings_for_plotting(plot_data):
    """Process embeddings with PCA for 2D and 3D visualization."""
    embeddings_array = plot_data["embeddings"]
    
    if embeddings_array is None or len(embeddings_array) == 0:
        print("No valid embeddings to process.")
        return None, None
        
    # Check dimensions
    n_samples, n_features = embeddings_array.shape
    print(f"Embeddings array shape: ({n_samples}, {n_features})")
    
    # Check requirements for PCA
    if n_samples <= 1:
        print("PCA requires at least 2 samples. Cannot perform dimensionality reduction.")
        return None, None
    
    # Process for 2D visualization
    embeddings_2d = None
    pca_dim_2d = min(2, n_samples, n_features)
    if pca_dim_2d == 2:
        try:
            pca_2d = PCA(n_components=pca_dim_2d)
            embeddings_2d = pca_2d.fit_transform(embeddings_array)
            print(f"PCA 2D reduction complete. Explained variance ratio: {pca_2d.explained_variance_ratio_}")
        except Exception as e:
            print(f"Error during 2D PCA: {e}")
    else:
        print(f"Cannot perform 2D PCA (need n_components=2, samples={n_samples}, features={n_features}).")
    
    # Process for 3D visualization
    embeddings_3d = None
    pca_dim_3d = min(3, n_samples, n_features)
    if pca_dim_3d == 3:
        try:
            pca_3d = PCA(n_components=pca_dim_3d)
            embeddings_3d = pca_3d.fit_transform(embeddings_array)
            print(f"PCA 3D reduction complete. Explained variance ratio: {pca_3d.explained_variance_ratio_}")
        except Exception as e:
            print(f"Error during 3D PCA: {e}")
    else:
        print(f"Cannot perform 3D PCA (need n_components=3, samples={n_samples}, features={n_features}).")
        
    return embeddings_2d, embeddings_3d

def build_sentence_impacts(analysis_results):
    """Build a dictionary of sentence impacts from RAG analysis results."""
    # Extract data from analysis_results
    context = analysis_results.get("context", "")
    query = analysis_results.get("query", "")
    response = analysis_results.get("response", "")
    token_impact = analysis_results.get("token_impact", {})
    pseudo_response = analysis_results.get("pseudo_response", "")
    threshold_config = analysis_results.get("threshold_config", {})
    thresholds = analysis_results.get("thresholds", {})
    
    context_thresholds = thresholds.get("context", [])
    query_thresholds = thresholds.get("query", [])
    
    # Separate token impacts by source
    context_impact = {k: v for k, v in token_impact.items() if k.startswith("context_")}
    query_impact = {k: v for k, v in token_impact.items() if k.startswith("query_")}
    
    # Extract sentences
    context_sentences = []
    for chunk in context:
        context_sentences.extend(extract_sentences(chunk))
    
    # Handle query - treat as single sentence if short
    query_sentences = extract_sentences(query) if len(query.split()) > 5 else [query]
    response_sentences = extract_sentences(response)
    
    print(f"Extracted {len(context_sentences)} context sentences, {len(query_sentences)} query sentences, "
          f"and {len(response_sentences)} response sentences.")
    
    # Build sentence impacts dictionary
    sentence_impacts = {}
    
    # Process context sentences
    for i, sentence in enumerate(context_sentences):
        impact = calculate_sentence_impact(sentence, "context", token_impact)
        impact = calculate_percentage(impact, context_impact)
        key = f"context_sentence_{i}"
        sentence_impacts[key] = {
            "text": sentence,
            "impact": impact,
            "source": "context",
            "color": get_sentence_color({"source": "context", "impact": impact}, 
                                        context_thresholds, query_thresholds)
        }
    
    # Process query sentences
    for i, sentence in enumerate(query_sentences):
        impact = calculate_sentence_impact(sentence, "query", token_impact)
        impact = calculate_percentage(impact, query_impact)
        key = f"query_sentence_{i}"
        sentence_impacts[key] = {
            "text": sentence,
            "impact": impact,
            "source": "query",
            "color": get_sentence_color({"source": "query", "impact": impact}, 
                                       context_thresholds, query_thresholds)
        }
    
    # Add response sentences
    for i, sentence in enumerate(response_sentences):
        key = f"response_sentence_{i}"
        sentence_impacts[key] = {
            "text": sentence,
            "impact": 0.0,
            "source": "response",
            "color": COLORBLIND_COLORS["response"]
        }
    
    # Add pseudo response if available
    if pseudo_response:
        sentence_impacts["pseudo_response"] = {
            "text": pseudo_response,
            "impact": 0.0,
            "source": "pseudo_response",
            "color": COLORBLIND_COLORS["pseudo_response"]
        }
    if response:
        sentence_impacts["w_response"] = {
            "text": response,
            "impact": 0.0,
            "source": "w_response",
            "color": COLORBLIND_COLORS["w_response"]
        }
    
    return sentence_impacts

def plot_embeddings(analysis_results, embedding_function, impact_filter=None):
    """
    Main entry point for plotting sentence embeddings.
    
    Args:
        analysis_results: Dictionary with analysis results from analyze_rag().
        embedding_function: Function to generate embeddings for sentences.
        impact_filter: Optional tuple (min_impact, max_impact) to filter sentences by impact.
    """
    # Step 1: Build sentence impacts
    sentence_impacts = build_sentence_impacts(analysis_results)
    
    # Step 2: Prepare data for plotting
    plot_data = prepare_plot_data(sentence_impacts, embedding_function, impact_filter)
    
    if plot_data["embeddings"] is None or len(plot_data["embeddings"]) == 0:
        print("No valid embeddings for plotting.")
        return
    
    # Step 3: Process embeddings with PCA
    embeddings_2d, embeddings_3d = process_embeddings_for_plotting(plot_data)
    
    # Step 4: Create plots
    if embeddings_2d is not None:
        print("Plotting 2D embeddings...")
        plot_data_2d = plot_data.copy()
        plot_data_2d["embeddings"] = embeddings_2d
        plot_2d_sentence_embeddings(plot_data_2d)
    else:
        print("Skipping 2D plot - insufficient dimensionality.")
    
    if embeddings_3d is not None:
        print("Plotting 3D embeddings...")
        plot_data_3d = plot_data.copy()
        plot_data_3d["embeddings"] = embeddings_3d
        plot_3d_sentence_embeddings(plot_data_3d)
    else:
        print("Skipping 3D plot - insufficient dimensionality.")
    
    # Print sentence lookup table
    print_sentence_lookup_table(plot_data)