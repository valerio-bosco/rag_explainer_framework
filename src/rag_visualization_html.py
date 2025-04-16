# rag_visualization_html.py
import numpy as np
import spacy

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

def visualize_results_html(analysis_results):
    """
    Generates an HTML visualization of the token and permutation impact,
    considering separate thresholds with levels.
    
    Args:
        analysis_results: Dictionary with analysis results from analyze_rag()
        
    Returns:
        HTML string representation of the analysis
    """
    token_impact = analysis_results["token_impact"]
    context = analysis_results["context"]
    query = analysis_results["query"]
    permutation_impact = analysis_results["permutation_impact"]
    threshold_config = analysis_results["threshold_config"]
    thresholds = analysis_results["thresholds"]
    
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
    context_colors = threshold_config.get("context", {}).get("colors", [])
    context_levels = threshold_config.get("context", {}).get("levels", [])
    avg_context = np.mean([v for k, v in token_impact.items() if k.startswith("context_")]) if [v for k, v in token_impact.items() if k.startswith("context_")] else 0
    std_context = np.std([v for k, v in token_impact.items() if k.startswith("context_")]) if [v for k, v in token_impact.items() if k.startswith("context_")] else 0
    for i, color in enumerate(context_colors):
        html += f'<div class="legend-item"><span class="legend-color" style="background-color: {color};"></span> Context (>{avg_context + context_levels[i]*std_context:.2f})</div>'

    query_colors = threshold_config.get("query", {}).get("colors", [])
    query_levels = threshold_config.get("query", {}).get("levels", [])
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

def save_html_report(analysis_results, filename="rag_explanation.html"):
    """
    Generates and saves an HTML report to a file.
    
    Args:
        analysis_results: Dictionary with analysis results
        filename: Output filename
    """
    html_output = visualize_results_html(analysis_results)
    with open(filename, "w") as f:
        f.write(html_output)
    print(f"\nExplanation saved to {filename}")