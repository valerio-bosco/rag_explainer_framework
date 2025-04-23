# rag_visualization_html.py
import hashlib
import numpy as np
import spacy

from rag_evaluation import tokenize_and_filter

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# def visualize_results_html(analysis_results):
#     """
#     Generates an HTML visualization of the token and permutation impact,
#     considering separate thresholds with levels.
    
#     Args:
#         analysis_results: Dictionary with analysis results from analyze_rag()
        
#     Returns:
#         HTML string representation of the analysis
#     """
#     token_impact = analysis_results["token_impact"]
     
#     context = analysis_results["context"]
#     query = analysis_results["query"] 
#     permutation_impact = analysis_results["permutation_impact"]
#     threshold_config = analysis_results["threshold_config"]
#     thresholds = analysis_results["thresholds"]
    
#     print(analysis_results)
#     html = """
#     <!DOCTYPE html>
#     <html>
#     <head>
#         <title>RAG Explainability Analysis</title>
#         <style>
#             body { font-family: sans-serif; margin: 20px; }
#             h1, h2 { color: #333; }
#             .explanation-section { margin-bottom: 20px; border: 1px solid #eee; padding: 15px; border-radius: 5px; }
#             .token-highlight { cursor: help; border-radius: 3px; padding: 0.1em 0.3em; color: black; }
#             .legend { margin-top: 10px; padding: 10px; border: 1px solid #ddd; border-radius: 5px; background-color: #f9f9f9; }
#             .legend-item { display: inline-block; margin-right: 15px; }
#             .legend-color { width: 20px; height: 15px; display: inline-block; vertical-align: middle; margin-right: 5px; border-radius: 3px; }
#             .permutation-item { margin-bottom: 5px; padding: 8px; border: 1px solid #eee; border-radius: 3px; }
#             .permutation-item strong { font-weight: bold; color: #555; }
#         </style>
#     </head>
#     """
#     html += f"""
#     <body>
#         <h1>RAG Explainability Analysis</h1>
#         <p>Context tokens and query tokens are highlighted based on their impact levels.</p>

#         <div class="explanation-section">
#             <h2>Token Impact</h2>
#             <p>The following highlights indicate the influence of each token on the response. Hover over a highlighted word to see its impact percentage.</p>
#             <div>
#                 <h3>Context:</h3>
#                 <p>
#     """
#     context_tokens = tokenize_and_filter(context)
#     context_impacts = {k.split("context_")[1]: v for k, v in token_impact.items() if k.startswith("context_")}

#     def get_highlight_class(impact, thresholds):
#         for threshold, color in reversed(thresholds):
#             if impact > threshold:
#                 if color == 'lightcoral':
#                     return "impact-level-1"
#                 elif color == 'salmon':
#                     return "impact-level-2"
#                 elif color == 'red':
#                     return "impact-level-3"
#                 elif color == 'yellow':
#                     return "impact-level-0"
#         return None

#     if context_impacts:
#         context_thresholds_with_colors = thresholds.get("context", [])
        
#         for token in context_tokens:
#             key = hashlib.md5(token.encode()).hexdigest()
#             if  key in context_impacts.keys():
#                 impact = context_impacts[key]
#                 highlight_class = get_highlight_class(impact, context_thresholds_with_colors)
#                 percentage = f"{impact * 100:.2f}%"
#                 if highlight_class:
#                     html += f"<span class='token-highlight {highlight_class}' title='Impact: {percentage}'>{token}</span> "
#                 else:
#                     html += f"{token} "
#             else:
#                 html += f"{token} "
#     else:
#         html += f"{context} "
#     html += "</p>"

#     html += """
#                     <h3>Query:</h3>
#                     <p>
#     """
#     query_tokens = tokenize_and_filter(query)
#     query_impacts = {k.split("query_")[1]: v for k, v in token_impact.items() if k.startswith("query_")}

#     def get_query_highlight_class(impact, thresholds):
#         for threshold, color in reversed(thresholds):
#             if impact > threshold:
#                 if color == 'lightblue':
#                     return "impact-level-1-query"
#                 elif color == 'cornflowerblue':
#                     return "impact-level-2-query"
#                 elif color == 'blue':
#                     return "impact-level-3-query"
#                 elif color == 'green':
#                     return "impact-level-0-query"
#         return None

#     if query_impacts:
#         query_thresholds_with_colors = thresholds.get("query", [])
#         for token in query_tokens:
#             key = hashlib.md5(token.encode()).hexdigest()
#             if  token in query_impacts.keys():
#                 impact = query_impacts[key]
#                 highlight_class = get_query_highlight_class(impact, query_thresholds_with_colors)
#                 percentage = f"{impact * 100:.2f}%"
#                 if highlight_class:
#                     html += f"<span class='token-highlight {highlight_class}' title='Impact: {percentage}'>{token}</span> "
#                 else:
#                     html += f"{token} "
#             else:
#                 html += f"{token} "
#     else:
#         html += f"{query} "
#     html += "</p>"

#     html += """
#                 </div>
#                 <div class="legend">
#                     <h3>Legend: Token Impact</h3>
#     """
#     context_colors = threshold_config.get("context", {}).get("colors", [])
#     context_levels = threshold_config.get("context", {}).get("levels", [])
#     avg_context = np.mean([v for k, v in token_impact.items() if k.startswith("context_")]) if [v for k, v in token_impact.items() if k.startswith("context_")] else 0
#     std_context = np.std([v for k, v in token_impact.items() if k.startswith("context_")]) if [v for k, v in token_impact.items() if k.startswith("context_")] else 0
#     for i, color in enumerate(context_colors):
#         html += f'<div class="legend-item"><span class="legend-color" style="background-color: {color};"></span> Context (>{avg_context + context_levels[i]*std_context:.2f})</div>'

#     query_colors = threshold_config.get("query", {}).get("colors", [])
#     query_levels = threshold_config.get("query", {}).get("levels", [])
#     avg_query = np.mean([v for k, v in token_impact.items() if k.startswith("query_")]) if [v for k, v in token_impact.items() if k.startswith("query_")] else 0
#     std_query = np.std([v for k, v in token_impact.items() if k.startswith("query_")]) if [v for k, v in token_impact.items() if k.startswith("query_")] else 0
#     for i, color in enumerate(query_colors):
#         html += f'<div class="legend-item"><span class="legend-color" style="background-color: {color};"></span> Query (>{avg_query + query_levels[i]*std_query:.2f})</div>'

#     html += """
#                 </div>
#             </div>

#             <div class="explanation-section">
#                 <h2>Context Permutation Impact (if analyzed)</h2>
#     """
#     if permutation_impact:
#         sorted_permutations = sorted(permutation_impact.items(), key=lambda item: item[1], reverse=True)
#         for context_perm, distance in sorted_permutations:
#             html += f"<div class='permutation-item'><strong>Distance: {distance:.4f}</strong> - Context: {context_perm}</div>"
#     else:
#         html += "<p>Context permutation analysis was not performed.</p>"

#     html += """
#             </div>

#         <style>
#             .impact-level-1 { background-color: lightcoral; }
#             .impact-level-2 { background-color: salmon; }
#             .impact-level-3 { background-color: red; }
#             .impact-level-0 { background-color: yellow; }
#             .impact-level-1-query { background-color: lightblue; }
#             .impact-level-2-query { background-color: cornflowerblue; }
#             .impact-level-3-query { background-color: blue; }
#             .impact-level-0-query { background-color: green; }
            
#         </style>
#     </body>
#     </html>
#     """
#     return html
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
    response = analysis_results["response"]
    thresholds = analysis_results["thresholds"]
    
    context_impacts = {k: v for k, v in token_impact.items() if k.startswith("context_")}
    # Get permutation impact if available
    permutation_impact = analysis_results.get("permutation_impact", {})
    
    # Helper function to get appropriate color for a value based on thresholds
    def get_color_for_value(value, threshold_list):
        for threshold, color in sorted(threshold_list, key=lambda x: x[0], reverse=True):
            if value >= threshold:
                return color
        return "inherit"  # Default color if below all thresholds
    
    # Helper function to calculate percentage impact based on max value
    def calculate_percentage(value, impact_dict):
        if not impact_dict:
            return 0
        max_value = max(float(v) for v in impact_dict.values()) 
        max_value = max(max_value, 1)  # Avoid division by zero 
        min_value = min(float(v) for v in impact_dict.values())
        min_value = min(min_value, -1)  # Avoid division by zero
        
        print("MAX:", max_value, "MIN:", min_value, "VALUE:", value)
        if max_value == 0:
            return 0
        return (float(value)-min_value) / (max_value- min_value) * 100
    
    # Process context
    context_sentences = tokenize_and_filter(context)
    context_html = []
    
    for sentence in context_sentences:
        token_hash = "context_"+hashlib.md5(sentence.encode()).hexdigest()
        if token_hash in context_impacts.keys():
            
            impact_value = context_impacts[token_hash]
            color = get_color_for_value(impact_value, thresholds["context"])
            percentage = calculate_percentage(impact_value, context_impacts)
            
            # Create span with tooltip showing impact percentage
            span = f'<span style="background-color: {color};" title="Impact: {percentage:.2f}%">{sentence}</span>'
            context_html.append(span)
        else:
            # If token not found in context impacts, add it without highlighting
            context_html.append(sentence)

    
    # Process query highlighting if query impacts exist
    query_sentences = tokenize_and_filter(query)
    query_impacts = {k: v for k, v in token_impact.items() if k.startswith("query_")}
    query_html = []
    
    # Map hashes for query tokens
    for token in query_sentences:
        token_hash = "query_" + hashlib.md5(token.encode()).hexdigest()
        if token_hash in query_impacts.keys():
            impact_value = query_impacts[token_hash]
            color = get_color_for_value(impact_value, thresholds["query"])
            percentage = calculate_percentage(impact_value, query_impacts)
            
            # Create span with tooltip showing impact percentage
            span = f'<span style="background-color: {color};" title="Impact: {percentage:.2f}%">{token}</span>'
            query_html.append(span) 
        else:
            # If token not found in query impacts, add it without highlighting
            query_html.append(token)
    
   
    
    # Format permutation impact information if available
    permutation_html = ""
    if permutation_impact:
        permutation_html = "<h3>Permutation Impact</h3><ul>"
        for perm, impact in sorted(permutation_impact.items(), key=lambda x: float(x[1]), reverse=True):
            percentage = (float(impact) / max(float(v) for v in permutation_impact.values())) * 100
            permutation_html += f"<li>Permutation {perm}: {percentage:.2f}%</li>"
        permutation_html += "</ul>"
    
    # Create legend for color thresholds
    context_legend = "<h4>Context Impact Legend</h4><ul>"
    for threshold, color in sorted(thresholds["context"], key=lambda x: x[0]):
        context_legend += f'<li><span style="display:inline-block; width:20px; height:20px; background-color:{color};"></span> > {float(calculate_percentage(threshold, context_impacts)):.4f}</li>'
    context_legend += "</ul>"
    
    query_legend = "<h4>Query Impact Legend</h4><ul>"
    for threshold, color in sorted(thresholds["query"], key=lambda x: x[0]):
        query_legend += f'<li><span style="display:inline-block; width:20px; height:20px; background-color:{color};"></span> > {float(calculate_percentage(threshold, query_impacts)):.4f}</li>'
    query_legend += "</ul>"
    
    # Build the complete HTML
    html = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>RAG Analysis Visualization</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            .container {{ max-width: 1200px; margin: 0 auto; }}
            .context-section, .query-section , .response-section{{ 
                margin-bottom: 20px; 
                padding: 20px; 
                border-radius: 5px;
                border: 1px solid #ddd;
            }}
            h2 {{ color: #333; }}
            span {{ padding: 2px 0; }}
            .legend-container {{ 
                display: flex; 
                justify-content: space-between;
                flex-wrap: wrap;
            }}
            .legend {{ 
                margin-right: 20px; 
                padding: 10px; 
                border: 1px solid #ddd; 
                border-radius: 5px; 
                margin-bottom: 20px;
            }}
            ul {{ list-style-type: none; padding-left: 10px; }}
            li {{ margin-bottom: 5px; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>RAG Analysis Visualization</h1>
            
            <div class="legend-container">
                <div class="legend">
                    {context_legend}
                </div>
                <div class="legend">
                    {query_legend}
                </div>
            </div>
            
            <div class="query-section">
                <h2>Query</h2>
                <p>{"".join(query_html)}</p>
            </div>
            
            <div class="context-section">
                <h2>Context</h2>
                <p>{"".join(context_html)}</p>
            </div>
             <div class="response-section">
                <h2>Response</h2>
                <p>{response}</p>
            </div>    
            {permutation_html}
        </div>
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