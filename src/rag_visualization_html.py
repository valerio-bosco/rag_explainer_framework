# rag_visualization_html.py
import hashlib
import numpy as np
import spacy

from rag_evaluation import tokenize_and_filter

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

def visualize_results_html(analysis_results):
    """
    Generates an elegant HTML visualization of the token and permutation impact,
    with a modern, clean design and interactive elements.
    
    Args:
        analysis_results: Dictionary with analysis results from analyze_rag()
        
    Returns:
        HTML string representation of the analysis
    """
    token_impact = analysis_results["token_impact"]
    sentence_impact = analysis_results["sentence_impact"]
    context = analysis_results["context"]
    query = analysis_results["query"]
    response = analysis_results["response"]
    thresholds = analysis_results["thresholds"]
    
    context_impacts = {k: v for k, v in token_impact.items() if k.startswith("context_")}
    permutation_impact = analysis_results.get("permutation_impact", {})
    
    # Helper function to get appropriate color for a value based on thresholds
    def get_color_for_value(value, threshold_list):
        for threshold, color in sorted(threshold_list, key=lambda x: x[0], reverse=True):
            if value >= threshold:
                return color
        return "transparent"  # Default transparent background if below all thresholds
    
    # Helper function to calculate percentage impact based on max and min values
    def calculate_percentage(value, impact_dict):
        if not impact_dict:
            return 0
        max_value = max(float(v) for v in impact_dict.values()) 
        max_value = max(max_value, 1)  # Avoid division by zero 
        min_value = min(float(v) for v in impact_dict.values())
        min_value = min(min_value, -1)  # Avoid division by zero
        
        if max_value == 0:
            return 0
        return (float(value)-min_value) / (max_value- min_value) * 100
    context_html = []
    chunk_count = 0
    
    for chunk in context:
        chunk_count += 1
        context_sentences = tokenize_and_filter(chunk)
        context_html.append(f"<div class='chunk-header'>Chunk {chunk_count}</div>")
        
        chunk_content = []
        for sentence in context_sentences:
            token_hash = "context_"+hashlib.md5(sentence.encode()).hexdigest()
            
            if token_hash in context_impacts.keys():
                impact_value = context_impacts[token_hash]
                color = get_color_for_value(impact_value, thresholds["context"])
                percentage = calculate_percentage(impact_value, context_impacts)
                
                # Calculate opacity for visual indication of impact
                opacity = min(0.9, max(0.15, percentage/100))
                
                # Create span with tooltip showing impact percentage
                span = f'<span class="highlight" style="background-color: {color}; opacity: {opacity};" data-impact="{percentage:.2f}%">{sentence}</span>'
                chunk_content.append(span)
            else:
                # If token not found in context impacts, add it without highlighting
                chunk_content.append(f'<span>{sentence}</span>')
                
        context_html.append("<div class='chunk-content'>" + "".join(chunk_content) + "</div>")
    
    # Process query highlighting with improved visualization
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
            
            # Calculate opacity for visual indication of impact
            opacity = min(0.9, max(0.15, percentage/100))
            
            # Create span with tooltip showing impact percentage
            span = f'<span class="highlight" style="background-color: {color}; opacity: {opacity};" data-impact="{percentage:.2f}%">{token}</span>'
            query_html.append(span) 
        else:
            # If token not found in query impacts, add it without highlighting
            query_html.append(token)
    
    # Process response sentences and their relationship to context
    response_sentences = tokenize_and_filter(response)
    response_sentence_impacts = {k: v for k, v in sentence_impact.items() if k.startswith("sentence_")}
    
    response_html = "<div class='response-analysis'>"
    response_html += "<h3>Response Sentence Analysis</h3>"
    response_html += "<div class='sentence-cards'>"

    for sentence in response_sentences:
        sentence_hash = "sentence_" + hashlib.md5(sentence.encode()).hexdigest()
        if sentence_hash in response_sentence_impacts.keys():
            impact_value_tuples = response_sentence_impacts[sentence_hash]
            impact_value_tuples = sorted(impact_value_tuples, key=lambda x: float(x[1]))
            
            # Create a card for each response sentence
            response_html += """
            <div class='sentence-card'>
                <div class='response-sentence'>
                    <svg class="quote-icon" viewBox="0 0 24 24" width="20" height="20">
                        <path d="M14,17H17L19,13V7H13V13H16M6,17H9L11,13V7H5V13H8L6,17Z" />
                    </svg>
                    <span>""" + sentence + """</span>
                </div>
                <div class='similar-contexts'>
                    <h4>Most similar context sentences</h4>
                    <ul>
            """
            
            # Add each related context sentence with a similarity score
            for idx, (context_sentence, impact_value) in enumerate(impact_value_tuples):
                # Create a visual indicator of similarity strength
                similarity_percentage = min(100, max(0, float(impact_value) * 100))
                # Color gradient: green (high) to amber (medium) to red (low)
                hue = min(120, max(0, similarity_percentage * 1.2))
                progress_color = f"hsl({hue}, 80%, 45%)"
                
                response_html += f"""
                    <li class="{'primary' if idx == 0 else ''}">
                        <div class='context-sentence'>
                            <svg class="link-icon" viewBox="0 0 24 24" width="16" height="16">
                                <path d="M3.9,12C3.9,10.29 5.29,8.9 7,8.9H11V7H7A5,5 0 0,0 2,12A5,5 0 0,0 7,17H11V15.1H7C5.29,15.1 3.9,13.71 3.9,12M8,13H16V11H8V13M17,7H13V8.9H17C18.71,8.9 20.1,10.29 20.1,12C20.1,13.71 18.71,15.1 17,15.1H13V17H17A5,5 0 0,0 22,12A5,5 0 0,0 17,7Z" />
                            </svg>
                            {context_sentence}
                        </div>
                        <div class='similarity-container'>
                            <div class='similarity-score'>Similarity: <strong>{float(impact_value):.3f}</strong></div>
                            <div class='similarity-bar-container'>
                                <div class='similarity-bar' style='width: {similarity_percentage}%; background-color: {progress_color};'></div>
                            </div>
                        </div>
                    </li>
                """
            
            response_html += """
                    </ul>
                </div>
            </div>
            """
    
    response_html += "</div></div>"
    
    # Format permutation impact information with improved visualization
    permutation_html = ""
    if permutation_impact:
        permutation_html = "<div class='permutation-section'>"
        permutation_html += "<h3>Permutation Impact Analysis</h3>"
        
        # Extract original value for comparison
        max_perm = permutation_impact["original"]
        
        # Create a table for permutation impacts
        permutation_html += """
        <table class="permutation-table">
            <thead>
                <tr>
                    <th>Permutation</th>
                    <th>Impact</th>
                    <th>Change</th>
                    <th>Visualization</th>
                </tr>
            </thead>
            <tbody>
        """
        
        # Add row for original arrangement
        permutation_html += f"""
            <tr class="original-row">
                <td>Original</td>
                <td>{float(max_perm):.4f}</td>
                <td>Baseline</td>
                <td>
                    <div class="permutation-bar-container">
                        <div class="permutation-bar baseline" style="width: 100%"></div>
                    </div>
                </td>
            </tr>
        """
        
        # Sort permutations by impact
        sorted_perms = sorted(
            [(k, v) for k, v in permutation_impact.items() if k != "original"],
            key=lambda x: float(x[1])
        )
        
        for perm, impact in sorted_perms:
            percentage_change = (float(impact) / max_perm) * 100 - 100
            is_improvement = percentage_change > 0
            
            # Calculate width based on relative impact
            bar_width = min(100, max(5, (float(impact) / max_perm) * 100))
            
            # Determine color based on improvement or decline
            bar_color = "var(--color-success)" if is_improvement else "var(--color-error)"
            
            permutation_html += f"""
                <tr>
                    <td>{perm}</td>
                    <td>{float(impact):.4f}</td>
                    <td class="{'' if is_improvement else 'negative'}">
                        <svg class="trend-icon" viewBox="0 0 24 24" width="16" height="16">
                            <path d="{'M16,6L12,2L8,6H11V13H13V6H16M8,18L12,22L16,18H13V11H11V18H8Z' if is_improvement else 'M16,18L12,22L8,18H11V11H13V18H16M8,6L12,2L16,6H13V13H11V6H8Z'}" />
                        </svg>
                        {abs(percentage_change):.2f}%
                    </td>
                    <td>
                        <div class="permutation-bar-container">
                            <div class="permutation-bar" style="width: {bar_width}%; background-color: {bar_color}"></div>
                        </div>
                    </td>
                </tr>
            """
        
        permutation_html += """
            </tbody>
        </table>
        </div>
        """
    
    # Create elegant legend with visual indicators
    context_legend = """
    <div class="legend-header">Context Impact</div>
    <div class="color-scale">
    """
    
    # Sort threshold colors from highest to lowest
    sorted_thresholds = sorted(thresholds["context"], key=lambda x: x[0], reverse=True)
    
    # Generate color scale stops
    for i, (threshold, color) in enumerate(sorted_thresholds):
        percentage = calculate_percentage(threshold, context_impacts)
        context_legend += f"""
        <div class="color-stop">
            <div class="color-sample" style="background-color: {color}"></div>
            <div class="threshold-value">{float(percentage):.1f}%+</div>
        </div>
        """
    
    context_legend += """
    </div>
    """
    
    query_legend = """
    <div class="legend-header">Query Impact</div>
    <div class="color-scale">
    """
    
    # Sort threshold colors from highest to lowest
    sorted_thresholds = sorted(thresholds["query"], key=lambda x: x[0], reverse=True)
    
    # Generate color scale stops
    for threshold, color in sorted_thresholds:
        percentage = calculate_percentage(threshold, query_impacts)
        query_legend += f"""
        <div class="color-stop">
            <div class="color-sample" style="background-color: {color}"></div>
            <div class="threshold-value">{float(percentage):.1f}%+</div>
        </div>
        """
    
    query_legend += """
    </div>
    """
    
    # Include JavaScript for interactive features
    javascript = """
    <script>
    document.addEventListener('DOMContentLoaded', function() {
        // Add hover tooltips to highlighted spans
        const highlights = document.querySelectorAll('.highlight');
        highlights.forEach(span => {
            span.addEventListener('mouseenter', function(e) {
                const impact = this.getAttribute('data-impact');
                
                // Create tooltip
                const tooltip = document.createElement('div');
                tooltip.className = 'tooltip';
                tooltip.textContent = `Impact: ${impact}`;
                
                // Position tooltip
                const rect = this.getBoundingClientRect();
                tooltip.style.left = `${rect.left + window.scrollX}px`;
                tooltip.style.top = `${rect.top + window.scrollY - 30}px`;
                
                // Add to DOM
                document.body.appendChild(tooltip);
                
                // Store reference
                this.tooltip = tooltip;
            });
            
            span.addEventListener('mouseleave', function() {
                if (this.tooltip) {
                    this.tooltip.remove();
                    this.tooltip = null;
                }
            });
        });
        
        // Add toggle functionality for sentence cards
        const sentenceCards = document.querySelectorAll('.sentence-card');
        sentenceCards.forEach(card => {
            card.addEventListener('click', function() {
                this.classList.toggle('expanded');
            });
        });
    });
    </script>
    """
    
    # Build the complete HTML with elegant styling
    html = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>RAG Analysis Visualization</title>
        <style>
            :root {{
                --color-primary: #3a86ff;
                --color-primary-light: #60a5fa;
                --color-primary-dark: #1e40af;
                --color-secondary: #4b5563;
                --color-accent: #8b5cf6;
                --color-success: #10b981;
                --color-warning: #f59e0b;
                --color-error: #ef4444;
                --color-text: #1f2937;
                --color-text-light: #6b7280;
                --color-text-lightest: #9ca3af;
                --color-background: #f9fafb;
                --color-card: #ffffff;
                --color-border: #e5e7eb;
                --color-highlight: #fef3c7;
                --shadow-sm: 0 1px 2px 0 rgba(0, 0, 0, 0.05);
                --shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.1), 0 1px 2px 0 rgba(0, 0, 0, 0.06);
                --shadow-md: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
                --shadow-lg: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
                --rounded-sm: 4px;
                --rounded: 8px;
                --rounded-md: 12px;
                --rounded-lg: 16px;
                --transition: all 0.3s ease;
                --font-sans: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
                --font-mono: 'Roboto Mono', 'SFMono-Regular', Consolas, 'Liberation Mono', Menlo, monospace;
            }}
            
            * {{
                box-sizing: border-box;
                margin: 0;
                padding: 0;
            }}
            
            body {{
                font-family: var(--font-sans);
                font-size: 16px;
                line-height: 1.5;
                color: var(--color-text);
                background-color: var(--color-background);
                padding: 2rem;
            }}
            
            .container {{
                max-width: 1200px;
                margin: 0 auto;
            }}
            
            /* Typography */
            h1, h2, h3, h4, h5, h6 {{
                margin: 0;
                line-height: 1.2;
                font-weight: 600;
                color: var(--color-text);
            }}
            
            h1 {{
                font-size: 2.5rem;
                font-weight: 700;
                margin-bottom: 2rem;
                color: var(--color-primary-dark);
                position: relative;
                display: inline-block;
            }}
            
            h1::after {{
                content: '';
                position: absolute;
                bottom: -10px;
                left: 0;
                width: 60px;
                height: 4px;
                background-color: var(--color-primary);
                border-radius: 2px;
            }}
            
            h2 {{
                font-size: 1.5rem;
                margin-bottom: 1rem;
                color: var(--color-primary-dark);
            }}
            
            h3 {{
                font-size: 1.25rem;
                margin-bottom: 1rem;
                color: var(--color-primary);
            }}
            
            h4 {{
                font-size: 1rem;
                color: var(--color-secondary);
                margin-bottom: 0.5rem;
            }}
            
            p {{
                margin-bottom: 1rem;
            }}
            
            /* Card Layouts */
            .section-card {{
                background-color: var(--color-card);
                border-radius: var(--rounded-md);
                padding: 1.5rem;
                margin-bottom: 2rem;
                box-shadow: var(--shadow);
                border: 1px solid var(--color-border);
                transition: var(--transition);
            }}
            
            .section-card:hover {{
                box-shadow: var(--shadow-md);
            }}
            
            /* Legend */
            .legend-container {{
                display: flex;
                justify-content: space-between;
                gap: 1.5rem;
                margin-bottom: 2rem;
                flex-wrap: wrap;
            }}
            
            .legend {{
                flex: 1;
                min-width: 200px;
                background-color: var(--color-card);
                padding: 1.25rem;
                border-radius: var(--rounded);
                box-shadow: var(--shadow);
                border: 1px solid var(--color-border);
            }}
            
            .legend-header {{
                font-weight: 600;
                margin-bottom: 1rem;
                color: var(--color-primary-dark);
                font-size: 1rem;
            }}
            
            .color-scale {{
                display: flex;
                flex-direction: column;
                gap: 0.75rem;
            }}
            
            .color-stop {{
                display: flex;
                align-items: center;
                gap: 0.5rem;
            }}
            
            .color-sample {{
                width: 20px;
                height: 20px;
                border-radius: var(--rounded-sm);
                border: 1px solid rgba(0,0,0,0.1);
            }}
            
            .threshold-value {{
                font-size: 0.875rem;
                color: var(--color-text-light);
            }}
            
            /* Query Section */
            .query-section {{
                background-color: var(--color-card);
                border-radius: var(--rounded-md);
                padding: 1.5rem;
                margin-bottom: 2rem;
                box-shadow: var(--shadow);
                border: 1px solid var(--color-border);
                position: relative;
                overflow: hidden;
            }}
            
            .query-section::before {{
                content: 'Query';
                position: absolute;
                top: 0;
                right: 0;
                background-color: var(--color-primary);
                color: white;
                font-size: 0.75rem;
                font-weight: 600;
                padding: 0.25rem 0.75rem;
                border-bottom-left-radius: var(--rounded-sm);
                text-transform: uppercase;
                letter-spacing: 0.05em;
            }}
            
            /* Context Section */
            .context-section {{
                background-color: var(--color-card);
                border-radius: var(--rounded-md);
                padding: 1.5rem;
                margin-bottom: 2rem;
                box-shadow: var(--shadow);
                border: 1px solid var(--color-border);
                position: relative;
                overflow: hidden;
            }}
            
            .context-section::before {{
                content: 'Context';
                position: absolute;
                top: 0;
                right: 0;
                background-color: var(--color-secondary);
                color: white;
                font-size: 0.75rem;
                font-weight: 600;
                padding: 0.25rem 0.75rem;
                border-bottom-left-radius: var(--rounded-sm);
                text-transform: uppercase;
                letter-spacing: 0.05em;
            }}
            
            .chunk-header {{
                font-weight: 600;
                margin-top: 1.25rem;
                margin-bottom: 0.5rem;
                padding-bottom: 0.5rem;
                border-bottom: 1px solid var(--color-border);
                color: var(--color-secondary);
                font-size: 0.9rem;
            }}
            
            .chunk-content {{
                line-height: 1.6;
                margin-bottom: 1rem;
            }}
            
            .highlight {{
                border-radius: 3px;
                transition: all 0.2s ease;
                cursor: pointer;
                padding: 2px 0;
                position: relative;
            }}
            
            .highlight:hover {{
                opacity: 0.8 !important;
            }}
            
            .tooltip {{
                position: absolute;
                background-color: rgba(0, 0, 0, 0.8);
                color: white;
                padding: 4px 8px;
                border-radius: 4px;
                font-size: 12px;
                pointer-events: none;
                z-index: 100;
                white-space: nowrap;
            }}
            
            /* Response Section */
            .response-section {{
                background-color: var(--color-card);
                border-radius: var(--rounded-md);
                padding: 1.5rem;
                margin-bottom: 2rem;
                box-shadow: var(--shadow);
                border: 1px solid var(--color-border);
                position: relative;
                overflow: hidden;
            }}
            
            .response-section::before {{
                content: 'Response';
                position: absolute;
                top: 0;
                right: 0;
                background-color: var(--color-accent);
                color: white;
                font-size: 0.75rem;
                font-weight: 600;
                padding: 0.25rem 0.75rem;
                border-bottom-left-radius: var(--rounded-sm);
                text-transform: uppercase;
                letter-spacing: 0.05em;
            }}
            
            /* Response Analysis */
            .response-analysis {{
                background-color: var(--color-card);
                border-radius: var(--rounded-md);
                padding: 1.5rem;
                margin-bottom: 2rem;
                box-shadow: var(--shadow);
                border: 1px solid var(--color-border);
            }}
            
            .sentence-cards {{
                display: grid;
                grid-template-columns: repeat(auto-fill, minmax(400px, 1fr));
                gap: 1.5rem;
                margin-top: 1.5rem;
            }}
            
            .sentence-card {{
                background-color: white;
                border-radius: var(--rounded);
                box-shadow: var(--shadow);
                overflow: hidden;
                border: 1px solid var(--color-border);
                transition: var(--transition);
                cursor: pointer;
            }}
            
            .sentence-card:hover {{
                transform: translateY(-4px);
                box-shadow: var(--shadow-md);
            }}
            
            .response-sentence {{
                padding: 1.25rem;
                background-color: #f0f7ff;
                border-left: 4px solid var(--color-primary);
                font-size: 1rem;
                font-weight: 500;
                color: var(--color-text);
                position: relative;
                display: flex;
                align-items: flex-start;
                gap: 0.5rem;
            }}
            
            .quote-icon {{
                fill: var(--color-primary);
                flex-shrink: 0;
                margin-top: 3px;
            }}
            
            .similar-contexts {{
                padding: 1.25rem;
            }}
            
            .similar-contexts h4 {{
                margin-top: 0;
                margin-bottom: 1rem;
                font-size: 0.9rem;
                color: var(--color-text-light);
                font-weight: 500;
            }}
            
            .similar-contexts ul {{
                list-style-type: none;
                padding: 0;
                margin: 0;
            }}
            
            .similar-contexts li {{
                padding: 0.75rem 0;
                border-bottom: 1px solid var(--color-border);
            }}
            
            .similar-contexts li.primary {{
                background-color: #fcfcfc;
            }}
            
            .similar-contexts li:last-child {{
                border-bottom: none;
            }}
            
            .context-sentence {{
                margin-bottom: 0.5rem;
                font-size: 0.9rem;
                color: var(--color-text);
                display: flex;
                align-items: flex-start;
                gap: 0.5rem;
            }}
            
            .link-icon {{
                fill: var(--color-text-lightest);
                flex-shrink: 0;
                margin-top: 3px;
            }}
            
            .similarity-container {{
                display: flex;
                align-items: center;
                margin-top: 0.5rem;
                gap: 1rem;
            }}
            
            .similarity-score {{
                font-size: 0.8rem;
                color: var(--color-text-light);
                white-space: nowrap;
            }}
            
            .similarity-score strong {{
                color: var(--color-text);
            }}
            
            .similarity-bar-container {{
                flex-grow: 1;
                height: 6px;
                background-color: #eee;
                border-radius: 3px;
                overflow: hidden;
            }}
            
            .similarity-bar {{
                height: 100%;
                border-radius: 3px;
                transition: width 0.3s ease;
            }}
            
            /* Permutation Section */
            .permutation-section {{
                background-color: var(--color-card);
                border-radius: var(--rounded-md);
                padding: 1.5rem;
                margin-bottom: 2rem;
                box-shadow: var(--shadow);
                border: 1px solid var(--color-border);
            }}
            
            .permutation-table {{
                width: 100%;
                border-collapse: collapse;
                margin-top: 1rem;
                font-size: 0.9rem;
            }}
            
            .permutation-table th,
            .permutation-table td {{
                padding: 0.75rem 1rem;
                text-align: left;
                border-bottom: 1px solid var(--color-border);
            }}
            
            .permutation-table th {{
                font-weight: 600;
                color: var(--color-text);
                background-color: #f9fafb;
            }}
            
            .permutation-table tbody tr:hover {{
                background-color: #f9fafb;
            }}
            
            .permutation-table .original-row {{
                background-color: #f0f7ff;
            }}
            
            .permutation-bar-container {{
                width: 100%;
                height: 8px;
                background-color: #eee;
                border-radius: 4px;
                overflow: hidden;
            }}
            
            .permutation-bar {{
                height: 100%;
                border-radius: 4px;
                transition: width 0.3s ease;
            }}
            
            .permutation-bar.baseline {{
                background-color: var(--color-primary);
            }}
            
            .trend-icon {{
                vertical-align: middle;
                margin-right: 4px;
                fill: currentColor;
            }}
            
            .negative {{
                color: var(--color-error);
            }}
            
            /* Responsive Adjustments */
            @media (max-width: 768px) {{
                body {{
                    padding: 1rem;
                }}
                
                .sentence-cards {{
                    grid-template-columns: 1fr;
                }}
            }}
        </style>
        {javascript}
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
            
            {response_html}
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