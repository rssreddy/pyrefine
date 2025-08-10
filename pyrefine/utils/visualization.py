"""
Visualization utilities for SELF-REFINE with Change-of-Thought framework.

Extracted from chart_script.py for creating LangGraph flowcharts.
"""

import plotly.graph_objects as go
from typing import Dict, List, Any, Optional


def create_flowchart_visualization(
    output_path: str = "langgraph_flowchart.png",
    width: int = 800,
    height: int = 900
) -> None:
    """
    Create a flowchart visualization of the LangGraph DAG topology.
    
    Args:
        output_path: Path to save the flowchart image
        width: Image width in pixels
        height: Image height in pixels
    """
    # Data from the LangGraph topology
    data = {
        "nodes": [
            {"id": "START", "type": "start", "label": "START", "x": 400, "y": 50},
            {"id": "initialize", "type": "process", "label": "initialize", 
             "description": "Set up session and reset components", "x": 400, "y": 150},
            {"id": "generate_initial", "type": "process", "label": "generate_initial", 
             "description": "Generate initial response using LLM", "x": 400, "y": 250},
            {"id": "capture_cot", "type": "process", "label": "capture_cot", 
             "description": "Extract Change-of-Thought reasoning", "x": 400, "y": 350},
            {"id": "critic_analysis", "type": "process", "label": "critic_analysis", 
             "description": "Analyze response and provide feedback", "x": 400, "y": 450},
            {"id": "check_stopping", "type": "decision", "label": "check_stopping", 
             "description": "Apply adaptive stopping criteria", "x": 400, "y": 550},
            {"id": "refine_response", "type": "process", "label": "refine_response", 
             "description": "Generate improved response based on feedback", "x": 150, "y": 450},
            {"id": "finalize", "type": "process", "label": "finalize", 
             "description": "Clean up and prepare final results", "x": 650, "y": 650},
            {"id": "END", "type": "end", "label": "END", "x": 650, "y": 750}
        ],
        "edges": [
            {"from": "START", "to": "initialize", "label": ""},
            {"from": "initialize", "to": "generate_initial", "label": ""},
            {"from": "generate_initial", "to": "capture_cot", "label": ""},
            {"from": "capture_cot", "to": "critic_analysis", "label": ""},
            {"from": "critic_analysis", "to": "check_stopping", "label": ""},
            {"from": "check_stopping", "to": "refine_response", "label": "continue"},
            {"from": "refine_response", "to": "capture_cot", "label": ""},
            {"from": "check_stopping", "to": "finalize", "label": "stop"},
            {"from": "finalize", "to": "END", "label": ""}
        ]
    }

    # Create figure
    fig = go.Figure()

    # Color mapping for node types
    color_map = {
        'start': '#1FB8CD',     # Strong cyan
        'process': '#2E8B57',   # Sea green  
        'decision': '#DB4545',  # Bright red
        'end': '#5D878F'        # Cyan
    }

    # Symbol mapping for node types
    symbol_map = {
        'start': 'circle',
        'process': 'square',
        'decision': 'diamond',
        'end': 'circle'
    }

    # Create node position lookup
    node_pos = {node['id']: (node['x'], node['y']) for node in data['nodes']}

    # Add edges with arrows
    for edge in data['edges']:
        from_pos = node_pos[edge['from']]
        to_pos = node_pos[edge['to']]
        
        # Add the line
        fig.add_trace(go.Scatter(
            x=[from_pos[0], to_pos[0]],
            y=[from_pos[1], to_pos[1]],
            mode='lines',
            line=dict(color='gray', width=2),
            showlegend=False,
            hoverinfo='skip'
        ))
        
        # Add arrow head using annotations
        fig.add_annotation(
            x=to_pos[0],
            y=to_pos[1],
            ax=from_pos[0],
            ay=from_pos[1],
            xref='x',
            yref='y',
            axref='x',
            ayref='y',
            showarrow=True,
            arrowhead=2,
            arrowsize=1,
            arrowwidth=2,
            arrowcolor='gray',
            text='',
            standoff=15
        )
        
        # Add edge labels for conditional branches
        if edge['label'] in ['continue', 'stop']:
            mid_x = (from_pos[0] + to_pos[0]) / 2
            mid_y = (from_pos[1] + to_pos[1]) / 2
            
            # Offset the label slightly to avoid overlap
            if edge['label'] == 'continue':
                mid_x -= 30
            else:
                mid_x += 30
                
            fig.add_annotation(
                x=mid_x,
                y=mid_y,
                text=edge['label'],
                showarrow=False,
                font=dict(size=10, color='black'),
                bgcolor='white',
                bordercolor='gray',
                borderwidth=1
            )

    # Group nodes by type for legend
    node_types = {}
    for node in data['nodes']:
        node_type = node['type']
        if node_type not in node_types:
            node_types[node_type] = []
        node_types[node_type].append(node)

    # Add nodes by type
    for node_type, nodes in node_types.items():
        x_coords = [node['x'] for node in nodes]
        y_coords = [node['y'] for node in nodes]
        labels = [node['label'] for node in nodes]
        descriptions = [node.get('description', node['label']) for node in nodes]
        
        # Create hover text with abbreviated descriptions
        hover_text = []
        for i, desc in enumerate(descriptions):
            # Keep full description for hover but truncate if too long
            if len(desc) > 50:
                hover_text.append(desc[:47] + '...')
            else:
                hover_text.append(desc)
        
        # Truncate node labels to fit in 15 characters
        display_labels = []
        for label in labels:
            if len(label) > 15:
                # Split on underscore and abbreviate
                parts = label.split('_')
                if len(parts) > 1:
                    abbreviated = parts[0][:8] + '_' + parts[1][:5]
                else:
                    abbreviated = label[:15]
                display_labels.append(abbreviated)
            else:
                display_labels.append(label)
        
        fig.add_trace(go.Scatter(
            x=x_coords,
            y=y_coords,
            mode='markers+text',
            marker=dict(
                color=color_map[node_type],
                size=40,
                symbol=symbol_map[node_type],
                line=dict(width=2, color='white')
            ),
            text=display_labels,
            textposition='middle center',
            textfont=dict(size=9, color='white'),
            name=node_type.title(),
            hovertext=hover_text,
            hoverinfo='text'
        ))

    # Update layout
    fig.update_layout(
        title="SELF-REFINE with Change-of-Thought: LangGraph DAG Topology",
        showlegend=True,
        legend=dict(orientation='h', yanchor='bottom', y=1.05, xanchor='center', x=0.5),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[0, 800]),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, autorange='reversed', range=[0, 800]),
        plot_bgcolor='white'
    )

    # Save the chart
    fig.write_image(output_path, width=width, height=height)


def create_confidence_progression_chart(
    confidence_data: List[float],
    output_path: str = "confidence_progression.png"
) -> None:
    """
    Create a chart showing confidence progression across iterations.
    
    Args:
        confidence_data: List of confidence values per iteration
        output_path: Path to save the chart
    """
    fig = go.Figure()
    
    iterations = list(range(len(confidence_data)))
    
    fig.add_trace(go.Scatter(
        x=iterations,
        y=confidence_data,
        mode='lines+markers',
        name='Confidence',
        line=dict(color='#2E8B57', width=3),
        marker=dict(size=8, color='#1FB8CD')
    ))
    
    fig.update_layout(
        title="Confidence Progression Across Iterations",
        xaxis_title="Iteration",
        yaxis_title="Confidence Score",
        yaxis=dict(range=[0, 1]),
        showlegend=False
    )
    
    fig.write_image(output_path)


def create_stopping_criteria_analysis(
    stopping_decisions: List[Dict[str, Any]],
    output_path: str = "stopping_analysis.png"
) -> None:
    """
    Create a visualization of stopping criteria decisions.
    
    Args:
        stopping_decisions: List of stopping decision data
        output_path: Path to save the chart
    """
    iterations = [d['iteration'] for d in stopping_decisions]
    confidences = [d['confidence'] for d in stopping_decisions]
    decisions = [d['should_stop'] for d in stopping_decisions]
    
    fig = go.Figure()
    
    # Add confidence line
    fig.add_trace(go.Scatter(
        x=iterations,
        y=confidences,
        mode='lines+markers',
        name='Decision Confidence',
        line=dict(color='#2E8B57', width=2),
        marker=dict(size=6)
    ))
    
    # Add stop/continue markers
    stop_iterations = [i for i, d in zip(iterations, decisions) if d]
    stop_confidences = [c for c, d in zip(confidences, decisions) if d]
    
    if stop_iterations:
        fig.add_trace(go.Scatter(
            x=stop_iterations,
            y=stop_confidences,
            mode='markers',
            name='Stop Decision',
            marker=dict(size=12, color='#DB4545', symbol='x')
        ))
    
    fig.update_layout(
        title="Stopping Criteria Analysis",
        xaxis_title="Iteration",
        yaxis_title="Decision Confidence",
        yaxis=dict(range=[0, 1])
    )
    
    fig.write_image(output_path)