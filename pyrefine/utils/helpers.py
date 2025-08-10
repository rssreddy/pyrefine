"""
Helper utility functions for SELF-REFINE with Change-of-Thought framework.
"""

import re
from typing import List, Any, Optional
from datetime import timedelta


def format_execution_time(seconds: float) -> str:
    """
    Format execution time in a human-readable format.
    
    Args:
        seconds: Execution time in seconds
        
    Returns:
        Formatted time string
    """
    if seconds < 1:
        return f"{seconds*1000:.0f}ms"
    elif seconds < 60:
        return f"{seconds:.2f}s"
    else:
        td = timedelta(seconds=seconds)
        return str(td)


def truncate_text(text: str, max_length: int = 100, suffix: str = "...") -> str:
    """
    Truncate text to a maximum length with optional suffix.
    
    Args:
        text: Text to truncate
        max_length: Maximum length before truncation
        suffix: Suffix to add when truncating
        
    Returns:
        Truncated text
    """
    if len(text) <= max_length:
        return text
    
    return text[:max_length - len(suffix)] + suffix


def calculate_similarity(text1: str, text2: str) -> float:
    """
    Calculate simple similarity between two texts based on word overlap.
    
    Args:
        text1: First text
        text2: Second text
        
    Returns:
        Similarity score between 0 and 1
    """
    if not text1 or not text2:
        return 0.0
    
    # Simple word-based similarity
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())
    
    if len(words1) == 0 and len(words2) == 0:
        return 1.0
    
    intersection = len(words1.intersection(words2))
    union = len(words1.union(words2))
    
    return intersection / union if union > 0 else 0.0


def extract_key_phrases(text: str, max_phrases: int = 5) -> List[str]:
    """
    Extract key phrases from text using simple heuristics.
    
    Args:
        text: Input text
        max_phrases: Maximum number of phrases to extract
        
    Returns:
        List of key phrases
    """
    # Simple extraction based on sentence structure
    sentences = text.split('.')
    phrases = []
    
    for sentence in sentences[:max_phrases]:
        sentence = sentence.strip()
        if len(sentence) > 10:  # Skip very short sentences
            # Extract the main clause (simplified)
            if ',' in sentence:
                main_clause = sentence.split(',')[0]
            else:
                main_clause = sentence
            
            phrases.append(main_clause.strip())
    
    return phrases[:max_phrases]


def clean_response_text(text: str) -> str:
    """
    Clean response text by removing extra whitespace and formatting.
    
    Args:
        text: Raw response text
        
    Returns:
        Cleaned text
    """
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove leading/trailing whitespace
    text = text.strip()
    
    # Normalize line breaks
    text = re.sub(r'\n\s*\n', '\n\n', text)
    
    return text


def format_feedback_summary(feedback_items: List[Any]) -> str:
    """
    Format a list of feedback items into a readable summary.
    
    Args:
        feedback_items: List of feedback items with severity and content
        
    Returns:
        Formatted summary string
    """
    if not feedback_items:
        return "No feedback items"
    
    summary_parts = []
    
    # Group by severity
    high_items = [item for item in feedback_items if getattr(item, 'severity', 'medium') == 'high']
    medium_items = [item for item in feedback_items if getattr(item, 'severity', 'medium') == 'medium']
    low_items = [item for item in feedback_items if getattr(item, 'severity', 'medium') == 'low']
    
    if high_items:
        summary_parts.append(f"{len(high_items)} high-priority issues")
    if medium_items:
        summary_parts.append(f"{len(medium_items)} medium-priority issues")
    if low_items:
        summary_parts.append(f"{len(low_items)} low-priority issues")
    
    return ", ".join(summary_parts)


def validate_api_key(api_key: Optional[str], provider: str) -> bool:
    """
    Validate API key format for different providers.
    
    Args:
        api_key: API key to validate
        provider: Provider name ('openai' or 'gemini')
        
    Returns:
        True if key format is valid
    """
    if not api_key:
        return False
    
    if provider.lower() == 'openai':
        # OpenAI keys typically start with 'sk-'
        return api_key.startswith('sk-') and len(api_key) > 20
    elif provider.lower() == 'gemini':
        # Gemini keys are typically longer alphanumeric strings
        return len(api_key) > 20 and api_key.replace('-', '').replace('_', '').isalnum()
    
    return True  # Default to valid for unknown providers


def create_session_id() -> str:
    """
    Create a unique session ID for tracking refinement sessions.
    
    Returns:
        Unique session ID string
    """
    import uuid
    from datetime import datetime
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    unique_id = str(uuid.uuid4())[:8]
    
    return f"selfrefine_{timestamp}_{unique_id}"


def parse_iteration_results(results: dict) -> dict:
    """
    Parse and summarize iteration results for reporting.
    
    Args:
        results: Raw results from refinement process
        
    Returns:
        Parsed and summarized results
    """
    summary = {
        'total_iterations': results.get('iteration_count', 0),
        'execution_time': format_execution_time(results.get('execution_time', 0)),
        'final_response_length': len(results.get('final_response', '')),
        'model_used': results.get('model_used', 'unknown'),
        'session_id': results.get('session_id', 'unknown')
    }
    
    # Add CoT analysis summary
    cot_analysis = results.get('cot_analysis', {})
    if cot_analysis:
        confidence_progression = cot_analysis.get('confidence_progression', [])
        if confidence_progression:
            summary['initial_confidence'] = confidence_progression[0]
            summary['final_confidence'] = confidence_progression[-1]
            summary['confidence_improvement'] = confidence_progression[-1] - confidence_progression[0]
    
    # Add stopping decision summary
    stopping_history = results.get('stopping_history', [])
    if stopping_history:
        final_decision = stopping_history[-1]
        summary['stopping_reason'] = final_decision.get('reason', 'unknown')
        summary['stopping_confidence'] = final_decision.get('confidence', 0.0)
    
    return summary