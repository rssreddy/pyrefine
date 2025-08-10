"""
Logging configuration for SELF-REFINE with Change-of-Thought framework.
"""

import logging
import logging.config
import sys
from pathlib import Path
from typing import Optional, Dict, Any


def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    format_type: str = "structured"
) -> None:
    """
    Set up logging configuration for the framework.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional log file path
        format_type: Format type ('simple', 'structured', 'json')
    """
    # Define log formats
    formats = {
        'simple': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        'structured': '%(asctime)s | %(levelname)-8s | %(name)-20s | %(funcName)-15s | %(message)s',
        'json': '{"timestamp": "%(asctime)s", "level": "%(levelname)s", "logger": "%(name)s", "function": "%(funcName)s", "message": "%(message)s"}'
    }
    
    log_format = formats.get(format_type, formats['structured'])
    
    # Create logging configuration
    config = {
        'version': 1,
        'disable_existing_loggers': False,
        'formatters': {
            'default': {
                'format': log_format,
                'datefmt': '%Y-%m-%d %H:%M:%S'
            }
        },
        'handlers': {
            'console': {
                'class': 'logging.StreamHandler',
                'level': level,
                'formatter': 'default',
                'stream': sys.stdout
            }
        },
        'loggers': {
            'pyrefine': {
                'level': level,
                'handlers': ['console'],
                'propagate': False
            },
            'langgraph': {
                'level': 'WARNING',  # Reduce LangGraph verbosity
                'handlers': ['console'],
                'propagate': False
            },
            'langchain': {
                'level': 'WARNING',  # Reduce LangChain verbosity
                'handlers': ['console'],
                'propagate': False
            }
        },
        'root': {
            'level': level,
            'handlers': ['console']
        }
    }
    
    # Add file handler if log file is specified
    if log_file:
        # Ensure log directory exists
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        config['handlers']['file'] = {
            'class': 'logging.handlers.RotatingFileHandler',
            'level': level,
            'formatter': 'default',
            'filename': log_file,
            'maxBytes': 10485760,  # 10MB
            'backupCount': 5
        }
        
        # Add file handler to loggers
        config['loggers']['pyrefine']['handlers'].append('file')
        config['root']['handlers'].append('file')
    
    # Apply configuration
    logging.config.dictConfig(config)


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance for the specified name.
    
    Args:
        name: Logger name (typically __name__)
        
    Returns:
        Logger instance
    """
    return logging.getLogger(name)


def set_log_level(level: str) -> None:
    """
    Set the log level for all framework loggers.
    
    Args:
        level: New log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    """
    # Convert string to logging level
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    
    # Update framework logger
    framework_logger = logging.getLogger('pyrefine')
    framework_logger.setLevel(numeric_level)
    
    # Update all handlers
    for handler in framework_logger.handlers:
        handler.setLevel(numeric_level)


def create_session_logger(session_id: str, log_dir: str = "logs") -> logging.Logger:
    """
    Create a session-specific logger for detailed tracking.
    
    Args:
        session_id: Unique session identifier
        log_dir: Directory to store session logs
        
    Returns:
        Session-specific logger
    """
    # Create log directory
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    
    # Create session logger
    logger_name = f"pyrefine.session.{session_id}"
    logger = logging.getLogger(logger_name)
    
    # Avoid duplicate handlers
    if not logger.handlers:
        # Create file handler for session
        log_file = log_path / f"session_{session_id}.log"
        handler = logging.FileHandler(log_file)
        
        # Set format
        formatter = logging.Formatter(
            '%(asctime)s | %(levelname)-8s | %(funcName)-15s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        handler.setFormatter(formatter)
        
        # Add handler to logger
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)  # Detailed logging for sessions
        logger.propagate = False  # Don't propagate to parent loggers
    
    return logger


def log_iteration_start(logger: logging.Logger, iteration: int, prompt: str) -> None:
    """
    Log the start of a refinement iteration.
    
    Args:
        logger: Logger instance
        iteration: Iteration number
        prompt: Current prompt or refinement instruction
    """
    logger.info(f"=== ITERATION {iteration} START ===")
    logger.debug(f"Prompt: {prompt[:200]}{'...' if len(prompt) > 200 else ''}")


def log_iteration_end(
    logger: logging.Logger, 
    iteration: int, 
    response: str, 
    confidence: float,
    execution_time: float
) -> None:
    """
    Log the end of a refinement iteration.
    
    Args:
        logger: Logger instance
        iteration: Iteration number
        response: Generated response
        confidence: Confidence score
        execution_time: Time taken for iteration
    """
    logger.info(f"=== ITERATION {iteration} END ===")
    logger.info(f"Confidence: {confidence:.3f}, Time: {execution_time:.2f}s")
    logger.debug(f"Response: {response[:200]}{'...' if len(response) > 200 else ''}")


def log_stopping_decision(
    logger: logging.Logger,
    iteration: int,
    should_stop: bool,
    reason: str,
    confidence: float
) -> None:
    """
    Log a stopping criteria decision.
    
    Args:
        logger: Logger instance
        iteration: Current iteration
        should_stop: Whether to stop refinement
        reason: Reason for the decision
        confidence: Decision confidence
    """
    decision = "STOP" if should_stop else "CONTINUE"
    logger.info(f"Stopping Decision [{iteration}]: {decision} - {reason} (conf: {confidence:.3f})")


def log_cot_analysis(
    logger: logging.Logger,
    iteration: int,
    num_steps: int,
    confidence: float,
    coherence: float,
    changes: list
) -> None:
    """
    Log Change-of-Thought analysis results.
    
    Args:
        logger: Logger instance
        iteration: Current iteration
        num_steps: Number of reasoning steps
        confidence: Overall confidence
        coherence: Reasoning coherence
        changes: List of thought changes
    """
    logger.info(f"CoT Analysis [{iteration}]: {num_steps} steps, "
                f"confidence={confidence:.3f}, coherence={coherence:.3f}")
    
    if changes:
        logger.debug(f"Thought changes: {'; '.join(changes[:3])}")


def log_critic_feedback(
    logger: logging.Logger,
    iteration: int,
    feedback_count: int,
    high_severity: int,
    medium_severity: int,
    low_severity: int
) -> None:
    """
    Log critic feedback summary.
    
    Args:
        logger: Logger instance
        iteration: Current iteration
        feedback_count: Total feedback items
        high_severity: Number of high severity issues
        medium_severity: Number of medium severity issues
        low_severity: Number of low severity issues
    """
    logger.info(f"Critic Feedback [{iteration}]: {feedback_count} items "
                f"(H:{high_severity}, M:{medium_severity}, L:{low_severity})")


def log_session_summary(
    logger: logging.Logger,
    session_id: str,
    total_iterations: int,
    execution_time: float,
    final_confidence: float,
    stopping_reason: str
) -> None:
    """
    Log session summary at completion.
    
    Args:
        logger: Logger instance
        session_id: Session identifier
        total_iterations: Total number of iterations
        execution_time: Total execution time
        final_confidence: Final confidence score
        stopping_reason: Reason for stopping
    """
    logger.info(f"=== SESSION {session_id} COMPLETE ===")
    logger.info(f"Iterations: {total_iterations}, Time: {execution_time:.2f}s")
    logger.info(f"Final Confidence: {final_confidence:.3f}")
    logger.info(f"Stopping Reason: {stopping_reason}")


# Initialize default logging when module is imported
setup_logging()