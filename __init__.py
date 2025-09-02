#!/usr/bin/env python3
"""
PPT Generator Package
Modular presentation generation system
"""
from .ppt_generator import PPTGenerator
from .config import DEFAULT_MODELS
from .document_processor import DocumentProcessor
from .retrieval_engine import RetrievalEngine

__version__ = "1.0.0"
__all__ = ["PPTGenerator", "DocumentProcessor", "RetrievalEngine", "DEFAULT_MODELS"]

# Convenience import for main functionality
def create_generator(models=None):
    """Create a PPTGenerator instance with optional custom models"""
    return PPTGenerator(models)
