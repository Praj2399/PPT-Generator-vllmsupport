#!/usr/bin/env python3
"""
Model factory for creating LLM and embedding instances
Supports Ollama, VLLM, and Mixed mode backends with graceful fallbacks
"""

import warnings
from typing import Optional, Any
from pydantic import BaseModel

try:
    from langchain_ollama import ChatOllama, OllamaEmbeddings
except ImportError:
    warnings.warn("LangChain Ollama not available")
    ChatOllama = OllamaEmbeddings = None

try:
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
except ImportError:
    warnings.warn("LangChain OpenAI not available - VLLM support disabled")
    ChatOpenAI = OpenAIEmbeddings = None

from .config import (
    OLLAMA_CONFIG, VLLM_CONFIG, FEATURES,
    get_backend_info
)


class ModelFactory:
    """Factory for creating model instances with backend flexibility"""
    
    @staticmethod
    def create_llm(model_role: str = "primary", schema: Optional[BaseModel] = None) -> Any:
        """Create LLM instance with automatic backend selection and fallback"""
        
        backend_info = get_backend_info()
        print(f"🔧 Creating LLM for {model_role}: backend={backend_info['generation_backend']}")
        
        # Try VLLM first if enabled
        if FEATURES["use_vllm_generation"] and ChatOpenAI:
            try:
                return ModelFactory._create_vllm_llm(model_role, schema)
            except Exception as e:
                print(f"  VLLM LLM creation failed: {e}")
                if not FEATURES["fallback_to_ollama"]:
                    raise
                print(" Falling back to Ollama...")
        
        # Use Ollama (default or fallback)
        if ChatOllama:
            return ModelFactory._create_ollama_llm(model_role, schema)
        
        raise RuntimeError("No LLM backend available (neither Ollama nor VLLM)")
    
    @staticmethod
    def create_embeddings() -> Any:
        """Create embeddings instance with automatic backend selection and fallback"""
        
        backend_info = get_backend_info()
        print(f"🔧 Creating embeddings: backend={backend_info['embedding_backend']}")
        
        # Try VLLM first if enabled
        if FEATURES["use_vllm_embeddings"] and OpenAIEmbeddings:
            try:
                return ModelFactory._create_vllm_embeddings()
            except Exception as e:
                print(f"  VLLM embeddings creation failed: {e}")
                if not FEATURES["fallback_to_ollama"]:
                    raise
                print(" Falling back to Ollama embeddings...")
        
        # Use Ollama (default or fallback)
        if OllamaEmbeddings:
            return ModelFactory._create_ollama_embeddings()
        
        raise RuntimeError("No embeddings backend available")
    
    @staticmethod
    def _create_vllm_llm(model_role: str, schema: Optional[BaseModel] = None) -> ChatOpenAI:
        """Create VLLM-based LLM (single instance, matches working document query code)"""
        base_config = {
            "model": VLLM_CONFIG["generation_model"],
            "base_url": VLLM_CONFIG["generation_url"],
            "api_key": VLLM_CONFIG["api_key"],
            "temperature": 0,
        }
        
        if schema:
            print("  VLLM: Structured output via prompting (no native json_schema)")
        
        llm = ChatOpenAI(**base_config)
        print(f" Created VLLM LLM: {model_role} at {VLLM_CONFIG['generation_url']}")
        return llm
    
    @staticmethod
    def _create_ollama_llm(model_role: str, schema: Optional[BaseModel] = None) -> ChatOllama:
        """Create Ollama-based LLM"""
        model_name = OLLAMA_CONFIG["generation_models"].get(model_role, 
                                                           OLLAMA_CONFIG["generation_models"]["primary"])
        
        base_config = {
            "model": model_name,
            "base_url": OLLAMA_CONFIG["base_url"],
            "temperature": 0,
        }
        
        # Add structured output if schema provided
        if schema:
            base_config["json_schema"] = schema.model_json_schema()
            print(f" Created Ollama LLM with structured output: {model_role}")
        else:
            print(f" Created Ollama LLM: {model_role}")
        
        return ChatOllama(**base_config)
    
    @staticmethod
    def _create_vllm_embeddings() -> OpenAIEmbeddings:
        """Create VLLM-based embeddings (single instance, matches working document query code)"""
        embeddings = OpenAIEmbeddings(
            model=VLLM_CONFIG["embedding_model"],
            base_url=VLLM_CONFIG["embedding_url"],
            api_key=VLLM_CONFIG["api_key"],
        )
        print(f" Created VLLM embeddings: {VLLM_CONFIG['embedding_model']} at {VLLM_CONFIG['embedding_url']}")
        return embeddings
    
    @staticmethod
    def _create_ollama_embeddings() -> OllamaEmbeddings:
        """Create Ollama-based embeddings"""
        embeddings = OllamaEmbeddings(
            model=OLLAMA_CONFIG["embedding_model"],
            base_url=OLLAMA_CONFIG["base_url"]
        )
        print(f" Created Ollama embeddings: {OLLAMA_CONFIG['embedding_model']} at {OLLAMA_CONFIG['base_url']}")
        return embeddings
