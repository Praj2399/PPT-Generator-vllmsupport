#!/usr/bin/env python3
"""
Global Vectorstore Cache for PPT Generator
Maintains a single vectorstore in memory across API requests
Replaces vectorstore when new documents are processed
"""

import os
from typing import Optional, Tuple, Any, Dict, List
from datetime import datetime
import pickle
from pathlib import Path


class GlobalVectorstoreCache:
    """
    Singleton-like cache that lives at module level
    Survives across multiple API requests within the same server session
    One presentation at a time - new generation replaces old
    """
    
    def __init__(self, cache_dir: str = "./cache"):
        """
        Initialize the global cache
        
        Args:
            cache_dir: Directory for optional disk persistence
        """
        # Core vectorstore storage
        self.current_vectorstore: Optional[Any] = None
        """Stores the actual Chroma/FAISS vectorstore object with embeddings"""
        
        self.current_splits: Optional[List] = None
        """Stores processed document chunks for BM25 keyword search"""
        
        # Metadata
        self.current_file_hash: Optional[str] = None
        """SHA256 hash of file paths + sizes for identification"""
        
        self.created_at: Optional[datetime] = None
        """Timestamp when vectorstore was created"""
        
        self.topic: Optional[str] = None
        """Original topic used for generation"""
        
        self.original_params: Dict = {}
        """All original generation parameters (tone, num_slides, etc.)"""
        
        # Optional disk persistence
        self.cache_dir = cache_dir
        self.enable_disk_persistence = False  # Set to True to enable disk saves
        
        # Statistics (optional, for debugging)
        self.stats = {
            "cache_hits": 0,
            "cache_misses": 0,
            "total_replacements": 0
        }
        
        # Create cache directory if disk persistence is enabled
        if self.enable_disk_persistence and self.cache_dir:
            Path(self.cache_dir).mkdir(parents=True, exist_ok=True)
    
    def set(self, 
            vectorstore: Any, 
            splits: List, 
            file_hash: str, 
            topic: str = None, 
            params: Dict = None) -> None:
        """
        Replace any existing cache with new vectorstore data
        
        Args:
            vectorstore: The vectorstore object (Chroma, FAISS, etc.)
            splits: Document chunks/splits used to create the vectorstore
            file_hash: Unique identifier for the source files
            topic: The presentation topic
            params: Generation parameters (tone, num_slides, model_role, etc.)
        
        Note: This REPLACES any existing vectorstore - one presentation at a time
        """
        # Clear old vectorstore (garbage collection will free memory)
        old_vectorstore = self.current_vectorstore
        
        # Set new data
        self.current_vectorstore = vectorstore
        self.current_splits = splits
        self.current_file_hash = file_hash
        self.created_at = datetime.now()
        self.topic = topic or self.topic  # Keep old topic if not provided
        self.original_params = params or self.original_params
        
        # Update statistics
        self.stats["total_replacements"] += 1
        
        print(f" Global vectorstore updated at {self.created_at}")
        print(f"   Topic: {self.topic}")
        print(f"   Chunks: {len(splits) if splits else 0}")
        print(f"   File hash: {file_hash[:8]}...")
        
        # Optional: Save to disk for persistence across restarts
        if self.enable_disk_persistence:
            self._save_to_disk()
        
        # Old vectorstore will be garbage collected
        del old_vectorstore
    
    def get(self) -> Tuple[Optional[Any], Optional[List]]:
        """
        Retrieve the existing vectorstore and splits
        
        Returns:
            Tuple of (vectorstore, splits) or (None, None) if cache is empty
        """
        if self.current_vectorstore is not None:
            self.stats["cache_hits"] += 1
            print(f" Using existing vectorstore from {self.created_at}")
            print(f"   Cache hits: {self.stats['cache_hits']}")
            return self.current_vectorstore, self.current_splits
        else:
            self.stats["cache_misses"] += 1
            
            # Optional: Try to load from disk if cache is empty
            if self.enable_disk_persistence and self._load_from_disk():
                return self.current_vectorstore, self.current_splits
            
            print(" No vectorstore in cache")
            return None, None
    
    def exists(self) -> bool:
        """
        Check if a vectorstore exists in cache
        
        Returns:
            True if vectorstore exists, False otherwise
        """
        return self.current_vectorstore is not None
    
    def clear(self) -> None:
        """
        Clear the cache completely
        Frees memory and removes any disk persistence
        """
        # Clear memory
        self.current_vectorstore = None
        self.current_splits = None
        self.current_file_hash = None
        self.created_at = None
        # Don't clear topic and params - might be useful for UI
        
        print(" Global vectorstore cleared")
        
        # Optional: Clear disk cache
        if self.enable_disk_persistence:
            self._clear_disk_cache()
    
    def get_info(self) -> Dict:
        """
        Get information about the current cache state
        
        Returns:
            Dictionary with cache metadata and statistics
        """
        return {
            "exists": self.exists(),
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "topic": self.topic,
            "original_params": self.original_params,
            "file_hash": self.current_file_hash,
            "chunks_count": len(self.current_splits) if self.current_splits else 0,
            "stats": self.stats,
            "disk_persistence": self.enable_disk_persistence
        }
    
    # ========== Optional Disk Persistence Methods ==========
    # These can be enabled by setting enable_disk_persistence = True
    
    def _save_to_disk(self) -> bool:
        """
        Save vectorstore to disk (for persistence across server restarts)
        Note: This uses pickle which has security implications - only for trusted environments
        
        Returns:
            True if save successful, False otherwise
        """
        if not self.cache_dir or not self.current_vectorstore:
            return False
        
        try:
            # Save metadata
            metadata = {
                "created_at": self.created_at,
                "topic": self.topic,
                "original_params": self.original_params,
                "file_hash": self.current_file_hash,
                "stats": self.stats
            }
            
            metadata_path = Path(self.cache_dir) / "cache_metadata.pkl"
            with open(metadata_path, 'wb') as f:
                pickle.dump(metadata, f)
            
            # For FAISS vectorstore, use native save method
            if hasattr(self.current_vectorstore, 'save_local'):
                vectorstore_path = Path(self.cache_dir) / "vectorstore"
                self.current_vectorstore.save_local(str(vectorstore_path))
                print(f" Vectorstore saved to disk: {vectorstore_path}")
            else:
                # For other vectorstores, attempt pickle (less reliable)
                vectorstore_path = Path(self.cache_dir) / "vectorstore.pkl"
                with open(vectorstore_path, 'wb') as f:
                    pickle.dump(self.current_vectorstore, f)
            
            # Save splits
            splits_path = Path(self.cache_dir) / "splits.pkl"
            with open(splits_path, 'wb') as f:
                pickle.dump(self.current_splits, f)
            
            return True
            
        except Exception as e:
            print(f"⚠️ Failed to save cache to disk: {e}")
            return False
    
    def _load_from_disk(self) -> bool:
        """
        Load vectorstore from disk if available
        
        Returns:
            True if load successful, False otherwise
        """
        if not self.cache_dir:
            return False
        
        try:
            # Load metadata
            metadata_path = Path(self.cache_dir) / "cache_metadata.pkl"
            if not metadata_path.exists():
                return False
            
            with open(metadata_path, 'rb') as f:
                metadata = pickle.load(f)
            
            # Restore metadata
            self.created_at = metadata.get("created_at")
            self.topic = metadata.get("topic")
            self.original_params = metadata.get("original_params", {})
            self.current_file_hash = metadata.get("file_hash")
            self.stats = metadata.get("stats", self.stats)
            
            # Load splits
            splits_path = Path(self.cache_dir) / "splits.pkl"
            if splits_path.exists():
                with open(splits_path, 'rb') as f:
                    self.current_splits = pickle.load(f)
            
            # Load vectorstore (implementation depends on vectorstore type)
            # This is a placeholder - actual implementation needs to handle specific vectorstore types
            vectorstore_path = Path(self.cache_dir) / "vectorstore"
            if vectorstore_path.exists():
                # For FAISS, you'd use FAISS.load_local()
                # This requires passing embeddings function which we don't have here
                print(" Disk loading requires embeddings function - skipping")
                return False
            
            print(f"💾 Cache loaded from disk (created at {self.created_at})")
            return True
            
        except Exception as e:
            print(f" Failed to load cache from disk: {e}")
            return False
    
    def _clear_disk_cache(self) -> None:
        """Clear all cached files from disk"""
        if not self.cache_dir:
            return
        
        try:
            cache_path = Path(self.cache_dir)
            if cache_path.exists():
                for file in cache_path.glob("*"):
                    if file.is_file():
                        file.unlink()
                    elif file.is_dir():
                        import shutil
                        shutil.rmtree(file)
                print("💾 Disk cache cleared")
        except Exception as e:
            print(f"⚠️ Failed to clear disk cache: {e}")
    
    def enable_persistence(self, enable: bool = True) -> None:
        """
        Enable or disable disk persistence
        
        Args:
            enable: True to enable disk persistence, False to disable
        """
        self.enable_disk_persistence = enable
        if enable:
            Path(self.cache_dir).mkdir(parents=True, exist_ok=True)
            print(f"💾 Disk persistence enabled: {self.cache_dir}")
        else:
            print("💾 Disk persistence disabled")


# ========== Create Single Global Instance ==========
# This is the singleton that persists across all API requests
GLOBAL_VS_CACHE = GlobalVectorstoreCache()

# Optional: Enable disk persistence by uncommenting the line below
# GLOBAL_VS_CACHE.enable_persistence(True)

print(" Global Vectorstore Cache initialized")
print(f"   Cache directory: {GLOBAL_VS_CACHE.cache_dir}")
print(f"   Disk persistence: {GLOBAL_VS_CACHE.enable_disk_persistence}")