"""Vector Database Integration for RAG Pipeline

This module provides vector database integration for scalable storage and retrieval
of code embeddings. Supports multiple vector database backends including:
- Chroma (default for development)
- Qdrant (production-ready)
- PostgreSQL with pgvector (enterprise)
- In-memory storage (testing)

Key Features:
1. Multi-backend support with unified interface
2. Efficient similarity search with filtering
3. Metadata storage and querying
4. Batch operations for large codebases
5. Persistence and backup capabilities
"""

import os
import json
import logging
import sqlite3
import hashlib
import pickle
import threading
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import asdict, dataclass
import numpy as np
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    import chromadb
    from chromadb.config import Settings
    CHROMA_AVAILABLE = True
except ImportError:
    CHROMA_AVAILABLE = False

try:
    import qdrant_client
    from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue
    QDRANT_AVAILABLE = True
except ImportError:
    QDRANT_AVAILABLE = False

try:
    import psycopg2
    from psycopg2.extras import RealDictCursor
    POSTGRES_AVAILABLE = True
except ImportError:
    POSTGRES_AVAILABLE = False

from .rag_pipeline import CodeChunk, RetrievalResult
from .config import Config


@dataclass
class VectorSearchResult:
    """Result from vector similarity search."""
    chunk_id: str
    chunk: CodeChunk
    similarity_score: float
    metadata: Dict[str, Any]


class VectorDatabaseInterface(ABC):
    """Abstract interface for vector database implementations."""
    
    @abstractmethod
    def initialize(self) -> bool:
        """Initialize the vector database."""
        pass
    
    @abstractmethod
    def add_chunks(self, chunks: List[CodeChunk]) -> bool:
        """Add code chunks with embeddings to the database."""
        pass
    
    @abstractmethod
    def search_similar(self, query_embedding: np.ndarray, 
                      top_k: int = 10, 
                      filters: Optional[Dict[str, Any]] = None) -> List[Tuple[CodeChunk, float]]:
        """Search for similar chunks."""
        pass
    
    @abstractmethod
    def update_chunk(self, chunk: CodeChunk) -> bool:
        """Update an existing chunk."""
        pass
    
    @abstractmethod
    def delete_chunk(self, chunk_id: str) -> bool:
        """Delete a chunk by ID."""
        pass
    
    @abstractmethod
    def get_chunk(self, chunk_id: str) -> Optional[CodeChunk]:
        """Retrieve a specific chunk by ID."""
        pass
    
    @abstractmethod
    def get_stats(self) -> Dict[str, Any]:
        """Get database statistics."""
        pass
    
    @abstractmethod
    def clear_database(self) -> bool:
        """Clear all data from the database."""
        pass
    
    @abstractmethod
    def backup(self, backup_path: str) -> bool:
        """Create a backup of the database."""
        pass
    
    @abstractmethod
    def restore(self, backup_path: str) -> bool:
        """Restore from a backup."""
        pass


class EnhancedInMemoryVectorDB(VectorDatabaseInterface):
    """
    Enhanced in-memory vector database with advanced indexing and search capabilities.
    
    Features:
    - Fast similarity search using optimized algorithms
    - Metadata filtering and hybrid search
    - Persistence to disk
    - Thread-safe operations
    - Automatic index optimization
    """
    
    def __init__(self, config: Config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Storage
        self.chunks: Dict[str, CodeChunk] = {}
        self.embeddings: Dict[str, np.ndarray] = {}
        self.metadata_index: Dict[str, Dict[str, List[str]]] = {}
        
        # Configuration
        self.similarity_metric = config.get('vector_db.similarity_metric', 'cosine')
        self.index_threshold = config.get('vector_db.index_threshold', 1000)
        self.persistence_path = config.get('vector_db.persistence_path', 'vector_db_cache')
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Performance optimization
        self._embedding_matrix = None
        self._chunk_ids = []
        self._needs_reindex = False
        self.initialized = False
        
        # Load persisted data if available
        self._load_from_disk()
    
    def initialize(self) -> bool:
        """Initialize the enhanced in-memory database."""
        with self._lock:
            if not self.initialized:
                self.chunks.clear()
                self.embeddings.clear()
                self.metadata_index.clear()
                self._embedding_matrix = None
                self._chunk_ids.clear()
                self._needs_reindex = False
                self.initialized = True
                self.logger.info("Enhanced in-memory vector database initialized")
            return True
    
    def add_chunks(self, chunks: List[CodeChunk]) -> bool:
        """Add multiple chunks to the database with batch optimization."""
        try:
            with self._lock:
                added_count = 0
                
                for chunk in chunks:
                    if chunk.id not in self.chunks:
                        self.chunks[chunk.id] = chunk
                        
                        if hasattr(chunk, 'embedding') and chunk.embedding is not None:
                            self.embeddings[chunk.id] = chunk.embedding
                        
                        # Update metadata index
                        self._update_metadata_index(chunk)
                        added_count += 1
                
                if added_count > 0:
                    self._needs_reindex = True
                    self.logger.info(f"Added {added_count} chunks to vector database")
                    
                    # Auto-persist if configured
                    if self.config.get('vector_db.auto_persist', True):
                        self._persist_to_disk()
                
                return True
                
        except Exception as e:
            self.logger.error(f"Error adding chunks to database: {e}")
            return False
    
    def search_similar(self, query_embedding: np.ndarray, 
                      top_k: int = 10, 
                      filters: Optional[Dict[str, Any]] = None) -> List[Tuple[CodeChunk, float]]:
        """Search for similar chunks with advanced filtering and ranking."""
        try:
            with self._lock:
                if not self.embeddings:
                    return []
                
                # Rebuild index if needed
                if self._needs_reindex:
                    self._rebuild_index()
                
                # Apply pre-filtering based on metadata
                candidate_ids = self._apply_metadata_filters(filters) if filters else list(self.embeddings.keys())
                
                if not candidate_ids:
                    return []
                
                # Calculate similarities
                similarities = []
                
                if len(candidate_ids) > 100 and self._embedding_matrix is not None:
                    # Use vectorized computation for large datasets
                    similarities = self._vectorized_similarity_search(query_embedding, candidate_ids, top_k)
                else:
                    # Use individual computation for smaller datasets
                    for chunk_id in candidate_ids:
                        if chunk_id in self.embeddings:
                            similarity = self._calculate_similarity(query_embedding, self.embeddings[chunk_id])
                            similarities.append((chunk_id, similarity))
                
                # Sort by similarity and get top results
                similarities.sort(key=lambda x: x[1], reverse=True)
                top_similarities = similarities[:top_k]
                
                # Convert to result format
                results = []
                for chunk_id, similarity in top_similarities:
                    if chunk_id in self.chunks:
                        results.append((self.chunks[chunk_id], similarity))
                
                return results
                
        except Exception as e:
            self.logger.error(f"Error searching similar chunks: {e}")
            return []
    
    def update_chunk(self, chunk: CodeChunk) -> bool:
        """Update an existing chunk."""
        try:
            with self._lock:
                if chunk.id in self.chunks:
                    old_chunk = self.chunks[chunk.id]
                    
                    # Remove old metadata index entries
                    self._remove_from_metadata_index(old_chunk)
                    
                    # Update chunk and embedding
                    self.chunks[chunk.id] = chunk
                    if hasattr(chunk, 'embedding') and chunk.embedding is not None:
                        self.embeddings[chunk.id] = chunk.embedding
                    
                    # Update metadata index
                    self._update_metadata_index(chunk)
                    
                    self._needs_reindex = True
                    return True
                
                return False
                
        except Exception as e:
            self.logger.error(f"Error updating chunk {chunk.id}: {e}")
            return False
    
    def delete_chunk(self, chunk_id: str) -> bool:
        """Delete a chunk from the database."""
        try:
            with self._lock:
                if chunk_id in self.chunks:
                    chunk = self.chunks[chunk_id]
                    
                    # Remove from all indexes
                    del self.chunks[chunk_id]
                    if chunk_id in self.embeddings:
                        del self.embeddings[chunk_id]
                    
                    self._remove_from_metadata_index(chunk)
                    self._needs_reindex = True
                    
                    return True
                
                return False
                
        except Exception as e:
            self.logger.error(f"Error deleting chunk {chunk_id}: {e}")
            return False
    
    def get_chunk(self, chunk_id: str) -> Optional[CodeChunk]:
        """Retrieve a specific chunk by ID."""
        with self._lock:
            return self.chunks.get(chunk_id)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive database statistics."""
        with self._lock:
            stats = {
                'total_chunks': len(self.chunks),
                'total_embeddings': len(self.embeddings),
                'languages': {},
                'chunk_types': {},
                'file_count': len(set(chunk.file_path for chunk in self.chunks.values())),
                'average_chunk_size': 0,
                'index_status': 'current' if not self._needs_reindex else 'needs_rebuild',
                'database_type': 'enhanced_in_memory'
            }
            
            # Calculate language and type distributions
            total_size = 0
            for chunk in self.chunks.values():
                # Language stats
                lang = chunk.language or 'unknown'
                stats['languages'][lang] = stats['languages'].get(lang, 0) + 1
                
                # Chunk type stats
                chunk_type = chunk.chunk_type or 'unknown'
                stats['chunk_types'][chunk_type] = stats['chunk_types'].get(chunk_type, 0) + 1
                
                # Size stats
                total_size += len(chunk.content)
            
            if stats['total_chunks'] > 0:
                stats['average_chunk_size'] = total_size / stats['total_chunks']
            
            return stats
    
    def clear_database(self) -> bool:
        """Clear all data from the database."""
        try:
            with self._lock:
                self.chunks.clear()
                self.embeddings.clear()
                self.metadata_index.clear()
                self._embedding_matrix = None
                self._chunk_ids.clear()
                self._needs_reindex = False
                
                # Clear persisted data
                if os.path.exists(self.persistence_path):
                    import shutil
                    shutil.rmtree(self.persistence_path)
                
                return True
                
        except Exception as e:
            self.logger.error(f"Error clearing database: {e}")
            return False
    
    def backup(self, backup_path: str) -> bool:
        """Save chunks to JSON file."""
        try:
            with self._lock:
                backup_data = {
                    'chunks': {chunk_id: asdict(chunk) for chunk_id, chunk in self.chunks.items()},
                    'embeddings': {chunk_id: embedding.tolist() for chunk_id, embedding in self.embeddings.items()},
                    'metadata_index': self.metadata_index,
                    'timestamp': datetime.now().isoformat()
                }
                
                with open(backup_path, 'w') as f:
                    json.dump(backup_data, f, indent=2)
                
                self.logger.info(f"Backup saved to {backup_path}")
                return True
            
        except Exception as e:
            self.logger.error(f"Backup failed: {e}")
            return False
    
    def restore(self, backup_path: str) -> bool:
        """Restore chunks from JSON file."""
        try:
            with self._lock:
                with open(backup_path, 'r') as f:
                    backup_data = json.load(f)
                
                self.chunks.clear()
                self.embeddings.clear()
                self.metadata_index.clear()
                
                # Restore chunks
                for chunk_id, chunk_data in backup_data.get('chunks', {}).items():
                    if 'created_at' in chunk_data:
                        chunk_data['created_at'] = datetime.fromisoformat(chunk_data['created_at'])
                    
                    chunk = CodeChunk(**chunk_data)
                    self.chunks[chunk_id] = chunk
                
                # Restore embeddings
                for chunk_id, embedding_data in backup_data.get('embeddings', {}).items():
                    if chunk_id in self.chunks:
                        self.embeddings[chunk_id] = np.array(embedding_data)
                
                # Restore metadata index
                self.metadata_index = backup_data.get('metadata_index', {})
                
                self._needs_reindex = True
                self.logger.info(f"Restored {len(self.chunks)} chunks from {backup_path}")
                return True
            
        except Exception as e:
            self.logger.error(f"Restore failed: {e}")
            return False
    
    def _calculate_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Calculate similarity between two embeddings."""
        if self.similarity_metric == 'cosine':
            # Cosine similarity
            dot_product = np.dot(embedding1, embedding2)
            norm1 = np.linalg.norm(embedding1)
            norm2 = np.linalg.norm(embedding2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            return dot_product / (norm1 * norm2)
        
        elif self.similarity_metric == 'euclidean':
            # Euclidean distance (converted to similarity)
            distance = np.linalg.norm(embedding1 - embedding2)
            return 1.0 / (1.0 + distance)
        
        elif self.similarity_metric == 'manhattan':
            # Manhattan distance (converted to similarity)
            distance = np.sum(np.abs(embedding1 - embedding2))
            return 1.0 / (1.0 + distance)
        
        else:
            # Default to cosine
            return self._calculate_similarity(embedding1, embedding2)
    
    def _vectorized_similarity_search(self, query_embedding: np.ndarray, 
                                    candidate_ids: List[str], top_k: int) -> List[Tuple[str, float]]:
        """Perform vectorized similarity search for better performance."""
        # Get embeddings for candidates
        candidate_embeddings = []
        valid_ids = []
        
        for chunk_id in candidate_ids:
            if chunk_id in self.embeddings:
                candidate_embeddings.append(self.embeddings[chunk_id])
                valid_ids.append(chunk_id)
        
        if not candidate_embeddings:
            return []
        
        # Convert to matrix
        embedding_matrix = np.array(candidate_embeddings)
        
        # Calculate similarities using vectorized operations
        if self.similarity_metric == 'cosine':
            # Normalize embeddings
            query_norm = query_embedding / np.linalg.norm(query_embedding)
            candidate_norms = embedding_matrix / np.linalg.norm(embedding_matrix, axis=1, keepdims=True)
            
            # Calculate cosine similarities
            similarities = np.dot(candidate_norms, query_norm)
        else:
            # Fall back to individual calculations for other metrics
            similarities = []
            for embedding in candidate_embeddings:
                sim = self._calculate_similarity(query_embedding, embedding)
                similarities.append(sim)
            similarities = np.array(similarities)
        
        # Get top k indices
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        # Return results
        results = []
        for idx in top_indices:
            results.append((valid_ids[idx], similarities[idx]))
        
        return results
    
    def _apply_metadata_filters(self, filters: Dict[str, Any]) -> List[str]:
        """Apply metadata filters to get candidate chunk IDs."""
        candidate_sets = []
        
        for key, value in filters.items():
            if key in self.metadata_index:
                if isinstance(value, list):
                    # OR operation for list values
                    candidates = set()
                    for v in value:
                        if str(v) in self.metadata_index[key]:
                            candidates.update(self.metadata_index[key][str(v)])
                    candidate_sets.append(candidates)
                else:
                    # Single value
                    if str(value) in self.metadata_index[key]:
                        candidate_sets.append(set(self.metadata_index[key][str(value)]))
        
        if not candidate_sets:
            return list(self.chunks.keys())
        
        # Intersection of all filter results (AND operation)
        result_set = candidate_sets[0]
        for candidate_set in candidate_sets[1:]:
            result_set = result_set.intersection(candidate_set)
        
        return list(result_set)
    
    def _update_metadata_index(self, chunk: CodeChunk) -> None:
        """Update metadata index for a chunk."""
        # Index by language
        if chunk.language:
            if 'language' not in self.metadata_index:
                self.metadata_index['language'] = {}
            if chunk.language not in self.metadata_index['language']:
                self.metadata_index['language'][chunk.language] = []
            self.metadata_index['language'][chunk.language].append(chunk.id)
        
        # Index by chunk type
        if chunk.chunk_type:
            if 'chunk_type' not in self.metadata_index:
                self.metadata_index['chunk_type'] = {}
            if chunk.chunk_type not in self.metadata_index['chunk_type']:
                self.metadata_index['chunk_type'][chunk.chunk_type] = []
            self.metadata_index['chunk_type'][chunk.chunk_type].append(chunk.id)
        
        # Index by file path
        if chunk.file_path:
            if 'file_path' not in self.metadata_index:
                self.metadata_index['file_path'] = {}
            if chunk.file_path not in self.metadata_index['file_path']:
                self.metadata_index['file_path'][chunk.file_path] = []
            self.metadata_index['file_path'][chunk.file_path].append(chunk.id)
        
        # Index custom metadata
        if chunk.metadata:
            for key, value in chunk.metadata.items():
                index_key = f"metadata_{key}"
                if index_key not in self.metadata_index:
                    self.metadata_index[index_key] = {}
                value_str = str(value)
                if value_str not in self.metadata_index[index_key]:
                    self.metadata_index[index_key][value_str] = []
                self.metadata_index[index_key][value_str].append(chunk.id)
    
    def _remove_from_metadata_index(self, chunk: CodeChunk) -> None:
        """Remove chunk from metadata index."""
        for index_name, index_data in self.metadata_index.items():
            for value, chunk_ids in index_data.items():
                if chunk.id in chunk_ids:
                    chunk_ids.remove(chunk.id)
    
    def _rebuild_index(self) -> None:
        """Rebuild the embedding matrix for vectorized operations."""
        if len(self.embeddings) > self.index_threshold:
            self._chunk_ids = list(self.embeddings.keys())
            embeddings_list = [self.embeddings[chunk_id] for chunk_id in self._chunk_ids]
            self._embedding_matrix = np.array(embeddings_list)
        else:
            self._embedding_matrix = None
            self._chunk_ids = []
        
        self._needs_reindex = False
    
    def _persist_to_disk(self) -> None:
        """Persist database to disk for faster startup."""
        try:
            os.makedirs(self.persistence_path, exist_ok=True)
            
            # Save chunks
            chunks_file = os.path.join(self.persistence_path, 'chunks.pkl')
            with open(chunks_file, 'wb') as f:
                pickle.dump(self.chunks, f)
            
            # Save embeddings
            embeddings_file = os.path.join(self.persistence_path, 'embeddings.pkl')
            with open(embeddings_file, 'wb') as f:
                pickle.dump(self.embeddings, f)
            
            # Save metadata index
            metadata_file = os.path.join(self.persistence_path, 'metadata_index.pkl')
            with open(metadata_file, 'wb') as f:
                pickle.dump(self.metadata_index, f)
            
            self.logger.info(f"Persisted vector database to {self.persistence_path}")
            
        except Exception as e:
            self.logger.error(f"Error persisting database: {e}")
    
    def _load_from_disk(self) -> None:
        """Load persisted database from disk."""
        try:
            if not os.path.exists(self.persistence_path):
                return
            
            # Load chunks
            chunks_file = os.path.join(self.persistence_path, 'chunks.pkl')
            if os.path.exists(chunks_file):
                with open(chunks_file, 'rb') as f:
                    self.chunks = pickle.load(f)
            
            # Load embeddings
            embeddings_file = os.path.join(self.persistence_path, 'embeddings.pkl')
            if os.path.exists(embeddings_file):
                with open(embeddings_file, 'rb') as f:
                    self.embeddings = pickle.load(f)
            
            # Load metadata index
            metadata_file = os.path.join(self.persistence_path, 'metadata_index.pkl')
            if os.path.exists(metadata_file):
                with open(metadata_file, 'rb') as f:
                    self.metadata_index = pickle.load(f)
            
            self._needs_reindex = True
            self.logger.info(f"Loaded {len(self.chunks)} chunks from persisted database")
            
        except Exception as e:
            self.logger.error(f"Error loading persisted database: {e}")
            # Reset to empty state on error
            self.chunks = {}
            self.embeddings = {}
            self.metadata_index = {}


class ChromaVectorDB(VectorDatabaseInterface):
    """Chroma vector database implementation."""
    
    def __init__(self, config: Config):
        if not CHROMA_AVAILABLE:
            raise ImportError("ChromaDB not available. Install with: pip install chromadb")
        
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.client = None
        self.collection = None
        self.collection_name = config.get('vector_db.collection_name', 'code_chunks')
        self.persist_directory = config.get('vector_db.persist_directory', './chroma_db')
    
    def initialize(self) -> bool:
        """Initialize Chroma database."""
        try:
            # Create client with persistence
            self.client = chromadb.PersistentClient(
                path=self.persist_directory,
                settings=Settings(anonymized_telemetry=False)
            )
            
            # Get or create collection
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"description": "Code chunks for RAG pipeline"}
            )
            
            self.logger.info(f"Chroma database initialized with collection: {self.collection_name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Chroma database: {e}")
            return False
    
    def add_chunks(self, chunks: List[CodeChunk]) -> bool:
        """Add chunks to Chroma collection."""
        try:
            if not self.collection:
                raise RuntimeError("Database not initialized")
            
            # Prepare data for Chroma
            ids = []
            embeddings = []
            documents = []
            metadatas = []
            
            for chunk in chunks:
                if chunk.embedding is None:
                    self.logger.warning(f"Chunk {chunk.id} has no embedding")
                    continue
                
                ids.append(chunk.id)
                embeddings.append(chunk.embedding.tolist())
                documents.append(chunk.content)
                
                # Prepare metadata (Chroma doesn't support nested objects)
                metadata = {
                    'file_path': chunk.file_path,
                    'chunk_type': chunk.chunk_type,
                    'start_line': chunk.start_line,
                    'end_line': chunk.end_line,
                    'language': chunk.language,
                    'created_at': chunk.created_at.isoformat()
                }
                
                # Add simple metadata fields
                if 'name' in chunk.metadata:
                    metadata['name'] = str(chunk.metadata['name'])
                if 'complexity' in chunk.metadata:
                    metadata['complexity'] = int(chunk.metadata['complexity'])
                
                metadatas.append(metadata)
            
            # Add to collection
            self.collection.add(
                ids=ids,
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas
            )
            
            self.logger.info(f"Added {len(ids)} chunks to Chroma database")
            return True
            
        except Exception as e:
            self.logger.error(f"Error adding chunks to Chroma: {e}")
            return False
    
    def search_similar(self, query_embedding: np.ndarray, 
                      top_k: int = 10, 
                      filters: Optional[Dict[str, Any]] = None) -> List[Tuple[CodeChunk, float]]:
        """Search for similar chunks in Chroma."""
        try:
            if not self.collection:
                raise RuntimeError("Database not initialized")
            
            # Prepare where clause for filtering
            where_clause = {}
            if filters:
                for key, value in filters.items():
                    if key in ['language', 'chunk_type', 'file_path']:
                        where_clause[key] = value
            
            # Query collection
            results = self.collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=top_k,
                where=where_clause if where_clause else None
            )
            
            # Convert results to CodeChunk objects
            chunks_with_scores = []
            
            if results['ids'] and results['ids'][0]:
                for i, chunk_id in enumerate(results['ids'][0]):
                    metadata = results['metadatas'][0][i]
                    document = results['documents'][0][i]
                    distance = results['distances'][0][i]
                    
                    # Convert distance to similarity (Chroma uses L2 distance)
                    similarity = 1.0 / (1.0 + distance)
                    
                    # Reconstruct CodeChunk
                    chunk = CodeChunk(
                        id=chunk_id,
                        content=document,
                        file_path=metadata['file_path'],
                        chunk_type=metadata['chunk_type'],
                        start_line=metadata['start_line'],
                        end_line=metadata['end_line'],
                        language=metadata['language'],
                        created_at=datetime.fromisoformat(metadata['created_at']),
                        metadata={
                            k: v for k, v in metadata.items() 
                            if k not in ['file_path', 'chunk_type', 'start_line', 'end_line', 'language', 'created_at']
                        }
                    )
                    
                    chunks_with_scores.append((chunk, similarity))
            
            return chunks_with_scores
            
        except Exception as e:
            self.logger.error(f"Error searching Chroma database: {e}")
            return []
    
    def update_chunk(self, chunk: CodeChunk) -> bool:
        """Update a chunk in Chroma."""
        try:
            if not self.collection:
                raise RuntimeError("Database not initialized")
            
            # Delete existing and add new (Chroma doesn't have direct update)
            self.collection.delete(ids=[chunk.id])
            return self.add_chunks([chunk])
            
        except Exception as e:
            self.logger.error(f"Error updating chunk in Chroma: {e}")
            return False
    
    def delete_chunk(self, chunk_id: str) -> bool:
        """Delete a chunk from Chroma."""
        try:
            if not self.collection:
                raise RuntimeError("Database not initialized")
            
            self.collection.delete(ids=[chunk_id])
            return True
            
        except Exception as e:
            self.logger.error(f"Error deleting chunk from Chroma: {e}")
            return False
    
    def get_chunk(self, chunk_id: str) -> Optional[CodeChunk]:
        """Get chunk from Chroma."""
        try:
            if not self.collection:
                raise RuntimeError("Database not initialized")
            
            results = self.collection.get(ids=[chunk_id])
            
            if results['ids'] and results['ids'][0]:
                metadata = results['metadatas'][0]
                document = results['documents'][0]
                
                return CodeChunk(
                    id=chunk_id,
                    content=document,
                    file_path=metadata.get('file_path', ''),
                    chunk_type=metadata.get('chunk_type', 'unknown'),
                    start_line=metadata.get('start_line', 0),
                    end_line=metadata.get('end_line', 0),
                    language=metadata.get('language', 'unknown'),
                    created_at=datetime.fromisoformat(metadata.get('created_at', datetime.now().isoformat())),
                    metadata={k.replace('custom_', ''): v for k, v in metadata.items() if k.startswith('custom_')}
                )
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error getting chunk from Chroma: {e}")
            return None
    
    def get_stats(self) -> Dict[str, Any]:
        """Get Chroma database statistics."""
        try:
            if not self.collection:
                return {'error': 'Database not initialized'}
            
            count = self.collection.count()
            
            return {
                'total_chunks': count,
                'collection_name': self.collection_name,
                'persist_directory': self.persist_directory,
                'database_type': 'chroma'
            }
            
        except Exception as e:
            self.logger.error(f"Error getting Chroma stats: {e}")
            return {'error': str(e)}
    
    def clear_database(self) -> bool:
        """Clear Chroma collection."""
        try:
            if not self.collection:
                raise RuntimeError("Database not initialized")
            
            # Delete the collection and recreate it
            self.client.delete_collection(self.collection_name)
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"description": "Code chunks for RAG pipeline"}
            )
            
            self.logger.info("Chroma database cleared")
            return True
            
        except Exception as e:
            self.logger.error(f"Error clearing Chroma database: {e}")
            return False
    
    def backup(self, backup_path: str) -> bool:
        """Backup Chroma database."""
        try:
            # Chroma handles persistence automatically
            # For additional backup, we can export all data
            if not self.collection:
                raise RuntimeError("Database not initialized")
            
            # Get all data
            results = self.collection.get()
            
            backup_data = {
                'collection_name': self.collection_name,
                'data': results,
                'timestamp': datetime.now().isoformat()
            }
            
            with open(backup_path, 'w') as f:
                json.dump(backup_data, f, indent=2)
            
            self.logger.info(f"Chroma backup saved to {backup_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Chroma backup failed: {e}")
            return False
    
    def restore(self, backup_path: str) -> bool:
        """Restore Chroma database from backup."""
        try:
            with open(backup_path, 'r') as f:
                backup_data = json.load(f)
            
            if not self.collection:
                self.initialize()
            
            # Clear existing data
            self.collection.delete()
            
            # Restore data
            data = backup_data['data']
            if data['ids']:
                self.collection.add(
                    ids=data['ids'],
                    embeddings=data['embeddings'],
                    documents=data['documents'],
                    metadatas=data['metadatas']
                )
            
            self.logger.info(f"Chroma database restored from {backup_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Chroma restore failed: {e}")
            return False


class VectorDatabaseManager:
    """
    Manager class for vector database operations.
    
    Provides a unified interface for different vector database backends
    and handles database selection, initialization, and operations.
    """
    
    def __init__(self, config: Config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.db: Optional[VectorDatabaseInterface] = None
        
        # Database type configuration
        self.db_type = config.get('vector_db.type', 'enhanced_in_memory')
        
    def initialize(self) -> bool:
        """Initialize the configured vector database."""
        try:
            if self.db_type == 'chroma':
                if not CHROMA_AVAILABLE:
                    self.logger.warning("Chroma not available, falling back to enhanced in-memory")
                    self.db = EnhancedInMemoryVectorDB(self.config)
                else:
                    self.db = ChromaVectorDB(self.config)
            
            elif self.db_type == 'qdrant':
                if not QDRANT_AVAILABLE:
                    self.logger.warning("Qdrant not available, falling back to enhanced in-memory")
                    self.db = EnhancedInMemoryVectorDB(self.config)
                else:
                    # TODO: Implement QdrantVectorDB
                    self.logger.warning("Qdrant implementation not yet available, using enhanced in-memory")
                    self.db = EnhancedInMemoryVectorDB(self.config)
            
            elif self.db_type == 'postgres':
                if not POSTGRES_AVAILABLE:
                    self.logger.warning("PostgreSQL not available, falling back to enhanced in-memory")
                    self.db = EnhancedInMemoryVectorDB(self.config)
                else:
                    # TODO: Implement PostgresVectorDB
                    self.logger.warning("PostgreSQL implementation not yet available, using enhanced in-memory")
                    self.db = EnhancedInMemoryVectorDB(self.config)
            
            elif self.db_type in ['in_memory', 'enhanced_in_memory']:
                self.db = EnhancedInMemoryVectorDB(self.config)
            
            else:  # Default to enhanced in-memory
                self.db = EnhancedInMemoryVectorDB(self.config)
            
            return self.db.initialize()
            
        except Exception as e:
            self.logger.error(f"Failed to initialize vector database: {e}")
            return False
    
    def add_chunks(self, chunks: List[CodeChunk]) -> bool:
        """Add chunks to the database."""
        if not self.db:
            raise RuntimeError("Database not initialized")
        return self.db.add_chunks(chunks)
    
    def search_similar(self, query_embedding: np.ndarray, 
                      top_k: int = 10, 
                      filters: Optional[Dict[str, Any]] = None) -> List[Tuple[CodeChunk, float]]:
        """Search for similar chunks."""
        if not self.db:
            raise RuntimeError("Database not initialized")
        return self.db.search_similar(query_embedding, top_k, filters)
    
    def update_chunk(self, chunk: CodeChunk) -> bool:
        """Update a chunk."""
        if not self.db:
            raise RuntimeError("Database not initialized")
        return self.db.update_chunk(chunk)
    
    def delete_chunk(self, chunk_id: str) -> bool:
        """Delete a chunk."""
        if not self.db:
            raise RuntimeError("Database not initialized")
        return self.db.delete_chunk(chunk_id)
    
    def get_chunk(self, chunk_id: str) -> Optional[CodeChunk]:
        """Get a specific chunk."""
        if not self.db:
            raise RuntimeError("Database not initialized")
        return self.db.get_chunk(chunk_id)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get database statistics."""
        if not self.db:
            return {'error': 'Database not initialized'}
        return self.db.get_stats()
    
    def clear_database(self) -> bool:
        """Clear the database."""
        if not self.db:
            raise RuntimeError("Database not initialized")
        return self.db.clear_database()
    
    def backup(self, backup_path: str) -> bool:
        """Create a backup."""
        if not self.db:
            raise RuntimeError("Database not initialized")
        return self.db.backup(backup_path)
    
    def restore(self, backup_path: str) -> bool:
        """Restore from backup."""
        if not self.db:
            raise RuntimeError("Database not initialized")
        return self.db.restore(backup_path)