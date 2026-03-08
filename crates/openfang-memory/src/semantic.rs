//! Semantic memory store with vector embedding support.
//!
//! Phase 1: SQLite LIKE matching (fallback when no embeddings).
//! Phase 2: Vector cosine similarity search using stored embeddings.
//!
//! Embeddings are stored as BLOBs in the `embedding` column of the memories table.
//! When a query embedding is provided, recall uses cosine similarity ranking.
//! When no embeddings are available, falls back to LIKE matching.

use chrono::Utc;
use openfang_types::agent::AgentId;
use openfang_types::error::{OpenFangError, OpenFangResult};
use openfang_types::memory::{MemoryFilter, MemoryFragment, MemoryId, MemorySource};
use rusqlite::Connection;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use tracing::debug;

/// Semantic store backed by SQLite with optional vector search.
#[derive(Clone)]
pub struct SemanticStore {
    conn: Arc<Mutex<Connection>>,
}

impl SemanticStore {
    /// Create a new semantic store wrapping the given connection.
    pub fn new(conn: Arc<Mutex<Connection>>) -> Self {
        Self { conn }
    }

    /// Store a new memory fragment (without embedding).
    pub fn remember(
        &self,
        agent_id: AgentId,
        content: &str,
        source: MemorySource,
        scope: &str,
        metadata: HashMap<String, serde_json::Value>,
    ) -> OpenFangResult<MemoryId> {
        self.remember_with_embedding(agent_id, content, source, scope, metadata, None)
    }

    /// Store a new memory fragment with an optional embedding vector.
    pub fn remember_with_embedding(
        &self,
        agent_id: AgentId,
        content: &str,
        source: MemorySource,
        scope: &str,
        metadata: HashMap<String, serde_json::Value>,
        embedding: Option<&[f32]>,
    ) -> OpenFangResult<MemoryId> {
        let conn = self
            .conn
            .lock()
            .map_err(|e| OpenFangError::Internal(e.to_string()))?;
        let id = MemoryId::new();
        let now = Utc::now().to_rfc3339();
        let source_str = serde_json::to_string(&source)
            .map_err(|e| OpenFangError::Serialization(e.to_string()))?;
        let meta_str = serde_json::to_string(&metadata)
            .map_err(|e| OpenFangError::Serialization(e.to_string()))?;
        let embedding_bytes: Option<Vec<u8>> = embedding.map(embedding_to_bytes);

        conn.execute(
            "INSERT INTO memories (id, agent_id, content, source, scope, confidence, metadata, created_at, accessed_at, access_count, deleted, embedding)
             VALUES (?1, ?2, ?3, ?4, ?5, 1.0, ?6, ?7, ?7, 0, 0, ?8)",
            rusqlite::params![
                id.0.to_string(),
                agent_id.0.to_string(),
                content,
                source_str,
                scope,
                meta_str,
                now,
                embedding_bytes,
            ],
        )
        .map_err(|e| OpenFangError::Memory(e.to_string()))?;
        Ok(id)
    }

    /// Search for memories using text matching (fallback, no embeddings).
    pub fn recall(
        &self,
        query: &str,
        limit: usize,
        filter: Option<MemoryFilter>,
    ) -> OpenFangResult<Vec<MemoryFragment>> {
        self.recall_with_embedding(query, limit, filter, None)
    }

    /// Search for memories using vector similarity when a query embedding is provided,
    /// falling back to LIKE matching otherwise.
    pub fn recall_with_embedding(
        &self,
        query: &str,
        limit: usize,
        filter: Option<MemoryFilter>,
        query_embedding: Option<&[f32]>,
    ) -> OpenFangResult<Vec<MemoryFragment>> {
        let conn = self
            .conn
            .lock()
            .map_err(|e| OpenFangError::Internal(e.to_string()))?;

        // Build SQL: fetch candidates (broader than limit for vector re-ranking)
        let fetch_limit = if query_embedding.is_some() {
            // Fetch more candidates for vector search re-ranking
            (limit * 10).max(100)
        } else {
            limit
        };

        let mut sql = String::from(
            "SELECT id, agent_id, content, source, scope, confidence, metadata, created_at, accessed_at, access_count, embedding
             FROM memories WHERE deleted = 0",
        );
        let mut params: Vec<Box<dyn rusqlite::types::ToSql>> = Vec::new();
        let mut param_idx = 1;

        // Text search filter (only when no embeddings — vector search handles relevance)
        if query_embedding.is_none() && !query.is_empty() {
            sql.push_str(&format!(" AND content LIKE ?{param_idx}"));
            params.push(Box::new(format!("%{query}%")));
            param_idx += 1;
        }

        // Apply filters
        if let Some(ref f) = filter {
            if let Some(agent_id) = f.agent_id {
                sql.push_str(&format!(" AND agent_id = ?{param_idx}"));
                params.push(Box::new(agent_id.0.to_string()));
                param_idx += 1;
            }
            if let Some(ref scope) = f.scope {
                sql.push_str(&format!(" AND scope = ?{param_idx}"));
                params.push(Box::new(scope.clone()));
                param_idx += 1;
            }
            if let Some(min_conf) = f.min_confidence {
                sql.push_str(&format!(" AND confidence >= ?{param_idx}"));
                params.push(Box::new(min_conf as f64));
                param_idx += 1;
            }
            if let Some(ref source) = f.source {
                let source_str = serde_json::to_string(source)
                    .map_err(|e| OpenFangError::Serialization(e.to_string()))?;
                sql.push_str(&format!(" AND source = ?{param_idx}"));
                params.push(Box::new(source_str));
                let _ = param_idx;
            }
        }

        sql.push_str(" ORDER BY accessed_at DESC, access_count DESC");
        sql.push_str(&format!(" LIMIT {fetch_limit}"));

        let mut stmt = conn
            .prepare(&sql)
            .map_err(|e| OpenFangError::Memory(e.to_string()))?;

        let param_refs: Vec<&dyn rusqlite::types::ToSql> =
            params.iter().map(|p| p.as_ref()).collect();
        let rows = stmt
            .query_map(param_refs.as_slice(), |row| {
                let id_str: String = row.get(0)?;
                let agent_str: String = row.get(1)?;
                let content: String = row.get(2)?;
                let source_str: String = row.get(3)?;
                let scope: String = row.get(4)?;
                let confidence: f64 = row.get(5)?;
                let meta_str: String = row.get(6)?;
                let created_str: String = row.get(7)?;
                let accessed_str: String = row.get(8)?;
                let access_count: i64 = row.get(9)?;
                let embedding_bytes: Option<Vec<u8>> = row.get(10)?;
                Ok((
                    id_str,
                    agent_str,
                    content,
                    source_str,
                    scope,
                    confidence,
                    meta_str,
                    created_str,
                    accessed_str,
                    access_count,
                    embedding_bytes,
                ))
            })
            .map_err(|e| OpenFangError::Memory(e.to_string()))?;

        let mut fragments = Vec::new();
        for row_result in rows {
            let (
                id_str,
                agent_str,
                content,
                source_str,
                scope,
                confidence,
                meta_str,
                created_str,
                accessed_str,
                access_count,
                embedding_bytes,
            ) = row_result.map_err(|e| OpenFangError::Memory(e.to_string()))?;

            let id = uuid::Uuid::parse_str(&id_str)
                .map(MemoryId)
                .map_err(|e| OpenFangError::Memory(e.to_string()))?;
            let agent_id = uuid::Uuid::parse_str(&agent_str)
                .map(openfang_types::agent::AgentId)
                .map_err(|e| OpenFangError::Memory(e.to_string()))?;
            let source: MemorySource =
                serde_json::from_str(&source_str).unwrap_or(MemorySource::System);
            let metadata: HashMap<String, serde_json::Value> =
                serde_json::from_str(&meta_str).unwrap_or_default();
            let created_at = chrono::DateTime::parse_from_rfc3339(&created_str)
                .map(|dt| dt.with_timezone(&Utc))
                .unwrap_or_else(|_| Utc::now());
            let accessed_at = chrono::DateTime::parse_from_rfc3339(&accessed_str)
                .map(|dt| dt.with_timezone(&Utc))
                .unwrap_or_else(|_| Utc::now());

            let embedding = embedding_bytes.as_deref().map(embedding_from_bytes);

            fragments.push(MemoryFragment {
                id,
                agent_id,
                content,
                embedding,
                metadata,
                source,
                confidence: confidence as f32,
                created_at,
                accessed_at,
                access_count: access_count as u64,
                scope,
            });
        }

        // If we have a query embedding, re-rank by cosine similarity
        if let Some(qe) = query_embedding {
            fragments.sort_by(|a, b| {
                let sim_a = a
                    .embedding
                    .as_deref()
                    .map(|e| cosine_similarity(qe, e))
                    .unwrap_or(-1.0);
                let sim_b = b
                    .embedding
                    .as_deref()
                    .map(|e| cosine_similarity(qe, e))
                    .unwrap_or(-1.0);
                sim_b
                    .partial_cmp(&sim_a)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            fragments.truncate(limit);
            debug!(
                "Vector recall: {} results from {} candidates",
                fragments.len(),
                fetch_limit
            );
        }

        // Update access counts for returned memories
        for frag in &fragments {
            let _ = conn.execute(
                "UPDATE memories SET access_count = access_count + 1, accessed_at = ?1 WHERE id = ?2",
                rusqlite::params![Utc::now().to_rfc3339(), frag.id.0.to_string()],
            );
        }

        Ok(fragments)
    }

    /// Hybrid recall: combines vector similarity + FTS5 full-text search using
    /// Reciprocal Rank Fusion, then applies temporal decay and MMR diversity.
    ///
    /// Falls back to `recall_with_embedding` when FTS5 is not available.
    #[allow(clippy::too_many_arguments)]
    pub fn hybrid_recall(
        &self,
        query: &str,
        limit: usize,
        filter: Option<MemoryFilter>,
        query_embedding: Option<&[f32]>,
        vector_weight: f32,
        text_weight: f32,
        temporal_decay_days: u32,
        mmr_lambda: f32,
    ) -> OpenFangResult<Vec<MemoryFragment>> {
        // If no query embedding, can't do hybrid — fall back
        let qe = match query_embedding {
            Some(qe) => qe,
            None => return self.recall(query, limit, filter),
        };

        // Step 1: Get vector candidates (more than limit for re-ranking)
        let fetch_limit = (limit * 10).max(50);
        let vector_results =
            self.recall_with_embedding(query, fetch_limit, filter.clone(), Some(qe))?;

        // Step 2: Get FTS5 candidates
        let fts_results = self.fts_search(query, fetch_limit, filter.as_ref())?;

        // Step 3: Reciprocal Rank Fusion (RRF)
        let k = 60.0f32; // RRF constant
        let mut scores: HashMap<String, (f32, MemoryFragment)> = HashMap::new();

        for (rank, frag) in vector_results.into_iter().enumerate() {
            let rrf_score = vector_weight / (k + rank as f32 + 1.0);
            let id_str = frag.id.0.to_string();
            scores
                .entry(id_str)
                .and_modify(|(s, _)| *s += rrf_score)
                .or_insert((rrf_score, frag));
        }

        for (rank, frag) in fts_results.into_iter().enumerate() {
            let rrf_score = text_weight / (k + rank as f32 + 1.0);
            let id_str = frag.id.0.to_string();
            scores
                .entry(id_str)
                .and_modify(|(s, _)| *s += rrf_score)
                .or_insert((rrf_score, frag));
        }

        // Step 4: Temporal decay
        let now = Utc::now();
        let mut scored_frags: Vec<(f32, MemoryFragment)> = scores.into_values().collect();

        if temporal_decay_days > 0 {
            let lambda = (2.0f32).ln() / temporal_decay_days as f32;
            for (score, frag) in &mut scored_frags {
                let age_days = now
                    .signed_duration_since(frag.created_at)
                    .num_days()
                    .max(0) as f32;
                *score *= (-lambda * age_days).exp();
            }
        }

        // Sort by score descending
        scored_frags.sort_by(|a, b| {
            b.0.partial_cmp(&a.0)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Step 5: MMR diversity re-ranking
        let selected = mmr_select(scored_frags, limit, mmr_lambda, qe);

        Ok(selected)
    }

    /// FTS5 full-text search on the memories_fts virtual table.
    fn fts_search(
        &self,
        query: &str,
        limit: usize,
        filter: Option<&MemoryFilter>,
    ) -> OpenFangResult<Vec<MemoryFragment>> {
        if query.trim().is_empty() {
            return Ok(vec![]);
        }

        let conn = self
            .conn
            .lock()
            .map_err(|e| OpenFangError::Internal(e.to_string()))?;

        // Check if FTS5 table exists (graceful degradation)
        let has_fts: bool = conn
            .prepare("SELECT 1 FROM sqlite_master WHERE type='table' AND name='memories_fts'")
            .and_then(|mut s| s.query_row([], |_| Ok(true)))
            .unwrap_or(false);

        if !has_fts {
            return Ok(vec![]);
        }

        // Sanitize query for FTS5: escape special characters
        let sanitized = query
            .replace('"', "\"\"")
            .replace(['*', ':'], "");

        // Build FTS5 query — join with memories to get full row data
        let mut sql = String::from(
            "SELECT m.id, m.agent_id, m.content, m.source, m.scope, m.confidence,
                    m.metadata, m.created_at, m.accessed_at, m.access_count, m.embedding
             FROM memories_fts fts
             JOIN memories m ON m.rowid = fts.rowid
             WHERE memories_fts MATCH ?1 AND m.deleted = 0",
        );
        let mut params: Vec<Box<dyn rusqlite::types::ToSql>> = Vec::new();
        params.push(Box::new(format!("\"{}\"", sanitized)));
        let mut param_idx = 2;

        if let Some(f) = filter {
            if let Some(agent_id) = f.agent_id {
                sql.push_str(&format!(" AND m.agent_id = ?{param_idx}"));
                params.push(Box::new(agent_id.0.to_string()));
                param_idx += 1;
            }
            if let Some(ref scope) = f.scope {
                sql.push_str(&format!(" AND m.scope = ?{param_idx}"));
                params.push(Box::new(scope.clone()));
                let _ = param_idx;
            }
        }

        sql.push_str(&format!(" ORDER BY rank LIMIT {limit}"));

        let mut stmt = conn
            .prepare(&sql)
            .map_err(|e| OpenFangError::Memory(e.to_string()))?;

        let param_refs: Vec<&dyn rusqlite::types::ToSql> =
            params.iter().map(|p| p.as_ref()).collect();

        let rows = stmt
            .query_map(param_refs.as_slice(), |row| {
                let id_str: String = row.get(0)?;
                let agent_str: String = row.get(1)?;
                let content: String = row.get(2)?;
                let source_str: String = row.get(3)?;
                let scope: String = row.get(4)?;
                let confidence: f64 = row.get(5)?;
                let meta_str: String = row.get(6)?;
                let created_str: String = row.get(7)?;
                let accessed_str: String = row.get(8)?;
                let access_count: i64 = row.get(9)?;
                let embedding_bytes: Option<Vec<u8>> = row.get(10)?;
                Ok((
                    id_str,
                    agent_str,
                    content,
                    source_str,
                    scope,
                    confidence,
                    meta_str,
                    created_str,
                    accessed_str,
                    access_count,
                    embedding_bytes,
                ))
            })
            .map_err(|e| OpenFangError::Memory(e.to_string()))?;

        let mut fragments = Vec::new();
        for row_result in rows {
            let (
                id_str,
                agent_str,
                content,
                source_str,
                scope,
                confidence,
                meta_str,
                created_str,
                accessed_str,
                access_count,
                embedding_bytes,
            ) = row_result.map_err(|e| OpenFangError::Memory(e.to_string()))?;

            let id = uuid::Uuid::parse_str(&id_str)
                .map(MemoryId)
                .map_err(|e| OpenFangError::Memory(e.to_string()))?;
            let agent_id = uuid::Uuid::parse_str(&agent_str)
                .map(openfang_types::agent::AgentId)
                .map_err(|e| OpenFangError::Memory(e.to_string()))?;
            let source: MemorySource =
                serde_json::from_str(&source_str).unwrap_or(MemorySource::System);
            let metadata: HashMap<String, serde_json::Value> =
                serde_json::from_str(&meta_str).unwrap_or_default();
            let created_at = chrono::DateTime::parse_from_rfc3339(&created_str)
                .map(|dt| dt.with_timezone(&Utc))
                .unwrap_or_else(|_| Utc::now());
            let accessed_at = chrono::DateTime::parse_from_rfc3339(&accessed_str)
                .map(|dt| dt.with_timezone(&Utc))
                .unwrap_or_else(|_| Utc::now());

            let embedding = embedding_bytes.as_deref().map(embedding_from_bytes);

            fragments.push(MemoryFragment {
                id,
                agent_id,
                content,
                embedding,
                metadata,
                source,
                confidence: confidence as f32,
                created_at,
                accessed_at,
                access_count: access_count as u64,
                scope,
            });
        }

        Ok(fragments)
    }

    /// Soft-delete a memory fragment.
    pub fn forget(&self, id: MemoryId) -> OpenFangResult<()> {
        let conn = self
            .conn
            .lock()
            .map_err(|e| OpenFangError::Internal(e.to_string()))?;
        conn.execute(
            "UPDATE memories SET deleted = 1 WHERE id = ?1",
            rusqlite::params![id.0.to_string()],
        )
        .map_err(|e| OpenFangError::Memory(e.to_string()))?;
        Ok(())
    }

    /// Update the embedding for an existing memory.
    pub fn update_embedding(&self, id: MemoryId, embedding: &[f32]) -> OpenFangResult<()> {
        let conn = self
            .conn
            .lock()
            .map_err(|e| OpenFangError::Internal(e.to_string()))?;
        let bytes = embedding_to_bytes(embedding);
        conn.execute(
            "UPDATE memories SET embedding = ?1 WHERE id = ?2",
            rusqlite::params![bytes, id.0.to_string()],
        )
        .map_err(|e| OpenFangError::Memory(e.to_string()))?;
        Ok(())
    }
}

/// Compute cosine similarity between two vectors.
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let mut dot = 0.0f32;
    let mut norm_a = 0.0f32;
    let mut norm_b = 0.0f32;
    for i in 0..a.len() {
        dot += a[i] * b[i];
        norm_a += a[i] * a[i];
        norm_b += b[i] * b[i];
    }
    let denom = norm_a.sqrt() * norm_b.sqrt();
    if denom < f32::EPSILON {
        0.0
    } else {
        dot / denom
    }
}

/// Maximal Marginal Relevance (MMR) selection.
///
/// Iteratively selects results that maximize: λ * relevance - (1-λ) * max_sim_to_selected
fn mmr_select(
    candidates: Vec<(f32, MemoryFragment)>,
    limit: usize,
    lambda: f32,
    query_embedding: &[f32],
) -> Vec<MemoryFragment> {
    if candidates.is_empty() || limit == 0 {
        return vec![];
    }

    let mut selected: Vec<MemoryFragment> = Vec::with_capacity(limit);
    let mut remaining: Vec<(f32, MemoryFragment)> = candidates;

    while selected.len() < limit && !remaining.is_empty() {
        let mut best_idx = 0;
        let mut best_mmr = f32::NEG_INFINITY;

        for (i, (score, frag)) in remaining.iter().enumerate() {
            let relevance = *score;

            // Max similarity to any already-selected result
            let max_sim = if selected.is_empty() {
                0.0
            } else {
                selected
                    .iter()
                    .filter_map(|s| {
                        let se = s.embedding.as_deref()?;
                        let fe = frag.embedding.as_deref()?;
                        Some(cosine_similarity(se, fe))
                    })
                    .fold(0.0f32, f32::max)
            };

            let mmr = lambda * relevance - (1.0 - lambda) * max_sim;
            if mmr > best_mmr {
                best_mmr = mmr;
                best_idx = i;
            }
        }

        let (_, frag) = remaining.remove(best_idx);
        selected.push(frag);
    }

    let _ = query_embedding; // used conceptually via scores; kept for API symmetry
    selected
}

/// Serialize embedding to bytes for SQLite BLOB storage.
fn embedding_to_bytes(embedding: &[f32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(embedding.len() * 4);
    for &val in embedding {
        bytes.extend_from_slice(&val.to_le_bytes());
    }
    bytes
}

/// Deserialize embedding from bytes.
fn embedding_from_bytes(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::migration::run_migrations;

    fn setup() -> SemanticStore {
        let conn = Connection::open_in_memory().unwrap();
        run_migrations(&conn).unwrap();
        SemanticStore::new(Arc::new(Mutex::new(conn)))
    }

    #[test]
    fn test_remember_and_recall() {
        let store = setup();
        let agent_id = AgentId::new();
        store
            .remember(
                agent_id,
                "The user likes Rust programming",
                MemorySource::Conversation,
                "episodic",
                HashMap::new(),
            )
            .unwrap();
        let results = store.recall("Rust", 10, None).unwrap();
        assert_eq!(results.len(), 1);
        assert!(results[0].content.contains("Rust"));
    }

    #[test]
    fn test_recall_with_filter() {
        let store = setup();
        let agent_id = AgentId::new();
        store
            .remember(
                agent_id,
                "Memory A",
                MemorySource::Conversation,
                "episodic",
                HashMap::new(),
            )
            .unwrap();
        store
            .remember(
                AgentId::new(),
                "Memory B",
                MemorySource::Conversation,
                "episodic",
                HashMap::new(),
            )
            .unwrap();
        let filter = MemoryFilter::agent(agent_id);
        let results = store.recall("Memory", 10, Some(filter)).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].content, "Memory A");
    }

    #[test]
    fn test_forget() {
        let store = setup();
        let agent_id = AgentId::new();
        let id = store
            .remember(
                agent_id,
                "To forget",
                MemorySource::Conversation,
                "episodic",
                HashMap::new(),
            )
            .unwrap();
        store.forget(id).unwrap();
        let results = store.recall("To forget", 10, None).unwrap();
        assert!(results.is_empty());
    }

    #[test]
    fn test_remember_with_embedding() {
        let store = setup();
        let agent_id = AgentId::new();
        let embedding = vec![0.1, 0.2, 0.3, 0.4];
        let id = store
            .remember_with_embedding(
                agent_id,
                "Rust is great",
                MemorySource::Conversation,
                "episodic",
                HashMap::new(),
                Some(&embedding),
            )
            .unwrap();
        assert_ne!(id.0.to_string(), "");
    }

    #[test]
    fn test_vector_recall_ranking() {
        let store = setup();
        let agent_id = AgentId::new();

        // Store 3 memories with embeddings pointing in different directions
        let emb_rust = vec![0.9, 0.1, 0.0, 0.0]; // "Rust" direction
        let emb_python = vec![0.0, 0.0, 0.9, 0.1]; // "Python" direction
        let emb_mixed = vec![0.5, 0.5, 0.0, 0.0]; // mixed

        store
            .remember_with_embedding(
                agent_id,
                "Rust is a systems language",
                MemorySource::Conversation,
                "episodic",
                HashMap::new(),
                Some(&emb_rust),
            )
            .unwrap();
        store
            .remember_with_embedding(
                agent_id,
                "Python is interpreted",
                MemorySource::Conversation,
                "episodic",
                HashMap::new(),
                Some(&emb_python),
            )
            .unwrap();
        store
            .remember_with_embedding(
                agent_id,
                "Both are popular",
                MemorySource::Conversation,
                "episodic",
                HashMap::new(),
                Some(&emb_mixed),
            )
            .unwrap();

        // Query with a "Rust"-like embedding
        let query_emb = vec![0.85, 0.15, 0.0, 0.0];
        let results = store
            .recall_with_embedding("", 3, None, Some(&query_emb))
            .unwrap();

        assert_eq!(results.len(), 3);
        // Rust memory should be first (highest cosine similarity)
        assert!(results[0].content.contains("Rust"));
        // Python memory should be last (lowest similarity)
        assert!(results[2].content.contains("Python"));
    }

    #[test]
    fn test_update_embedding() {
        let store = setup();
        let agent_id = AgentId::new();
        let id = store
            .remember(
                agent_id,
                "No embedding yet",
                MemorySource::Conversation,
                "episodic",
                HashMap::new(),
            )
            .unwrap();

        // Update with embedding
        let emb = vec![1.0, 0.0, 0.0];
        store.update_embedding(id, &emb).unwrap();

        // Verify the embedding is stored by doing vector recall
        let query_emb = vec![1.0, 0.0, 0.0];
        let results = store
            .recall_with_embedding("", 10, None, Some(&query_emb))
            .unwrap();
        assert_eq!(results.len(), 1);
        assert!(results[0].embedding.is_some());
        assert_eq!(results[0].embedding.as_ref().unwrap().len(), 3);
    }

    #[test]
    fn test_mixed_embedded_and_non_embedded() {
        let store = setup();
        let agent_id = AgentId::new();

        // One memory with embedding, one without
        store
            .remember_with_embedding(
                agent_id,
                "Has embedding",
                MemorySource::Conversation,
                "episodic",
                HashMap::new(),
                Some(&[1.0, 0.0]),
            )
            .unwrap();
        store
            .remember(
                agent_id,
                "No embedding",
                MemorySource::Conversation,
                "episodic",
                HashMap::new(),
            )
            .unwrap();

        // Vector recall should rank embedded memory higher
        let results = store
            .recall_with_embedding("", 10, None, Some(&[1.0, 0.0]))
            .unwrap();
        assert_eq!(results.len(), 2);
        // Embedded memory should rank first
        assert_eq!(results[0].content, "Has embedding");
    }

    #[test]
    fn test_hybrid_recall_combines_results() {
        let store = setup();
        let agent_id = AgentId::new();

        // Memory with embedding pointing in "rust" direction but text about Python
        store
            .remember_with_embedding(
                agent_id,
                "Python is a great scripting language",
                MemorySource::Conversation,
                "episodic",
                HashMap::new(),
                Some(&[0.9, 0.1, 0.0, 0.0]),
            )
            .unwrap();
        // Memory about Rust with embedding pointing elsewhere
        store
            .remember_with_embedding(
                agent_id,
                "Rust systems programming is fast and safe",
                MemorySource::Conversation,
                "episodic",
                HashMap::new(),
                Some(&[0.0, 0.0, 0.9, 0.1]),
            )
            .unwrap();

        let query_emb = vec![0.85, 0.15, 0.0, 0.0];
        let results = store
            .hybrid_recall(
                "Rust",
                10,
                None,
                Some(&query_emb),
                0.6,
                0.4,
                0,   // no temporal decay
                1.0, // pure relevance, no diversity
            )
            .unwrap();

        // Both should be returned (vector finds Python, FTS finds Rust)
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_hybrid_recall_without_embedding_falls_back() {
        let store = setup();
        let agent_id = AgentId::new();
        store
            .remember(
                agent_id,
                "Rust is awesome",
                MemorySource::Conversation,
                "episodic",
                HashMap::new(),
            )
            .unwrap();

        // No query embedding → should fall back to text search
        let results = store
            .hybrid_recall("Rust", 10, None, None, 0.6, 0.4, 0, 0.7)
            .unwrap();
        assert_eq!(results.len(), 1);
        assert!(results[0].content.contains("Rust"));
    }

    #[test]
    fn test_mmr_select_diversity() {
        // Two nearly identical candidates + one diverse candidate
        let agent_id = AgentId::new();
        let now = Utc::now();

        let frag_a = MemoryFragment {
            id: MemoryId::new(),
            agent_id,
            content: "A".to_string(),
            embedding: Some(vec![1.0, 0.0, 0.0]),
            metadata: HashMap::new(),
            source: MemorySource::Conversation,
            confidence: 1.0,
            created_at: now,
            accessed_at: now,
            access_count: 0,
            scope: "episodic".to_string(),
        };
        let frag_b = MemoryFragment {
            id: MemoryId::new(),
            agent_id,
            content: "B".to_string(),
            embedding: Some(vec![0.99, 0.01, 0.0]),
            metadata: HashMap::new(),
            source: MemorySource::Conversation,
            confidence: 1.0,
            created_at: now,
            accessed_at: now,
            access_count: 0,
            scope: "episodic".to_string(),
        };
        let frag_c = MemoryFragment {
            id: MemoryId::new(),
            agent_id,
            content: "C".to_string(),
            embedding: Some(vec![0.0, 0.0, 1.0]),
            metadata: HashMap::new(),
            source: MemorySource::Conversation,
            confidence: 1.0,
            created_at: now,
            accessed_at: now,
            access_count: 0,
            scope: "episodic".to_string(),
        };

        let candidates = vec![
            (1.0, frag_a),
            (0.95, frag_b),
            (0.5, frag_c),
        ];

        let query_emb = vec![1.0, 0.0, 0.0];
        // Low lambda = favor diversity
        let selected = mmr_select(candidates, 2, 0.3, &query_emb);
        assert_eq!(selected.len(), 2);
        // First should be A (highest score), second should be C (diverse)
        assert_eq!(selected[0].content, "A");
        assert_eq!(selected[1].content, "C");
    }
}
