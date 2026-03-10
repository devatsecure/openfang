//! Pluggable context engine for assembling LLM context.
//!
//! The `ContextEngine` trait abstracts how context is assembled before each
//! LLM call. The `DefaultContextEngine` preserves the existing behavior:
//! recall memories, append to system prompt, push messages.
//!
//! Future engines can implement different strategies: summary-based compaction,
//! RAG-style retrieval, sliding window, etc.

use async_trait::async_trait;
use openfang_memory::MemorySubstrate;
use openfang_types::agent::AgentManifest;
use openfang_types::memory::{Memory, MemoryFilter, MemoryFragment};
use openfang_types::message::Message;
use tracing::{debug, warn};

use crate::embedding::EmbeddingDriver;

/// True when the embedding failure is likely "model not available" (e.g. Ollama nomic-embed-text not pulled).
/// We log these at debug to avoid spamming when using text fallback by design.
fn is_embedding_unavailable_fallback(e: &crate::embedding::EmbeddingError) -> bool {
    let msg = e.to_string().to_lowercase();
    msg.contains("not found")
        || msg.contains("404")
        || msg.contains("model")
        || msg.contains("nomic")
        || msg.contains("connection refused")
        || msg.contains("connection reset")
}

/// Assembled context ready for an LLM call.
pub struct AssembledContext {
    /// The system prompt (with recalled memories appended).
    pub system_prompt: String,
    /// The messages to send to the LLM.
    pub messages: Vec<Message>,
    /// Recalled memories (for inspection/logging).
    pub recalled_memories: Vec<MemoryFragment>,
}

/// Trait for pluggable context assembly strategies.
#[async_trait]
pub trait ContextEngine: Send + Sync {
    /// Assemble context for an LLM call given session state and a query.
    async fn assemble(
        &self,
        agent_id: openfang_types::agent::AgentId,
        query: &str,
        manifest: &AgentManifest,
        memory: &MemorySubstrate,
        embedding_driver: Option<&(dyn EmbeddingDriver + Send + Sync)>,
        messages: &[Message],
    ) -> Result<AssembledContext, String>;
}

/// Default context engine — preserves existing recall + prompt assembly behavior.
pub struct DefaultContextEngine;

#[async_trait]
impl ContextEngine for DefaultContextEngine {
    async fn assemble(
        &self,
        agent_id: openfang_types::agent::AgentId,
        query: &str,
        manifest: &AgentManifest,
        memory: &MemorySubstrate,
        embedding_driver: Option<&(dyn EmbeddingDriver + Send + Sync)>,
        messages: &[Message],
    ) -> Result<AssembledContext, String> {
        // Recall relevant memories — prefer vector similarity when available
        let memories = if let Some(emb) = embedding_driver {
            match emb.embed_one(query).await {
                Ok(query_vec) => {
                    debug!("ContextEngine: vector recall (dims={})", query_vec.len());
                    memory
                        .recall_with_embedding_async(
                            query,
                            5,
                            Some(MemoryFilter {
                                agent_id: Some(agent_id),
                                ..Default::default()
                            }),
                            Some(&query_vec),
                        )
                        .await
                        .unwrap_or_default()
                }
                Err(e) => {
                    if is_embedding_unavailable_fallback(&e) {
                        debug!("ContextEngine: embedding unavailable (e.g. Ollama model not installed), using text fallback: {e}");
                    } else {
                        warn!("ContextEngine: embedding failed, falling back: {e}");
                    }
                    memory
                        .recall(
                            query,
                            5,
                            Some(MemoryFilter {
                                agent_id: Some(agent_id),
                                ..Default::default()
                            }),
                        )
                        .await
                        .unwrap_or_default()
                }
            }
        } else {
            memory
                .recall(
                    query,
                    5,
                    Some(MemoryFilter {
                        agent_id: Some(agent_id),
                        ..Default::default()
                    }),
                )
                .await
                .unwrap_or_default()
        };

        // Build system prompt with recalled memories
        let mut system_prompt = manifest.model.system_prompt.clone();
        if !memories.is_empty() {
            let mem_pairs: Vec<(String, String)> = memories
                .iter()
                .map(|m| (String::new(), m.content.clone()))
                .collect();
            system_prompt.push_str("\n\n");
            system_prompt
                .push_str(&crate::prompt_builder::build_memory_section(&mem_pairs));
        }

        Ok(AssembledContext {
            system_prompt,
            messages: messages.to_vec(),
            recalled_memories: memories,
        })
    }
}

/// Hybrid context engine — uses FTS5 full-text + vector + temporal decay + MMR diversity.
///
/// Falls back gracefully when embeddings aren't available (e.g. WhatsApp assistant
/// without a local embedding model). In that case, FTS5 still provides much better
/// recall than the basic LIKE text search used by `DefaultContextEngine`.
pub struct HybridContextEngine {
    pub vector_weight: f32,
    pub text_weight: f32,
    pub temporal_decay_days: u32,
    pub mmr_lambda: f32,
}

impl HybridContextEngine {
    pub fn from_config(config: &openfang_types::config::MemoryConfig) -> Self {
        Self {
            vector_weight: config.vector_weight,
            text_weight: config.text_weight,
            temporal_decay_days: config.temporal_decay_days,
            mmr_lambda: config.mmr_lambda,
        }
    }
}

#[async_trait]
impl ContextEngine for HybridContextEngine {
    async fn assemble(
        &self,
        agent_id: openfang_types::agent::AgentId,
        query: &str,
        manifest: &AgentManifest,
        memory: &MemorySubstrate,
        embedding_driver: Option<&(dyn EmbeddingDriver + Send + Sync)>,
        messages: &[Message],
    ) -> Result<AssembledContext, String> {
        // Get embedding vector if driver is available (may fail gracefully)
        let query_vec = if let Some(emb) = embedding_driver {
            match emb.embed_one(query).await {
                Ok(v) => {
                    debug!("HybridContextEngine: vector recall (dims={})", v.len());
                    Some(v)
                }
                Err(e) => {
                    if is_embedding_unavailable_fallback(&e) {
                        debug!("HybridContextEngine: embedding unavailable (e.g. Ollama model not installed), FTS5-only: {e}");
                    } else {
                        warn!("HybridContextEngine: embedding failed, FTS5-only mode: {e}");
                    }
                    None
                }
            }
        } else {
            debug!("HybridContextEngine: no embedding driver, FTS5-only mode");
            None
        };

        let filter = Some(MemoryFilter {
            agent_id: Some(agent_id),
            ..Default::default()
        });

        let memories = memory
            .hybrid_recall_async(
                query,
                5,
                filter,
                query_vec.as_deref(),
                self.vector_weight,
                self.text_weight,
                self.temporal_decay_days,
                self.mmr_lambda,
            )
            .await
            .unwrap_or_default();

        // Build system prompt with recalled memories
        let mut system_prompt = manifest.model.system_prompt.clone();
        if !memories.is_empty() {
            debug!(
                "HybridContextEngine: recalled {} memories for agent {}",
                memories.len(),
                manifest.name
            );
            let mem_pairs: Vec<(String, String)> = memories
                .iter()
                .map(|m| (String::new(), m.content.clone()))
                .collect();
            system_prompt.push_str("\n\n");
            system_prompt
                .push_str(&crate::prompt_builder::build_memory_section(&mem_pairs));
        }

        Ok(AssembledContext {
            system_prompt,
            messages: messages.to_vec(),
            recalled_memories: memories,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_context_engine_implements_trait() {
        // Compile-time check that DefaultContextEngine implements ContextEngine
        let _engine: Box<dyn ContextEngine> = Box::new(DefaultContextEngine);
    }

    #[test]
    fn test_assembled_context_fields() {
        let ctx = AssembledContext {
            system_prompt: "You are a helpful agent.".to_string(),
            messages: vec![],
            recalled_memories: vec![],
        };
        assert!(!ctx.system_prompt.is_empty());
        assert!(ctx.messages.is_empty());
        assert!(ctx.recalled_memories.is_empty());
    }
}
