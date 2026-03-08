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
                    warn!("ContextEngine: embedding failed, falling back: {e}");
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
