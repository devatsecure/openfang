# OpenClaw-Inspired Features Design

## Goal
Port three high-value features from OpenClaw into OpenFang: LLM Task Tool, Hybrid Memory Search, and Pluggable Context Engine.

## Priority Order
1. **LLM Task Tool** — fastest impact on workflow reliability
2. **Hybrid Memory Search** — improves recall quality for all agents
3. **Context Engine trait** — architectural enabler for future strategies

---

## Feature 1: LLM Task Tool

**Problem:** Workflows chain full agent sessions (researcher → strategist → twitter). Each step loads tools, recalls memories, manages session — heavy and prone to proxy timeouts.

**Solution:** New `llm_task` builtin tool. Makes a single stateless LLM call with a prompt + optional input, returns JSON. No tools, no memory, no session. Fast, cheap, reliable.

**API:**
```json
{
  "name": "llm_task",
  "input": {
    "prompt": "Extract 3 tweet-worthy insights from this research",
    "input": "<research text>",
    "output_format": "json",
    "model": "optional-override",
    "max_tokens": 800
  }
}
```

**Implementation:**
- Add to `tool_runner.rs` alongside existing tools
- Add `ToolDefinition` to `builtin_tool_definitions()`
- Use the agent's existing driver to make a single completion call
- No session, no memory, no tool loop — just prompt → response
- Optional JSON schema validation on output

**Files:** `crates/openfang-runtime/src/tool_runner.rs`

---

## Feature 2: Hybrid Memory Search

**Problem:** Memory recall is either vector OR text matching. Misses results that match keywords but not semantics (and vice versa). No recency weighting — 3-month-old research ranks same as yesterday's.

**Solution:** Combine vector similarity + SQLite FTS5 full-text search, merge with configurable weights, apply temporal decay and MMR diversity re-ranking.

**Architecture:**
```
Query → [Vector Search] → scored results ─┐
                                           ├─ Merge (RRF) → Temporal Decay → MMR → Top-K
Query → [FTS5 Search]  → scored results ─┘
```

**Implementation:**
1. Add FTS5 virtual table to memories migration
2. New `hybrid_recall()` method in `SemanticStore`
3. Reciprocal Rank Fusion (RRF) to merge vector + text scores
4. Temporal decay: `score *= exp(-λ * age_days)`, configurable half-life (default 30 days)
5. MMR diversity: iteratively select results that maximize relevance while minimizing similarity to already-selected results

**Config (in `[memory]` section):**
```toml
[memory]
hybrid_search = true          # enable hybrid (default: true)
temporal_decay_days = 30      # half-life in days (0 = disabled)
mmr_lambda = 0.7              # 0 = max diversity, 1 = max relevance
vector_weight = 0.6           # weight for vector results in RRF
text_weight = 0.4             # weight for FTS results in RRF
```

**Files:**
- `crates/openfang-memory/src/semantic.rs` (add hybrid_recall, FTS5)
- `crates/openfang-memory/src/migration.rs` (FTS5 table)
- `crates/openfang-types/src/config.rs` (MemoryConfig fields)

---

## Feature 3: Pluggable Context Engine

**Problem:** Context assembly is hardcoded in `agent_loop.rs` — recall memories, append to system prompt, push messages, apply budget. Can't swap strategies (e.g., summary-based compaction, RAG-style retrieval, sliding window).

**Solution:** Extract a `ContextEngine` trait that the agent loop calls. Default implementation preserves current behavior. Future engines can implement different strategies.

**Trait:**
```rust
#[async_trait]
pub trait ContextEngine: Send + Sync {
    /// Assemble context for an LLM call given session history and a query.
    async fn assemble(
        &self,
        session: &Session,
        query: &str,
        manifest: &AgentManifest,
        memory: &MemorySubstrate,
        embedding_driver: Option<&(dyn EmbeddingDriver + Send + Sync)>,
    ) -> Result<AssembledContext, String>;

    /// Post-turn lifecycle: persist context, trigger compaction if needed.
    async fn post_turn(
        &self,
        session: &mut Session,
        response: &str,
    ) -> Result<(), String>;
}

pub struct AssembledContext {
    pub system_prompt: String,
    pub messages: Vec<Message>,
    pub recalled_memories: Vec<MemoryFragment>,
}
```

**Implementation:**
- New file `crates/openfang-runtime/src/context_engine.rs`
- `DefaultContextEngine` — extracts current logic from `agent_loop.rs`
- Agent loop calls `engine.assemble()` instead of inline recall + prompt building
- Engine is selected per-agent via manifest or config
- Register in `lib.rs`

**Files:**
- Create: `crates/openfang-runtime/src/context_engine.rs`
- Modify: `crates/openfang-runtime/src/agent_loop.rs`
- Modify: `crates/openfang-runtime/src/lib.rs`

---

## Execution Order
1. LLM Task Tool (smallest, self-contained)
2. Hybrid Memory (medium, touches memory + migration)
3. Context Engine (largest, refactors agent_loop)

Each feature gets its own commit. Build + test + clippy gate between each.
