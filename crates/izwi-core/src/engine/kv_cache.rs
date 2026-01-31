//! KV Cache Manager with paged attention support.
//!
//! Implements memory-efficient KV cache management following vLLM's paged attention
//! design. Key features:
//! - Block-based memory allocation
//! - Efficient free block tracking with doubly-linked list
//! - Sequence-to-block mapping
//! - Memory usage tracking

use std::collections::{HashMap, VecDeque};
use tracing::debug;

use super::types::{BlockId, RequestId};

/// Configuration for the KV cache.
#[derive(Debug, Clone)]
pub struct KVCacheConfig {
    /// Number of transformer layers
    pub num_layers: usize,
    /// Number of attention heads per layer
    pub num_heads: usize,
    /// Dimension of each attention head
    pub head_dim: usize,
    /// Number of tokens per block
    pub block_size: usize,
    /// Maximum number of blocks to allocate
    pub max_blocks: usize,
    /// Data type size in bytes (2 for float16, 4 for float32)
    pub dtype_bytes: usize,
}

impl Default for KVCacheConfig {
    fn default() -> Self {
        Self {
            num_layers: 24,
            num_heads: 16,
            head_dim: 64,
            block_size: 16,
            max_blocks: 1024,
            dtype_bytes: 2, // float16
        }
    }
}

impl KVCacheConfig {
    /// Calculate memory per block in bytes.
    pub fn block_memory_bytes(&self) -> usize {
        // 2 (K+V) * block_size * num_heads * head_dim * dtype_bytes * num_layers
        2 * self.block_size * self.num_heads * self.head_dim * self.dtype_bytes * self.num_layers
    }

    /// Calculate total memory for all blocks.
    pub fn total_memory_bytes(&self) -> usize {
        self.block_memory_bytes() * self.max_blocks
    }

    /// Calculate number of blocks needed for a sequence length.
    pub fn blocks_for_tokens(&self, num_tokens: usize) -> usize {
        (num_tokens + self.block_size - 1) / self.block_size
    }
}

/// A single KV cache block.
#[derive(Debug, Clone)]
pub struct KVBlock {
    /// Block ID
    pub id: BlockId,
    /// Number of tokens stored in this block
    pub num_tokens: usize,
    /// Reference count (for copy-on-write / prefix caching)
    pub ref_count: usize,
    /// Hash of block content (for prefix caching)
    pub content_hash: Option<u64>,
}

impl KVBlock {
    fn new(id: BlockId) -> Self {
        Self {
            id,
            num_tokens: 0,
            ref_count: 1,
            content_hash: None,
        }
    }

    fn reset(&mut self) {
        self.num_tokens = 0;
        self.ref_count = 1;
        self.content_hash = None;
    }
}

/// Block allocator using a free list.
pub struct BlockAllocator {
    config: KVCacheConfig,
    /// All blocks
    blocks: Vec<KVBlock>,
    /// Free block IDs (LIFO for cache locality)
    free_list: VecDeque<BlockId>,
    /// Number of allocated blocks
    num_allocated: usize,
}

impl BlockAllocator {
    /// Create a new block allocator.
    pub fn new(config: KVCacheConfig) -> Self {
        let max_blocks = config.max_blocks;
        let blocks: Vec<KVBlock> = (0..max_blocks).map(KVBlock::new).collect();
        let free_list: VecDeque<BlockId> = (0..max_blocks).collect();

        Self {
            config,
            blocks,
            free_list,
            num_allocated: 0,
        }
    }

    /// Check if n blocks can be allocated.
    pub fn can_allocate(&self, n: usize) -> bool {
        self.free_list.len() >= n
    }

    /// Allocate n blocks, returning their IDs.
    /// Uses LIFO allocation - recently freed blocks are reused first for better cache locality.
    pub fn allocate(&mut self, n: usize) -> Option<Vec<BlockId>> {
        if !self.can_allocate(n) {
            return None;
        }

        let mut block_ids = Vec::with_capacity(n);
        for _ in 0..n {
            // Use pop_back for LIFO - recently freed blocks are at the back
            // This improves CPU cache locality as recently used memory is more likely to be hot
            if let Some(id) = self.free_list.pop_back() {
                self.blocks[id].reset();
                block_ids.push(id);
                self.num_allocated += 1;
            }
        }

        Some(block_ids)
    }

    /// Free a single block.
    pub fn free(&mut self, block_id: BlockId) {
        if block_id < self.blocks.len() {
            let block = &mut self.blocks[block_id];
            block.ref_count = block.ref_count.saturating_sub(1);

            if block.ref_count == 0 {
                self.free_list.push_back(block_id);
                self.num_allocated = self.num_allocated.saturating_sub(1);
            }
        }
    }

    /// Free multiple blocks.
    pub fn free_blocks(&mut self, block_ids: &[BlockId]) {
        for &id in block_ids {
            self.free(id);
        }
    }

    /// Get block by ID.
    pub fn get_block(&self, block_id: BlockId) -> Option<&KVBlock> {
        self.blocks.get(block_id)
    }

    /// Get mutable block by ID.
    pub fn get_block_mut(&mut self, block_id: BlockId) -> Option<&mut KVBlock> {
        self.blocks.get_mut(block_id)
    }

    /// Get number of free blocks.
    pub fn num_free(&self) -> usize {
        self.free_list.len()
    }

    /// Get number of allocated blocks.
    pub fn num_allocated(&self) -> usize {
        self.num_allocated
    }

    /// Get total memory used in bytes.
    pub fn memory_used_bytes(&self) -> usize {
        self.num_allocated * self.config.block_memory_bytes()
    }

    /// Get total memory capacity in bytes.
    pub fn memory_capacity_bytes(&self) -> usize {
        self.config.total_memory_bytes()
    }
}

/// KV Cache Manager - manages KV cache for all sequences.
pub struct KVCacheManager {
    config: KVCacheConfig,
    /// Block allocator
    allocator: BlockAllocator,
    /// Mapping from request ID to allocated block IDs
    request_blocks: HashMap<RequestId, Vec<BlockId>>,
    /// Block table: maps (request_id, block_index) to physical block ID
    /// This enables non-contiguous block allocation
    block_table: HashMap<RequestId, Vec<BlockId>>,
}

impl KVCacheManager {
    /// Create a new KV cache manager.
    pub fn new(config: KVCacheConfig) -> Self {
        let allocator = BlockAllocator::new(config.clone());

        Self {
            config,
            allocator,
            request_blocks: HashMap::new(),
            block_table: HashMap::new(),
        }
    }

    /// Check if n blocks can be allocated.
    pub fn can_allocate(&self, n: usize) -> bool {
        self.allocator.can_allocate(n)
    }

    /// Allocate blocks for a request.
    pub fn allocate(&mut self, request_id: &RequestId, num_blocks: usize) -> Vec<BlockId> {
        if let Some(block_ids) = self.allocator.allocate(num_blocks) {
            self.request_blocks
                .entry(request_id.clone())
                .or_insert_with(Vec::new)
                .extend(block_ids.iter().copied());

            self.block_table
                .entry(request_id.clone())
                .or_insert_with(Vec::new)
                .extend(block_ids.iter().copied());

            debug!(
                "Allocated {} blocks for request {}: {:?}",
                num_blocks, request_id, block_ids
            );

            block_ids
        } else {
            Vec::new()
        }
    }

    /// Allocate additional blocks for an existing request (for extension during decode).
    pub fn extend(&mut self, request_id: &RequestId, additional_blocks: usize) -> Vec<BlockId> {
        self.allocate(request_id, additional_blocks)
    }

    /// Free all blocks for a request.
    pub fn free(&mut self, request_id: &RequestId) {
        if let Some(block_ids) = self.request_blocks.remove(request_id) {
            debug!(
                "Freeing {} blocks for request {}: {:?}",
                block_ids.len(),
                request_id,
                block_ids
            );
            self.allocator.free_blocks(&block_ids);
        }
        self.block_table.remove(request_id);
    }

    /// Get blocks allocated to a request.
    pub fn get_blocks(&self, request_id: &RequestId) -> Option<&[BlockId]> {
        self.request_blocks.get(request_id).map(|v| v.as_slice())
    }

    /// Get the block table for a request.
    pub fn get_block_table(&self, request_id: &RequestId) -> Option<&[BlockId]> {
        self.block_table.get(request_id).map(|v| v.as_slice())
    }

    /// Update token count in a block.
    pub fn update_block_tokens(&mut self, block_id: BlockId, num_tokens: usize) {
        if let Some(block) = self.allocator.get_block_mut(block_id) {
            block.num_tokens = num_tokens;
        }
    }

    /// Get number of blocks needed for a number of tokens.
    pub fn blocks_for_tokens(&self, num_tokens: usize) -> usize {
        self.config.blocks_for_tokens(num_tokens)
    }

    /// Get statistics.
    pub fn stats(&self) -> KVCacheStats {
        KVCacheStats {
            total_blocks: self.config.max_blocks,
            allocated_blocks: self.allocator.num_allocated(),
            free_blocks: self.allocator.num_free(),
            num_sequences: self.request_blocks.len(),
            memory_used_bytes: self.allocator.memory_used_bytes(),
            memory_capacity_bytes: self.allocator.memory_capacity_bytes(),
        }
    }

    /// Get configuration.
    pub fn config(&self) -> &KVCacheConfig {
        &self.config
    }
}

/// KV cache statistics.
#[derive(Debug, Clone)]
pub struct KVCacheStats {
    pub total_blocks: usize,
    pub allocated_blocks: usize,
    pub free_blocks: usize,
    pub num_sequences: usize,
    pub memory_used_bytes: usize,
    pub memory_capacity_bytes: usize,
}

impl KVCacheStats {
    /// Get utilization as a percentage (0.0 - 1.0).
    pub fn utilization(&self) -> f64 {
        if self.total_blocks == 0 {
            return 0.0;
        }
        self.allocated_blocks as f64 / self.total_blocks as f64
    }
}

// ============================================================================
// Streaming KV Cache for Continuous Audio Prefill
// ============================================================================

/// Configuration for the streaming KV cache.
#[derive(Debug, Clone)]
pub struct StreamingKVCacheConfig {
    /// Base KV cache configuration
    pub base_config: KVCacheConfig,
    /// Maximum context window size (tokens)
    pub max_context_tokens: usize,
    /// Sliding window size for context (tokens)
    pub sliding_window_tokens: usize,
    /// Enable token eviction when window is full
    pub enable_eviction: bool,
    /// Number of tokens to evict at once when needed
    pub eviction_batch_size: usize,
}

impl Default for StreamingKVCacheConfig {
    fn default() -> Self {
        Self {
            base_config: KVCacheConfig::default(),
            max_context_tokens: 4096,
            sliding_window_tokens: 2048,
            enable_eviction: true,
            eviction_batch_size: 256,
        }
    }
}

/// Token position in the streaming context.
#[derive(Debug, Clone, Copy)]
pub struct TokenPosition {
    /// Absolute position since stream start
    pub absolute: usize,
    /// Position within current window
    pub window: usize,
    /// Block index containing this token
    pub block_idx: usize,
    /// Offset within the block
    pub block_offset: usize,
}

/// Streaming sequence state for continuous prefill.
#[derive(Debug, Clone)]
pub struct StreamingSequence {
    /// Request ID
    pub request_id: String,
    /// Current token count in the sequence
    pub token_count: usize,
    /// Absolute token position (total tokens seen)
    pub absolute_position: usize,
    /// Window start position (for sliding window)
    pub window_start: usize,
    /// Allocated block IDs
    pub block_ids: Vec<BlockId>,
    /// Whether this sequence is in prefill mode
    pub in_prefill: bool,
    /// Last update timestamp
    pub last_update: std::time::Instant,
}

impl StreamingSequence {
    /// Create a new streaming sequence.
    pub fn new(request_id: String) -> Self {
        Self {
            request_id,
            token_count: 0,
            absolute_position: 0,
            window_start: 0,
            block_ids: Vec::new(),
            in_prefill: true,
            last_update: std::time::Instant::now(),
        }
    }

    /// Get current window size.
    pub fn window_size(&self) -> usize {
        self.absolute_position - self.window_start
    }

    /// Check if window needs to slide.
    pub fn needs_window_slide(&self, max_window: usize) -> bool {
        self.window_size() >= max_window
    }
}

/// Streaming KV Cache Manager for continuous audio prefill.
///
/// Unlike the standard KV cache which grows linearly, this cache
/// implements a sliding window approach suitable for streaming audio:
/// - Continuous token ingestion during speech
/// - Window-based context management
/// - Automatic eviction of old tokens
pub struct StreamingKVCacheManager {
    config: StreamingKVCacheConfig,
    /// Base allocator
    allocator: BlockAllocator,
    /// Active streaming sequences
    sequences: HashMap<String, StreamingSequence>,
    /// Block table for each sequence
    block_table: HashMap<String, Vec<BlockId>>,
    /// Statistics
    total_tokens_processed: u64,
    total_evictions: u64,
}

impl StreamingKVCacheManager {
    /// Create a new streaming KV cache manager.
    pub fn new(config: StreamingKVCacheConfig) -> Self {
        let allocator = BlockAllocator::new(config.base_config.clone());

        Self {
            config,
            allocator,
            sequences: HashMap::new(),
            block_table: HashMap::new(),
            total_tokens_processed: 0,
            total_evictions: 0,
        }
    }

    /// Start a new streaming sequence.
    pub fn start_sequence(&mut self, request_id: &str) -> bool {
        if self.sequences.contains_key(request_id) {
            return false;
        }

        let sequence = StreamingSequence::new(request_id.to_string());
        self.sequences.insert(request_id.to_string(), sequence);
        self.block_table.insert(request_id.to_string(), Vec::new());
        true
    }

    /// Append tokens to a streaming sequence.
    ///
    /// This is the core method for continuous prefill - it handles:
    /// - Block allocation for new tokens
    /// - Window sliding when needed
    /// - Eviction of old tokens
    pub fn append_tokens(&mut self, request_id: &str, num_tokens: usize) -> Option<Vec<BlockId>> {
        let config = self.config.clone();

        // Check if sequence exists and if we need eviction
        let needs_eviction = {
            let sequence = self.sequences.get_mut(request_id)?;
            sequence.last_update = std::time::Instant::now();
            sequence.needs_window_slide(config.sliding_window_tokens) && config.enable_eviction
        };

        // Evict if needed (separate borrow scope)
        if needs_eviction {
            self.evict_old_tokens(request_id);
        }

        // Calculate blocks needed
        let (window_size, current_blocks) = {
            let sequence = self.sequences.get(request_id)?;
            let current = self
                .block_table
                .get(request_id)
                .map(|b| b.len())
                .unwrap_or(0);
            (sequence.window_size(), current)
        };

        let total_tokens_after = window_size + num_tokens;
        let blocks_needed = config.base_config.blocks_for_tokens(total_tokens_after);

        let additional_blocks = if blocks_needed > current_blocks {
            blocks_needed - current_blocks
        } else {
            0
        };

        // Allocate additional blocks if needed
        if additional_blocks > 0 {
            if !self.allocator.can_allocate(additional_blocks) {
                // Try to evict from other sequences
                if !self.try_evict_for_space(additional_blocks, request_id) {
                    return None;
                }
            }

            if let Some(blocks) = self.allocator.allocate(additional_blocks) {
                if let Some(table) = self.block_table.get_mut(request_id) {
                    table.extend(blocks);
                }
            } else {
                return None;
            }
        }

        // Update sequence state
        if let Some(seq) = self.sequences.get_mut(request_id) {
            seq.token_count += num_tokens;
            seq.absolute_position += num_tokens;
            seq.block_ids = self
                .block_table
                .get(request_id)
                .cloned()
                .unwrap_or_default();
        }

        self.total_tokens_processed += num_tokens as u64;

        Some(
            self.block_table
                .get(request_id)
                .cloned()
                .unwrap_or_default(),
        )
    }

    /// Evict old tokens from a sequence to make room for new ones.
    fn evict_old_tokens(&mut self, request_id: &str) {
        let config = self.config.clone();

        let Some(sequence) = self.sequences.get_mut(request_id) else {
            return;
        };

        let tokens_to_evict = config.eviction_batch_size;
        let blocks_to_free = config.base_config.blocks_for_tokens(tokens_to_evict);

        // Free oldest blocks
        if let Some(table) = self.block_table.get_mut(request_id) {
            let blocks_freed: Vec<BlockId> =
                table.drain(..blocks_to_free.min(table.len())).collect();
            for block_id in blocks_freed {
                self.allocator.free(block_id);
            }
        }

        // Update window start
        sequence.window_start += tokens_to_evict;
        sequence.token_count = sequence.token_count.saturating_sub(tokens_to_evict);

        self.total_evictions += 1;
    }

    /// Try to evict from other sequences to make space.
    fn try_evict_for_space(&mut self, blocks_needed: usize, exclude_request: &str) -> bool {
        let mut blocks_freed = 0;

        // Find candidates for eviction (oldest sequences first)
        let mut candidates: Vec<_> = self
            .sequences
            .iter()
            .filter(|(id, _)| *id != exclude_request)
            .map(|(id, seq)| (id.clone(), seq.last_update, seq.block_ids.len()))
            .collect();

        candidates.sort_by_key(|(_, time, _)| *time);

        for (request_id, _, _num_blocks) in candidates {
            if blocks_freed >= blocks_needed {
                break;
            }

            // Evict entire sequence if needed
            if let Some(table) = self.block_table.get(&request_id) {
                for &block_id in table {
                    self.allocator.free(block_id);
                    blocks_freed += 1;
                }
            }
            self.block_table.remove(&request_id);
            self.sequences.remove(&request_id);
        }

        blocks_freed >= blocks_needed
    }

    /// End a streaming sequence and free all resources.
    pub fn end_sequence(&mut self, request_id: &str) {
        if let Some(table) = self.block_table.remove(request_id) {
            for block_id in table {
                self.allocator.free(block_id);
            }
        }
        self.sequences.remove(request_id);
    }

    /// Get sequence info.
    pub fn get_sequence(&self, request_id: &str) -> Option<&StreamingSequence> {
        self.sequences.get(request_id)
    }

    /// Get block table for a sequence.
    pub fn get_block_table(&self, request_id: &str) -> Option<&[BlockId]> {
        self.block_table.get(request_id).map(|v| v.as_slice())
    }

    /// Mark sequence as transitioning from prefill to decode.
    pub fn end_prefill(&mut self, request_id: &str) {
        if let Some(seq) = self.sequences.get_mut(request_id) {
            seq.in_prefill = false;
        }
    }

    /// Check if sequence is in prefill mode.
    pub fn is_in_prefill(&self, request_id: &str) -> bool {
        self.sequences
            .get(request_id)
            .map(|s| s.in_prefill)
            .unwrap_or(false)
    }

    /// Get statistics.
    pub fn stats(&self) -> StreamingKVCacheStats {
        StreamingKVCacheStats {
            base_stats: KVCacheStats {
                total_blocks: self.config.base_config.max_blocks,
                allocated_blocks: self.allocator.num_allocated(),
                free_blocks: self.allocator.num_free(),
                num_sequences: self.sequences.len(),
                memory_used_bytes: self.allocator.memory_used_bytes(),
                memory_capacity_bytes: self.allocator.memory_capacity_bytes(),
            },
            total_tokens_processed: self.total_tokens_processed,
            total_evictions: self.total_evictions,
            active_sequences: self.sequences.len(),
            sequences_in_prefill: self.sequences.values().filter(|s| s.in_prefill).count(),
        }
    }
}

/// Statistics for the streaming KV cache.
#[derive(Debug, Clone)]
pub struct StreamingKVCacheStats {
    /// Base cache statistics
    pub base_stats: KVCacheStats,
    /// Total tokens processed across all sequences
    pub total_tokens_processed: u64,
    /// Total number of eviction operations
    pub total_evictions: u64,
    /// Number of active streaming sequences
    pub active_sequences: usize,
    /// Number of sequences currently in prefill mode
    pub sequences_in_prefill: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_block_allocator() {
        let config = KVCacheConfig {
            max_blocks: 10,
            ..Default::default()
        };
        let mut allocator = BlockAllocator::new(config);

        assert_eq!(allocator.num_free(), 10);
        assert_eq!(allocator.num_allocated(), 0);

        // Allocate 3 blocks
        let blocks = allocator.allocate(3).unwrap();
        assert_eq!(blocks.len(), 3);
        assert_eq!(allocator.num_free(), 7);
        assert_eq!(allocator.num_allocated(), 3);

        // Free 1 block
        allocator.free(blocks[0]);
        assert_eq!(allocator.num_free(), 8);
        assert_eq!(allocator.num_allocated(), 2);
    }

    #[test]
    fn test_kv_cache_manager() {
        let config = KVCacheConfig {
            max_blocks: 100,
            block_size: 16,
            ..Default::default()
        };
        let mut manager = KVCacheManager::new(config);

        // Allocate for request 1
        let blocks1 = manager.allocate(&"req1".to_string(), 5);
        assert_eq!(blocks1.len(), 5);

        // Allocate for request 2
        let blocks2 = manager.allocate(&"req2".to_string(), 3);
        assert_eq!(blocks2.len(), 3);

        let stats = manager.stats();
        assert_eq!(stats.allocated_blocks, 8);
        assert_eq!(stats.num_sequences, 2);

        // Free request 1
        manager.free(&"req1".to_string());
        let stats = manager.stats();
        assert_eq!(stats.allocated_blocks, 3);
        assert_eq!(stats.num_sequences, 1);
    }
}
