"""
Sparse Attention Index Estimator Module

用于预测下一层 sparse attention 需要的 sparse_index。
参考 NSA Indexer 的思路：基于当前层的 Q 和 KV cache 计算 attention score，选择 topk 个最重要的 token。

预测流程:
- 第 i 层结束时，用当前层的 hidden_states 和 KV cache 预测第 i+1 层的 sparse_index
- 存储预测的 index 到 forward_batch.next_layer_sparse_predictions[i+1]
- 第 i+1 层使用预测的 index 进行 dispatch_decode
- 返回绝对位置索引，由调用方转换为相对索引
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List, TYPE_CHECKING
from sglang.srt.layers.linear import ReplicatedLinear
from sglang.srt.utils import add_prefix

import logging
logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch


class SparseIndexEstimator(nn.Module):
    """
    Sparse Index Estimator for USC (Unified Sparse Cache).
    
    基于全序列的 attention score 预测下一层需要的 sparse index：
    1. 用当前层的 hidden_states 生成 query
    2. 计算 query 与全序列 KV 的 attention score（分块计算避免 OOM）
    3. 选择 score 最高的 topk 个 token
    
    返回绝对位置索引（在全序列中的位置）。
    """
    
    def __init__(
        self,
        hidden_size: int,
        n_heads: int,
        head_dim: int,
        kv_lora_rank: int = 512,
        qk_rope_head_dim: int = 64,
        prefix: str = "",
        quant_config=None,
        chunk_size: int = 1024,  
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.kv_lora_rank = kv_lora_rank
        self.qk_rope_head_dim = qk_rope_head_dim
        self.chunk_size = chunk_size
        
        # Query projection: hidden_state -> query 
        self.query_proj = ReplicatedLinear(
            hidden_size,
            kv_lora_rank,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("query_proj", prefix),
        )
        
        # softmax scale
        self.softmax_scale = kv_lora_rank ** -0.5
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        seq_lens: List[int],
        topk: int = 256,
        kv_cache: Optional[torch.Tensor] = None,
        layer_id: int = 0,
        forward_batch: Optional["ForwardBatch"] = None,
    ) -> torch.Tensor:
        """
        基于全序列 attention score 预测 sparse index。
        
        流程：
        1. 用 hidden_states 生成 query
        2. 计算 query 与全序列 KV 的 attention score
        3. 选择 score 最高的 topk 个 token
        
        Args:
            hidden_states: [batch, hidden_size] 当前 token 的 hidden states
            seq_lens: list of int, 每个 sequence 的长度
            topk: 需要选择的 topk 数量
            kv_cache: [total_tokens, 1, kv_lora_rank + qk_rope_head_dim] KV cache buffer
            layer_id: 当前层的 ID
            forward_batch: ForwardBatch 对象（用于获取 paged attention 的索引信息）
            
        Returns:
            predicted_sparse_index: [batch, topk] 预测的 sparse index
                - 绝对索引（在全序列中的位置），-1 表示无效位置
        """
        num_tokens = hidden_states.shape[0]
        device = hidden_states.device
        dtype = hidden_states.dtype
        
        # 初始化输出为 -1
        predicted_sparse_index = torch.full(
            (num_tokens, topk),
            -1,
            device=device,
            dtype=torch.long,
        )
        
        if kv_cache is None or forward_batch is None:
            return predicted_sparse_index
        
        # Normalize seq_lens to list[int]
        if isinstance(seq_lens, torch.Tensor):
            seq_lens_list = seq_lens.tolist()
        else:
            seq_lens_list = list(seq_lens)

        # If CP is enabled, prefer global seq len from CP metadata
        cp_total_seq_len = None
        try:
            from sglang.srt.layers.attention.nsa.utils import nsa_use_prefill_cp
            if nsa_use_prefill_cp(forward_batch) and hasattr(forward_batch, "nsa_cp_metadata"):
                total_len = forward_batch.nsa_cp_metadata.total_seq_lens
                if isinstance(total_len, torch.Tensor):
                    total_len = int(total_len.item())
                if isinstance(total_len, int) and total_len > 0:
                    cp_total_seq_len = total_len
        except Exception:
            cp_total_seq_len = None
        
        # DEBUG: 检查 seq_lens 是否正常
        if len(seq_lens_list) == 0 or (len(seq_lens_list) > 0 and sum(seq_lens_list) == 0):
            logger.warning(
                f"[IndexEstimator WARN] Layer {layer_id}: seq_lens empty or all zeros! "
                f"seq_lens={seq_lens_list}, num_tokens={num_tokens}"
            )
        
        # 获取 token_to_batch_idx（用于将每个 token 映射到对应的 batch/sequence）
        token_to_batch_idx = None
        if forward_batch is not None and hasattr(forward_batch, "attn_backend"):
            try:
                token_to_batch_idx = forward_batch.attn_backend.get_token_to_batch_idx()
            except Exception:
                token_to_batch_idx = None
        
        # 生成 query: [num_tokens, kv_lora_rank]
        query = self.query_proj(hidden_states)[0]
        
        for token_idx in range(num_tokens):
            # 选择该 token 对应的 batch_idx
            if token_to_batch_idx is not None:
                batch_idx = int(token_to_batch_idx[token_idx].item())
            else:
                batch_idx = token_idx
            
            if batch_idx >= len(seq_lens_list) or seq_lens_list[batch_idx] <= 0:
                continue
                
            if cp_total_seq_len is not None:
                seq_len_i = int(cp_total_seq_len)
            else:
                seq_len_i = int(seq_lens_list[batch_idx])
            cur_topk = min(topk, seq_len_i) 
            
            if cur_topk <= 0:
                continue
            
            top_indices = self._compute_topk_full_sequence(
                query[token_idx],
                seq_len_i,
                cur_topk,
                kv_cache,
                forward_batch,
                batch_idx,
                device,
                dtype,
                debug=(token_idx == 0),
                layer_id=layer_id,
            )
            predicted_sparse_index[token_idx, :cur_topk] = top_indices

        return predicted_sparse_index
    
    def _compute_topk_full_sequence(
        self,
        query: torch.Tensor,  # [kv_lora_rank]
        seq_len: int,
        cur_topk: int,
        kv_cache: torch.Tensor,
        forward_batch: "ForwardBatch",
        batch_idx: int,
        device: torch.device,
        dtype: torch.dtype,
        debug: bool = False,
        layer_id: int = 0,
    ) -> torch.Tensor:
        """
        在全序列中基于 attention score 选择 topk 个 token。
        
        使用分块计算避免 OOM
        1. 将全序列分成多个 chunk
        2. 每个 chunk 计算 attention score
        3. 合并所有 chunk 的 score, 选择全局 topk
        
        返回绝对位置索引（在全序列中的位置 [0, seq_len-1]）。
        """
        # 获取该请求在 pool 中的索引
        req_pool_idx = forward_batch.req_pool_indices[batch_idx].item()
        
        # 获取全序列的 token 位置
        token_locs = forward_batch.req_to_token_pool.req_to_token[req_pool_idx, :seq_len]
        if debug:
            try:
                token_locs_min = int(token_locs.min().item()) if token_locs.numel() > 0 else -1
                token_locs_max = int(token_locs.max().item()) if token_locs.numel() > 0 else -1
                logger.info(
                    f"[IndexEstimator DEBUG] layer={layer_id} batch_idx={batch_idx} "
                    f"seq_len={seq_len} token_locs_min={token_locs_min} "
                    f"token_locs_max={token_locs_max} "
                    f"kv_cache_size={kv_cache.shape[0]}"
                )
            except Exception:
                logger.warning("[IndexEstimator DEBUG] failed to log token_locs range")
        
        # 分块计算 attention score
        query_f32 = query.float().unsqueeze(0)  # [1, kv_lora_rank]
        all_scores = []
        
        for chunk_start in range(0, seq_len, self.chunk_size):
            chunk_end = min(chunk_start + self.chunk_size, seq_len)
            chunk_locs = token_locs[chunk_start:chunk_end]
            
            # 获取这个 chunk 的 keys
            chunk_keys = kv_cache[chunk_locs, 0, :self.kv_lora_rank]  # [chunk_size, kv_lora_rank]
            chunk_keys_f32 = chunk_keys.float()
            
            # 计算 attention score: query @ keys^T
            chunk_scores = torch.matmul(query_f32, chunk_keys_f32.t()).squeeze(0)  # [chunk_size]
            chunk_scores = chunk_scores * self.softmax_scale
            
            all_scores.append(chunk_scores)
        
        # 合并所有 chunk 的 score
        scores = torch.cat(all_scores, dim=0)  # [seq_len]
        
        # 选择 score 最高的 topk 个 token (index in [0, seq_len))
        _, top_indices = torch.topk(scores, cur_topk, sorted=False)
        
        # 按位置排序，保持因果顺序
        top_indices, _ = torch.sort(top_indices)
        
        # Convert local positions to global token ids
        # token_locs stores global token ids for this sequence
        top_global_indices = token_locs[top_indices]
        
        return top_global_indices
    