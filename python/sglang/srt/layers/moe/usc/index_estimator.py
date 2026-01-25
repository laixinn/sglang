"""
MoE Index Prediction Module

This module provides index prediction functionality for MoE (Mixture of Experts) models.
It predicts the next layer's expert indices based on current hidden states, enabling
prefetching and caching optimizations.
"""

from typing import Optional, Tuple, Dict, Any
import torch
import torch.nn as nn

from sglang.srt.layers.moe.topk import TopK, TopKOutput


class MoEIndexPredictor:
    """
    Predicts next layer's MoE expert indices using current hidden states.
    
    This predictor:
    1. Uses gate module to compute router logits
    2. Uses topk module to select top-k experts (same logic as actual model)
    3. Stores predictions in forward_batch for next layer to use
    """
    
    def __init__(
        self,
        next_gate_fn: nn.Module,
        next_topk_fn: TopK,
    ):
        self.next_gate_fn = next_gate_fn
        self.next_topk_fn = next_topk_fn
    
    @torch.no_grad()
    def predict(
        self,
        hidden_states: torch.Tensor,
    ) -> TopKOutput:
        if hidden_states.shape[0] == 0 or self.next_gate_fn is None or self.next_topk_fn is None:
            topk_output = self.topk.empty_topk_output(hidden_states.device)
        else:
            router_logits = self.next_gate_fn(hidden_states)
            
            topk_output = self.next_topk_fn(hidden_states, router_logits)

        return topk_output
    
    @torch.no_grad()
    def predict_and_store(
        self,
        hidden_states: torch.Tensor,
        next_layer_gate: Optional[torch.nn.Module],
        next_layer_topk: Optional[TopK],
        layer_id: int,
        forward_batch: Any, 
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Predict next layer's expert indices and store in forward_batch.
        
        Args:
            hidden_states: Current layer's output [num_tokens, hidden_size]
            next_layer_gate: Next MoE layer's gate module (or None if last layer)
            next_layer_topk: Next MoE layer's topk module (contains correction_bias, renormalize, etc.)
            layer_id: Current layer's ID
            forward_batch: ForwardBatch object to store predictions
            
        Returns:
            Tuple of (topk_idx, topk_weights) if prediction was made, None otherwise
        """
        if next_layer_gate is None or next_layer_topk is None:
            return None
        
        if hidden_states.shape[0] == 0:
            return None
        
        # Compute prediction using gate and topk (same as actual model)
        topk_idx, topk_weights = self.predict(
            hidden_states, next_layer_gate, next_layer_topk,
        )
        
        # Store in forward_batch
        next_layer_id = layer_id + 1
        if not hasattr(forward_batch, "next_layer_index_predictions"):
            object.__setattr__(forward_batch, "next_layer_index_predictions", {})
        
        forward_batch.next_layer_index_predictions[next_layer_id] = {
            "topk_idx": topk_idx,
            "topk_weights": topk_weights,
        }
        
        return topk_idx, topk_weights
