"""
MoE Index Prediction Module

This module provides index prediction functionality for MoE (Mixture of Experts) models.
It predicts the next layer's expert indices based on current hidden states, enabling
prefetching and caching optimizations.
"""

from typing import Optional, Tuple, Dict, Any
import os
import torch
import torch.nn.functional as F

from sglang.srt.layers.moe.topk import TopK, TopKOutputChecker


class MoEIndexPredictor:
    """
    Predicts next layer's MoE expert indices using current hidden states.
    
    This predictor:
    1. Uses gate module to compute router logits
    2. Uses topk module to select top-k experts (same logic as actual model)
    3. Stores predictions in forward_batch for next layer to use
    """
    
    def __init__(self):
        pass
    
    @torch.no_grad()
    def predict(
        self,
        hidden_states: torch.Tensor,
        gate: torch.nn.Module,
        topk: TopK,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict expert indices for given hidden states.
        
        Uses the same gate and topk modules as the actual model to ensure
        prediction logic is identical.
        
        Args:
            hidden_states: Input tensor [num_tokens, hidden_size]
            gate: MoEGate module (computes router logits)
            topk: TopK module (selects top-k experts with correction_bias, renormalize, etc.)
            
        Returns:
            topk_idx: Expert indices [num_tokens, top_k]
            topk_weights: Expert weights [num_tokens, top_k]
        """
        if gate is None or topk is None:
            raise ValueError("gate and topk cannot be None for index prediction")
        
        top_k = topk.topk_config.top_k
        
        if hidden_states.shape[0] == 0:
            return (
                torch.empty((0, top_k), dtype=torch.int64, device=hidden_states.device),
                torch.empty((0, top_k), dtype=torch.float32, device=hidden_states.device),
            )
        
        # Step 1: Compute router logits using gate (same as model)
        router_logits = gate(hidden_states)
        
        # Step 2: Select top-k experts using topk module (same as model)
        # Use forward_native to ensure we get StandardTopKOutput
        topk_output = topk.forward_native(hidden_states, router_logits)
        
        # Extract topk_idx and topk_weights from output
        if TopKOutputChecker.format_is_standard(topk_output):
            topk_weights = topk_output.topk_weights
            topk_idx = topk_output.topk_ids
        else:
            # Fallback: should not happen with forward_native
            raise ValueError(f"Unexpected topk output format: {topk_output.format}")
        
        return topk_idx, topk_weights
    
    
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

