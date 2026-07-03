from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, final

from sglang.srt.entrypoints.openai.protocol import PromptTokensDetails, UsageInfo


@final
class UsageProcessor:
    """Stateless helpers that turn raw token counts into a UsageInfo."""

    @staticmethod
    def _build_details(
        cached: int, disagg_prefill_prefix_len: Optional[int] = None
    ) -> Optional[PromptTokensDetails]:
        if cached <= 0 and disagg_prefill_prefix_len is None:
            return None
        return PromptTokensDetails(
            cached_tokens=cached,
            disagg_prefill_prefix_len=disagg_prefill_prefix_len,
        )

    @staticmethod
    def calculate_response_usage(
        responses: List[Dict[str, Any]],
        n_choices: int = 1,
        enable_cache_report: bool = False,
    ) -> UsageInfo:
        completion_tokens = sum(
            r["meta_info"].get("completion_tokens", 0) for r in responses
        )
        prompt_tokens = sum(
            responses[i]["meta_info"].get("prompt_tokens", 0)
            for i in range(0, len(responses), n_choices)
        )

        # some API don't have reasoning_tokens semantics
        reasoning_tokens = sum(
            r["meta_info"].get("reasoning_tokens", 0) for r in responses
        )

        cached_details = None
        if enable_cache_report:
            cached_total = sum(
                responses[i]["meta_info"].get("cached_tokens", 0)
                for i in range(0, len(responses), n_choices)
            )
            disagg_vals = [
                responses[i]["meta_info"].get("disagg_prefill_prefix_len")
                for i in range(0, len(responses), n_choices)
            ]
            disagg_total = (
                sum(v for v in disagg_vals if v is not None)
                if any(v is not None for v in disagg_vals)
                else None
            )
            cached_details = UsageProcessor._build_details(
                cached_total, disagg_total
            )

        return UsageProcessor.calculate_token_usage(
            prompt_tokens=prompt_tokens,
            reasoning_tokens=reasoning_tokens,
            completion_tokens=completion_tokens,
            cached_tokens=cached_details,
        )

    @staticmethod
    def calculate_streaming_usage(
        prompt_tokens: Mapping[int, int],
        reasoning_tokens: Mapping[int, int],
        completion_tokens: Mapping[int, int],
        cached_tokens: Mapping[int, int],
        n_choices: int,
        enable_cache_report: bool = False,
        disagg_prefill_prefix_lens: Optional[Mapping[int, Optional[int]]] = None,
    ) -> UsageInfo:
        # index % n_choices == 0 marks the first choice of a prompt
        total_prompt_tokens = sum(
            tok for idx, tok in prompt_tokens.items() if idx % n_choices == 0
        )
        total_reasoning_tokens = sum(reasoning_tokens.values())
        total_completion_tokens = sum(completion_tokens.values())

        cached_details = None
        if enable_cache_report:
            cached_total = sum(
                tok for idx, tok in cached_tokens.items() if idx % n_choices == 0
            )
            disagg_total = None
            if disagg_prefill_prefix_lens:
                vals = [
                    v
                    for idx, v in disagg_prefill_prefix_lens.items()
                    if idx % n_choices == 0 and v is not None
                ]
                if vals:
                    disagg_total = sum(vals)
            cached_details = UsageProcessor._build_details(
                cached_total, disagg_total
            )

        return UsageProcessor.calculate_token_usage(
            prompt_tokens=total_prompt_tokens,
            reasoning_tokens=total_reasoning_tokens,
            completion_tokens=total_completion_tokens,
            cached_tokens=cached_details,
        )

    @staticmethod
    def calculate_token_usage(
        prompt_tokens: int,
        completion_tokens: int,
        reasoning_tokens: Optional[int] = 0,
        cached_tokens: Optional[PromptTokensDetails] = None,
    ) -> UsageInfo:
        """Calculate token usage information"""
        return UsageInfo(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            prompt_tokens_details=cached_tokens,
            reasoning_tokens=reasoning_tokens,
        )
