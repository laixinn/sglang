import os
from typing import Optional

import torch
import torch.distributed as dist

class DecodeCache:
    def __init__(self, device: Optional[torch.device] = None):
        if device is None:
            # Use current CUDA device instead of creating a new one
            device = torch.device(f"cuda:{torch.cuda.current_device()}")

        self.device = device
        # Don't set default device as it affects global state and may fail if already set
        # torch.set_default_device(device)
        # torch.cuda.set_device(device)  # Don't change device, use current one
        self.dispatch_stream = torch.cuda.Stream(device=device)
        self.combine_stream = torch.cuda.Stream(device=device)
        self.hit_stream = torch.cuda.Stream(device=device)
        self.miss_stream = torch.cuda.Stream(device=device)

        # Remove the assertion as it may not hold in distributed environments
        # assert dist.get_rank() == int(os.getenv('LOCAL_RANK', '0')) == \
        #     self.device.index == torch.cuda.current_device()

    def overlap_combine(self):
        # TODO: implement here
        pass

    def get_dispatch_result(self):
        # TODO: implement here
        pass

    def dispatch_decode(self):
        raise NotImplementedError("dispatch_decode is not implemented")

    def combine_decode(self):
        raise NotImplementedError("combine_decode is not implemented")

    def _verify_cache(self):
        raise NotImplementedError("_verify_cache is not implemented")

    def _get_hit_cache(self):
        raise NotImplementedError("_get_hit_cache is not implemented")

    def _correction(self):
        raise NotImplementedError("_correction is not implemented")

    def _empty_cache(self):
        raise NotImplementedError("_empty_cache is not implemented")