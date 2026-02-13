#pragma once

enum NamedBarriers : uint32_t {
    sScale_and_sS_ready = 0,
    sScale_and_sS_free = 1,
    oBuf_free_and_sL_ready = 2,
    epilogue_r2s_ready = 3,
    batch_loop_sync = 4,
    warpgroup0_sync = 5
};
