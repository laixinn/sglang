"""
Decode-to-Prefill (D->P) KV cache replication via D2PKVManager.

After decode finishes generating tokens for a request, the decode server
fire-and-forget sends the decode-generated KV cache back to the prefill
server. The prefill server inserts it into its radix cache so future
multi-turn requests get higher prefill-side cache hits.

Mooncake backend only for V1.

Architecture — per-request sender/receiver, same pattern as forward PD:

- D2PKVManager subclasses MooncakeKVManager. Two instances per direction.

- Decode side: D2PKVManager in PREFILL mode.
  Per-request D2PKVSender objects (same interface as MooncakeKVSender)
  follow request_status transitions:
  Bootstrapping → WaitingForInput → Transferring → Success/Failed.
  capture_and_enqueue() creates a sender; sender.connect() asks the
  D2P bootstrap server to forward D2P_REQ to the prefill-side D2P
  manager. The bootstrap_thread is receive-only (blocking recv_multipart).
  The scheduler polls senders each tick via poll_d2p_senders():
    WaitingForInput → init() + send() + Transferring (external trigger)
    Success/Failed → clear() (dec_lock_ref + status cleanup)

- Prefill side: D2PKVManager in DECODE mode.
  start_decode_thread() dispatches D2P_REQ via handle_d2p_request().
  The scheduler event loop calls process_d2p_incoming() to allocate
  from leftover space, create _D2PReceiver (with SNDTIMEO), and poll
  for RDMA completion + radix cache insertion.

- Bootstrap: A D2PKVBootstrapServer instance at port
  (disaggregation_bootstrap_port + 1) on both sides.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, List, Optional

import numpy as np
import numpy.typing as npt
import requests
import torch
import zmq
from aiohttp import web

from sglang.srt.disaggregation.base.conn import KVArgs, KVPoll
from sglang.srt.disaggregation.common.conn import PrefillRankInfo
from sglang.srt.disaggregation.mooncake.conn import (
    KVArgsRegisterInfo,
    MooncakeKVBootstrapServer,
    MooncakeKVManager,
    MooncakeKVReceiver,
    MooncakeKVSender,
    TransferInfo,
)
from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.mem_cache.base_prefix_cache import InsertParams, MatchPrefixParams
from sglang.srt.mem_cache.radix_cache import RadixCache, RadixKey
from sglang.srt.utils.network import NetworkAddress

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.managers.scheduler import Scheduler

logger = logging.getLogger(__name__)



def get_d2p_bootstrap_port(server_args) -> int:
    return server_args.disaggregation_bootstrap_port + 1


# ---------------------------------------------------------------------------
# D2P bootstrap server
# ---------------------------------------------------------------------------


class D2PKVBootstrapServer(MooncakeKVBootstrapServer):
    """Bootstrap server for D2P KV replication.

    Runs on both prefill and decode sides at port
    (disaggregation_bootstrap_port + 1). Extends the base Mooncake bootstrap
    server with a ``/d2p_route`` endpoint following the same PUT/GET pattern
    as the base ``/route``:

    - **PUT** ``/d2p_route``: called by :meth:`D2PKVSender.connect` on the
      decode side. The server selects the target prefill rank from its
      registered table, then forwards the D2P_REQ control message to the
      prefill-side D2PKVManager(DECODE) over a cached ZMQ PUSH socket.
    - **GET** ``/d2p_route``: returns the target rank info for a given
      decode rank, enabling the receiver to discover connect endpoints.

    ``D2PKVSender.connect()`` is a thin HTTP PUT; all routing logic
    (rank selection, ZMQ socket management) lives here.
    """

    D2P_REQUEST_HEADER = b"D2P_REQ"
    D2P_SNDTIMEO_MS = 5000

    def __init__(self, host: str, port: int):
        self._d2p_ctx = zmq.Context()
        self._d2p_socket_cache: Dict[str, zmq.Socket] = {}
        self._d2p_socket_locks: Dict[str, threading.Lock] = {}
        self._d2p_socket_cache_lock = threading.Lock()
        super().__init__(host, port)

    def _setup_routes(self):
        super()._setup_routes()
        self.app.router.add_route("*", "/d2p_route", self._handle_d2p_route)

    # -- /d2p_route dispatch -----------------------------------------------

    async def _handle_d2p_route(self, request: web.Request):
        if request.method == "PUT":
            return await self._handle_d2p_route_put(request)
        elif request.method == "GET":
            return await self._handle_d2p_route_get(request)
        return web.Response(text="Method not allowed", status=405)

    # -- PUT: forward D2P_REQ to target rank (called by D2PKVSender) -------

    async def _handle_d2p_route_put(self, request: web.Request):
        """Forward a D2P request to the selected prefill-side D2P manager.

        The decode side supplies request metadata via HTTP PUT. This server
        owns the registered rank table and selects the target rank, then
        delivers the D2P_REQ control message over its cached ZMQ PUSH socket.
        """
        if not self._is_ready():
            return web.Response(
                text=(
                    "Prefill server not fully registered yet "
                    f"({self._registered_count} workers registered)."
                ),
                status=503,
            )

        try:
            data = await request.json()
            room = int(data["room"])
            token_ids = data["token_ids"]
            prompt_len = int(data["prompt_len"])
            num_tokens = int(data["num_tokens"])
            d2p_bootstrap_addr = str(data["d2p_bootstrap_addr"])
            if not isinstance(token_ids, list):
                raise ValueError("token_ids must be a list")
            if not d2p_bootstrap_addr:
                raise ValueError("d2p_bootstrap_addr must be non-empty")
            token_ids_bytes = np.asarray(token_ids, dtype=np.int32).tobytes()
        except (KeyError, TypeError, ValueError) as e:
            return web.Response(text=f"Invalid D2P request: {e}", status=400)

        try:
            async with self.lock:
                bootstrap_info = self._select_d2p_target(data)
        except (KeyError, TypeError, ValueError) as e:
            return web.Response(text=f"D2P target not found: {e}", status=404)

        sock, lock = self._connect_d2p_target(bootstrap_info)
        msg = [
            self.D2P_REQUEST_HEADER,
            str(room).encode("ascii"),
            token_ids_bytes,
            str(prompt_len).encode("ascii"),
            str(num_tokens).encode("ascii"),
            d2p_bootstrap_addr.encode("ascii"),
        ]
        try:
            with lock:
                sock.send_multipart(msg)
        except zmq.Again:
            logger.warning("D2P bootstrap forward timed out for room=%d", room)
            return web.Response(text="D2P forward timed out", status=504)
        except zmq.ZMQError as e:
            logger.warning("D2P bootstrap forward failed for room=%d: %s", room, e)
            return web.Response(text=f"D2P forward failed: {e}", status=500)

        return web.Response(text="OK", status=200)

    # -- GET: return target rank info for discovery ------------------------

    async def _handle_d2p_route_get(self, request: web.Request):
        """Return D2P target rank info for a given decode rank.

        Query params mirror the base ``/route`` GET: ``decode_tp_rank``,
        ``decode_attn_tp_size``, ``decode_pp_rank``, ``prefill_dp_rank``,
        ``prefill_cp_rank``.  Returns the selected rank's ip/port as JSON.
        """
        if not self._is_ready():
            return web.Response(
                text=(
                    "Prefill server not fully registered yet "
                    f"({self._registered_count} workers registered)."
                ),
                status=503,
            )

        data = dict(request.query)
        try:
            async with self.lock:
                info = self._select_d2p_target(data)
        except (KeyError, TypeError, ValueError) as e:
            return web.Response(text=f"D2P target not found: {e}", status=404)

        return web.json_response(
            {"rank_ip": info.rank_ip, "rank_port": info.rank_port},
            status=200,
        )

    # -- rank selection & ZMQ socket management ----------------------------

    def _select_d2p_target(self, data: dict) -> PrefillRankInfo:
        prefill_dp_rank = int(data.get("prefill_dp_rank", 0))
        prefill_cp_rank = int(data.get("prefill_cp_rank", 0))
        tp_group_table = self.prefill_port_table[prefill_dp_rank][prefill_cp_rank]

        if "target_tp_rank" in data and data["target_tp_rank"] is not None:
            target_tp_rank = int(data["target_tp_rank"])
        else:
            decode_attn_tp_size = int(
                data.get("decode_attn_tp_size", self.attn_tp_size)
            )
            decode_tp_rank = int(
                data.get(
                    "decode_tp_rank",
                    int(data.get("decode_engine_rank", 0)) % decode_attn_tp_size,
                )
            )
            if decode_attn_tp_size == self.attn_tp_size:
                target_tp_rank = decode_tp_rank
            elif decode_attn_tp_size > self.attn_tp_size:
                ratio = decode_attn_tp_size // self.attn_tp_size
                target_tp_rank = decode_tp_rank // ratio
            else:
                ratio = self.attn_tp_size // decode_attn_tp_size
                target_tp_rank = decode_tp_rank * ratio

        pp_group_table = tp_group_table[target_tp_rank]
        requested_pp_rank = data.get("target_pp_rank", data.get("decode_pp_rank", 0))
        target_pp_rank = int(requested_pp_rank)
        if target_pp_rank not in pp_group_table:
            target_pp_rank = min(pp_group_table)

        return pp_group_table[target_pp_rank]

    def _connect_d2p_target(self, bootstrap_info: PrefillRankInfo):
        na = NetworkAddress(bootstrap_info.rank_ip, bootstrap_info.rank_port)
        endpoint = na.to_tcp()
        with self._d2p_socket_cache_lock:
            if endpoint not in self._d2p_socket_cache:
                sock = self._d2p_ctx.socket(zmq.PUSH)
                if na.is_ipv6:
                    sock.setsockopt(zmq.IPV6, 1)
                sock.setsockopt(zmq.SNDTIMEO, self.D2P_SNDTIMEO_MS)
                sock.setsockopt(zmq.LINGER, 0)
                sock.connect(endpoint)
                self._d2p_socket_cache[endpoint] = sock
                self._d2p_socket_locks[endpoint] = threading.Lock()
            return self._d2p_socket_cache[endpoint], self._d2p_socket_locks[endpoint]

    # -- static helper for D2PKVSender.connect() ---------------------------

    @staticmethod
    def put_d2p_request(
        bootstrap_addr: str,
        room: int,
        task: D2PReplicationTask,
        kv_mgr: D2PKVManager,
    ) -> bool:
        """PUT a D2P request to the remote D2PKVBootstrapServer.

        Called by D2PKVSender.connect(). All payload construction lives here
        so the sender is a thin caller.
        """
        url = f"http://{bootstrap_addr}/d2p_route"
        payload = {
            "room": room,
            "token_ids": task.token_ids,
            "prompt_len": task.prompt_len,
            "num_tokens": len(task.src_kv_indices),
            "d2p_bootstrap_addr": kv_mgr._d2p_bootstrap_addr,
            "decode_engine_rank": kv_mgr.kv_args.engine_rank,
            "decode_attn_tp_size": kv_mgr.attn_tp_size,
            "decode_tp_rank": kv_mgr.attn_tp_rank,
            "decode_pp_rank": kv_mgr.pp_rank,
            "prefill_dp_rank": 0,
            "prefill_cp_rank": 0,
        }
        try:
            response = requests.put(url, json=payload, timeout=5)
        except requests.RequestException as e:
            logger.warning(
                "D2P: bootstrap request failed room=%d addr=%s: %s",
                room, bootstrap_addr, e,
            )
            return False

        if response.status_code == 200:
            return True

        logger.info(
            "D2P: bootstrap server rejected room=%d addr=%s status=%s body=%s",
            room, bootstrap_addr, response.status_code, response.text,
        )
        return False


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class D2PReplicationTask:
    token_ids: List[int]
    prompt_len: int
    src_kv_indices: npt.NDArray[np.int32]
    bootstrap_addr: str


@dataclass
class D2PPendingRequest:
    room: int
    token_ids: List[int]
    prompt_len: int
    num_tokens: int
    d2p_bootstrap_addr: str


# ---------------------------------------------------------------------------
# Shared: build KVArgs from a KV pool for reverse managers
# ---------------------------------------------------------------------------


def _build_reverse_kv_args(scheduler: Scheduler, forward_mgr: MooncakeKVManager) -> KVArgs:
    """Build KVArgs for a reverse manager from the scheduler's KV pool."""
    kv_args = KVArgs()
    kv_args.engine_rank = forward_mgr.kv_args.engine_rank
    kv_args.pp_rank = forward_mgr.pp_rank
    kv_args.system_dp_rank = forward_mgr.system_dp_rank

    kv_pool = scheduler.token_to_kv_pool_allocator.get_kvcache()
    kv_args.prefill_start_layer = getattr(kv_pool, "start_layer", 0)
    kv_args.prefill_end_layer = getattr(kv_pool, "end_layer", None)

    kv_data_ptrs, kv_data_lens, kv_item_lens = kv_pool.get_contiguous_buf_infos()
    kv_args.kv_data_ptrs = kv_data_ptrs
    kv_args.kv_data_lens = kv_data_lens
    kv_args.kv_item_lens = kv_item_lens
    kv_args.page_size = kv_pool.page_size

    if not forward_mgr.is_mla_backend:
        kv_args.kv_head_num = getattr(kv_pool, "head_num", 0)
        kv_args.total_kv_head_num = getattr(
            forward_mgr.kv_args, "total_kv_head_num",
            kv_args.kv_head_num * forward_mgr.attn_tp_size,
        )

    kv_args.aux_data_ptrs = []
    kv_args.aux_data_lens = []
    kv_args.aux_item_lens = []
    kv_args.state_types = []
    kv_args.state_data_ptrs = []
    kv_args.state_data_lens = []
    kv_args.state_item_lens = []
    kv_args.state_dim_per_tensor = []

    kv_args.ib_device = scheduler.server_args.disaggregation_ib_device
    kv_args.gpu_id = scheduler.ps.gpu_id
    kv_args.mla_compression_ratios = None

    from sglang.srt.mem_cache.deepseek_v4_memory_pool import DeepSeekV4TokenToKVPool

    if isinstance(kv_pool, DeepSeekV4TokenToKVPool):
        kv_args.mla_compression_ratios = list(kv_pool.compression_ratios)

    return kv_args


# ---------------------------------------------------------------------------
# Per-request D2P sender (decode side)
# ---------------------------------------------------------------------------


class D2PKVSender(MooncakeKVSender):
    """Per-request D2P sender on the decode side.

    Same interface as MooncakeKVSender. Adds connect() for D2P bootstrap
    (HTTP PUT to D2PKVBootstrapServer) and overrides clear() to release
    the radix tree lock.

    Lifecycle (in request_status):
      Bootstrapping → WaitingForInput → Transferring → Success/Failed

    Created inside capture_and_enqueue(). Polls are driven by the decode
    scheduler via D2PKVManager.poll_d2p_senders(), which handles the
    WaitingForInput → init() + send() transition externally (same pattern
    as the forward PD path in prefill.py).
    """

    def __init__(
        self,
        mgr: D2PKVManager,
        bootstrap_addr: str,
        bootstrap_room: int,
        dest_tp_ranks: List[int],
        pp_rank: int,
    ):
        super().__init__(mgr, bootstrap_addr, bootstrap_room, dest_tp_ranks, pp_rank)
        self.task: Optional[D2PReplicationTask] = None
        self._locked_node = None

    def connect(self) -> bool:
        """Ask the D2P bootstrap server to forward D2P_REQ."""
        if self.task is None:
            logger.warning("D2P: sender.connect() called without a task")
            return False
        return D2PKVBootstrapServer.put_d2p_request(
            self.task.bootstrap_addr,
            self.bootstrap_room,
            self.task,
            self.kv_mgr,
        )

    def clear(self) -> None:
        if self._locked_node is not None:
            self.kv_mgr.tree_cache.dec_lock_ref(self._locked_node)
            self._locked_node = None
        super().clear()


# ---------------------------------------------------------------------------
# D2P receiver with SNDTIMEO (prefill side)
# ---------------------------------------------------------------------------


class _D2PReceiver(MooncakeKVReceiver):
    """MooncakeKVReceiver with SNDTIMEO on PUSH sockets to prevent
    the prefill scheduler from hanging indefinitely when the decode-side
    D2P bootstrap PULL queue fills up."""

    @classmethod
    def _connect(cls, endpoint, is_ipv6=False):
        sock, lock = super()._connect(endpoint, is_ipv6)
        sock.setsockopt(zmq.SNDTIMEO, D2PKVBootstrapServer.D2P_SNDTIMEO_MS)
        sock.setsockopt(zmq.LINGER, 0)
        return sock, lock


# ---------------------------------------------------------------------------
# D2PKVManager
# ---------------------------------------------------------------------------


class D2PKVManager(MooncakeKVManager):
    """Reverse KV manager for D2P replication.

    Shares the forward manager's transfer engine (register_buffer_to_engine
    is a no-op). Registers to the D2P bootstrap server at
    (disaggregation_bootstrap_port + 1).

    PREFILL mode (decode side — sender):
      capture_and_enqueue() creates a D2PKVSender (same interface as
      MooncakeKVSender). The sender's connect() posts D2P_REQ metadata
      to the D2P bootstrap server. The bootstrap_thread is receive-only
      (blocking recv_multipart), same as mooncake's start_prefill_thread.
      When TransferInfo arrives (prefill responded), update_status →
      WaitingForInput. poll_d2p_senders() handles transitions:
      WaitingForInput → init() + send() + Transferring → Success/Failed → clear().

    DECODE mode (prefill side — receiver):
      start_decode_thread() dispatches D2P_REQ via handle_d2p_request().
      process_d2p_incoming() is called from the scheduler event loop to
      handle allocation (leftover space), _D2PReceiver creation,
      completion polling, and radix cache insertion.
    """

    def __init__(
        self,
        args: KVArgs,
        disaggregation_mode: DisaggregationMode,
        server_args,
        is_mla_backend: Optional[bool] = False,
    ):
        if disaggregation_mode not in (
            DisaggregationMode.PREFILL,
            DisaggregationMode.DECODE,
        ):
            raise ValueError(
                f"D2PKVManager requires PREFILL or DECODE mode, got {disaggregation_mode}"
            )

        self._d2p_bootstrap_port = get_d2p_bootstrap_port(server_args)

        self.tree_cache = None

        if disaggregation_mode == DisaggregationMode.PREFILL:
            self._d2p_senders: Dict[int, D2PKVSender] = {}
            self._room_counter = 1_000_000_000
            self._d2p_bootstrap_addr = None
        elif disaggregation_mode == DisaggregationMode.DECODE:
            self._pending_alloc: deque = deque()
            self._active_receivers: Dict[int, tuple] = {}
            self._allocated_rooms: Dict[int, dict] = {}
            self.token_to_kv_pool_allocator = None

        super().__init__(args, disaggregation_mode, server_args, is_mla_backend)
        self.bootstrap_port = self._d2p_bootstrap_port

        if disaggregation_mode == DisaggregationMode.DECODE:
            self.register_to_bootstrap()

    def register_to_bootstrap(self):
        self.bootstrap_port = self._d2p_bootstrap_port
        super().register_to_bootstrap()

    def register_buffer_to_engine(self):
        pass

    # ------------------------------------------------------------------
    # DECODE mode: D2P bootstrap thread (prefill side)
    # ------------------------------------------------------------------

    def start_decode_thread(self):
        """Override: receive D2P_REQ and status sync on our own socket.
        No staging/AUX_DATA — D2P only transfers KV data."""

        def d2p_decode_thread():
            while True:
                msg = self.server_socket.recv_multipart()
                if msg[0] == D2PKVBootstrapServer.D2P_REQUEST_HEADER:
                    self.handle_d2p_request(msg)
                    continue

                bootstrap_room, status, prefill_rank = msg
                status = int(status.decode("ascii"))
                bootstrap_room = int(bootstrap_room.decode("ascii"))
                if status == KVPoll.Success:
                    if bootstrap_room in self.request_status:
                        self.update_status(bootstrap_room, KVPoll.Success)
                elif status == KVPoll.Failed:
                    self.update_status(bootstrap_room, KVPoll.Failed)

        threading.Thread(target=d2p_decode_thread, daemon=True).start()

    # ------------------------------------------------------------------
    # PREFILL mode: bootstrap thread (decode side)
    # ------------------------------------------------------------------

    def start_prefill_thread(self):
        """Override: blocking recv loop for incoming TransferInfo / KVArgs.

        D2P_REQ forwarding is handled by the D2P bootstrap server, so this
        thread is receive-only — same as mooncake's start_prefill_thread.
        """

        def bootstrap_thread():
            while True:
                msg = self.server_socket.recv_multipart()
                room_str = msg[0].decode("ascii")
                mooncake_session_id = msg[3].decode("ascii")

                if room_str == "None":
                    self.decode_kv_args_table[mooncake_session_id] = (
                        KVArgsRegisterInfo.from_zmq(msg)
                    )
                    with self.session_lock:
                        if mooncake_session_id in self.failed_sessions:
                            self.failed_sessions.remove(mooncake_session_id)
                        if mooncake_session_id in self.session_failures:
                            del self.session_failures[mooncake_session_id]
                    logger.debug(
                        f"D2P: registered KVArgs from {mooncake_session_id}"
                    )
                else:
                    required_dst_info_num = int(msg[7].decode("ascii"))
                    room = int(room_str)
                    if room not in self.transfer_infos:
                        self.transfer_infos[room] = {}
                    self.transfer_infos[room][mooncake_session_id] = (
                        TransferInfo.from_zmq(msg)
                    )
                    if len(self.transfer_infos[room]) == required_dst_info_num:
                        self.req_to_decode_prefix_len[room] = next(
                            (
                                info.decode_prefix_len
                                for info in self.transfer_infos[room].values()
                                if info.decode_prefix_len is not None
                            ),
                            0,
                        )
                        self.update_status(room, KVPoll.WaitingForInput)

        threading.Thread(target=bootstrap_thread, daemon=True).start()

    # ------------------------------------------------------------------
    # Init helpers (called from scheduler after construction)
    # ------------------------------------------------------------------

    def init_d2p_sender(self, scheduler: Scheduler):
        self.tree_cache = scheduler.tree_cache

        if scheduler.server_args.dist_init_addr:
            d2p_host = NetworkAddress.parse(
                scheduler.server_args.dist_init_addr
            ).resolved().host
        else:
            bind_host = scheduler.server_args.host
            d2p_host = (
                self.local_ip
                if bind_host in ("0.0.0.0", "::", "")
                else bind_host
            )
        d2p_port = get_d2p_bootstrap_port(scheduler.server_args)
        self._d2p_bootstrap_addr = NetworkAddress(
            d2p_host, d2p_port
        ).to_host_port_str()
        logger.info("D2P sender initialized (reverse PREFILL manager)")

    def init_d2p_receiver(self, scheduler: Scheduler):
        self.tree_cache = scheduler.tree_cache
        self.token_to_kv_pool_allocator = scheduler.token_to_kv_pool_allocator
        logger.info("D2P receiver initialized (reverse DECODE manager)")

    # ------------------------------------------------------------------
    # PREFILL mode (decode side — sender)
    # ------------------------------------------------------------------

    def _next_room(self) -> int:
        self._room_counter += 1
        return self._room_counter

    def capture_and_enqueue(self, req: Req):
        """Called AFTER release_kv_cache from the scheduler thread.

        Creates a D2PKVSender, captures KV indices from the radix tree,
        and asks the D2P bootstrap server to forward D2P_REQ.
        """
        token_ids = list(
            (req.origin_input_ids + req.output_ids)[: req.kv_committed_len_saved]
        )
        prompt_len = len(req.origin_input_ids)
        if len(token_ids) <= prompt_len:
            return

        match_result = self.tree_cache.match_prefix(
            MatchPrefixParams(key=RadixKey(token_ids))
        )
        matched_len = len(match_result.device_indices)
        if matched_len <= prompt_len:
            return

        last_node = match_result.last_device_node
        self.tree_cache.inc_lock_ref(last_node)

        src_kv_indices = (
            match_result.device_indices[prompt_len:].cpu().numpy().astype(np.int32)
        )

        bootstrap_addr = NetworkAddress(
            req.bootstrap_host, self._d2p_bootstrap_port
        ).to_host_port_str()

        room = self._next_room()
        sender = D2PKVSender(self, bootstrap_addr, room, [], 0)
        sender.task = D2PReplicationTask(
            token_ids=token_ids,
            prompt_len=prompt_len,
            src_kv_indices=src_kv_indices,
            bootstrap_addr=bootstrap_addr,
        )
        sender._locked_node = last_node
        self._d2p_senders[room] = sender

        if not sender.connect():
            self.update_status(room, KVPoll.Failed)

        logger.info(
            f"D2P: enqueued room={room}, "
            f"new_tokens={len(src_kv_indices)}, bootstrap={bootstrap_addr}"
        )

    def poll_d2p_senders(self):
        """Poll all D2P senders and handle transitions.

        Called from the decode scheduler each tick (in process_decode_queue).
        Handles WaitingForInput → init() + send() externally, matching the
        forward PD pattern in prefill.py.
        """
        done_rooms = []
        for room, sender in self._d2p_senders.items():
            status = sender.poll()
            if status == KVPoll.WaitingForInput:
                sender.init(len(sender.task.src_kv_indices))
                sender.send(sender.task.src_kv_indices)
                self.update_status(room, KVPoll.Transferring)
            elif status == KVPoll.Success:
                logger.info(f"D2P: transfer success room={room}")
                sender.clear()
                done_rooms.append(room)
            elif status == KVPoll.Failed:
                logger.warning(f"D2P: transfer failed room={room}")
                sender.clear()
                done_rooms.append(room)

        for room in done_rooms:
            self._d2p_senders.pop(room)

    # ------------------------------------------------------------------
    # DECODE mode (prefill side — receiver)
    # ------------------------------------------------------------------

    def handle_d2p_request(self, msg):
        """Called from forward manager's bootstrap_thread on D2P_REQ.

        msg layout: [D2P_REQ, room, token_ids_bytes, prompt_len,
                      num_tokens, d2p_bootstrap_addr]
        """
        room = int(msg[1].decode("ascii"))
        token_ids = list(np.frombuffer(msg[2], dtype=np.int32))
        prompt_len = int(msg[3].decode("ascii"))
        num_tokens = int(msg[4].decode("ascii"))
        d2p_bootstrap_addr = msg[5].decode("ascii")

        # Pre-resolve parallel info on background thread so the scheduler
        # doesn't need to do a blocking HTTP fetch.
        try:
            self.try_ensure_parallel_info(d2p_bootstrap_addr)
        except Exception:
            logger.debug("D2P: parallel info pre-fetch failed for %s", d2p_bootstrap_addr)

        pending = D2PPendingRequest(
            room=room,
            token_ids=token_ids,
            prompt_len=prompt_len,
            num_tokens=num_tokens,
            d2p_bootstrap_addr=d2p_bootstrap_addr,
        )
        self._pending_alloc.append(pending)
        logger.info(f"D2P prefill: request room={room}, tokens={num_tokens}")

    def process_d2p_incoming(self):
        """Called from the prefill scheduler event loop each tick."""
        self._process_allocations()
        self._process_active_receivers()

    def _process_allocations(self):
        processed = 0
        while self._pending_alloc and processed < 16:
            req = self._pending_alloc[0]
            processed += 1

            if not self.try_ensure_parallel_info(req.d2p_bootstrap_addr):
                logger.debug("D2P: decode reverse manager not registered yet, deferring")
                break

            self._pending_alloc.popleft()
            try:
                self._handle_alloc(req)
            except Exception as e:
                logger.warning(f"D2P alloc failed for room {req.room}: {e}")

    def _free_d2p_indices(self, kv_indices: npt.NDArray[np.int32]):
        target_device = self.token_to_kv_pool_allocator.release_pages.device
        dst = torch.from_numpy(kv_indices.astype(np.int64)).to(target_device)
        self.token_to_kv_pool_allocator.free(dst)

    def _handle_alloc(self, req: D2PPendingRequest):
        num_tokens = req.num_tokens
        if self.token_to_kv_pool_allocator.available_size() < num_tokens:
            logger.info(f"D2P: not enough pool space for {num_tokens} tokens, skipping")
            return

        dst_indices = self.token_to_kv_pool_allocator.alloc(num_tokens)
        if dst_indices is None:
            logger.info("D2P: pool allocation returned None, skipping")
            return

        dst_kv_indices = dst_indices.cpu().numpy().astype(np.int32)

        alloc_info = {
            "token_ids": req.token_ids,
            "prompt_len": req.prompt_len,
            "dst_kv_indices": dst_kv_indices,
            "alloc_time": time.monotonic(),
        }
        self._allocated_rooms[req.room] = alloc_info

        try:
            receiver = _D2PReceiver(
                self, req.d2p_bootstrap_addr, req.room
            )
            receiver.init(prefill_dp_rank=0)

            if getattr(receiver, "conclude_state", None) == KVPoll.Failed:
                logger.warning(f"D2P: receiver init failed for room {req.room}")
                self.token_to_kv_pool_allocator.free(dst_indices)
                self._allocated_rooms.pop(req.room, None)
                return

            receiver.send_metadata(dst_kv_indices)
        except zmq.Again:
            logger.warning(f"D2P: receiver ZMQ send timed out for room {req.room}")
            self.token_to_kv_pool_allocator.free(dst_indices)
            self._allocated_rooms.pop(req.room, None)
            return
        except Exception as e:
            logger.warning(f"D2P: receiver setup failed for room {req.room}: {e}")
            self.token_to_kv_pool_allocator.free(dst_indices)
            self._allocated_rooms.pop(req.room, None)
            return

        self._active_receivers[req.room] = (receiver, alloc_info)
        logger.info(
            f"D2P prefill: receiver created room={req.room}, "
            f"dst_tokens={num_tokens}"
        )

    _D2P_RECEIVER_TIMEOUT_S = 60.0

    def _process_active_receivers(self):
        done_rooms = []
        now = time.monotonic()
        for room, (receiver, alloc_info) in self._active_receivers.items():
            status = receiver.poll()
            if status == KVPoll.Success:
                logger.info(f"D2P prefill: transfer complete room={room}")
                try:
                    self._insert_into_radix_cache(
                        token_ids=alloc_info["token_ids"],
                        prompt_len=alloc_info["prompt_len"],
                        new_kv_indices=alloc_info["dst_kv_indices"],
                    )
                except Exception as e:
                    logger.warning(f"D2P radix insert failed room={room}: {e}")
                    self._free_d2p_indices(alloc_info["dst_kv_indices"])
                receiver.clear()
                done_rooms.append(room)
            elif status == KVPoll.Failed:
                logger.warning(f"D2P prefill: transfer failed room={room}")
                self._free_d2p_indices(alloc_info["dst_kv_indices"])
                receiver.clear()
                done_rooms.append(room)
            elif now - alloc_info.get("alloc_time", now) > self._D2P_RECEIVER_TIMEOUT_S:
                logger.warning(f"D2P prefill: receiver timed out room={room}")
                self._free_d2p_indices(alloc_info["dst_kv_indices"])
                receiver.clear()
                done_rooms.append(room)

        for room in done_rooms:
            self._active_receivers.pop(room, None)
            self._allocated_rooms.pop(room, None)

    def _insert_into_radix_cache(
        self,
        token_ids: List[int],
        prompt_len: int,
        new_kv_indices: npt.NDArray[np.int32],
    ):
        if not isinstance(self.tree_cache, RadixCache):
            return

        existing_match = self.tree_cache.match_prefix(
            MatchPrefixParams(key=RadixKey(token_ids[:prompt_len]))
        )
        existing_indices = existing_match.device_indices

        new_indices_tensor = torch.from_numpy(new_kv_indices.astype(np.int64)).to(
            existing_indices.device
        )
        full_indices = torch.cat([existing_indices, new_indices_tensor])

        full_key = RadixKey(token_ids[: len(full_indices)])
        result = self.tree_cache.insert(
            InsertParams(key=full_key, value=full_indices)
        )
        if result.prefix_len > len(existing_indices):
            dup_indices = full_indices[len(existing_indices) : result.prefix_len]
            if len(dup_indices) > 0:
                self.token_to_kv_pool_allocator.free(dup_indices)

        logger.debug(
            f"D2P: inserted {len(new_kv_indices)} tokens into radix cache "
            f"(prefix_len={result.prefix_len}, total={len(full_indices)})"
        )
