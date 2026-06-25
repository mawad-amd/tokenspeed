"""Breakable CUDA graph capture for variable-shape (prefill / extend) forwards.

A *breakable* CUDA graph captures a forward as an ordered list of zero-arg
callables -- each is either a captured ``CUDAGraph.replay`` (a "graph segment")
or an eager Python function (a "break"). At designated break points (attention /
KV-cache ops, whose metadata is data-dependent and cannot be captured) the
current stream capture is ended, the op runs eagerly, and a fresh segment begins
capturing the remainder. Replay simply calls each segment in order.

This is the ``torch.compile``-free alternative to piecewise CUDA graphs. It is
modeled on vLLM's ``BreakableCUDAGraphWrapper`` (the homogeneous segment-list
design) and sglang's breakable graph (the eager-copy output handoff). Unlike a
full prefill graph, attention -- the only batch/length-aware op and the source of
the host-side ``max_seq_len_q`` scalar -- stays eager, so it never enters a graph.
Keeping all KV-cache reads/writes in the eager breaks also makes them honor the
per-layer transfer consumer index naturally (see
``docs/design/prefill-breakable-cudagraph.md``).

Address-stability contract (the load-bearing invariant):

* All segments share one CUDA mempool, so graph-allocated intermediates keep
  stable device addresses across replays.
* The runner must copy live inputs into the *same* static input buffers used at
  capture before calling :meth:`BreakableCapture.replay`.
* Break-point outputs must land at the *same* address each replay. We achieve
  this by allocating a destination buffer in the captured segment (pool-pinned)
  and copying the eager op's result into it; the next segment reads that address.
"""

from __future__ import annotations

import functools
import inspect
import threading
from contextlib import contextmanager
from typing import Any, Callable, Iterator

import torch

__all__ = [
    "BreakableCapture",
    "active_forward",
    "break_here",
    "break_point",
    "current_forward_ctx",
    "is_breakable_capture_active",
    "weak_ref_tensor",
]


# Ambient per-forward context (the model's ``ForwardContext``), mirroring
# vLLM/SGLang's ``set_forward_context``/``get_forward_context``. An eager break
# runs once at capture and again on every replay; the args it closed over at
# capture are the *dummy* batch's, hence stale. Rather than thread the live
# context through ``replay()`` (which would conflate graph mechanics with forward
# semantics), the runner publishes it here for the duration of capture / each
# replay, and breaks rebind their captured context to it by identity (see
# :func:`break_here`). Breaks therefore read live ``ctx`` fields exactly like the
# eager path -- no per-model singleton reach-around, no frozen-scalar workarounds.
_ambient = threading.local()


@contextmanager
def active_forward(ctx: Any) -> Iterator[None]:
    """Publish ``ctx`` as the ambient forward context for the enclosed block.

    The runner wraps both capture and each replay in this so breaks see the live
    context. Re-entrant (saves/restores the previous value).
    """
    prev = getattr(_ambient, "ctx", None)
    _ambient.ctx = ctx
    try:
        yield
    finally:
        _ambient.ctx = prev


def current_forward_ctx() -> Any:
    """The ambient forward context, or ``None`` outside an :func:`active_forward`."""
    return getattr(_ambient, "ctx", None)


def weak_ref_tensor(t: Any) -> Any:
    """Reference a break-point tensor without pinning its cudagraph mempool slot.

    vLLM/sglang back this with a C op returning a non-owning view that shares
    storage. We don't have that op yet, so this is the identity: correct, but a
    strong ref keeps the pool slot alive, so keep ``--prefill-graph-max-tokens``
    modest. TODO: replace with a ``tokenspeed-kernel`` non-owning-view op.
    """
    return t


class BreakableCapture:
    """Thread-local context manager that captures a breakable graph.

    Usage::

        cap = BreakableCapture(pool=shared_pool)
        with cap:
            model_forward(...)        # attention calls hit break_here()
        # later, after copying live inputs into the static buffers:
        cap.replay()

    Args:
        pool: An optional CUDA mempool id (as returned by
            ``torch.cuda.graph_pool_handle()`` or ``CUDAGraph.pool()``) shared by
            all segments. If ``None``, the first segment allocates a fresh pool
            and the rest reuse it.
        stream: An optional dedicated capture stream. CUDA forbids stream capture
            on the default stream, so if ``None`` a fresh side stream is created
            (mirroring what ``torch.cuda.graph()`` does internally).
    """

    _tls = threading.local()

    def __init__(
        self, pool: Any | None = None, stream: torch.cuda.Stream | None = None
    ) -> None:
        self.pool = pool
        self.segments: list[Callable[[], Any]] = []
        self._current_graph: torch.cuda.CUDAGraph | None = None
        self._capturing = False
        self._stream = stream if stream is not None else torch.cuda.Stream()
        self._stream_ctx: Any | None = None

    @classmethod
    def current(cls) -> "BreakableCapture | None":
        return getattr(cls._tls, "active", None)

    # -- capture lifecycle -------------------------------------------------

    def __enter__(self) -> "BreakableCapture":
        if BreakableCapture.current() is not None:
            raise RuntimeError("Nested BreakableCapture is not supported.")
        # Capture on a dedicated side stream; make it observe all prior work
        # (warmup, static-buffer init) issued on the entry stream.
        self._stream.wait_stream(torch.cuda.current_stream())
        self._stream_ctx = torch.cuda.stream(self._stream)
        self._stream_ctx.__enter__()
        BreakableCapture._tls.active = self
        self._begin_segment()
        return self

    def __exit__(self, *exc: object) -> bool:
        try:
            self._end_segment()
        finally:
            BreakableCapture._tls.active = None
            if self._stream_ctx is not None:
                self._stream_ctx.__exit__(*exc)
                self._stream_ctx = None
            # Eager break ops ran on the side stream during capture; make the
            # entry stream observe them before any subsequent replay/work.
            torch.cuda.current_stream().wait_stream(self._stream)
        return False

    def _begin_segment(self) -> None:
        assert not self._capturing
        graph = torch.cuda.CUDAGraph()
        if self.pool is not None:
            graph.capture_begin(pool=self.pool)
        else:
            graph.capture_begin()
        self._current_graph = graph
        self._capturing = True

    def _end_segment(self) -> None:
        if not self._capturing:
            return
        assert self._current_graph is not None
        self._current_graph.capture_end()
        self.segments.append(self._current_graph.replay)
        # Share the pool across all subsequent segments so their intermediates
        # are co-located and addresses stay stable across the whole replay.
        if self.pool is None:
            self.pool = self._current_graph.pool()
        self._current_graph = None
        self._capturing = False

    def add_eager(self, fn: Callable[[], Any]) -> Any:
        """End the current segment, run ``fn`` eagerly, record it, start a new one.

        ``fn`` is a zero-arg callable that performs the break-point op and writes
        its result into a stable (pool-pinned) address. It is stored verbatim and
        re-invoked on every :meth:`replay`.
        """
        assert self._capturing, "add_eager called outside an active capture"
        self._end_segment()
        result = fn()
        self.segments.append(fn)
        self._begin_segment()
        return result

    # -- replay ------------------------------------------------------------

    def replay(self) -> None:
        """Replay all segments in order.

        Breaks read the live forward context from the ambient :func:`active_forward`
        scope (the runner wraps replay in it), so this stays a pure graph primitive.
        """
        for run in self.segments:
            run()

    @property
    def num_segments(self) -> int:
        return len(self.segments)


def is_breakable_capture_active() -> bool:
    """True while a :class:`BreakableCapture` is open AND currently capturing."""
    cap = BreakableCapture.current()
    return cap is not None and cap._capturing


def break_here(
    fn: Callable[..., torch.Tensor],
    dst: torch.Tensor,
    *args: Any,
    **kwargs: Any,
) -> torch.Tensor:
    """Run ``fn(*args, **kwargs)`` as an eager break, landing its result in ``dst``.

    ``dst`` must be a tensor allocated in the *current* captured segment (so it is
    pool-pinned at a stable address). At capture and on every replay, ``fn`` runs
    eagerly and its result is copied into ``dst`` (unless ``fn`` already wrote
    ``dst`` in place and returned it). The following graph segment reads ``dst``.

    Outside an active capture (eager forward, or breakable disabled) this is a
    transparent pass-through: ``fn`` runs and its result is copied into ``dst``.

    Args/kwargs are bound once at capture time, with two live exceptions: (1) tensor
    args alias persistent storage (the static input buffers / pool-pinned segment
    intermediates), so they carry live values at replay; (2) the per-forward
    :class:`ForwardContext` is rebound by identity to the live context each replay
    (see :meth:`BreakableCapture.replay`), so ``fn`` may read live ``ctx`` fields
    (``forward_mode``, ``bs``, ``num_extends``, ``global_num_tokens``, ...) exactly
    like the eager path. **Other (loose) non-tensor scalars are still frozen** to
    their capture-time value, so route any remaining per-request quantity through
    ``ctx`` / ``forward_*_metadata`` rather than a bare scalar arg.

    Args:
        fn: The break-point op (e.g. attention). Returns a tensor.
        dst: Pool-pinned destination buffer the downstream segment reads from.
        *args, **kwargs: Forwarded to ``fn`` (see the freezing note above).

    Returns:
        ``dst`` (the stable handoff buffer).
    """
    cap = BreakableCapture.current()
    if cap is None or not cap._capturing:
        _land_in(dst, fn(*args, **kwargs))
        return dst

    # Weak-ref tensor closures so the recorded replay_fn does not pin pool slots.
    weak_args = tuple(weak_ref_tensor(a) for a in args)
    weak_kwargs = {k: weak_ref_tensor(v) for k, v in kwargs.items()}
    weak_dst = weak_ref_tensor(dst)
    # The ambient forward context at capture (the dummy batch's). At replay it is
    # rebound by identity to the live context, so the break reads live ctx fields.
    captured_ctx = current_forward_ctx()

    def replay_fn() -> torch.Tensor:
        live_ctx = current_forward_ctx()

        def sub(a: Any) -> Any:
            return live_ctx if a is captured_ctx else a

        _land_in(
            weak_dst,
            fn(
                *(sub(a) for a in weak_args),
                **{k: sub(v) for k, v in weak_kwargs.items()},
            ),
        )
        return weak_dst

    return cap.add_eager(replay_fn)


def break_point(out: "str | Callable[..., Any]") -> Callable:
    """Mark a sequence-mixing method as an eager breakable-graph break point.

    Decorate a sequence-mixing method (attention / MLA / linear-mixer / sparse
    indexer ``forward``) and it runs as an eager break under a breakable capture --
    the surrounding token-shaped compute (norms, MoE, projections, collectives) is
    captured around it automatically, while everything inside the method stays
    eager -- or a zero-overhead direct call when not capturing. This is the one
    decorator every model uses to mark a break; it wraps the lower-level
    :func:`break_here` primitive (allocates the pool-pinned handoff buffer and
    lands the eager result into it for the next captured segment to read).

    The output handoff buffer's shape/dtype/device are resolved from ``out``, which
    is one of:

    * the NAME of a method parameter whose shape/dtype/device match the output
      (e.g. ``"hidden_states"`` for a coarse whole-attention break returning
      ``[tokens, hidden]``); empty inputs (0 rows) short-circuit to that tensor.
    * a callable ``out(*args, **kwargs)`` returning either a reference TENSOR (use
      its shape/dtype/device; 0 rows short-circuits) or an explicit
      ``(shape, dtype, device)`` tuple -- for a NARROW break whose output shape
      matches no single input (e.g. MLA attention, where the output is
      ``[tokens, heads*v_head_dim]`` but ``q`` is ``[tokens, heads*qk_head_dim]``).

    Inside the method ``ctx`` is live (see :func:`break_here`), so write the body
    exactly like the eager path -- no stale-capture handling.
    """

    def decorator(method: Callable) -> Callable:
        if callable(out):
            getter = out
        else:
            idx = list(inspect.signature(method).parameters).index(out)

            def getter(*args: Any, **kwargs: Any) -> torch.Tensor:
                return kwargs[out] if out in kwargs else args[idx]

        @functools.wraps(method)
        def wrapper(*args: Any, **kwargs: Any) -> torch.Tensor:
            ref = getter(*args, **kwargs)
            if isinstance(ref, torch.Tensor):
                if ref.shape[0] == 0:
                    return ref
                shape, dtype, device = ref.shape, ref.dtype, ref.device
            else:
                shape, dtype, device = ref
            if not is_breakable_capture_active():
                return method(*args, **kwargs)
            dst = torch.empty(shape, dtype=dtype, device=device)
            return break_here(method, dst, *args, **kwargs)

        return wrapper

    return decorator


def _land_in(dst: torch.Tensor, result: torch.Tensor) -> None:
    """Copy ``result`` into ``dst`` at a stable address.

    ``dst`` is the (possibly token-padded) handoff buffer the next graph segment
    reads. ``result`` may cover only the real (unpadded) leading rows -- e.g. a
    varlen attention kernel writes only ``sum(cu_seqlens_q)`` rows -- so we copy
    into the matching leading slice. Padded rows are left as-is (discarded by the
    final output slice). No-op when the op already wrote ``dst`` in place.
    """
    if result is dst:
        return
    if result.shape == dst.shape:
        dst.copy_(result)
    else:
        dst.narrow(0, 0, result.shape[0]).copy_(result)
