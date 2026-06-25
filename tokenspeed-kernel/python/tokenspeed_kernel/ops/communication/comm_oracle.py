# Copyright (c) 2026 LightSeek Foundation
# Copyright (c) 2026 AMD
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Communication kernel SelectionOracle for TokenSpeed.

Implements the decision tables from The Communication Algorithms Bible
(Appendix A: Algorithm Decision Guide). Selects between iris (one-shot,
symmetric heap) and NCCL (ring/tree) based on message size, rank count,
and GPU architecture.

Decision logic:
- Intra-node XGMI (MI300X/MI355X), message < 4MB → iris (one-shot)
- Intra-node XGMI, message >= 4MB → NCCL ring (bandwidth-optimal)
- Intra-node NVLink → NCCL (iris not yet optimized for NVLink)
- Multi-node → NCCL NET (iris is intra-node only)
"""

from __future__ import annotations

import logging
from typing import Any

from tokenspeed_kernel.selection import SelectionOracle

logger = logging.getLogger(__name__)

# Thresholds derived from NCCL tuning.cc and empirical testing.
# See comms-bible Appendix A for rationale.
_IRIS_MAX_BYTES_ALLREDUCE = 4 * 1024 * 1024  # 4 MB
_IRIS_MAX_BYTES_REDUCE_SCATTER = 2 * 1024 * 1024  # 2 MB
_IRIS_MAX_BYTES_ALL_GATHER = 8 * 1024 * 1024  # 8 MB

# AMD GPU arch prefixes that support iris XGMI one-shot
_IRIS_SUPPORTED_ARCHS = {"gfx942", "gfx950"}


class CommSelectionOracle(SelectionOracle):
    """Per-family oracle for communication kernels.

    Scores iris kernels higher for small intra-node messages on AMD XGMI,
    scores NCCL higher for large messages and multi-node topologies.

    The score range is [0, 20):
    - 18-19: strong preference (iris for tiny messages on XGMI)
    - 14-16: moderate preference
    - 10: neutral (default)
    - 4-6: mild deprioritize
    - 0-2: strong deprioritize (iris for multi-node)
    """

    def adjust(
        self,
        spec: Any,
        platform: Any,
        traits: dict[str, Any] | None,
    ) -> int:
        if traits is None:
            return 10

        msg_bytes = traits.get("message_bytes", 0)
        is_intra_node = traits.get("intra_node", True)
        collective = traits.get("collective", "allreduce")
        world_size = traits.get("world_size", 1)

        is_iris = "iris" in getattr(spec, "name", "").lower()
        is_nccl = "nccl" in getattr(spec, "name", "").lower()
        is_amd = getattr(platform, "is_amd", False)
        gpu_arch = getattr(platform, "gpu_arch", "")

        xgmi_supported = any(gpu_arch.startswith(a) for a in _IRIS_SUPPORTED_ARCHS)

        if is_iris:
            return self._score_iris(
                msg_bytes, is_intra_node, collective, xgmi_supported, is_amd, world_size
            )
        elif is_nccl:
            return self._score_nccl(
                msg_bytes, is_intra_node, collective, xgmi_supported, world_size
            )

        return 10

    def _score_iris(
        self,
        msg_bytes: int,
        is_intra_node: bool,
        collective: str,
        xgmi_supported: bool,
        is_amd: bool,
        world_size: int,
    ) -> int:
        if not is_amd or not xgmi_supported:
            return 2  # iris requires AMD XGMI

        if not is_intra_node:
            return 0  # iris is intra-node only

        if world_size > 8:
            return 2  # iris targets single-node (max 8 GPUs on MI300X/MI355X)

        threshold = self._get_threshold(collective)

        if msg_bytes <= threshold:
            if msg_bytes <= 64 * 1024:
                return 19  # strong preference: one-shot wins big for tiny messages
            return 16  # moderate preference
        else:
            return 6  # mild deprioritize: NCCL ring is better for large messages

    def _score_nccl(
        self,
        msg_bytes: int,
        is_intra_node: bool,
        collective: str,
        xgmi_supported: bool,
        world_size: int,
    ) -> int:
        if not is_intra_node:
            return 18  # strong preference: NCCL NET for multi-node

        threshold = self._get_threshold(collective)

        if xgmi_supported and msg_bytes <= threshold:
            return 6  # iris is better for small intra-node on XGMI
        elif msg_bytes > threshold:
            return 16  # NCCL ring is bandwidth-optimal for large messages
        else:
            return 10

    @staticmethod
    def _get_threshold(collective: str) -> int:
        if collective == "allreduce":
            return _IRIS_MAX_BYTES_ALLREDUCE
        elif collective == "reduce_scatter":
            return _IRIS_MAX_BYTES_REDUCE_SCATTER
        elif collective == "all_gather":
            return _IRIS_MAX_BYTES_ALL_GATHER
        return _IRIS_MAX_BYTES_ALLREDUCE
