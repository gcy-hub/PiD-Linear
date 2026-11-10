"""Nonoverlapping query regions with halo keys and shared quadrant memory."""
from .sliding_qdm import SlidingQDM
from .local import halo_attention


class QDMHalo(SlidingQDM):
    def local_attention(self, inputs):
        return halo_attention(inputs.q_pos.to(inputs.v.dtype), inputs.k_pos.to(inputs.v.dtype),
                              inputs.v, inputs.layout, self.region_size, self.halo)
