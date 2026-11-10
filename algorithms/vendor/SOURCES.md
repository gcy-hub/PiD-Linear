# Operator sources

- `../local.py`: extracted from `flux2/flux2_qdm/vendor/clear_attention.py`, CLEAR-QDM adaptation a4262ad. CLEAR upstream: https://github.com/Huage001/CLEAR, revision ddca13160b7c2fe5d340c8a421db72b7bdaa6ba9. Apache-2.0, `CLEAR-LICENSE`. Circular neighborhood and learned pooling retained; shared framework layout replaces the local layout declaration.
- `gates.py`, `rms_norm.py`: existing OpenVDN adaptations from `flux2/flux2_qdm/vendor/`. https://github.com/OpenVDN/vdn-minimax-h3, reference revision e262cb5770b3dc992d4f44f251cd17bbfe0c4d8b. Apache-2.0, `LICENSE`.
- `../qdm.py`: extracted from `flux2/flux2_qdm/memory.py`; fixed-four FP32 solve and grid helpers shared with the PiD/FLUX3 implementations.
- `../cross_scan_kda.py`: extracted from `flux_kda/attention.py`; calls installed FLA `fla.layers.kda.KimiDeltaAttention` without replacing its recurrence. https://github.com/fla-org/flash-linear-attention (Apache-2.0).
