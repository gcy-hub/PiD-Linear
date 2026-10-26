# Vendored sources

The files `dit_t2i_DeCo.py`, `rmsnorm.py`, `rope.py`, `time_embed.py`,
`patch_embed.py`, `swiglu.py`, and `attention_op.py` are copied from
DeCo at commit `0792af05c9d8dce6c61e5636d136488f264065c7`. Only imports
in the transformer module are changed to package-relative imports; trailing whitespace is normalized. Official
attention already uses PyTorch SDPA. Upstream: https://github.com/Zehong-Ma/DeCo.

`gates.py` is the Apache-2.0 OutputGate adaptation from OpenVDN/vdn-minimax-h3,
carried by this repository's QDM research implementation. No external checkout
is required at runtime.
