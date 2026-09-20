# Notices

`fwht` and `Packed` in `src/_prism_hadamard_qwen35_shim.py` are ported from
`runtime/runtime.py` as shipped inside the Prism Hadamard MLX packs
(e.g. `prism-ml/Ternary-Bonsai-2-27B-mlx-2bit`). That file is distributed under
the MIT License, Copyright © 2023 Apple Inc.

They are reproduced rather than reimplemented so this adapter's numerics match
the pack's reference loader exactly.

This project is not affiliated with Prism ML, Apple, or oMLX.
