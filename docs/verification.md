# Verification record

Verified on 2026-11-13 using Python 3.11, PyTorch 2.8.0+cu128, Cellpose 4.2.1.1 and the pinned DINOv3 commit. Model operations used CPU.

- 25 pytest cases passed. Coverage includes dense loss values, detached teacher targets, forward geometry, actual optimizer updates, frozen patch projection, validation aggregation, data/flow pairing, split overlap, SHA-256 checking, Hugging Face conversion, RNG/optimizer resumption and notebook-relative path resolution.
- All three notebooks executed in a real Jupyter kernel from the `notebooks/` working directory. Notebook 01 performs an actual tiny-DINO training step and checkpoint round trip. Notebook 02 was run in its default inspection mode, with private-data loading and full-epoch execution disabled. Notebook 03 renders the historical aggregate results.
- The public model factory loaded the official ViT-S encoder and produced a finite `[1, 3, 64, 64]` output. It also loaded the original project's ViT-S 310-epoch checkpoint with `strict=True` and produced a finite output.
- A wheel was built and its package import/model construction checked in a separate target installation. The Cellpose license and third-party notices are included in the wheel.
- A separate code review identified a relative-path issue in Notebook 02. The correction resolves paths against `local_paths.json` and is covered by a regression test.

The historical 500-epoch experiments were not rerun by this verification. The original training and evaluation results are recorded separately in `results/historical-summary.json`. The isolated notebook tests use synthetic input and validate program behavior, not segmentation accuracy.

The existing Windows runtime emitted a pyreadline console-cleanup message in some shell runs after pytest completed. The notebook kernel tests completed successfully. Jupyter also emitted a local TCP transport warning during kernel startup.
