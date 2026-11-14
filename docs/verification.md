# Implementation validation

Verified on 2026-11-13 using Python 3.11, PyTorch 2.8.0+cu128, Cellpose 4.2.1.1 and the pinned DINOv3 commit. Model operations used CPU.

- 33 pytest cases passed. Coverage includes dense loss values, detached teacher targets, forward geometry, actual optimizer updates, frozen patch projection, validation aggregation, data/flow pairing, split overlap, SHA-256 checking, Hugging Face conversion, RNG/optimizer resumption and notebook-relative path resolution.
- All three notebooks executed in a real Jupyter kernel from the `notebooks/` working directory. Notebook 01 performs an actual tiny-DINO training step and checkpoint round trip. Notebook 02 was run in its default inspection mode, with private-data loading and full-epoch execution disabled. Notebook 03 renders the historical aggregate results.
- The public model factory loaded the official ViT-S encoder and produced a finite `[1, 3, 64, 64]` output. It also loaded the original project's ViT-S 310-epoch checkpoint with `strict=True` and produced a finite output.
- A wheel was built and its package import/model construction checked in a separate target installation. The Cellpose license and third-party notices are included in the wheel.
- Notebook 02 resolves relative data and checkpoint paths against `local_paths.json`. This behavior is covered by a regression test.

## Numerical comparison with the original implementation

The original ViT-S constructor and the public model factory were initialized with seed 100 on CPU. All state-dict keys and tensors were identical. Both models received the same synthetic input of shape `[1, 3, 64, 64]`. Seed 20 was restored before each forward to align training-time masks. The maximum absolute difference in the segmentation output was 0 in both training and evaluation modes. The auxiliary compatibility output was excluded from the comparison.

The original and public loss functions were also compared on synthetic tensors of shape `[2, 3, 16, 16]`. Supervised loss, distillation loss and their gradients with respect to the prediction were identical for these inputs. These checks establish equivalence for the tested cases and do not establish equivalence of complete training trajectories across devices or library versions.

## Full-experiment reproduction

The historical 500-epoch experiments were not rerun by this verification. The original training and evaluation results are recorded separately in `results/historical-summary.json`. The isolated notebook tests use synthetic input and validate program behavior, not segmentation accuracy.

Notebook 02 now uses `EpochStore` to save best/last checkpoints and restore validation-selection metadata together with optimizer and RNG state. Its default inspection mode does not execute private-data training. Reproducing the reported scores requires the original inputs, initialization and evaluation procedure, with a separate execution record for the modular implementation.

## Additional ViT-S distillation validation

Tests cover full-model S initialization, frozen teacher parameters, optimizer isolation, explicit selection of `train_eligible`, best-checkpoint selection across resume, rejection of nonfinite validation loss, complete epoch iteration and instance-level metric aggregation. The real 500-epoch supervised S checkpoint and ViT-L teacher passed a CUDA smoke run using 13 training images and 2 validation images. Peak allocated GPU memory was 11.98 GiB and teacher gradients remained absent. The full 500-epoch additional distillation run is tracked separately; its final segmentation metrics are not inferred from this smoke test.

## Completed ViT-S distillation

The public training units completed 500 additional epochs. The checkpoint selected at epoch 470 was evaluated on 64 independent test images using the recorded BF16 protocol. Its SHA-256 is `0ee39d2d8bd26c1b3b5514359d3d503ae3c150f04d0be7c21f132268c466d0d7`. Aggregate metrics are recorded in `results/vits-distillation.json`.
