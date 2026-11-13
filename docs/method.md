# CPDINO model and training specification

## Output-level knowledge distillation

Student output `p` and teacher output `t` have shape `[N, 3, H, W]`. Channels 0–1 are flow and channel 2 is the foreground logit. Labels `y` have `[foreground, flow_y, flow_x]` after removing the instance-ID plane.

```text
L_task = 0.5 × MSE(p_flow, 5 × y_flow) + BCEWithLogits(p_cell, y_foreground > 0.5)
L_KD   = MSE(p_flow, stopgrad(t_flow)) + MSE(p_cell, stopgrad(t_cell))
L      = L_task + lambda × L_KD
```

The historical ViT-B run uses lambda 1, a directly fine-tuned ViT-B student and a frozen Pro_ver2 ViT-L teacher. Validation uses only L_task so comparisons are independent of the teacher target. This is dense output regression distillation. The recipe matches flow values and raw foreground logits.

## ViT-S backbone substitution

The ViT-S encoder comes from `facebook/dinov3-vits16-pretrain-lvd1689m`. It has 12 blocks, embedding width 384, 6 attention heads and MLP ratio 4. CPDINO adds a linear head producing 192 values per spatial token.

```mermaid
flowchart TD
    I[384 × 384 image] --> P[16 × 16 conv / stride 8 / padding 4]
    P --> T[2304 spatial tokens + CLS + 4 registers]
    T --> V[12 DINOv3-S blocks / width 384]
    V --> N[LayerNorm / remove 5 auxiliary tokens]
    N --> H[Linear 384 → 192]
    H --> W[Fixed W2 / reshape to 384 × 384 × 3]
    W --> F[Flow y / Flow x / foreground logit]
    F --> M[Cellpose dynamics / instance masks]
```

The original adapter temporarily replaced the upstream ViT-B factory to create the S variant. `AdaptedCPDINO` now takes the encoder explicitly and inherits the upstream forward path. Four register tokens and patch-16 input are checked at the boundary. Factory construction uses `pretrained=False`; checkpoint loading is a separate unit with a hash check.

The Transformers checkpoint conversion joins Q/K/V, inserts the zero K bias, maps layer scale and MLP names, and checks every tensor shape. Extra and missing tensors fail. The source-format conversion is currently verified for the official ViT-S checkpoint used in this project. Raw DINOv3 encoder state dicts can also be loaded.

## Optimization parameters

Upstream CPDINO applies the patch projection using `.data`; this blocks gradients into its weights. The adapter marks these parameters frozen explicitly. Transformer blocks and the linear readout are optimized. W2 and diameter parameters are fixed. The inherited mask token is unused by this segmentation forward path.

The historical S experiment uses lambda 0, AdamW, LR 1e-5, weight decay 0.1, 500 epochs, batch 29, 384-pixel tiles and validation every 10 epochs. B distillation uses batch 13. Exact dataset iteration is in the notebook, and checkpoint metadata includes the recipe and dataset file manifest fingerprint.

## Reproduction limits

The CPU unit notebook verifies execution with synthetic inputs. Public results were imported from the prior full experiment. Private images and checkpoint access are required to repeat those scores. The new modular implementation preserves the model forward and loss path, with explicit data/weight validation. It has been checked against the official ViT-S weights and a historical S state dict, but its entire 500-epoch trajectory has not been repeated.

Notebook resumptions are supported at epoch boundaries. NumPy, Python and Torch RNG, model and optimizer are restored. The experiment configuration and dataset fingerprint must match. Cross-device/library-version runs can have different numerical trajectories.

Direct B starts from a Cellpose segmentation checkpoint. S starts from a general DINOv3 encoder with a newly initialized segmentation head. An architecture-only accuracy comparison would require controlling this initialization difference as well as data, training budget and evaluation settings.

The training notebook saves `last.pt` at epoch boundaries. The original campaign selected `best.pt` by supervised validation loss; automatic best-checkpoint selection is not implemented in the public notebook. Historical test scores therefore refer to the original campaign artifacts.
