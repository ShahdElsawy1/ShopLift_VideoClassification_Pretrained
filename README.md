# Video Classification and Interior Image Editing

This repository holds three notebooks that use pretrained models in two areas:

1. **Shoplifting detection**: fine-tuning VideoMAE (Kinetics-400) to classify surveillance clips as normal or shoplifting.
2. **Interior image editing**: editing a living-room photo with Qwen-Image-2.1 (instruction-based editing) and with SAM3 plus SDXL inpainting (mask-based editing).

## Repository contents

| Notebook | Task | Platform |
|---|---|---|
| `finetuned-v1.ipynb` | Fine-tune VideoMAE-B for shoplifting detection | Kaggle (Tesla T4) |
| `qwen-image-editing.ipynb` | Prompt-based room editing with Qwen-Image-2.1 in 4-bit | Kaggle (2x T4) |
| `SAM_Image_Editing.ipynb` | Text-prompted segmentation (SAM3), recolor, and SDXL inpainting | Colab |

---

## 1. Shoplifting detection with VideoMAE

`finetuned-v1.ipynb` fine-tunes `MCG-NJU/videomae-base-finetuned-kinetics` for binary video classification and compares it with a CNN + BiLSTM trained from scratch.

### Dataset

- 855 clips (704x576): 531 normal, 324 shoplifting. Labels come from the folder names.
- The data is downloaded from a Google Drive zip link set in `DATA_URL`. The dataset is not included in this repository.

### Pipeline

1. **Frame cache.** 48 frames per clip, sampled from equal temporal segments, resized to 224x224 and stored as `.npy`.
2. **Duplicate removal.** ResNet50 frame embeddings, centred by the dataset mean, flag near-identical clips.
   - Pairs with aligned frame cosine of at least 0.90 are duplicates; one copy is kept. This dropped 218 of 855 clips (25.5%), leaving 637 (324 shoplifting, 313 normal).
   - Pairs of at least 0.80 are grouped so a group never spans train, validation, and test.
3. **Shortcut check.** A logistic regression on metadata and trivial clip statistics (duration, fps, motion, brightness, contrast) reaches AUC 0.705 under grouped CV. Model AUC should be judged against this baseline, not against 0.5.
4. **Group-aware split.** `StratifiedGroupKFold` gives 478 train, 82 validation, 77 test clips. A leakage audit confirms the maximum similarity across splits stays below 0.80.
5. **Augmentation.** One flip, crop, and color jitter is applied consistently across all frames of a clip.
6. **Frames per clip.** A short sweep over T = 8 and T = 16 selected T = 8 (validation F1 0.897 vs 0.833).
7. **Training.**
   - AdamW with learning rate 3e-5 for the backbone and 3e-4 for the head.
   - Cosine schedule with warmup, class-weighted cross-entropy with label smoothing 0.1.
   - Mixed precision, gradient accumulation, early stopping (patience 3, up to 10 epochs).
8. **Evaluation.** Flip test-time augmentation, threshold chosen on validation, Wilson and group-bootstrap confidence intervals, and error analysis.

### Results (test set, 77 clips)

| Threshold | Accuracy | Precision | Recall | F1 | AUC |
|---|---|---|---|---|---|
| 0.50 | 0.974 | 0.952 | 1.000 | 0.976 | 0.999 |
| 0.64 (chosen on validation) | 0.961 | 0.951 | 0.975 | 0.963 | 0.999 |

Group-bootstrap 95% CI for accuracy at threshold 0.64: 0.909 to 1.000.

Comparison with the scratch CNN + BiLSTM (3.27M parameters, F1 0.896 at threshold 0.5): the scratch model was evaluated on its own split, so the two test sets differ. Re-run both on the same split before drawing conclusions. The test set is small, so read the confidence intervals rather than the point estimates.

The final model is saved to `/kaggle/working/videomae_shoplifting`, and the comparison table to `comparison.csv`.

### Configuration

Main settings are at the top of the notebook: `DATA_URL`, `T_DEFAULT`, `EPOCHS`, `PATIENCE`, `LR_BODY`, `LR_HEAD`, `DUP_THR`, `GROUP_THR`, `RUN_SWEEP`, and `RUN_CV`. Set `RUN_CV = True` for grouped k-fold cross-validation (about 5x the training time).

---

## 2. Interior image editing

Both notebooks edit the same kind of input, a living-room photo, with different approaches.

### `qwen-image-editing.ipynb`: Qwen-Image-2.1

- Loads `Qwen/Qwen-Image-2.1` with 4-bit NF4 quantization (bitsandbytes).
- Splits the model across two GPUs: transformer and VAE on `cuda:0`, text encoder on `cuda:1`, with hooks that move tensors between devices.
- Input is resized so the longest side is 512, with dimensions rounded to multiples of 32. Inference uses 30 steps and a fixed seed of 42.
- Example edits: replace a sofa with a dark green velvet sofa; replace a wall painting with a colorful one.
- Peak GPU memory is about 12.3 GB on GPU 0 and 9.6 GB on GPU 1.

Requires `transformers>=5.17`, `accelerate`, `bitsandbytes`, `pillow`, and `diffusers` installed from GitHub.

### `SAM_Image_Editing.ipynb`: SAM3 and SDXL inpainting

1. **Segmentation.** `facebook/sam3` segments the object named in a text prompt (for example `"sofa"`).
2. **Recolor baseline.** The mask is used to shift hue and saturation in HSV space toward a target color.
3. **Inpainting.** The mask is dilated (25x25 kernel) and passed to `diffusers/stable-diffusion-xl-1.0-inpainting-0.1` to generate a new object (a dark red sofa and a gold sofa in the examples).
4. **Compositing.** The result is blended back with a Gaussian-blurred mask to keep the rest of the image unchanged.

`facebook/sam3` is a gated model. Request access on Hugging Face and log in with `notebook_login()` before running.

---

## Setup

- Python 3.10 or later and a CUDA GPU. The notebooks were run on Kaggle and Colab.
- Install the packages listed in the first cell of each notebook.
- Change the input paths (`DATA_URL`, and the `/kaggle/input/...` and `/content/...` image paths) to match your environment.

## Repository structure

```
.
├── finetuned-v1.ipynb
├── qwen-image-editing.ipynb
├── SAM_Image_Editing.ipynb
└── README.md
```

## Models used

- [MCG-NJU/videomae-base-finetuned-kinetics](https://huggingface.co/MCG-NJU/videomae-base-finetuned-kinetics)
- [Qwen/Qwen-Image-2.1](https://huggingface.co/Qwen/Qwen-Image-2.1)
- [facebook/sam3](https://huggingface.co/facebook/sam3)
- [diffusers/stable-diffusion-xl-1.0-inpainting-0.1](https://huggingface.co/diffusers/stable-diffusion-xl-1.0-inpainting-0.1)# ShopLift_VideoClassification_Pretrained
