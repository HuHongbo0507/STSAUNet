# FIRST-Net

A standalone FIRST-Net implementation for datasets under F:\FIRST-NET. The model combines RPCANet-style unfolded background/target separation with five-frame learned deformable alignment and center-frame mask prediction.

## Implementation choices

- The FIRST-Net paper describes frame-wise feature separation followed by deformable temporal alignment and reconstruction. This implementation follows that structure and predicts the center-frame mask.
- The unfolding stage keeps separate learnable background-proximal and sparse-target-gradient operators. It avoids explicit SVD.
- The alignment layer predicts 18 offsets for a 3x3 deformable kernel and uses PyTorch grid_sample for bilinear sampling. It does not require MMCV compiled operators.
- DeRef_Net's ISTA_TDANNet is designed around 11x11 images, XML annotations, and hard-coded .mat initialization paths. Those parts are not reused because the provided datasets contain raster frames and pixel masks.
- NUDT-MIRSDT and NUDT-MIRSDT1 prefer train1.txt/test1.txt because they point to images. The train.txt/test.txt variants point to Mix MAT files and are accepted as fallback split lists. Validation uses unlisted complete sequences when present; if every sequence is already assigned to train/test, validation holds out whole sequences from the training list. The stale absolute paths in train2.txt/test2.txt are not used.
- Inputs are grayscale and normalized to [0,1]. Images use bilinear resizing, masks use nearest-neighbor resizing and are binarized.
- Sequence edges use replicated frames by default. Select reflect for reflected indexing.

## FIRST_dataset pairing needs confirmation

The image directory contains groups 00003, 00004, 00008, 00009, and 00010 with frame IDs 0-6. The mask directory contains groups 00001-00005 with frame IDs 1-7. Exact filename pairing is not possible. Visual and pixel checks confirm that 00003-0 aligns with 00003-1 and 00004-0 aligns with 00004-1, so those two mappings are recorded in first_dataset_pairs.csv with an offset of +1. The other three mappings cannot be established reliably from filenames and must be confirmed before FIRST_dataset training. The loader stops with an actionable error rather than assigning masks by sort order.

Edit first_dataset_pairs.csv in this folder after confirming the correspondence. Each row means: image_sequence,mask_sequence,frame_offset. For example, 00003,00003,1 pairs image 00003-0.png with mask 00003-1.png. Each image sequence must map to exactly one mask sequence, and every mask file must be used once.

## Environment

Use Python 3.10 or newer and install a PyTorch build that matches your OS and CUDA version. Then install the remaining requirements from this folder:

    python -m pip install -r requirements.txt

MMCV, MMEngine, OpenCV, and SciPy are not required for model training. If OpenCV is installed, object component metrics are computed faster; otherwise a pure Python fallback is used.

## Validate the datasets

From this folder:

    python verify_data.py --dataset NUDT-MIRSDT
    python verify_data.py --dataset NUDT-MIRSDT1

To validate FIRST_dataset after its mapping file is complete:

    python verify_data.py --dataset FIRST_dataset

The validator checks image/mask pairing, readability, dimensions, sequence-level splits, and same-sequence five-frame windows. It does not change data.

## Train

Defaults to the full NUDT-MIRSDT dataset at 256x256:

    python train.py

Choose another dataset:

    python train.py --dataset NUDT-MIRSDT --epochs 100 --batch-size 2 --image-size 256

Run the shortened NUDT-MIRSDT1 subset:

    python train.py --dataset NUDT-MIRSDT1 --epochs 100 --batch-size 2

Training outputs go to runs/<dataset>-<timestamp>/ by default. Each run contains config.json, history.csv, train.log, last.pt, and best.pt. For a small GPU, lower batch size, image size, feature channels, or unfolding channels. Use --image-size 0 to preserve native resolution; if native sizes vary, use batch size 1.

Important options include --window-size (must be odd), --boundary (replicate or reflect), --stages, --feature-channels, --max-offset, --reconstruction-weight, and --alignment-weight. Adam uses a default learning rate of 1e-4. Reconstruction and alignment weights default to 0.01.

## Evaluate

Use test sequences and a saved checkpoint:

    python evaluate.py --checkpoint runs/NUDT-MIRSDT-YYYYMMDD-HHMMSS/best.pt --dataset NUDT-MIRSDT --split test

If --dataset is omitted, the checkpoint's dataset is used. Evaluation reports pixel IoU, Precision, Recall, F1, object-level PD (centroid tolerance 3 pixels), and pixel false-alarm rate FA (false-positive pixels divided by background pixels). PD and FA definitions should be checked against a paper's protocol before comparing values.

## Infer on one sequence

    python infer.py --checkpoint runs/NUDT-MIRSDT-YYYYMMDD-HHMMSS/best.pt --dataset NUDT-MIRSDT --sequence Sequence1

Inference writes probability maps and 0/255 binary masks under predictions/<dataset>/<sequence>/. Original frame names and dimensions are retained.

## Minimal model check

After installing PyTorch, run:

    python -c "import torch; from firstnet.model import FIRSTNet; m=FIRSTNet(); x=torch.rand(1,5,1,64,64); y=m(x); print({k: tuple(v.shape) for k,v in y.items()})"

The logits and reconstruction should each have shape (1,1,64,64).
