# FIRST-Net

A standalone FIRST-Net implementation for datasets under F:\FIRST-NET. The model combines RPCANet-style joint temporal-channel background/target separation with deformable alignment and center-frame mask prediction. The generic default is five frames; the FIRST_prepared experiment below explicitly uses seven.

## Implementation choices

- The earlier implementation used frame-wise separation. The joint_tc experiment below changes unfolding to jointly update the temporal block, followed by deformable alignment and center-frame prediction.
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

## FIRST_prepared joint unfolding experiment (2026-11-03)

The model now defaults to `unfolding_mode=joint_tc`. A sample is `[T,1,H,W]`;
DataLoader returns `[B,T,1,H,W]`. Before FeatureSeparation, only T and C are
packed to `[B,T*C,H,W]`, with `packed[b,t*C+c]=frames[b,t,c]`. Each background
proximal, sparse gradient and reconstruction operator uses Conv2d with groups=1
across all TC channels. All six stages therefore update the seven-frame block
jointly. The frame dimension is restored before per-frame feature extraction,
deformable alignment and center-frame fusion. The later B*T feature extraction
is intentional; unfolding has already used the temporal block.

`framewise` is retained only as an explicit legacy option. Old checkpoints without
an unfolding_mode field are loaded in framewise mode by evaluate.py and infer.py;
the new experiment starts from random weights and stores its joint mode.

The reconstruction term is now the mean of center-frame fused reconstruction MSE
and all-frame final unfolded reconstruction MSE. This connects the final merge
operator to supervision. The total remains BCEWithLogits + soft IoU + 0.01 times
reconstruction + 0.01 times alignment at the default weights. This joint unfolding
and additional unfolded reconstruction constraint are experiment changes, rather
than an assertion that the earlier framewise implementation already had them.

Use the prepared active manifests without re-splitting. Each sample uses all
seven frames, predicts frame 4, and preserves the existing 256x256 crops.

```powershell
Set-Location 'F:\FIRST-NET\FIRSTNet'
& 'D:\Anaconda3\envs\py310rpca\python.exe' .\verify_data.py --dataset FIRST_prepared
& 'D:\Anaconda3\envs\py310rpca\python.exe' -u .\train.py --data-root 'F:\FIRST-NET' --dataset FIRST_prepared --epochs 20 --batch-size 2 --workers 0 --window-size 7 --image-size 256 --device cuda --lr 1e-4 --seed 1 --unfolding-mode joint_tc --stages 6 --unfold-channels 32 --feature-channels 32 --reconstruction-weight 0.01 --alignment-weight 0.01 --check-inputs --evaluate-at-end
```

`--check-inputs` checks active split equality and parent isolation, full-size real
training samples, forward hooks for all stages, exact TC ordering/round-trip,
neighbor perturbation with unchanged center, batch independence, finite losses
and all gradients, and one Adam update on a throwaway model. Formal training is
re-seeded afterward. Only an actual CUDA OOM triggers batch-size fallback to 1.

Runs have environment.json, source/ and source_sha256.json, data_audit.json,
input_check.json, config.json, train.log, status.json, history.csv, validation/
per epoch, atomic best.pt and last.pt, and training_summary.json. After exactly
20 full epochs, `--evaluate-at-end` loads the validation-IoU-selected best.pt,
runs a single test pass at fixed threshold 0.5, and writes test_metrics.json,
summary.json, plus two example clips per available test source. Examples include
center image, true mask, probability PNG/float NPY, binary prediction and a montage.

Grouped validation/test reports contain pooled pixel metrics overall, by source,
and by annotation type. bbox_weak is a filled-box metric and must be interpreted
separately from pixel_mask. WideIRSTD is training only. Object-level PD uses
8-connected component centroids, greedy one-to-one matching within 3 pixels;
FA is false-positive pixels divided by ground-truth background pixels. These are
implementation definitions, not automatic comparability with other papers.

To inspect progress, read runs/<run>/status.json or the tail of train.log. Only
status=completed means training and requested final testing both finished;
training_summary.json alone means training finished but testing may still run.
A traceback and status=failed identify an actual failure.

## Controlled detection-improvement options

`--sampling uniform|real_synthetic|source_balanced` defaults to the original
uniform policy. Balanced policies use prepared manifest groups and replacement,
with exactly len(dataset) draws per epoch; no extra optimizer steps. Source,
annotation and unique-clip draws are saved in sampling/epoch-XXX.json.
`--segmentation-loss baseline|hard_bce` defaults to the original objective.
Hard BCE selects all positives plus the largest-loss background pixels, with
`--hard-negative-ratio 3 --minimum-negatives 256`; soft IoU and the existing
auxiliary loss definitions/weights are retained. The checkpoint saves the loss
configuration; evaluate.py reconstructs it. New options do not change inference
architecture or old checkpoint loading.

Real ROC/PR, probability caches, complexity profiling, tests and the bounded
20-epoch single-factor/combined study are in F:\FIRST-NET\Metric; see its README.md.
Only validation selects the winner. Existing FIRST_prepared and first7 runs remain
as independent preserved records.
