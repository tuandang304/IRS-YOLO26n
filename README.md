# IRS-YOLO26n: training-time infrared supervision for nano RGB vehicle detectors

Code for the paper *Training-Time Infrared Supervision for Nano RGB Vehicle Detectors in Low-Light UAV
Imagery: Gains and an Evaluation Blind Spot* (Computers, Materials & Continua, under review).

The recipe **P** uses the paired thermal-infrared (IR) frames of DroneVehicle only during training:

1. **IR soft labels.** Vehicles annotated in the IR image but not in the RGB image (matched by box IoU)
   enter the detection loss as down-weighted targets (ω = 0.3).
2. **Thermal-guided feature distillation (TGD).** 1×1 adapters align the student's neck features with
   those of a frozen YOLO26n IR teacher inside ground-truth regions; IR-only cells are excluded from
   the background term.

Both parts are removed after training: the deployed model is the unmodified RGB YOLO26n (or
YOLO11n / YOLOv10n). IRS-YOLO26n is YOLO26n trained with P at a 704-pixel input. Evaluation reports
COCO AP50:95 against **both** the RGB and the IR annotation, overall and for night images.

## Repository layout

| Path | Contents |
|---|---|
| `src/lowlight.py` | P: IR soft labels, TGD, plain feature distillation (`tgd_plain`), Ultralytics trainer hooks |
| `src/job_main.py` | Job driver: data preparation, training (in resumable chunks), prediction, latency |
| `src/dv_prep.py`, `src/dv_pairs.py` | DroneVehicle download (pinned Hugging Face revision), conversion to horizontal boxes, RGB–IR box matching (IR-only vehicles) |
| `src/stage0_manifest.py` | Illumination bands from mean RGB luminance (night / twilight / day) |
| `src/zdce.py` | Zero-DCE++ enhancement baseline |
| `src/evaluate.py`, `src/score_*.py`, `src/compare.py` | Evaluator (AP against RGB or IR labels, per band), paired bootstrap |
| `src/analysis_*.py`, `src/rev1_boot.py` | Recall of IR-only vehicles, threshold/IoU checks, seed×image bootstrap |
| `src/make_results.py`, `src/make_tables_extra.py`, `src/make_rev1_tables.py` | Tables and statistics (seed-paired differences, *t* CIs, Holm correction) |
| `src/make_figures.py`, `src/make_arch_figure.py` | Figures |
| `src/pseudo_labels.py`, `src/dfine_job.py` | Exploratory variants from proxy screening (fusion pseudo-labels, D-FINE); not part of the main results |
| `configs/queues/*.json` | Job specifications of every run, as launched (original submission) |
| `configs/queues/revision/*.json` | Additional seeds 1 and 2 (revision) |
| `splits/` | Image identifiers per split and illumination band, luminance calibration, `manifest.json` with SHA-256 hashes |
| `scripts/run_queue.py` | Runs a queue file on the local GPU |

## Setup

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
```

Pretrained checkpoints: `yolo26n.pt` and `yolo11n.pt` are downloaded by Ultralytics on first use;
download `yolov10n.pt` (Ultralytics release) to `weights/yolov10n.pt`. The Zero-DCE++ baseline needs the
official Zero-DCE++ weights `Epoch99.pth` at `inputs/zdcepp_Epoch99.pth`.

Data: `job_main.py` downloads DroneVehicle from the Hugging Face mirror `McCheng/DroneVehicle`
(revision `947bd13f`) and prepares it once under `$DV_DATA` (default `data/dronevehicle`). The official
release is <https://github.com/VisDrone/DroneVehicle>. The resulting image lists should match `splits/`.

## Training and prediction

Each queue file is a list of jobs; `chunks` lists the epochs at which training stops and resumes (the
paper's runs were trained in 15-epoch chunks on one NVIDIA L4, 60 epochs, batch 32). Outputs go to
`runs/remote/<job name>/ws/out/<job name>/`.

```bash
python scripts/run_queue.py configs/queues/stage4_vm1.json --only b1   # YOLO26n at 640 (checkpoint passed to the P jobs via `inputs`)
python scripts/run_queue.py configs/queues/stage4_vm2.json --only t_ir # IR teacher (YOLO26n on IR frames, 640)
mkdir -p runs/remote/t_ir_final/ws/out/t_ir_final/weights && \
  cp runs/remote/t_ir/ws/out/t_ir/weights/best.pt runs/remote/t_ir_final/ws/out/t_ir_final/weights/
python scripts/run_queue.py configs/queues/f_b1_704.json               # baseline YOLO26n, 704
python scripts/run_queue.py configs/queues/f_p_704.json                # IRS-YOLO26n (P), 704
```

Main models and their job specifications (seed 0 in `configs/queues/`, seeds 1–2 in `configs/queues/revision/`):

| Model in the paper | Job name | Queue file (seed 0) |
|---|---|---|
| YOLO26n (baseline) | `b1_704` | `f_b1_704.json` |
| YOLO26n + P (IRS-YOLO26n) | `p_704` | `f_p_704.json` |
| IR soft labels only (ω = 0.3) | `m2_soft_704` | `r3_vm_b.json` |
| Union of RGB and IR labels (ω = 1) | `m1_union_704` | `r3_vm_b.json` |
| Plain feature distillation | `m3_plainkd_704` | `r3_vm_b.json` |
| Zero-DCE++ + YOLO26n | `b2_704` | `f_b2_704.json` |
| YOLO11n / YOLO11n + P | `yolo11n_704` / `p11_704` | `f_yolo11n_704.json` / `f_p11_704.json` |
| YOLOv10n / YOLOv10n + P | `v10_704` / `p10_704` | `r3_vm_a.json` |
| YOLO26n / YOLO26n + P at 640 | `b1` / `p_640` | `stage4_vm1.json` / `f_p_640.json` |
| IR teacher | `t_ir` | `stage4_vm2.json` |

Jobs pass checkpoints of earlier jobs through `inputs` (for example `b1` and the IR teacher), so train those first. Some runs appear in more than one queue file because they were resumed after the cloud VM was
preempted; the job specification is identical. Prediction jobs (`kind: predict`, names starting with
`test_`) write `preds_<split>.json` (score threshold 0.001, at most 300 detections per image).

## Evaluation and statistics

```bash
python src/score_local.py runs/remote/test_p_704/ws/out/test_p_704/preds_test.json test --labels rgb ir
python src/compare.py A=<preds_a.json> B=<preds_b.json> --split test --labels rgb   # paired bootstrap
python src/make_rev1_tables.py      # three-seed statistics: per-seed differences, t CI, seed×image bootstrap, Holm
```

The table and figure scripts read predictions and scores from `runs/`; see the docstring at the top
of each script for its inputs.

## License and citation

This code builds on Ultralytics and is released under the GNU Affero General Public License v3.0
(see `LICENSE`). Trained weights and predictions are available from the corresponding author on
reasonable request. If you use this code, please cite the paper (citation to be added on publication).
