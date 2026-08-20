# GitHub Release Pattern Followed

The repository follows common practices used by strong ML research code
releases, with extra caution for medical-image data boundaries.

## Public Patterns Used

The release shape was checked against:

- Papers with Code, "Tips for Publishing Research Code":
  https://github.com/paperswithcode/releasing-research-code
- SegFormer official implementation:
  https://github.com/NVlabs/SegFormer
- nnU-Net official repository:
  https://github.com/MIC-DKFZ/nnUNet
- MMSegmentation toolbox and benchmark:
  https://github.com/open-mmlab/mmsegmentation

The shared pattern is:

- short project identity and architecture/result visual near the top;
- installable package or clear dependency specification;
- exact training and evaluation entry points;
- table of results in the README plus machine-readable result files;
- configs separated from code;
- model/data/reproducibility documentation;
- release notes and citation metadata;
- clear policy for checkpoints, pretrained models, datasets, and large artifacts.

## How DEFT-Net Implements This Pattern

- `src/deftnet/`: importable PyTorch package.
- `configs/`: current and legacy YAML configs.
- `scripts/train_phase1.py`, `scripts/train.py`, `scripts/evaluate.py`, and
  `scripts/infer.py`: two-stage training and evaluation entry points.
- `scripts/per_image_metrics.py`, `scripts/paired_stats.py`,
  `scripts/profile_model.py`: reproducibility utilities.
- `experiments/pooled_public_fixed_notta_seed_summary.csv`: current public
  aggregate result table.
- `experiments/mechanism_controls_three_seed_summary.csv` and
  `experiments/full_data_controls_three_seed_summary.csv`: mechanism evidence.
- `experiments/fusion_hsaf_vs_mean.json`: mechanism-control summary.
- `docs/`: data, model-card, reproducibility, artifact-roadmap, release notes,
  and experiment-design summaries.
- `.gitignore`: prevents datasets, checkpoints, archives, Office manuscripts,
  and local run outputs from being committed accidentally.

## Medical-Image Specific Guardrails

Unlike a generic computer-vision demo repo, DEFT-Net deliberately withholds:

- raw angiography images;
- third-party dataset archives;
- trained weights until release is approved;
- prediction masks if filenames or metadata could disclose private information;
- manuscript DOCX/PDF drafts.

This is why the public repository is a code and aggregate-evidence release
rather than an unrestricted data-and-checkpoint artifact.
