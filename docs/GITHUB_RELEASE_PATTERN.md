# GitHub Release Pattern Followed

The repository follows common practices used by strong ML research code releases:

- installable package under `src/`;
- quick-start commands in the README;
- visual architecture figure in the first viewport;
- dataset-layout section without redistributing restricted data;
- config-driven training and evaluation scripts;
- fixed-protocol result table committed as CSV;
- clear distinction between main results and sensitivity protocols;
- model-card, data, citation, license, and reproducibility docs;
- `.gitignore` that prevents datasets, checkpoints, archives, and local paper
  artifacts from being committed.

This shape is inspired by public research-code conventions from Papers with
Code's release checklist, NeurIPS reproducibility guidance, and mature
segmentation repositories such as SegFormer, nnU-Net, and MMSegmentation.
