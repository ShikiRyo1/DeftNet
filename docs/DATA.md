# Data

This repository does not redistribute third-party datasets, coronary angiography
images, clinical images, masks, prediction masks, or dataset archives.

## Expected Local Layout

```text
data_root/
  train_images/
  train_masks/
  val_images/
  val_masks/
  test_images/
  test_masks/
  true_test_images/
  true_test_masks/
```

Masks are binary vessel masks. Images are converted to grayscale by the provided
dataset class.

## v77 Benchmark Scope

The manuscript-facing v77 benchmark is a pooled public coronary angiography
benchmark assembled from DCA/DCAE, XCAD, and ARCADE after binary vessel-mask
harmonization.

Because patient/procedure identifiers are not consistently available across the
public releases, conclusions are restricted to fixed-protocol image-level
benchmark comparisons. This repository therefore does not claim external
clinical validation, patient-level validation, or procedure-level validation.

## Release Policy

Before publishing data-derived artifacts, check:

- dataset redistribution terms;
- patient/privacy restrictions;
- whether sample visualizations contain clinical metadata;
- whether filenames or source metadata can reveal private information;
- whether prediction masks can be released under the dataset license.

For that reason, this repository includes non-sensitive architecture and result
figures, but not raw angiography images, dataset archives, prediction masks, or
trained weights.
