# Data

This repository does not redistribute third-party datasets or clinical images.

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

## Release Policy

Before publishing data-derived artifacts, check:

- dataset redistribution terms;
- patient/privacy restrictions;
- whether sample visualizations contain clinical metadata;
- whether prediction masks can be released under the dataset license.

For that reason, this repository includes non-sensitive architecture and result
figures, but not raw angiography images, dataset archives, or trained weights.
