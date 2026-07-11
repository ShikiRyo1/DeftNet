# Data scope and local layout

This repository does not redistribute third-party angiograms, clinical images,
reference masks, dataset archives, prediction exports, or identifying metadata.

## Expected local layout

```text
data_root/
  train_images/    train_masks/
  val_images/      val_masks/
  test_images/     test_masks/
```

Image and mask files are matched by filename or stem. Inputs are converted to
grayscale and resized to `512 x 512`; binary masks use nearest-neighbour
resampling.

## Current benchmark scope

The manuscript benchmark combines 1,760 released coronary angiography images
from DCA/DCAE, XCAD, and ARCADE after conversion to a common single-channel
image and binary vessel-foreground representation. The fixed released-image
split is 1,408/176/176 (80%/10%/10%) for training, validation, and held-out
testing. All model selection is confined to training and validation; the test
split is evaluated at threshold `0.5` without test-time augmentation.

The five Phase-I perspective folds are created only inside the 1,408-image
training split. Each expert omits a different internal fold and trains on the
other four. These folds do not replace the validation or held-out test split.

Where public releases expose reliable sequence or group identifiers, users
should construct group-disjoint partitions and record the grouping key in the
split manifest. The aggregate package is described as released-image evidence;
it is not presented as patient-, procedure-, site-, or external clinical
validation where those identifiers are unavailable.

## Harmonization and provenance checks

Before training, record and verify:

- source release and license for every image-mask pair;
- conversion from source annotations to binary vessel foreground;
- removal of empty, invalid, or duplicate records;
- dimensions and checksums after harmonization;
- split membership and any available sequence/group identifier;
- absence of exact image or mask duplicates across final partitions.

`scripts/make_split_manifest.py` generates image/mask checksums for a prepared
local split. `scripts/make_perspective_folds.py` records the deterministic
internal Phase-I fold assignment.

## Release policy

Only artifacts permitted by each source license should be published. Do not
commit raw clinical images, metadata-bearing screenshots, private filenames,
prediction masks, or checkpoints derived from restricted data without a
separate release review.
