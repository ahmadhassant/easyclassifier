# EasyVision – design draft

*Draft, 10 October 2026 (copy check and dataset-building practices added). Computer vision as its own project, built on the
rules of EasyClassifier and EasyResearch, and plugged into EasyResearch
Desktop as one more task.*

## 1. Why a separate project

* **Size.** Image work needs torchvision, pretrained weights (tens to hundreds
  of MB each), image libraries and, for segmentation, realistically a GPU.
  Bundling this into the release-1 desktop would double the download for
  users who only analyse tables and signals.
* **Time.** Release 1 is nearly done; vision is months of work. Separate
  releases keep release 1 on schedule.
* **Different evaluation problems.** Images bring their own, very common forms
  of leakage (section 2), and their own measures (Dice, IoU, mAP).
* **The architecture allows it.** The desktop loads tasks from module
  manifests and adapters. An `easyvision` package with its own adapter appears
  as the "Computer vision" task already shown in the sidebar, without changes
  to the other four tasks.

## 2. The honest story

Image models are reported with accuracies that often do not survive contact
with new data, and in most cases the reason is not the network but the
evaluation. As a reviewer one sees the same mistakes again and again. They are
easy to make in good faith and hard to detect from a paper, because the
methods section rarely says *when* augmentation was applied or *how* images
were split.

The evidence is strong:

* In brain MRI classification, splitting 2D slices instead of subjects
  inflated slice-level test accuracy by 29% to 55% across four datasets; on
  data with **random labels**, the leaky split still gave about 96% accuracy
  against the expected 50% (Yagis et al., 2021).
* About 3.3% of the CIFAR-10 and 10% of the CIFAR-100 test images have
  near-duplicates in the training set; on cleaned test sets the accuracy of
  well-known networks dropped, with error rates rising by up to 2.7 points
  (Barz & Denzler, 2020).
* Of the COVID-19 imaging models reviewed by Roberts et al. (2021), none was
  judged of potential clinical use, because of methodological flaws and
  biases, including duplicated images and patients across sets and
  combined datasets with different sources per class.
* Models can learn the hospital, the scanner or text markers instead of the
  disease ("shortcut learning"; Geirhos et al., 2020; DeGrave et al., 2021).
* Leakage is a cross-disciplinary problem in machine-learning-based science
  (Kapoor & Narayanan, 2023); medical imaging has its own long list of
  methodological failures (Varoquaux & Cheplygina, 2022).

**Augmentation** deserves a special place. It is a legitimate way to make
training harder to overfit (Shorten & Khoshgoftaar, 2019), but it is abused in
three ways that inflate results:

1. **Augmenting before splitting.** The dataset is enlarged offline (each
   image rotated, flipped, shifted …) and the enlarged set is then split. A
   test image's rotated twin is in the training set, so the test measures
   recognition of near-copies.
2. **Augmenting to "balance" classes before splitting.** The minority class
   is multiplied with augmented copies, which is oversampling with the same
   leak as above, and it also changes the class proportions the measures
   are computed on.
3. **Undisclosed test-time augmentation**, or augmentation applied to the test
   set and counted as more test cases.

EasyVision's position is simple: **a test image, and every image derived from
it or from the same subject, is never seen during training, model selection
or early stopping.** The tool enforces this by design, states it in the
report, and shows the user what would have happened otherwise (section 8).

### Leaks and what EasyVision does about them

| Leak | How it happens | What EasyVision does | What the report says |
|---|---|---|---|
| Augmentation before splitting | offline enlargement, then split | augmentation exists only as an on-the-fly transform inside the training part of each split; there is no option to write augmented images to disk | the transforms used, and that they were applied to training images only |
| Augmentation to balance classes | minority class multiplied before splitting | not offered; imbalance is handled with class weights and balanced accuracy | class sizes and how imbalance was handled |
| Several images per subject | patients, specimens, animals, plants, slides, 3D-scan slices, video frames, image tiles | a **group** (subject) column is part of the data layout; all images of a group stay in one fold; the tool asks for it and warns when file names suggest groups but none was given | number of subjects and images, and that splitting was by subject |
| Copies passed off as new images | the same image saved several times, or flipped, rotated, cropped or recoloured and saved as a new sample | copy check before anything else (section 2.1); copies are removed, never counted as data | files supplied, copies found by kind, unique images used |
| Near-duplicates | re-uploads, crops, the same photograph in two classes | perceptual-hash check across the whole dataset before splitting; duplicates are shown and either merged into one group or removed | how many near-duplicates were found and what was done |
| Preprocessing statistics from all images | normalisation mean and SD computed on all images | pretrained networks use their fixed published normalisation; anything learned (e.g. for a network trained from scratch) is learned on training images only | normalisation used |
| Early stopping on the test fold | the epoch with the best test score is kept | early stopping uses a validation part taken from the training images (by group) only | the stopping rule |
| Choosing the best of several networks | the winner's comparison score is reported | as in EasyClassifier: nested cross-validation (small data) or an untouched final test set (larger data) | the honest final score, separate from the comparison |
| Undisclosed test-time augmentation | predictions averaged over test transforms | off by default; if switched on, applied to every model alike and named in the report | whether test-time augmentation was used |
| Pretraining overlap | public benchmark images may be in the data used to pretrain the network | a warning when a well-known public dataset is detected (by name or hashes); cannot be fully checked | the pretrained weights used and this caveat |
| Shortcuts | text markers, rulers, borders, site-specific scanners | optional "site" column for grouping by hospital or device; occlusion maps of correctly and wrongly classified images | a reminder to check the maps for non-biological cues |
| Classes from different sources | each class collected from a different hospital, camera, website or dataset | the "source predicts class" check (section 2.1) | the check's result |

### 2.1 Copies passed off as new data, and other ways to build a dataset dishonestly

A common practice, met in student work and in submitted papers alike, is to
**make more data by copying it**: the same image is saved under several
names, or flipped, mirrored, rotated, cropped, resized, recoloured or
re-compressed and then added to the dataset as if it were a new case. The
dataset looks larger, the classes look balanced, and when it is split the
copies of a test image sit in the training set, so the accuracy is inflated
and the reported sample size is false. EasyVision does not allow this.

**Copy check (always on, cannot be switched off), before inspection or
splitting:**

1. **Exact copies**: identical files (file hash) and identical pixels under a
   different name or format (hash of the decoded pixels).
2. **Geometric copies**: the image flipped horizontally or vertically, or
   rotated by 90, 180 or 270 degrees. Each image's perceptual hash is
   computed for all eight flip-and-rotation variants and the smallest is kept
   as its signature, so a flipped or rotated copy has the same signature as
   the original.
3. **Photometric and size copies**: resized, re-compressed, or with changed
   brightness, contrast or colour; caught by a perceptual hash, which ignores
   such changes, within a small distance.
4. **Crops, zooms and small rotations**: features from the pretrained network,
   with very high similarity between two images, mark a likely copy. These
   pairs are shown side by side for the user to confirm, because genuinely
   similar images (for example neighbouring microscopy fields) can also be
   close.

**What happens:**

* Copies within the same class: one image is kept, the copies are removed
  and are not counted as data.
* The same image in two different classes: the analysis is **refused** until
  the user resolves it, because one image cannot belong to two classes.
* Likely copies the user confirms are removed; pairs the user marks as
  different images are kept, but placed in the same group (so they are never
  split between training and test), and the decision is listed in the report.
* The report and the Methods paragraph always give the honest counts: "N
  files were supplied; C were copies (E exact, G flipped or rotated, P
  resized or recoloured, K crops confirmed by the user) and were removed;
  the analysis used U unique images from S subjects."

**Other dataset-building practices the tool refuses or exposes:**

* **Reporting the augmented size as the dataset size** ("10,000 images" after
  augmentation): impossible, because augmentation never creates files, and
  the report states only unique images and subjects.
* **Patches or tiles of one image counted as independent cases** (for example
  many crops of one slide or photograph): patches are grouped by their source
  image or subject, which the layout requires when patches are supplied.
* **Classes collected from different sources** ("Frankenstein" datasets, where
  for example all diseased images come from one hospital or website and all
  healthy ones from another): before training, a simple model tries to
  predict the class from properties that should not carry the answer (image
  size, file format, colour mode, compression, camera or scanner fields,
  border colour). If these predict the class well, the tool warns that the
  network may learn the source instead of the content, and the warning is in
  the report.
* **Choosing the split or the random seed that gives the best result**: the
  seed is fixed and recorded, and cross-validation reports every fold, not
  the best one.
* **Removing "difficult" test images after seeing the results**: images can
  only be excluded before the analysis, with a reason, and the exclusions
  are listed in the report.
* **Using the test images to choose the stopping epoch or the model**: not
  possible; see the table above.

## 3. Rules (the same as EasyClassifier, applied to images)

1. Everything learned from data is learned on training images only, inside
   every split.
2. Splits are by subject (group) whenever images share a source.
3. Copies are never data: exact, flipped, rotated, resized or recoloured
   copies are removed before the analysis, and only unique images are
   counted (section 2.1).
4. The selected model receives a separate, honest final score.
5. Default settings fixed in advance, with fixed seeds; no hyperparameter
   search. Fine-tuning a pretrained network with one documented recipe is
   training, not tuning.
6. Every run writes a plain-language report with a Methods paragraph, every
   method cited, the leak checks performed, figures, predictions, the trained
   model, `settings.json`, `versions.txt` and a log.

## 4. Scope and phases

**Phase 1 – image classification (first release).**
Data layout: a folder with one subfolder per class, or a CSV/Excel table with
columns *file*, *class* and optionally *group* (subject) and *site*. Common
formats (PNG, JPEG, TIFF, BMP); DICOM and NIfTI later. Group detection: a
column in the table, or a pattern in file names (for example
`patient012_img3.png`) that the user confirms.

**Phase 2 – segmentation.** Images with masks (same file name in a `masks`
folder). U-Net-style network with a pretrained encoder; Dice and IoU per class;
overlay figures. Same grouping and splitting rules.

**Phase 3 – detection (only if users need it).** Bounding boxes; mAP. The most
demanding phase in data preparation and computing time.

Out of scope for now: 3D volumes as 3D (they are split by subject and analysed
as 2D slices with subject-level evaluation), video, self-supervised
pretraining.

## 5. Models (phase 1)

All with fixed defaults and seeds:

* **Pretrained CNNs as fixed feature extractors** + logistic regression (a
  "linear probe"): fast on a laptop CPU, a strong baseline for small data.
* **Fine-tuned pretrained networks**: a small ResNet, an EfficientNet and a
  small vision transformer; one documented recipe (optimizer, learning rate,
  epochs, early stopping on a validation part of the training images).
* **A classical baseline**: colour and texture features + random forest, to
  show what a simple method achieves.
* KNN with the Hassanat distance on the extracted features, as in the other
  cores.

Augmentation: a light, fixed, documented set (horizontal flip, small rotation,
random resized crop, small brightness change), on the fly, training images
only; the user can switch it off or remove transforms that make no sense for
the data (for example flips for text or for left-right anatomy).

## 6. Evaluation

* Group-aware stratified cross-validation; nested cross-validation for small
  data, a group-wise untouched final test set for larger data (the same rule
  as the other cores, counted in subjects).
* Measures: balanced accuracy (selection), accuracy, precision, recall, F1,
  MCC, ROC AUC; with several images per subject, results both per image and
  per subject (averaged predictions).
* A random-label negative control can be run with one click: the honest
  pipeline must give chance-level results.

## 7. Outputs

The standard results folder, plus:

* **"What was checked"**: copies removed (by kind), groups, near-duplicates,
  the source-predicts-class check, augmentation scope,
  stopping rule, test-time augmentation, pretraining caveat.
* **A reviewer checklist** in the report: the questions a reviewer should be
  able to answer from the paper, filled in for this analysis.
* Figures: class examples, confusion matrix, ROC curves, model comparison,
  occlusion maps, near-duplicate pairs (if any), learning curve.

## 8. The demonstration (and a paper)

As EasyClassifier's honesty benchmark showed for tables, the strongest
argument is a measured one. On public datasets, EasyVision will report the
same models evaluated five ways:

1. honest (group-wise split, augmentation inside training only);
2. augmentation before splitting;
3. image-level instead of subject-level splitting;
4. with near-duplicates left in;
5. with flipped and rotated copies added as new images, the practice of
   section 2.1;

plus a random-label control for each. The gap between (1) and (2)–(5) is the
inflation a reviewer cannot see in a paper. This is a publishable result in
its own right and a strong case for the tool.

## 9. Architecture

* A separate repository and pip package `easyvision`, depending on
  EasyResearch's shared parts (`common/output.py`, `common/references.py`),
  PyTorch and torchvision.
* Desktop: a `vision.json` manifest and an adapter. The desktop currently
  lists only modules whose input is a table (`input_kind == "tabular"`); it
  needs a folder chooser and an image preview for an `images` input kind.
* Pretrained weights are downloaded once and cached; the portable build can
  bundle a small set so that it works offline.

## 10. Computing

Phase 1 runs on a laptop CPU for a few thousand images with the linear probe,
and slower with fine-tuning (the progress bar already supports per-epoch
updates). A GPU is used automatically when present. Segmentation on realistic
data needs a GPU; this is where the server version with paid GPU access fits.

## 11. Open decisions

* Name: EasyVision, or a "vision" core inside EasyResearch with a separate
  release? (This draft assumes a separate package and repository.)
* First datasets for the demonstration (public, with subject identifiers,
  for example chest X-ray or dermatology collections with patient IDs).
* Which pretrained networks to offer by default, given download size and the
  licences of their weights.
* Whether DICOM is needed in phase 1.

## References

* Barz, B., & Denzler, J. (2020). Do we train on test data? Purging CIFAR of
  near-duplicates. Journal of Imaging, 6(6), 41.
  https://doi.org/10.3390/jimaging6060041
* DeGrave, A. J., Janizek, J. D., & Lee, S.-I. (2021). AI for radiographic
  COVID-19 detection selects shortcuts over signal. Nature Machine
  Intelligence, 3, 610-619. https://doi.org/10.1038/s42256-021-00338-7
* Geirhos, R., et al. (2020). Shortcut learning in deep neural networks.
  Nature Machine Intelligence, 2, 665-673.
* Kapoor, S., & Narayanan, A. (2023). Leakage and the reproducibility crisis
  in machine-learning-based science. Patterns, 4(9), 100804.
* Roberts, M., et al. (2021). Common pitfalls and recommendations for using
  machine learning to detect and prognosticate for COVID-19 using chest
  radiographs and CT scans. Nature Machine Intelligence, 3, 199-217.
  https://doi.org/10.1038/s42256-021-00307-0
* Shorten, C., & Khoshgoftaar, T. M. (2019). A survey on image data
  augmentation for deep learning. Journal of Big Data, 6, 60.
* Varoquaux, G., & Cheplygina, V. (2022). Machine learning for medical
  imaging: methodological failures and recommendations for the future. npj
  Digital Medicine, 5, 48.
* Yagis, E., Atnafu, S. W., García Seco de Herrera, A., Marzi, C., Scheda, R.,
  Giannelli, M., Tessa, C., Citi, L., & Diciotti, S. (2021). Effect of data
  leakage in brain MRI classification using 2D convolutional neural networks.
  Scientific Reports, 11, 22544. https://doi.org/10.1038/s41598-021-01681-w
