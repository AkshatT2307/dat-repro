# Implemented

The selected Marmousi2 NMO time stack is now downloaded, prepared and runnable. See [RUN_128.md](RUN_128.md) for the final fixed-epoch protocol; the earlier planning notes below are superseded.

# Small 2D seismic extension of Fast IDR

Recommendation: use a **processed 2D seismic amplitude section from AGL Elastic
Marmousi (Marmousi2)** for the first controlled Gaussian-denoising experiment.
This is an extension of IDR to seismic data, not a result reproduced from the
original IDR paper. No seismic download/training has been started.

## Dataset choice

- Marmousi2 includes precomputed seismic sections, so no wave-equation simulation
  or migration is needed. Choose a time-domain NMO stack or migrated section,
  not the P/S-wave velocity or density model. AGL lists NMOstack_time.segy.zip
  at 6,936 KB and Kirchhoff_PreSTM_time.segy.zip at 7,543 KB.
  Source: https://www.agl.uh.edu/downloads/downloads.htm
- SEG documents the dataset and its attribution license:
  https://wiki.seg.org/wiki/AGL_Elastic_Marmousi
- The full original archive is listed at 153.7 MB on the Zenodo mirror, which
  includes processed data and license material:
  https://zenodo.org/records/23469300
  A second mirror is listed at:
  https://huggingface.co/datasets/shaowinw/seismic-benchmark-models
  Metadata was reviewed; binary downloads have not been tested. Direct SEG wiki
  requests return 403 in this environment; indexed SEG text was available.
- Mobil AVO Viking Graben Line 12 is a real 2D marine alternative. The original
  SEG catalog describes a 25 km line with prestack records. It is a later field
  test, not the easiest initial supervised-vs-IDR comparison: a noise-free paired
  reference is not established by its documentation.
  https://wiki.seg.org/wiki/Open_data
- Alaska 16-81 is another small real 2D line; raw and processed stack data are
  documented, but these are not automatically aligned noisy/clean training pairs.
  https://wiki.seg.org/wiki/Alaska_2D_land_line_16-81

## Proposed experiment

1. Read floating-point seismic amplitudes from one processed SEG-Y section.
   Preserve amplitude sign; do not train on rendered screenshots, color maps,
   uint8 quantization, or the RGB image pipeline's clipping/0.5 offset.
2. Partition by disjoint spatial regions BEFORE extracting patches; leave a
   buffer at least one patch wide. Do not randomly split overlapping patches,
   or put different processed versions of the same spatial region in train/test.
3. Use a one-input/one-output-channel U-Net with the Fast IDR training strategy.
   Normalize using training-region statistics only and retain the inverse scale.
4. Add known Gaussian noise at documented amplitude/SNR levels. IDR trains with
   noisy/refined targets; a separately trained supervised U-Net uses the original
   section patches as references. Use identical held-out corruptions for both.
   The original processed section is a reference for REMOVING ADDED NOISE, not
   proof of an entirely artifact-free physical signal.
5. Report SNR improvement, RMSE, waveform correlation, residual signal leakage,
   and consistent-scale section/trace plots. PSNR/SSIM can supplement these with
   a clearly defined amplitude range. No uint8 clipping for metric calculation.
6. Interpret this as an in-section held-out pilot. A single geological model
   cannot establish broad seismic or real-field generalization.

## Compute budget

The existing ImageNet log at step 4108 records batch 64, 256x256 patches and
2.7405 seconds/step, averaged over the run. That rate implies:

| Optimizer steps | One model | Fast IDR + supervised, assuming equal speed |
|---|---:|---:|
| 5,000 | 3.81 hours | 7.61 hours |
| 50,000 | 38.06 hours | 76.13 hours |

These are reference extrapolations, NOT measured seismic timings. Supervised
training may be faster. A smaller dataset alone does not reduce fixed-step
training cost. Single-channel inputs alone do not make the hidden U-Net layers
three times cheaper.

For a low-compute pilot, propose 128x128 patches, batch 8-16, and initially 5k
steps with validation; benchmark before choosing the full budget. Retaining
50k steps is also possible and should be used if that is the chosen protocol.
Smaller batches process fewer patches at the same optimizer-step count; log
that distinction. Do not change the ongoing ImageNet experiment.

## Selected protocol (updated user decision)

Choose **Marmousi2 NMOstack_time**, a precomputed 2D seismic amplitude section.
Use native 128x128 time-by-trace patches, a single-channel U-Net, and 10,000
optimizer steps per model. Initial LR 3e-4, halved at 5k and 8k. Compare Fast IDR
and supervised with identical patches/splits/noise and batch sizes. Fast IDR
continues to refresh its target network every epoch. Unlike the ImageNet branch,
do not downsample a native seismic patch just to match the image protocol:
native patches preserve the original time and trace sampling. If future seismic
resampling is desired, apply a documented anti-alias filter and update sampling
metadata before adding noise.

The choice and schedule are now settled; the seismic amplitude loader, splits,
normalization, and metrics still need implementing separately. Do not pass SEG-Y
or rendered seismic screenshots into the RGB ImageNet training script. No
seismic training or data download has been started.
