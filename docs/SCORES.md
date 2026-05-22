# AEAL / AEAR Segmentation — Scores

52 patients, patient-level splits. Dice = standard (pixel-exact), held-out.

| Experiment | Dice | NSD
|---|---|---|
| Fixed atlas-box crop | 0.01 | 0.12
| Tight crop — single split | 0.50 | 0.71
| Tight crop — 5-fold CV, single model | 0.56 | 0.76
| Tight crop — 5-fold CV, ensemble | 0.62 | 0.79
| Bone-window — 5-fold CV, single model | 0.61 | 0.79
| Bone-window — 5-fold CV, ensemble | 0.61 | 0.77
| End-to-end (localizer + Stage-2) | 0.58 | 0.74
| 3D U-Net — single split | 0.64 | 0.86

 2p Dice / NSD are the appropriate metrics.

