| Corruption | Severity | n | PSNR model | PSNR input | gain | PSNR median | SSIM model | SSIM input | gain | SSIM median | L1 model | L1 input |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| clean | none | 40 | 22.90 | n/a | n/a | n/a | 0.6599 | 1.0000 | -0.3401 | n/a | 0.0495 | 0.0000 |
| salt_pepper | low | 40 | 22.79 | 20.10 | +2.69 | 33.86 | 0.6553 | 0.5878 | +0.0675 | 0.9482 | 0.0503 | 0.0150 |
| salt_pepper | medium | 40 | 22.59 | 15.80 | +6.79 | 32.63 | 0.6465 | 0.3207 | +0.3258 | 0.9409 | 0.0518 | 0.0402 |
| salt_pepper | high | 40 | 22.27 | 13.10 | +9.17 | 30.11 | 0.6337 | 0.1954 | +0.4383 | 0.9219 | 0.0545 | 0.0748 |
| blur | low | 40 | 22.93 | 35.54 | -12.61 | n/a | 0.6589 | 0.9713 | -0.3124 | n/a | 0.0495 | 0.0101 |
| blur | medium | 40 | 22.91 | 29.26 | -6.34 | n/a | 0.6554 | 0.8848 | -0.2294 | n/a | 0.0497 | 0.0211 |
| blur | high | 40 | 22.80 | 26.35 | -3.55 | n/a | 0.6485 | 0.7961 | -0.1476 | n/a | 0.0506 | 0.0301 |
| occlusion | low | 40 | 21.65 | 18.06 | +3.59 | n/a | 0.6349 | 0.8807 | -0.2458 | n/a | 0.0560 | 0.0375 |
| occlusion | medium | 40 | 19.97 | 14.13 | +5.84 | n/a | 0.6044 | 0.7728 | -0.1685 | n/a | 0.0660 | 0.0796 |
| occlusion | high | 40 | 18.06 | 12.25 | +5.81 | n/a | 0.5675 | 0.6627 | -0.0953 | n/a | 0.0827 | 0.1222 |

gain = model minus baseline (positive means the model is better than the baseline). Clean rows have no corruption, so their input is identical to the target and gains are n/a.
