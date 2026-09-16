\# Denoising of Microscopic Images



Comparison of classical and deep-learning denoising methods on microscopic images, evaluated using PSNR across Gaussian, Poisson, and Uniform noise.



\## Methods compared

\- \*\*Classical filters:\*\* Median, Wiener, BM3D

\- \*\*Deep learning:\*\* Zero-Shot Noise2Noise (ZS-N2N) — a lightweight (\~20k parameter) self-supervised network that trains from scratch on a single noisy image, requiring no external dataset or pretrained weights.



\## Results



| Noise Type | Median | Wiener | BM3D  | ZS-N2N (DL) |

|------------|--------|--------|-------|-------------|

| Gaussian   | 27.09  | 25.51  | 29.64 | 28.75       |

| Poisson    | 27.92  | 26.02  | 34.15 | 33.88       |

| Uniform    | 28.95  | 29.63  | 33.39 | 33.03       |



\*(PSNR in dB. See `report/` for full methodology and analysis.)\*



\## Repo structure

\- `notebooks/` — original research notebooks (classical filters + ZS-N2N)

\- `report/` — full write-up with methodology, related work, and results

\- `deployment/` — containerized version of the full comparison pipeline (Docker)



\## Running the deployed version



```bash

cd deployment

docker build -t denoiser .

docker run -v ${PWD}:/app/data denoiser --input /app/data/your\_image.jpg --output /app/data/comparison.png --add-noise g --noise-level 25 --epochs 500

```



Outputs a single comparison image showing Ground Truth, Noisy, Median, Wiener, BM3D, and ZS-N2N results side by side with PSNR for each.

