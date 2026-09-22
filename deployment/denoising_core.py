"""
Core denoising comparison logic: Median / Wiener / BM3D vs. ZS-N2N.

Same logic as before - this file just exposes it as a function
(run_comparison_bytes) that both a CLI script and a web API can call,
instead of only being runnable from the command line.
"""
import io

import bm3d
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from scipy.signal import wiener as scipy_wiener


def calc_psnr(clean, pred):
    mse = np.mean((clean - pred) ** 2)
    return 10 * np.log10(1 / mse) if mse > 0 else float("inf")


def bytes_to_grayscale(image_bytes, size=256):
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("Could not decode image - is this a valid image file?")
    img = cv2.resize(img, (size, size))
    return img.astype(np.float32) / 255.0


def add_noise(image, noise_type, noise_level, seed=42):
    rng = np.random.default_rng(seed)
    if noise_type == "g":
        noise = rng.normal(0, noise_level / 255, size=image.shape)
        return np.clip(image + noise, 0, 1).astype(np.float32)
    if noise_type == "p":
        noisy = rng.poisson(image * noise_level) / noise_level
        return np.clip(noisy, 0, 1).astype(np.float32)
    if noise_type == "u":
        noise = rng.uniform(-noise_level / 2, noise_level / 2, size=image.shape)
        return np.clip(image + noise, 0, 1).astype(np.float32)
    return image.astype(np.float32)


def median_filter(image_uint8, kernel=3):
    return cv2.medianBlur(image_uint8, kernel).astype(np.float32)


def wiener_filter(image, kernel=3):
    noise_variance = np.var(image - cv2.blur(image, (kernel, kernel)))
    return scipy_wiener(image, (kernel, kernel), noise_variance).astype(np.float32)


def bm3d_filter(image, sigma_psd=0.1):
    return bm3d.bm3d(image, sigma_psd).astype(np.float32)


class Network(nn.Module):
    def __init__(self, n_chan, chan_embed=48):
        super().__init__()
        self.act = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.conv1 = nn.Conv2d(n_chan, chan_embed, 3, padding=1)
        self.conv2 = nn.Conv2d(chan_embed, chan_embed, 3, padding=1)
        self.conv3 = nn.Conv2d(chan_embed, n_chan, 1)

    def forward(self, x):
        x = self.act(self.conv1(x))
        x = self.act(self.conv2(x))
        return self.conv3(x)


def pair_downsampler(img):
    c = img.shape[1]
    f1 = torch.FloatTensor([[[[0, 0.5], [0.5, 0]]]]).to(img.device).repeat(c, 1, 1, 1)
    f2 = torch.FloatTensor([[[[0.5, 0], [0, 0.5]]]]).to(img.device).repeat(c, 1, 1, 1)
    return F.conv2d(img, f1, stride=2, groups=c), F.conv2d(img, f2, stride=2, groups=c)


def zsn2n_denoise(noisy_np, epochs=300, lr=0.001, step_size=1000, gamma=0.5):
    noisy_t = torch.from_numpy(noisy_np).unsqueeze(0).unsqueeze(0)
    model = Network(1)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

    for _ in range(epochs):
        n1, n2 = pair_downsampler(noisy_t)
        p1, p2 = n1 - model(n1), n2 - model(n2)
        loss_res = 0.5 * (F.mse_loss(n1, p2) + F.mse_loss(n2, p1))
        denoised = noisy_t - model(noisy_t)
        d1, d2 = pair_downsampler(denoised)
        loss_cons = 0.5 * (F.mse_loss(p1, d1) + F.mse_loss(p2, d2))
        loss = loss_res + loss_cons
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

    with torch.no_grad():
        denoised = torch.clamp(noisy_t - model(noisy_t), 0, 1)
    return denoised.squeeze().numpy()


def run_comparison_bytes(image_bytes, noise_type="g", noise_level=25.0, epochs=300):
    """
    Takes raw image bytes (e.g. an uploaded file), runs all four denoising
    methods, and returns the comparison grid as PNG bytes + a dict of PSNR
    results. This is the function both the CLI and the API call - the one
    place the actual pipeline logic lives.
    """
    clean = bytes_to_grayscale(image_bytes)
    noisy = add_noise(clean, noise_type, noise_level)

    imgs = {"Noisy": noisy}
    psnr = {"Noisy": calc_psnr(clean, noisy)}

    m = median_filter((noisy * 255).astype(np.uint8), 3).astype(np.float32) / 255.0
    imgs["Median"] = m
    psnr["Median"] = calc_psnr(clean, m)

    w = wiener_filter(noisy, 3)
    imgs["Wiener"] = w
    psnr["Wiener"] = calc_psnr(clean, w)

    b = bm3d_filter(noisy, 0.1)
    imgs["BM3D"] = b
    psnr["BM3D"] = calc_psnr(clean, b)

    z = zsn2n_denoise(noisy, epochs=epochs)
    imgs["ZS-N2N (DL)"] = z
    psnr["ZS-N2N (DL)"] = calc_psnr(clean, z)

    fig, axes = plt.subplots(1, len(imgs) + 1, figsize=(4 * (len(imgs) + 1), 4))
    axes[0].imshow(clean, cmap="gray")
    axes[0].set_title("Ground Truth")
    axes[0].axis("off")

    for ax, (name, img) in zip(axes[1:], imgs.items()):
        ax.imshow(img, cmap="gray")
        ax.set_title(f"{name}\nPSNR: {psnr[name]:.2f} dB")
        ax.axis("off")

    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", dpi=120)
    plt.close(fig)
    buf.seek(0)

    return buf.getvalue(), psnr
