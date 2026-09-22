"""
FastAPI wrapper around denoising_core.py.

This is the "receptionist's desk" - it stays running, listens for
requests, and calls your actual denoising logic (denoising_core.py)
whenever someone sends an image.
"""
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import Response

from denoising_core import run_comparison_bytes

app = FastAPI(
    title="Denoising Comparison API",
    description="Upload an image; get back a side-by-side comparison of Median, Wiener, BM3D, and ZS-N2N denoising.",
)


@app.get("/health")
def health():
    """Cloud Run and Kubernetes both use endpoints like this to check
    'is this container actually working', not just 'is it running'."""
    return {"status": "ok"}


@app.post("/denoise")
async def denoise(
    file: UploadFile = File(...),
    add_noise: str = Query("g", enum=["g", "p", "u", "n"]),
    noise_level: float = Query(25.0),
    epochs: int = Query(300, le=1000, description="Capped at 1000 to keep API response times reasonable"),
):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Please upload an image file")

    image_bytes = await file.read()

    try:
        png_bytes, psnr = run_comparison_bytes(
            image_bytes, noise_type=add_noise, noise_level=noise_level, epochs=epochs
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing failed: {e}")

    return Response(
        content=png_bytes,
        media_type="image/png",
        headers={"X-PSNR-Results": str(psnr)},
    )
