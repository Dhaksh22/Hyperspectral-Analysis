# utils.py
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import io
import pandas as pd

def load_npz_image(path):
    data = np.load(path, allow_pickle=True)  # <--- added allow_pickle
    if 'red' in data and 'nir' in data:
        red = data['red']
        nir = data['nir']
        green = data.get('green', red)
        return {'red': red, 'nir': nir, 'green': green}
    else:
        keys = list(data.keys())
        return {'red': data[keys[0]], 'nir': data[keys[1]]}

def compute_ndvi(nir, red, eps=1e-8):
    # NDVI = (NIR - RED) / (NIR + RED)
    ndvi = (nir - red) / (nir + red + eps)
    ndvi = np.clip(ndvi, -1, 1)
    return ndvi

def ndvi_to_colormap_image(ndvi, cmap="RdYlGn"):
    import matplotlib
    norm = matplotlib.colors.Normalize(vmin=-1, vmax=1)
    cmap = matplotlib.cm.get_cmap(cmap)

    # If NDVI has >2 dimensions, take first band
    if ndvi.ndim > 2:
        ndvi = ndvi[:, :, 0]   # or np.mean(ndvi, axis=2)

    rgba = cmap(norm(ndvi))   # shape (H,W,4)
    rgba_uint8 = (rgba * 255).astype(np.uint8)

    return Image.fromarray(rgba_uint8)


def overlay_mask_on_rgb(red, green, nir, mask, alpha=0.5):
    """
    Create a pseudo-RGB image and overlay a red mask where 'mask' == 1.
    """

    import numpy as np
    from PIL import Image

    # Ensure inputs are 2D float32
    def to_2d(band):
        if band.ndim > 2:
            band = band[:, :, 0]   # take first band if extra dims
        return band.astype(np.float32)

    red   = to_2d(red)
    green = to_2d(green)
    nir   = to_2d(nir)

    # Normalize each band to 0–255
    def norm_band(b):
        b_min, b_max = b.min(), b.max()
        if b_max - b_min < 1e-8:
            return np.zeros_like(b, dtype=np.uint8)
        return ((b - b_min) / (b_max - b_min) * 255).astype(np.uint8)

    red, green, nir = norm_band(red), norm_band(green), norm_band(nir)

    # Stack into pseudo-RGB
    rgb = np.stack([red, green, nir], axis=2)  # (H,W,3)
    pil = Image.fromarray(rgb)

    # --- FIX mask handling ---
    if mask.ndim > 2:
        mask = mask[:, :, 0]   # pick first band if 3D
    mask = mask.astype(np.uint8)
    mask_img = Image.fromarray(mask * 255).convert("L")

    # Create overlay
    mask_overlay = Image.new("RGBA", pil.size)
    mask_overlay.paste((255, 0, 0, int(255 * alpha)), mask_img)

    pil = pil.convert("RGBA")
    out = Image.alpha_composite(pil, mask_overlay)

    return out.convert("RGB")


def read_sensor_csv(path):
    df = pd.read_csv(path, parse_dates=['timestamp'])
    return df

def simple_fusion_alert(ndvi_mean, soil_moisture_mean, ndvi_thresh=0.3, soil_thresh=0.2):
    alert = (ndvi_mean < ndvi_thresh) and (soil_moisture_mean < soil_thresh)
    score = max(0, (ndvi_thresh - ndvi_mean) / (ndvi_thresh + 1e-8) +
                   (soil_thresh - soil_moisture_mean) / (soil_thresh + 1e-8))
    if alert:
        msg = f"⚠️ Stress detected! NDVI={ndvi_mean:.2f}, Soil={soil_moisture_mean:.2f}, Risk Score={score:.2f}"
    else:
        msg = f"✅ Healthy. NDVI={ndvi_mean:.2f}, Soil={soil_moisture_mean:.2f}"
    return alert, score, msg

def compute_ndmi(nir, swir, eps=1e-8):
    ndmi = (nir - swir) / (nir + swir + eps)
    return np.clip(ndmi, -1, 1)
