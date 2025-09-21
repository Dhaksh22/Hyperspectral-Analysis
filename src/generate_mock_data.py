# generate_mock_data.py
import numpy as np
import pandas as pd
import os
from datetime import datetime, timedelta

os.makedirs("sample_data", exist_ok=True)

# create a fake RGB and NIR bands: shape (H,W)
H, W = 256, 256
np.random.seed(0)
# base vegetation pattern
base = np.clip(np.random.normal(loc=0.6, scale=0.12, size=(H,W)), 0, 1)
# create red and nir bands
red = (base * 0.5 + np.random.normal(0, 0.05, (H,W))).clip(0,1)
nir = (base * 0.8 + np.random.normal(0, 0.05, (H,W))).clip(0,1)
green = (base * 0.6 + np.random.normal(0, 0.05, (H,W))).clip(0,1)

# simulate a stressed patch: reduce nir in a rectangle
nir[80:140, 120:180] *= 0.5
red[80:140, 120:180] *= 1.05

# save as npz
np.savez("sample_data/sample_tile.npz", red=red.astype(np.float32), nir=nir.astype(np.float32), 
         green=green.astype(np.float32))

# create sample sensor CSV (time series)
start = datetime.now() - timedelta(days=9)
rows = []
for i in range(10):
    t = start + timedelta(days=i)
    # make soil moisture drop slightly when stress occurs
    soil = 0.35 if i < 6 else 0.18
    temp = 28 + np.random.randn()*1.2
    rh = 75 - i*1.5 + np.random.randn()*2
    leaf_wet = 0 if i%2==0 else 1
    rows.append([t.isoformat(), soil + np.random.normal(0,0.02), temp, rh, leaf_wet])

df = pd.DataFrame(rows, columns=["timestamp","soil_moisture","air_temp","humidity","leaf_wetness"])
df.to_csv("sample_data/sensors.csv", index=False)
print("Saved sample_data/sample_tile.npz and sample_data/sensors.csv")
