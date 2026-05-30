# Training-Scale Sensitivity Analysis: LST Downscaling in Google Earth Engine

---

## Purpose

This script is a modified version of the main LST downscaling framework, used specifically for the training-scale sensitivity analysis reported in **Supplementary Material S1** (Table S1) of the paper above.

The analysis tests whether the choice of 300 m as the training scale meaningfully affects model performance, by comparing results at 150 m and 300 m under identical conditions.

---

## What is different from the main script

Three changes were made relative to the main archived script (`landsat-lst-downscaling-10m`):

| Change | Location | Purpose |
|---|---|---|
| Added `MAX_EVAL = 2000` | Section 3 (Parameters) | Caps sample size so both scales use the same number of training/test points for a fair comparison |
| Added `sampleStack` reproject | Section 10.5 | Pins the coarse projection before sampling to avoid GEE memory errors at non-300 m scales |
| Added `.limit(MAX_EVAL)` to split | Section 10.5 | Applies the sample cap after random column assignment |

> **Important:** The absolute RMSE values from this sensitivity script are higher than those in the main Table 3 of the paper, because the sample size here (~2,000) is smaller than in the main run (~6,000). Table S1 should only be read as a within-test comparison between the two scales, not as a comparison against the main results.

---

## How to reproduce Table S1

### Step 1 — Open in Google Earth Engine

Go to [code.earthengine.google.com](https://code.earthengine.google.com) and paste the script, or import it directly.

### Step 2 — Run at 150 m

In Section 3, set:
```javascript
var COARSE_SCALE = 150;
```
Run the script for all years (`YEARS = [2021, 2022, 2023, 2024, 2025]`). Record the test RMSE, MAE, R², and ΔRMSE for each year from the console.

### Step 3 — Run at 300 m

Change to:
```javascript
var COARSE_SCALE = 300;
```
Run again. Record the same metrics.

Keep all other parameters identical between both runs. The key fixed settings are:

```javascript
var ALGORITHM            = 'RF';
var ABLATION_OPTICAL_ONLY = false;   // full model (optical + SAR)
var INDEX_STRATEGY       = 'NATIVE_20M';
var MAX_EVAL             = 2000;
var SEED                 = 0;
var RF_NUMBER_OF_TREES   = 500;
var RF_VARIABLES_PER_SPLIT = 6;
```

### Step 4 — Why 600 m was not tested

Aggregating the 10 m predictor bands to 600 m requires (600/10)² = 3,600 input pixels per output cell. This exceeds the `maxPixels: 1024` setting retained for reproducibility with the main script. The 600 m scale therefore cannot be evaluated cleanly under these settings.

---

## Required GEE assets

These are the same assets used by the main script (Warsaw case study):

| Asset | Path |
|---|---|
| Study boundary | `projects/ee-abdurrahmanzaki20/assets/warsaw` |
| Precipitation data | `projects/ee-abdurrahmanzaki20/assets/waw_precipitation` |

To adapt for another study area, replace these assets and adjust the `aoi` rectangle, `YEARS`, and season months as described in the HOW TO ADAPT section at the top of the script.

---

## Satellite data sources (all accessed via GEE)

| Data | GEE collection ID |
|---|---|
| Landsat 8 Collection 2 Level 2 | `LANDSAT/LC08/C02/T1_L2` |
| Landsat 9 Collection 2 Level 2 | `LANDSAT/LC09/C02/T1_L2` |
| Sentinel-2 Level-2A | `COPERNICUS/S2_SR_HARMONIZED` |
| Sentinel-1 GRD | `COPERNICUS/S1_GRD` |


## License

MIT License. See `LICENSE` file.

---

## Contact

Abdurrahman Zaki
Department of Geoinformation, Adam Mickiewicz University, Poznań, Poland
abdzak@amu.edu.pl | abdurrahman.zaki20@pwk.undip.ac.id
