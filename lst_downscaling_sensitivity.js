// =============================================================================
// LST DOWNSCALING FRAMEWORK: Landsat 8/9 → 10 m via Sentinel-2/1
// Multi-Algorithm Comparison (GBT / RF / SVM / CART) with Grid Search
// =============================================================================
// SENSITIVITY ANALYSIS VERSION
// This script is a modified version of the main LST downscaling script,
// archived separately for the training-scale sensitivity analysis reported
// in Supplementary Material S1.
//
// Changes from the main archived script:
//   - Added MAX_EVAL (line ~143): caps sample size to ~2000 for fair comparison
//   - Added sampleStack reproject (Section 10.5): pins projection before sampling
//     to avoid memory errors at non-300 m coarse scales
//   - Added .limit(MAX_EVAL) to the 70/30 split (Section 10.5)
//
// To reproduce Table S1:
//   1. Set COARSE_SCALE = 150, run all years, record metrics
//   2. Set COARSE_SCALE = 300, run all years, record metrics
//   All other parameters must remain identical between the two runs.
// =============================================================================
// 
// DESCRIPTION:
//   This script downscales Landsat 8/9 land surface temperature (LST) from
//   30 m to 10 m spatial resolution using machine learning and predictor
//   variables derived from Sentinel-2 and Sentinel-1 imagery.
//   The workflow includes:
//     1. Cloud masking and dry-day filtering
//     2. LST retrieval (built-in and custom emissivity-corrected)
//     3. Spectral index computation from Sentinel-2 (with configurable
//        resolution strategy to minimize SWIR resampling artifacts)
//     4. SAR feature extraction from Sentinel-1
//     5. ML model training at configurable coarse resolution (COARSE_SCALE)
//        — set to 150 m or 300 m for sensitivity analysis
//     6. SAR ablation test (optical-only vs optical+SAR predictors)
//     7. Prediction at fine resolution (10 m)
//     8. Residual correction to preserve coarse-scale consistency
//
//   Supported algorithms:
//     - GBT  (Gradient Boosted Trees)  — ee.Classifier.smileGradientTreeBoost
//     - RF   (Random Forest)           — ee.Classifier.smileRandomForest
//     - SVM  (Support Vector Machine)  — ee.Classifier.libsvm (EPSILON_SVR)
//     - CART (Classification & Regression Trees) — ee.Classifier.smileCart
//
//   Grid search:
//     Set GRID_SEARCH_ENABLED = true to sweep hyperparameters for the
//     selected algorithm. Results are printed to the console. After
//     inspecting, set the best parameters and switch back to false.
//
//   Display and export options:
//     Multiple toggles available for printing diagnostics (scene info,
//     model statistics, variable importance) and exporting outputs
//     (LST composites, predictor bands, downscaled LST).
//
// CITATION:
//   Manuscript submitted for publication / under review
//
// HOW TO ADAPT FOR YOUR STUDY AREA:
//   1. Replace 'aoi' rectangle with your area coordinates
//   2. Replace 'study_boundary' asset with your boundary shapefile
//   3. Update YEARS array for your study period
//   4. Set MANUAL_CRS to your local coordinate system if USE_AUTO_UTM = false
//   5. Adjust SUMMER_START_MONTH and SUMMER_END_MONTH based on your study
//      area's hemisphere (Northern: June-August; Southern: December-February)
//   6. Adjust CLOUD_COVER_MAX threshold based on your region's cloud frequency
//   7. For precipitation: toggle USE_LOCAL_PRECIP to false to use ERA5,
//      OR upload your own precipitation FeatureCollection with required
//      properties: 'date' (string 'YYYY-MM-dd') and 'daily_precip_mm' (number)
//   8. Set ALGORITHM and tune hyperparameters via grid search for your study area
//
// REQUIRED ASSETS (for Warsaw case study):
//   - projects/ee-abdurrahmanzaki20/assets/warsaw         (boundary)
//   - projects/ee-abdurrahmanzaki20/assets/waw_precipitation
//
// =============================================================================

// ======================== SECTION 1: STUDY AREA ==============================
// Define your Area of Interest (AOI) and optional boundary polygon.
// For a new study area, change the rectangle coordinates and boundary asset.

var study_boundary = ee.FeatureCollection("projects/ee-abdurrahmanzaki20/assets/warsaw");
var aoi = ee.Geometry.Rectangle([20.75, 52.05, 21.30, 52.40]);
Map.centerObject(aoi, 10);

// Toggle: true = clip outputs to study_boundary; false = clip to aoi rectangle
var CROP_TO_BOUNDARY = true;

// Helper: returns the clipping geometry based on toggle
function clipRegion() {
  return CROP_TO_BOUNDARY ? study_boundary.geometry() : aoi;
}


// ======================== SECTION 2: COORDINATE REFERENCE SYSTEM ==============
// Auto-detect UTM zone from AOI centroid for portability.
// If you prefer a specific local CRS (e.g., EPSG:2178 for Poland), set it below.

var USE_AUTO_UTM = false; // true = auto-detect; false = use MANUAL_CRS
var MANUAL_CRS = 'EPSG:2178'; // Only used if USE_AUTO_UTM = false

function getProjectedCRS(geometry) {
  if (!USE_AUTO_UTM) return MANUAL_CRS;
  
  // Calculate UTM zone on client side to avoid ee.String issues
  var centroid = geometry.centroid(1).coordinates().getInfo();
  var lon = centroid[0];
  var lat = centroid[1];
  
  // Determine UTM zone number
  var zone = Math.floor((lon + 180) / 6) + 1;
  
  // Format zone number with leading zero if needed
  var zoneStr = zone < 10 ? '0' + zone : '' + zone;
  
  // North or South hemisphere
  var epsg = lat >= 0 ? 'EPSG:326' + zoneStr : 'EPSG:327' + zoneStr;
  
  return epsg;
}

var PROJ_CRS = getProjectedCRS(aoi);
print('Using coordinate system:', PROJ_CRS);


// ======================== SECTION 3: PARAMETERS ==============================
// Temporal, filtering, model, and export parameters.
// Future scholars: adjust these for your study area and time period.

// --- Temporal ---
var YEARS = [2021, 2022, 2023, 2024, 2025];
var SUMMER_START_MONTH = 6; // June
var SUMMER_END_MONTH   = 8; // August

// --- Precipitation / dry-day filtering ---
var PRECIP_THRESHOLD      = 1; // mm/day: max precipitation for "dry day"
var PRECIP_THRESHOLD_PREV = 1; // mm/day: max precip on the day before

// Toggle: true = use uploaded local precipitation asset
//         false = use ERA5-Land hourly reanalysis (global, ~11 km)
var USE_LOCAL_PRECIP = true;

// Local precipitation asset (Warsaw case study).
// FORMAT REQUIRED: FeatureCollection with properties:
//   - 'date' (string 'YYYY-MM-dd') or 'system:time_start' (millis)
//   - 'daily_precip_mm' (number: total daily precipitation in mm)
var LOCAL_PRECIP_ASSET = 'projects/ee-abdurrahmanzaki20/assets/waw_precipitation';

// --- Cloud masking ---
var CLOUD_COVER_MAX = 20;    // Max metadata cloud cover % for image selection
var CLOUD_BUFFER_M  = 0;     // Buffer (meters) around cloud pixels.
                              // 0 = no buffer (used for Warsaw).
                              // Try 100-300 for sensitivity analysis.
                              // Note: may increase data gaps in cloudy regions.

// --- Compositing ---
var USE_MEDIAN = false; // true = median composite; false = mean composite

// --- Sentinel-1 subsampling ---
// GEE may exceed memory for large areas with many S1 scenes.
// Increase STEP to reduce scenes (2 = every 2nd scene = 50% reduction).
var S1_SUBSAMPLE_STEP = 2;  // 1 = no subsampling; 2 = 50%; 3 = 66% reduction. Doing subsampling means trading temporal coverage for memory safety.

// --- Sentinel-2 index computation strategy ---
// 'NATIVE_20M': Compute SWIR indices at native 20 m (10 m bands aggregated
//               to 20 m first), then bilinear resample indices to 10 m.
//               Fewest edge artifacts, but loses some 10 m spatial detail.
// 'BILINEAR_BAND': Resample B11/B12 to 10 m via bilinear before computing
//                  indices. Preserves 10 m detail but may produce edge
//                  artifacts from resolution mismatch between sharp 10 m
//                  and smooth resampled 20 m bands.
// 'NEAREST' : Default GEE behavior. Fastest, but produces blocky artifacts.
var INDEX_STRATEGY = 'NATIVE_20M';


// =============================================================================
// >>>  ALGORITHM SELECTION  <<<
// =============================================================================
// Choose one: 'GBT', 'RF', 'SVM', 'CART'
// All algorithms use the same training data, prediction, residual correction,
// and validation workflow. Only the classifier changes.
var ALGORITHM = 'RF';

// --- SAR ablation test ---
// Compares model performance with and without Sentinel-1 derived predictors.
// true  = optical-only predictors (NDVI, NDBI, BSI, MNDWI, ALBEDO) - 5 predictors
// false = full model with optical + SAR predictors (+ VV, VH, VV_VH_RATIO) - 8 predictors
var ABLATION_OPTICAL_ONLY = false;

// --- Sampling ---
var NUM_PIXELS = 40000;          // request high; let supply fill it
var MAX_EVAL   = 2000;           // then cap to a common size both scales can supply
var SEED       = 0;      // Random seed for reproducibility

// --- Downscaling resolution ---
var COARSE_SCALE = 300; // Coarse training resolution (m)
var FINE_SCALE   = 10;  // Fine prediction resolution (m)


// =============================================================================
// >>>  HYPERPARAMETERS PER ALGORITHM  <<<
// =============================================================================
// These are the FIXED hyperparameters used when GRID_SEARCH_ENABLED = false.
// After running grid search, update these with the best values found.

// --- GBT (Gradient Boosted Trees) ---
// Determined through grid search across 2021-2025 summers.
var GBT_NUMBER_OF_TREES = 500;
var GBT_SHRINKAGE       = 0.05;   // Learning rate
var GBT_SAMPLING_RATE   = 1;      // Fraction of samples per tree
var GBT_MAX_NODES       = 25;     // Max terminal nodes per tree
var GBT_LOSS            = 'LeastAbsoluteDeviation'; // Robust to outliers

// --- RF (Random Forest) ---
var RF_NUMBER_OF_TREES    = 500;
// Adjusted automatically: 6 for full model (8 predictors), 3 for optical-only (5 predictors)
var RF_VARIABLES_PER_SPLIT = ABLATION_OPTICAL_ONLY ? 3 : 6;
var RF_MIN_LEAF_POPULATION = 1;     // Min samples in a leaf node
var RF_BAG_FRACTION        = 0.5;   // Fraction of input to bag per tree
var RF_MAX_NODES           = null;  // null = no limit

// --- SVM (Support Vector Machine, EPSILON_SVR) ---
var SVM_KERNEL_TYPE = 'RBF';   // 'LINEAR', 'POLY', 'RBF', 'SIGMOID'
var SVM_COST        = 100;      // Regularization parameter
var SVM_GAMMA       = 0.1;     // Kernel coefficient for RBF

// --- CART (Classification and Regression Trees) ---
var CART_MAX_NODES           = 100; // null = no limit
var CART_MIN_LEAF_POPULATION = 10;    // Min samples in a leaf node


// =============================================================================
// >>>  GRID SEARCH CONFIGURATION  <<<
// =============================================================================
// Set GRID_SEARCH_ENABLED = true to run a hyperparameter sweep for the
// selected ALGORITHM. Results are printed to the console as a table and chart.
//
// WORKFLOW:
//   1. Set GRID_SEARCH_ENABLED = true and ALGORITHM = 'GBT' (or RF/SVM/CART)
//   2. Run script → inspect console for RMSE per parameter combination
//   3. Note the best parameters
//   4. Update the fixed hyperparameters above with the best values
//   5. Set GRID_SEARCH_ENABLED = false → run the full downscaling
//
// Grid search uses one selected year (GRID_SEARCH_YEAR) only (to save compute).
// Each combination trains a separate model on GEE servers.
// Keep grids small (≤ 27 combinations) to avoid quota issues.

var GRID_SEARCH_ENABLED = false;
var GRID_SEARCH_YEAR = 2025; // Which year to use for grid search

// --- GBT grid ---
// Key parameters: numberOfTrees (model complexity) × shrinkage (learning rate)
var GBT_GRID_NUM_TREES = [100, 300, 500];
var GBT_GRID_SHRINKAGE = [0.01, 0.02, 0.05];
var GBT_GRID_MAX_NODES = [10, 25, 50]; // Optional 3rd parameter

// --- RF grid ---
// Key parameters: numberOfTrees × variablesPerSplit
var RF_GRID_NUM_TREES         = [100, 300, 500];
var RF_GRID_VARIABLES_PER_SPLIT = ABLATION_OPTICAL_ONLY ? [2, 3, 4] : [2, 4, 6];
var RF_GRID_MIN_LEAF_POP      = [1, 5, 10]; // Optional 3rd parameter

// --- SVM grid ---
// Key parameters: cost × gamma
var SVM_GRID_COST  = [1, 10, 100];
var SVM_GRID_GAMMA = [0.01, 0.1, 1.0];

// --- CART grid ---
// Key parameters: maxNodes × minLeafPopulation
// Use -1 to represent null (unlimited) for maxNodes in the grid
var CART_GRID_MAX_NODES    = [-1, 50, 100];  // -1 = null (unlimited)
var CART_GRID_MIN_LEAF_POP = [1, 5, 10];

// How many grid dimensions to use: 2 or 3
// 2 = tune first two parameter arrays only (faster, fewer combinations)
// 3 = tune all three parameter arrays (more thorough, slower)
// Only applies to GBT and RF which have 3 grid arrays defined.
// SVM and CART always use 2 dimensions.
var GRID_DIMENSIONS = 2;


// --- Display toggles ---
var PRINT_SCENE_INFO           = false;   // Print scene count, dates, and times per satellite
var SHOW_TRAIN_TEST_POINTS     = false;  // Add train and test samples as layers
var PRINT_IMPORTANCE           = false;  // Variable importance (GBT, RF, CART only)
var PRINT_MODEL_STATS          = true;  // Model performance statistics

// --- Export toggles ---
var EXPORT_STD_LST_TO_DRIVE    = false;  // Export standard (Collection 2) LST
var EXPORT_CORR_LST_TO_DRIVE   = false;  // Export emissivity-corrected LST
var EXPORT_PREDICTORS_TO_DRIVE = false;  // Export all predictor bands per year
var EXPORT_DOWNSCALED_TO_DRIVE = false;  // Export 10 m downscaled LST



// --- Visualization parameters ---
var VIS_LST = {
  min: 15, max: 40,
  palette: ['blue', 'cyan', 'yellow', 'red']
};


// ======================== SECTION 4: PRECIPITATION & DRY-DAY FILTER ==========
// Identifies dry days (precip < threshold on current AND previous day)
// and creates date filters used to subset Landsat, Sentinel-2, and Sentinel-1.

var startYear = YEARS[0];
var endYear   = YEARS[YEARS.length - 1];
var bufferDays = 1;
var startDate = ee.Date.fromYMD(startYear, SUMMER_START_MONTH, 1)
                  .advance(-bufferDays, 'day');
var endDate   = ee.Date.fromYMD(endYear, SUMMER_END_MONTH, 31)
                  .advance(1, 'day');

// Build precipitation collection based on source toggle
var precipCol;
if (USE_LOCAL_PRECIP) {
  // Local precipitation asset
  var precipFC = ee.FeatureCollection(LOCAL_PRECIP_ASSET);
  precipCol = precipFC.map(function(f) {
    var dateStr = ee.String(f.get('date'));
    var date = ee.Date.parse('YYYY-MM-dd', dateStr);
    var precip = ee.Number(f.get('daily_precip_mm'));
    return ee.Feature(f.geometry(), {
      mean_precip: precip,
      'system:time_start': date.millis()
    });
  });
} else {
  // ERA5-Land hourly total precipitation (global).
  // Aggregated to daily sum over the AOI.
  // Note: ERA5 precipitation is in meters; convert to mm (* 1000).
  var era5 = ee.ImageCollection('ECMWF/ERA5_LAND/HOURLY')
    .filterDate(startDate, endDate)
    .filterBounds(aoi)
    .select('total_precipitation');

  // Generate daily dates
  var nDays = endDate.difference(startDate, 'day').round();
  var dayList = ee.List.sequence(0, nDays.subtract(1));
  precipCol = ee.FeatureCollection(dayList.map(function(offset) {
    var d = startDate.advance(offset, 'day');
    var dEnd = d.advance(1, 'day');
    var dailySum = era5.filterDate(d, dEnd).reduce(ee.Reducer.sum());
    var meanPrecip = dailySum.reduceRegion({
      reducer: ee.Reducer.mean(),
      geometry: aoi,
      scale: 11132,
      bestEffort: true
    });
    var val = ee.Number(ee.Algorithms.If(
      meanPrecip.contains('total_precipitation_sum'),
      ee.Number(meanPrecip.get('total_precipitation_sum')).multiply(1000),
      0
    ));
    return ee.Feature(null, {
      mean_precip: val,
      'system:time_start': d.millis()
    });
  }));
}

// Add previous-day precipitation using a temporal join
function addPrevPrecip(collection) {
  var sorted = collection.sort('system:time_start');
  var previous = sorted.map(function(el) {
    var originalDate = ee.Date(el.get('system:time_start'));
    var shiftedDate = originalDate.advance(-1, 'day');
    return el.set('system:time_start', shiftedDate.millis());
  });
  var join = ee.Join.saveFirst('prev');
  var filter = ee.Filter.equals({
    leftField: 'system:time_start',
    rightField: 'system:time_start'
  });
  var joined = join.apply(sorted, previous, filter);
  return joined.map(function(el) {
    var prevEl = ee.Feature(el.get('prev'));
    var prevPrecip = ee.Algorithms.If(prevEl, prevEl.get('mean_precip'), 0);
    return el.set('prev_precip', prevPrecip);
  });
}

var precipWithPrev = addPrevPrecip(precipCol);

// Identify dry days in summer months
var dryDays = precipWithPrev
  .filter(ee.Filter.lt('mean_precip', PRECIP_THRESHOLD))
  .filter(ee.Filter.lt('prev_precip', PRECIP_THRESHOLD_PREV))
  .filter(ee.Filter.calendarRange(SUMMER_START_MONTH, SUMMER_END_MONTH, 'month'));

// Build an OR-chained date filter for satellite collections
var dryDateFilters = dryDays.toList(dryDays.size()).map(function(el) {
  var date = ee.Date(ee.Feature(el).get('system:time_start'));
  return ee.Filter.date(date, date.advance(1, 'day'));
});

var dryFilter = ee.Algorithms.If(
  dryDateFilters.size().eq(0),
  ee.Filter.eq('system:index', 'non_existent'),
  dryDateFilters.slice(1).iterate(function(f, combined) {
    return ee.Filter.or(combined, f);
  }, dryDateFilters.get(0))
);


// ======================== SECTION 5: CLOUD MASKING FUNCTIONS ==================

/**
 * Landsat 8/9 cloud mask using QA_PIXEL band.
 * Bit flags used:
 *   Bit 0: Fill
 *   Bit 1: Dilated Cloud
 *   Bit 2: Cirrus (high confidence)
 *   Bit 3: Cloud
 *   Bit 4: Cloud Shadow
 *   Bit 5: Snow
 *   Bits 8-9: Cloud Confidence (< Medium)
 *   Bits 10-11: Cloud Shadow Confidence (< Medium)
 *   Bits 12-13: Snow/Ice Confidence (< Medium)
 *   Bits 14-15: Cirrus Confidence (< Medium)
 * Optionally buffers cloud cores by CLOUD_BUFFER_M meters.
 */
function maskL2Clouds(img) {
  var qa = img.select('QA_PIXEL');

  var mask = qa.bitwiseAnd(1 << 0).eq(0)      // Bit 0: fill
    .and(qa.bitwiseAnd(1 << 1).eq(0))          // Bit 1: dilated cloud
    .and(qa.bitwiseAnd(1 << 2).eq(0))          // Bit 2: cirrus
    .and(qa.bitwiseAnd(1 << 3).eq(0))          // Bit 3: cloud
    .and(qa.bitwiseAnd(1 << 4).eq(0))          // Bit 4: cloud shadow
    .and(qa.bitwiseAnd(1 << 5).eq(0))          // Bit 5: snow
    .and(qa.bitwiseAnd(3 << 8).lt(2 << 8))     // Bits 8-9: cloud conf < medium
    .and(qa.bitwiseAnd(3 << 10).lt(2 << 10))   // Bits 10-11: shadow conf < medium
    .and(qa.bitwiseAnd(3 << 12).lt(2 << 12))   // Bits 12-13: snow conf < medium
    .and(qa.bitwiseAnd(3 << 14).lt(2 << 14));  // Bits 14-15: cirrus conf < medium

  var result = img.updateMask(mask);

  // Optional cloud buffer
  if (CLOUD_BUFFER_M > 0) {
    var cloudCore = qa.bitwiseAnd(1 << 1).neq(0)
      .or(qa.bitwiseAnd(1 << 3).neq(0))
      .or(qa.bitwiseAnd(1 << 4).neq(0))
      .or(qa.bitwiseAnd(1 << 2).neq(0));
    cloudCore = cloudCore
      .reproject({crs: PROJ_CRS, scale: 30})
      .focal_max({
        kernel: ee.Kernel.circle({radius: CLOUD_BUFFER_M, units: 'meters'}),
        iterations: 1
      });
    var safeFromClouds = cloudCore.unmask(0).not();
    result = result.updateMask(safeFromClouds);
  }
  return result;
}

/**
 * Sentinel-2 cloud mask using Scene Classification Layer (SCL).
 * Masked classes:
 *   3: Cloud shadow
 *   8: Cloud medium probability
 *   9: Cloud high probability
 *   10: Cirrus
 * Optionally buffers cloud cores by CLOUD_BUFFER_M meters.
 */
function maskS2Clouds(img) {
  var scl = img.select('SCL');
  var cloudCore = scl.eq(3).or(scl.eq(8)).or(scl.eq(9)).or(scl.eq(10));

  var result = img;
  if (CLOUD_BUFFER_M > 0) {
    var nearCloud = cloudCore
      .reproject({crs: PROJ_CRS, scale: 10})
      .focal_max({
        kernel: ee.Kernel.circle({radius: CLOUD_BUFFER_M, units: 'meters'}),
        iterations: 1
      });
    var safeFromClouds = nearCloud.unmask(0).not();
    result = img.updateMask(safeFromClouds);
  } else {
    // Still mask cloud pixels themselves even when buffer = 0
    result = img.updateMask(cloudCore.not());
  }
  return result;
}


// ======================== SECTION 6: LST RETRIEVAL ===========================

/** Convert Landsat Collection 2 Level 2 ST_B10 to LST in Celsius */
function addLSTc(img) {
  var lstK = img.select('ST_B10').multiply(0.00341802).add(149.0);
  var lstC = lstK.subtract(273.15).rename('LST_C');
  return img.addBands(lstC);
}

/**
 * Custom LST using NDVI-based emissivity estimation.
 * Follows two-endmember mixing model:
 *   emissivity_vegetation = 0.982 (Rajan et al., 2022; https://doi.org/10.1007/s10661-022-09796-x)
 *   emissivity_soil = 0.971 (Rajan et al., 2022; https://doi.org/10.1007/s10661-022-09796-x)
 *   emissivity_water = 0.99 (Deng & Wu, 2013; https://doi.org/10.1016/j.rse.2012.12.020)
 * Proportional vegetation cover (Pv) derived from NDVI percentile normalization.
 */
function addLocalEmissivity(img) {
  var sr = img.select(['SR_B4', 'SR_B5']).multiply(0.0000275).add(-0.2);
  var ndvi = sr.normalizedDifference(['SR_B5', 'SR_B4']).rename('NDVI');

  var p = ndvi.reduceRegion({
    reducer: ee.Reducer.percentile([2, 98]),
    geometry: aoi, scale: 30, bestEffort: true, maxPixels: 1e13
  });
  var ndviMin = ee.Number(p.get('NDVI_p2'));
  var ndviMax = ee.Number(p.get('NDVI_p98'));
  var Pv = ndvi.unitScale(ndviMin, ndviMax).clamp(0, 1).pow(2).rename('PV');

  var eps_v = 0.982, eps_s = 0.971;
  var emiss = Pv.multiply(eps_v)
    .add(ee.Image(1).subtract(Pv).multiply(eps_s))
    .rename('EMISSIVITY');

  // Water bodies from QA_PIXEL bit 7
  var water = img.select('QA_PIXEL').bitwiseAnd(1 << 7).gt(0);
  emiss = emiss.where(water, 0.99);

  return img.addBands([ndvi, Pv, emiss]);
}

/** Compute custom LST from emissivity and atmospheric parameters */
function correctedLST(img) {
  var emiss = img.select('EMISSIVITY');
  var atran = img.select('ST_ATRAN').multiply(1e-4);
  var urad  = img.select('ST_URAD').multiply(1e-3);
  var drad  = img.select('ST_DRAD').multiply(1e-3);
  var trad  = img.select('ST_TRAD').multiply(1e-3);

  var ls = trad.subtract(urad)
    .subtract(atran.multiply(ee.Image(1).subtract(emiss)).multiply(drad))
    .divide(atran.multiply(emiss))
    .updateMask(trad.mask());
  ls = ls.updateMask(ls.gt(0));

  var spacecraft = ee.String(img.get('SPACECRAFT_ID'));
  var k1 = ee.Image.constant(ee.Number(ee.Algorithms.If(
    spacecraft.equals('LANDSAT_8'), 774.8853, 799.0284)));
  var k2 = ee.Image.constant(ee.Number(ee.Algorithms.If(
    spacecraft.equals('LANDSAT_8'), 1321.0789, 1329.2405)));

  var lstK = k2.divide(ee.Image(1).add(k1.divide(ls)).log());
  var lstC = lstK.subtract(273.15).rename('LST_C_CORR');
  return img.addBands(lstC);
}


// ======================== SECTION 7: PREDICTOR VARIABLES ======================
// Spectral indices from Sentinel-2 and SAR features from Sentinel-1.

/**
 * Compute spectral indices from Sentinel-2 bands.
 * 
 * When INDEX_STRATEGY = 'NATIVE_20M':
 *   SWIR-based indices (NDBI, BSI, MNDWI, ALBEDO) are computed at 20 m native
 *   resolution by aggregating 10 m bands (B2, B3, B4, B8) to match B11/B12,
 *   then bilinear-resampled to 10 m. This avoids mixed-pixel artifacts caused
 *   by nearest-neighbor resampling of 20 m SWIR bands to the 10 m grid.
 *
 * When INDEX_STRATEGY = 'BILINEAR_BAND':
 *   B11/B12 are bilinear-resampled to 10 m before computing indices. Preserves
 *   10 m detail but may produce edge artifacts from resolution mismatch.
 *
 * When INDEX_STRATEGY = 'NEAREST':
 *   All indices are computed at 10 m directly (GEE default nearest-neighbor
 *   resampling for B11/B12). Fastest but may produce blocky edge artifacts.
 *
 * NDVI is always computed at 10 m since both B8 and B4 are native 10 m.
 */
function addCovariates(img) {
  var b2  = img.select('B2').multiply(0.0001);
  var b3  = img.select('B3').multiply(0.0001);
  var b4  = img.select('B4').multiply(0.0001);
  var b8  = img.select('B8').multiply(0.0001);
  var b11 = img.select('B11').multiply(0.0001);
  var b12 = img.select('B12').multiply(0.0001);

  // NDVI — both bands native 10 m, always computed directly
  var den  = b8.add(b4);
  var ndvi = b8.subtract(b4).divide(den).updateMask(den.gt(0)).rename('NDVI');

  var ndbi, bsi, mndwi, albedo;

  if (INDEX_STRATEGY === 'NATIVE_20M') {
    // ── Compute indices at 20 m, resample results to 10 m ──
    // Both 10 m and 20 m bands share the same pixel grid during computation,
    // eliminating resolution mismatch artifacts at land-water boundaries.
    var proj10 = ee.Projection(PROJ_CRS).atScale(10);
    var proj20 = ee.Projection(PROJ_CRS).atScale(20);

    // Aggregate 10 m bands to 20 m grid
    var b2_20 = b2.setDefaultProjection(proj10)
                  .reduceResolution({reducer: ee.Reducer.mean(), maxPixels: 4})
                  .reproject({crs: proj20});
    var b3_20 = b3.setDefaultProjection(proj10)
                  .reduceResolution({reducer: ee.Reducer.mean(), maxPixels: 4})
                  .reproject({crs: proj20});
    var b4_20 = b4.setDefaultProjection(proj10)
                  .reduceResolution({reducer: ee.Reducer.mean(), maxPixels: 4})
                  .reproject({crs: proj20});
    var b8_20 = b8.setDefaultProjection(proj10)
                  .reduceResolution({reducer: ee.Reducer.mean(), maxPixels: 4})
                  .reproject({crs: proj20});

    // Indices computed at 20 m, then bilinear resampled to 10 m
    var ndbi_20 = b11.subtract(b8_20).divide(b11.add(b8_20));
    ndbi = ndbi_20.resample('bilinear').rename('NDBI');

    var bsi_num_20 = b11.add(b4_20).subtract(b8_20).subtract(b2_20);
    var bsi_den_20 = b11.add(b4_20).add(b8_20).add(b2_20);
    bsi = bsi_num_20.divide(bsi_den_20).resample('bilinear').rename('BSI');

    var mndwi_20 = b3_20.subtract(b11).divide(b3_20.add(b11));
    mndwi = mndwi_20.resample('bilinear').rename('MNDWI');

    var albedo_20 = b2_20.multiply(0.2266)
      .add(b3_20.multiply(0.1236))
      .add(b4_20.multiply(0.1573))
      .add(b8_20.multiply(0.3417))
      .add(b11.multiply(0.1170))
      .add(b12.multiply(0.0338));
    albedo = albedo_20.resample('bilinear').rename('ALBEDO');

  } else if (INDEX_STRATEGY === 'BILINEAR_BAND') {
    // ── Resample B11/B12 to 10 m first, then compute indices ──
    // Preserves full 10 m detail from visible/NIR bands, but resolution
    // mismatch between sharp 10 m and smooth 20 m can cause edge artifacts.
    var proj20b = ee.Projection(PROJ_CRS).atScale(20);
    var b11r = img.select('B11').setDefaultProjection(proj20b)
                  .resample('bilinear').multiply(0.0001);
    var b12r = img.select('B12').setDefaultProjection(proj20b)
                  .resample('bilinear').multiply(0.0001);

    ndbi = b11r.subtract(b8).divide(b11r.add(b8)).rename('NDBI');
    bsi = b11r.add(b4).subtract(b8).subtract(b2)
      .divide(b11r.add(b4).add(b8).add(b2)).rename('BSI');
    mndwi = b3.subtract(b11r).divide(b3.add(b11r)).rename('MNDWI');
    albedo = b2.multiply(0.2266).add(b3.multiply(0.1236))
      .add(b4.multiply(0.1573)).add(b8.multiply(0.3417))
      .add(b11r.multiply(0.1170)).add(b12r.multiply(0.0338))
      .rename('ALBEDO');

  } else {
    // ── NEAREST: default GEE behavior ──
    // Fastest. B11/B12 implicitly nearest-neighbor resampled to 10 m.
    ndbi = b11.subtract(b8).divide(b11.add(b8)).rename('NDBI');
    bsi = b11.add(b4).subtract(b8).subtract(b2)
      .divide(b11.add(b4).add(b8).add(b2)).rename('BSI');
    mndwi = b3.subtract(b11).divide(b3.add(b11)).rename('MNDWI');
    albedo = b2.multiply(0.2266).add(b3.multiply(0.1236))
      .add(b4.multiply(0.1573)).add(b8.multiply(0.3417))
      .add(b11.multiply(0.1170)).add(b12.multiply(0.0338))
      .rename('ALBEDO');
  }

  return img.addBands([ndvi, ndbi, bsi, mndwi, albedo]);
}

// List of covariate band names used in model training/prediction
var COVARIATES_FULL = ['NDVI', 'NDBI', 'BSI', 'MNDWI', 'ALBEDO', 'VV', 'VH', 'VV_VH_RATIO'];
var COVARIATES_OPTICAL = ['NDVI', 'NDBI', 'BSI', 'MNDWI', 'ALBEDO'];
var COVARIATES = ABLATION_OPTICAL_ONLY ? COVARIATES_OPTICAL : COVARIATES_FULL;

// ======================== SECTION 8: SATELLITE COLLECTIONS ===================

// Landsat 8 + 9 Collection 2 Level 2, filtered to summer dry days
var ls = ee.ImageCollection('LANDSAT/LC08/C02/T1_L2')
  .merge(ee.ImageCollection('LANDSAT/LC09/C02/T1_L2'))
  .filterBounds(aoi)
  .filter(ee.Filter.calendarRange(SUMMER_START_MONTH, SUMMER_END_MONTH, 'month'))
  .filter(ee.Filter.lt('CLOUD_COVER', CLOUD_COVER_MAX))
  .map(maskL2Clouds)
  .map(addLSTc)
  .filter(dryFilter);

// Sentinel-2 Level-2A, filtered to summer dry days
var sentinel = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
  .filter(ee.Filter.calendarRange(SUMMER_START_MONTH, SUMMER_END_MONTH, 'month'))
  .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', CLOUD_COVER_MAX))
  .filterBounds(aoi)
  .select(['B2', 'B3', 'B4', 'B8', 'B11', 'B12', 'SCL'])
  .map(function(img) {
    return maskS2Clouds(img).select(['B2', 'B3', 'B4', 'B8', 'B11', 'B12']);
  })
  .filter(dryFilter);

// Sentinel-1 GRD IW, filtered to summer dry days
var sentinel1 = ee.ImageCollection('COPERNICUS/S1_GRD')
  .filter(ee.Filter.calendarRange(SUMMER_START_MONTH, SUMMER_END_MONTH, 'month'))
  .filter(ee.Filter.eq('instrumentMode', 'IW'))
  .filterBounds(aoi)
  .select(['VV', 'VH'])
  .filter(dryFilter);

// =============================================================================
// >>>  SECTION 9: CLASSIFIER BUILDER & GRID SEARCH  <
// =============================================================================

/**
 * Build a classifier for the given algorithm and parameter object.
 * All classifiers are set to REGRESSION output mode.
 *
 * @param {string} algorithm - One of 'GBT', 'RF', 'SVM', 'CART'
 * @param {Object} params    - Algorithm-specific hyperparameters (optional overrides)
 * @returns {ee.Classifier}  - Untrained classifier in REGRESSION mode
 */
function buildClassifier(algorithm, params) {
  params = params || {};
  var clf;

  if (algorithm === 'GBT') {
    clf = ee.Classifier.smileGradientTreeBoost({
      numberOfTrees: params.numberOfTrees  || GBT_NUMBER_OF_TREES,
      shrinkage:     params.shrinkage      || GBT_SHRINKAGE,
      samplingRate:  params.samplingRate    || GBT_SAMPLING_RATE,
      maxNodes:      params.maxNodes        || GBT_MAX_NODES,
      loss:          params.loss            || GBT_LOSS,
      seed:          SEED
    });

  } else if (algorithm === 'RF') {
    var rfArgs = {
      numberOfTrees:    params.numberOfTrees    || RF_NUMBER_OF_TREES,
      minLeafPopulation: params.minLeafPopulation || RF_MIN_LEAF_POPULATION,
      bagFraction:      params.bagFraction      || RF_BAG_FRACTION,
      seed:             SEED
    };
    // variablesPerSplit and maxNodes: only include if non-null
    var vps = (params.variablesPerSplit !== undefined)
      ? params.variablesPerSplit : RF_VARIABLES_PER_SPLIT;
    if (vps !== null) { rfArgs.variablesPerSplit = vps; }

    var rfMaxN = (params.maxNodes !== undefined) ? params.maxNodes : RF_MAX_NODES;
    if (rfMaxN !== null) { rfArgs.maxNodes = rfMaxN; }

    clf = ee.Classifier.smileRandomForest(rfArgs);

  } else if (algorithm === 'SVM') {
    clf = ee.Classifier.libsvm({
      svmType:    'EPSILON_SVR',
      kernelType: params.kernelType || SVM_KERNEL_TYPE,
      cost:       params.cost       || SVM_COST,
      gamma:      params.gamma      || SVM_GAMMA
    });

  } else if (algorithm === 'CART') {
    var cartArgs = {
      minLeafPopulation: params.minLeafPopulation || CART_MIN_LEAF_POPULATION
    };
    var cartMaxN = (params.maxNodes !== undefined) ? params.maxNodes : CART_MAX_NODES;
    if (cartMaxN !== null) { cartArgs.maxNodes = cartMaxN; }

    clf = ee.Classifier.smileCart(cartArgs);

  } else {
    throw new Error('Unknown algorithm: ' + algorithm +
      '. Choose GBT, RF, SVM, or CART.');
  }

  return clf.setOutputMode('REGRESSION');
}


/**
 * Evaluate a trained classifier on a test set.
 * Returns RMSE, MAE, and R² as ee.Number objects.
 */
function evaluateModel(trainedClassifier, testData) {
  var predicted = testData.classify(trainedClassifier, 'predicted');

  var withSqDiff = predicted.map(function(f) {
    var sqDiff = ee.Number(f.get('LST_C'))
      .subtract(ee.Number(f.get('predicted'))).pow(2);
    return f.set('sq_diff', sqDiff);
  });
  var rmse = ee.Number(withSqDiff.reduceColumns({
    reducer: ee.Reducer.mean(), selectors: ['sq_diff']
  }).get('mean')).sqrt();

  var withAbsDiff = predicted.map(function(f) {
    var absDiff = ee.Number(f.get('LST_C'))
      .subtract(ee.Number(f.get('predicted'))).abs();
    return f.set('abs_diff', absDiff);
  });
  var mae = ee.Number(withAbsDiff.reduceColumns({
    reducer: ee.Reducer.mean(), selectors: ['abs_diff']
  }).get('mean'));

  var meanLST = testData.reduceColumns(ee.Reducer.mean(), ['LST_C']).get('mean');
  var withSSTot = testData.map(function(f) {
    return f.set('ss_tot', ee.Number(f.get('LST_C'))
      .subtract(ee.Number(meanLST)).pow(2));
  });
  var ssRes = withSqDiff.reduceColumns({
    reducer: ee.Reducer.sum(), selectors: ['sq_diff']
  }).get('sum');
  var ssTot = withSSTot.reduceColumns({
    reducer: ee.Reducer.sum(), selectors: ['ss_tot']
  }).get('sum');
  var r2 = ee.Number(1).subtract(ee.Number(ssRes).divide(ee.Number(ssTot)));

  return {rmse: rmse, mae: mae, r2: r2};
}


/**
 * Run grid search for the selected algorithm.
 * Builds a parameter grid, trains one model per combination on trainData,
 * evaluates on testData, and prints results to the console.
 *
 * @param {string} algorithm
 * @param {ee.FeatureCollection} trainData
 * @param {ee.FeatureCollection} testData
 * @param {number} year
 */
function runGridSearch(algorithm, trainData, testData, year) {
  var grid = [];  // Array of {params: {...}, label: '...'}
  var paramNames = [];

  if (algorithm === 'GBT') {
    paramNames = ['nTrees', 'shrinkage', 'maxNodes'];
    GBT_GRID_NUM_TREES.forEach(function(nt) {
      GBT_GRID_SHRINKAGE.forEach(function(sh) {
        if (GRID_DIMENSIONS >= 3) {
          GBT_GRID_MAX_NODES.forEach(function(mn) {
            grid.push({
              params: {numberOfTrees: nt, shrinkage: sh, maxNodes: mn},
              label: 'nT=' + nt + ' sh=' + sh + ' mN=' + mn
            });
          });
        } else {
          grid.push({
            params: {numberOfTrees: nt, shrinkage: sh},
            label: 'nT=' + nt + ' sh=' + sh
          });
        }
      });
    });

  } else if (algorithm === 'RF') {
    paramNames = ['nTrees', 'varPerSplit', 'minLeafPop'];
    RF_GRID_NUM_TREES.forEach(function(nt) {
      RF_GRID_VARIABLES_PER_SPLIT.forEach(function(vps) {
        if (GRID_DIMENSIONS >= 3) {
          RF_GRID_MIN_LEAF_POP.forEach(function(mlp) {
            grid.push({
              params: {numberOfTrees: nt, variablesPerSplit: vps,
                       minLeafPopulation: mlp},
              label: 'nT=' + nt + ' vps=' + vps + ' mlp=' + mlp
            });
          });
        } else {
          grid.push({
            params: {numberOfTrees: nt, variablesPerSplit: vps},
            label: 'nT=' + nt + ' vps=' + vps
          });
        }
      });
    });

  } else if (algorithm === 'SVM') {
    paramNames = ['cost', 'gamma'];
    SVM_GRID_COST.forEach(function(c) {
      SVM_GRID_GAMMA.forEach(function(g) {
        grid.push({
          params: {cost: c, gamma: g},
          label: 'C=' + c + ' γ=' + g
        });
      });
    });

  } else if (algorithm === 'CART') {
    paramNames = ['maxNodes', 'minLeafPop'];
    CART_GRID_MAX_NODES.forEach(function(mn) {
      CART_GRID_MIN_LEAF_POP.forEach(function(mlp) {
        var actualMN = (mn === -1) ? null : mn;
        grid.push({
          params: {maxNodes: actualMN, minLeafPopulation: mlp},
          label: 'mN=' + (mn === -1 ? 'none' : mn) + ' mlp=' + mlp
        });
      });
    });
  }

  print('══════════════════════════════════════════════════');
  print('GRID SEARCH - ' + algorithm + ' - ' + year +
    ' - ' + grid.length + ' combinations');
  print('══════════════════════════════════════════════════');

  // Train and evaluate each combination
  var resultFeatures = [];
  grid.forEach(function(combo, idx) {
    var clf = buildClassifier(algorithm, combo.params);
    var trained = clf.train({
      features: trainData,
      classProperty: 'LST_C',
      inputProperties: COVARIATES
    });
    var metrics = evaluateModel(trained, testData);

    resultFeatures.push(ee.Feature(null, {
      combo_index: idx,
      label: combo.label,
      rmse: metrics.rmse,
      mae: metrics.mae,
      r2: metrics.r2
    }));
  });

  var resultFC = ee.FeatureCollection(resultFeatures);

  // Print sorted table (best RMSE first)
  print('Grid search results (sorted by RMSE):',
    resultFC.sort('rmse'));

  // Chart: RMSE per combination
  var labels = grid.map(function(g) { return g.label; });
  var rmseValues = resultFeatures.map(function(f) {
    return ee.Feature(f).get('rmse');
  });

  var chart = ui.Chart.feature.byFeature({
    features: resultFC.sort('combo_index'),
    xProperty: 'label',
    yProperties: ['rmse']
  }).setChartType('ColumnChart')
    .setOptions({
      title: 'Grid Search - ' + algorithm + ' - ' + year + ' - Test RMSE',
      hAxis: {title: 'Parameter Combination', slantedText: true, slantedTextAngle: 45},
      vAxis: {title: 'RMSE (°C)'},
      legend: {position: 'none'},
      colors: ['#1a73e8']
    });
  print(chart);

  // Also chart R² for comparison
  var chartR2 = ui.Chart.feature.byFeature({
    features: resultFC.sort('combo_index'),
    xProperty: 'label',
    yProperties: ['r2']
  }).setChartType('ColumnChart')
    .setOptions({
      title: 'Grid Search - ' + algorithm + ' - ' + year + ' - Test R²',
      hAxis: {title: 'Parameter Combination', slantedText: true, slantedTextAngle: 45},
      vAxis: {title: 'R²'},
      legend: {position: 'none'},
      colors: ['#34a853']
    });
  print(chartR2);
}


// =============================================================================
// >>>  SECTION 9b: SCENE INFORMATION HELPER  <
// =============================================================================
// Prints the number of scenes, acquisition dates, and times for each satellite
// collection used in a given year. Useful for reproducibility and reporting.

/**
 * Print scene metadata for a satellite image collection.
 * Shows total count, and a list of date + time for each scene.
 *
 * @param {ee.ImageCollection} collection - Filtered image collection
 * @param {string} sensorName            - Display name (e.g., 'Landsat 8')
 * @param {number} year                  - Year being processed
 */
function printSceneInfo(collection, sensorName, year) {
  var count = collection.size();
  print('────────────────────────────────────────');
  print('Scene info - ' + sensorName + ' - ' + year + ' - count:', count);

  // Extract date-time strings from each image.
  // Use ee.Algorithms.If to guard against empty collections, because
  // toList(0) throws an error in GEE (e.g., Landsat 9 before Sept 2021).
  var dateTimeList = ee.Algorithms.If(
    count.gt(0),
    collection.toList(count).map(function(img) {
      img = ee.Image(img);
      var dateTime = ee.Date(img.get('system:time_start'));
      return ee.String(dateTime.format('YYYY-MM-dd HH:mm:ss'));
    }),
    ee.List([])
  );
  print('Scene info - ' + sensorName + ' - ' + year + ' - dates:', dateTimeList);
}





// ======================== SECTION 10: MAIN PROCESSING LOOP ===================

function processSummerYear(year) {
  // --- 10.1: Landsat LST composites ---
  var lsColFiltered = ls.filter(ee.Filter.calendarRange(year, year, 'year'));
  
  // Standard LST (Collection 2 built-in product)
  var stdLsCol = lsColFiltered.map(function(img) {
    return img.select('LST_C').rename('LST_C_STD');
  });

  // Corrected LST (NDVI-emissivity method)
  var corrLsColRecalc = lsColFiltered.map(addLocalEmissivity).map(correctedLST);
  var corrLsCol = corrLsColRecalc.map(function(img) {
    return img.select('LST_C_CORR');
  });

  // Composites
  var count = stdLsCol.count();
  var compStd  = ee.Image(ee.Algorithms.If(USE_MEDIAN,
    stdLsCol.median(), stdLsCol.mean()));
  var compCorr = ee.Image(ee.Algorithms.If(USE_MEDIAN,
    corrLsCol.median(), corrLsCol.mean()));

  var qualityMask = count.gte(1);

  var stdLsComp = compStd.updateMask(qualityMask).set('year', year)
    .clip(clipRegion());
  var corrLsComp = compCorr.updateMask(qualityMask).set('year', year)
    .clip(clipRegion());

  // Add both LST composites to map
  Map.addLayer(stdLsComp,  VIS_LST, 'Standard LST 30 m - ' + year, false);
  Map.addLayer(corrLsComp, VIS_LST, 'Corrected LST 30 m - ' + year, false);

  // --- Exports: standard and corrected LST ---
  if (EXPORT_STD_LST_TO_DRIVE) {
    Export.image.toDrive({
      image: stdLsComp.select('LST_C_STD'),
      description: 'Standard_LST_Summer_' + year,
      folder: 'LST_Downscaling',
      fileNamePrefix: 'Standard_LST_Summer_' + year,
      region: clipRegion(), scale: 30, maxPixels: 1e13, crs: PROJ_CRS
    });
  }
  if (EXPORT_CORR_LST_TO_DRIVE) {
    Export.image.toDrive({
      image: corrLsComp.select('LST_C_CORR'),
      description: 'Corrected_LST_Summer_' + year,
      folder: 'LST_Downscaling',
      fileNamePrefix: 'Corrected_LST_Summer_' + year,
      region: clipRegion(), scale: 30, maxPixels: 1e13, crs: PROJ_CRS
    });
  }

  // --- 10.2: Scene information (counts, dates, times) ---
  if (PRINT_SCENE_INFO) {
    // Split Landsat into L8 and L9 for separate reporting
    var ls8Year = lsColFiltered
      .filter(ee.Filter.eq('SPACECRAFT_ID', 'LANDSAT_8'));
    var ls9Year = lsColFiltered
      .filter(ee.Filter.eq('SPACECRAFT_ID', 'LANDSAT_9'));

    var sentFiltered_info = sentinel
      .filter(ee.Filter.calendarRange(year, year, 'year'));

    print('========================================');
    print('SCENE INFORMATION - ' + year);
    print('========================================');
    printSceneInfo(ls8Year,           'Landsat 8',   year);
    printSceneInfo(ls9Year,           'Landsat 9',   year);
    printSceneInfo(sentFiltered_info, 'Sentinel-2',  year);
    if (!ABLATION_OPTICAL_ONLY) {
      var sent1Filtered_info = sentinel1
        .filter(ee.Filter.calendarRange(year, year, 'year'));
      printSceneInfo(sent1Filtered_info,'Sentinel-1',  year);
    }
  }

  // --- 10.3: Sentinel-2 (filtering by year) ---
  var sentFiltered = sentinel.filter(ee.Filter.calendarRange(year, year, 'year'));

  // --- 10.4: Sentinel-1 SAR composites (skipped in ablation mode) ---
  var vv, vh, vv_vh_ratio;
  if (!ABLATION_OPTICAL_ONLY) {
    var sent1Filtered = sentinel1.filter(ee.Filter.calendarRange(year, year, 'year'));

    var sortedSent1 = sent1Filtered.sort('system:time_start');
    var sent1List = sortedSent1.toList(sortedSent1.size());
    var subsampleIndices = ee.List.sequence(0, sortedSent1.size().subtract(1), S1_SUBSAMPLE_STEP);
    var subsampledSent1 = ee.ImageCollection(subsampleIndices.map(function(i) {
      return ee.Image(sent1List.get(i));
    }));

    vv = subsampledSent1.select('VV').mean().rename('VV');
    vh = subsampledSent1.select('VH').mean().rename('VH');
    vv_vh_ratio = vv.subtract(vh).rename('VV_VH_RATIO');
  }

  // --- 10.5: Downscaling ---
  var sentMean = sentFiltered.mean().clip(aoi);

  // Coarse LST for training
  var coarseLST = corrLsComp.select('LST_C_CORR')
    .rename('LST_C')
    .reproject({crs: PROJ_CRS, scale: 30})
    .reduceResolution({
      reducer: ee.Reducer.mean(), maxPixels: 1024, bestEffort: true
    })
    .reproject({crs: PROJ_CRS, scale: COARSE_SCALE});

  // Coarse covariates
  var coarseCov = addCovariates(sentMean);
  if (!ABLATION_OPTICAL_ONLY) {
    coarseCov = coarseCov.addBands([vv, vh, vv_vh_ratio]);
  }
  coarseCov = coarseCov
    .reproject({crs: PROJ_CRS, scale: FINE_SCALE})
    .reduceResolution({
      reducer: ee.Reducer.mean(), maxPixels: 1024, bestEffort: true
    })
    .reproject({crs: PROJ_CRS, scale: COARSE_SCALE})
    .updateMask(coarseLST.mask());

  // Settle the combined stack on the coarse grid before sampling.
  // This pins the projection so sample() reads computed pixels instead of
  // re-running the reduceResolution pyramid per tile (avoids memory error
  // at non-300 m coarse scales). Keeps full AOI extent and valid-pixel pool.
  var sampleStack = coarseLST.addBands(coarseCov)
    .reproject({crs: PROJ_CRS, scale: COARSE_SCALE});

  // Random sampling
  var fullData = sampleStack.sample({
    numPixels: NUM_PIXELS,
    region: aoi,
    scale: COARSE_SCALE,
    geometries: true,
    seed: SEED,
    tileScale: 16
  });

  // 70/30 split
  var fullDataWithRandom = fullData.randomColumn('random', SEED).limit(MAX_EVAL);
  var trainData = fullDataWithRandom.filter(ee.Filter.lt('random', 0.7));
  var testData  = fullDataWithRandom.filter(ee.Filter.gte('random', 0.7));
  
  if (PRINT_MODEL_STATS) {
    print('Sample sizes - ' + year + ' - train / test:', trainData.size(), testData.size());
  }
  if (SHOW_TRAIN_TEST_POINTS) {
    Map.addLayer(trainData, {color: 'blue'}, 'Train ' + year);
    Map.addLayer(testData,  {color: 'red'},  'Test ' + year);
  }

  // =========================================================================
  // >>>  10.6: GRID SEARCH (conditional)  <<<
  // =========================================================================
  if (GRID_SEARCH_ENABLED) {
    runGridSearch(ALGORITHM, trainData, testData, year);
    // When grid search is enabled, skip the rest of the processing
    // to save compute. Inspect results, update params, then re-run
    // with GRID_SEARCH_ENABLED = false.
    return;
  }

  // =========================================================================
  // >>>  10.7: MODEL TRAINING (uses buildClassifier for selected algorithm)  <<<
  // =========================================================================
  var classifier = buildClassifier(ALGORITHM).train({
    features: trainData,
    classProperty: 'LST_C',
    inputProperties: COVARIATES
  });

  if (PRINT_MODEL_STATS) {
    // Build a dictionary of current hyperparameters for console output
    var hpDict;
    if (ALGORITHM === 'GBT') {
      hpDict = ee.Dictionary({
        algorithm: ALGORITHM,
        numPixels: NUM_PIXELS,
        numberOfTrees: GBT_NUMBER_OF_TREES,
        shrinkage: GBT_SHRINKAGE,
        samplingRate: GBT_SAMPLING_RATE,
        maxNodes: GBT_MAX_NODES,
        loss: GBT_LOSS,
        seed: SEED
      });
    } else if (ALGORITHM === 'RF') {
      hpDict = ee.Dictionary({
        algorithm: ALGORITHM,
        numPixels: NUM_PIXELS,
        numberOfTrees: RF_NUMBER_OF_TREES,
        variablesPerSplit: RF_VARIABLES_PER_SPLIT || 'default',
        minLeafPopulation: RF_MIN_LEAF_POPULATION,
        bagFraction: RF_BAG_FRACTION,
        maxNodes: RF_MAX_NODES || 'unlimited',
        seed: SEED
      });
    } else if (ALGORITHM === 'SVM') {
      hpDict = ee.Dictionary({
        algorithm: ALGORITHM,
        numPixels: NUM_PIXELS,
        svmType: 'EPSILON_SVR',
        kernelType: SVM_KERNEL_TYPE,
        cost: SVM_COST,
        gamma: SVM_GAMMA
      });
    } else if (ALGORITHM === 'CART') {
      hpDict = ee.Dictionary({
        algorithm: ALGORITHM,
        numPixels: NUM_PIXELS,
        maxNodes: CART_MAX_NODES || 'unlimited',
        minLeafPopulation: CART_MIN_LEAF_POPULATION,
        seed: SEED
      });
    }
    print(ALGORITHM + ' hyperparameters - ' + year + ':', hpDict);
  }

  // Variable importance (tree-based only: GBT, RF, CART)
  if (PRINT_IMPORTANCE && ALGORITHM !== 'SVM') {
    print('Variable importance - ' + ALGORITHM + ' - ' + year + ':',
      classifier.explain().get('importance'));
  }

  // --- 10.8: Fine-resolution prediction ---
  var fineCov = addCovariates(sentMean);
  if (!ABLATION_OPTICAL_ONLY) {
    fineCov = fineCov.addBands([vv, vh, vv_vh_ratio]);
  }
  fineCov = fineCov.clip(clipRegion());

  var downscaledRaw = fineCov.classify(classifier, 'LST_C_DS');

  // --- 10.9: Residual correction ---
  var predictedCoarse = coarseCov.classify(classifier, 'LST_C_DS');
  var residualCoarse = coarseLST.subtract(predictedCoarse).rename('RESIDUAL');

  var residualFine = residualCoarse
    .resample('bilinear')
    .reproject({crs: PROJ_CRS, scale: FINE_SCALE})
    .clip(clipRegion());

  var downscaledLST = downscaledRaw.add(residualFine)
    .rename('LST_C_DS')
    .clip(clipRegion())
    .set('year', year)
    .set('algorithm', ALGORITHM)
    .set('ablation_optical_only', ABLATION_OPTICAL_ONLY);
  
  var ablationLabel = ABLATION_OPTICAL_ONLY ? ', Optical Only' : '';
  Map.addLayer(downscaledLST, VIS_LST,
    'Downscaled LST 10 m (' + ALGORITHM + ablationLabel + ') - ' + year, false);

  // --- 10.10: Exports ---
  if (EXPORT_DOWNSCALED_TO_DRIVE) {
    var ablationTag = ABLATION_OPTICAL_ONLY ? '_OpticalOnly' : '_Full';
    Export.image.toDrive({
      image: downscaledLST,
      description: 'Downscaled_LST_' + ALGORITHM + '_' + INDEX_STRATEGY + ablationTag + '_Summer_' + year,
      folder: 'LST_Downscaling',
      fileNamePrefix: 'Downscaled_LST_' + ALGORITHM + '_' + INDEX_STRATEGY + ablationTag + '_Summer_' + year,
      region: clipRegion(), scale: FINE_SCALE, maxPixels: 1e13, crs: PROJ_CRS
    });
  }

  // Export all predictor variables at fine resolution
  if (EXPORT_PREDICTORS_TO_DRIVE) {
    var predictorStack = fineCov.select(COVARIATES);
    COVARIATES.forEach(function(bandName) {
      Export.image.toDrive({
        image: predictorStack.select(bandName),
        description: bandName + '_Summer_' + year,
        folder: 'LST_Downscaling',
        fileNamePrefix: bandName + '_Summer_' + year,
        region: clipRegion(), scale: FINE_SCALE, maxPixels: 1e13, crs: PROJ_CRS
      });
    });
  }

  // --- 10.11: Accuracy assessment ---
  var metrics = evaluateModel(classifier, testData);

  // Train metrics (for overfitting check)
  var metricsTrain = evaluateModel(classifier, trainData);

  if (PRINT_MODEL_STATS) {
    print(ALGORITHM + ' test metrics - ' + year + ':', ee.Dictionary({
      rmse_test: metrics.rmse, mae_test: metrics.mae, r2_test: metrics.r2
    }));
    print(ALGORITHM + ' train RMSE - ' + year + ':', metricsTrain.rmse);
    print(ALGORITHM + ' ΔRMSE (test - train) - ' + year + ':',
      metrics.rmse.subtract(metricsTrain.rmse));
  }
}


// ======================== SECTION 11: RUN =====================================
if (GRID_SEARCH_ENABLED) {
  // Grid search runs on the GRID_SEARCH_YEAR only to save compute.
  // After finding best params, switch GRID_SEARCH_ENABLED = false to run all years.
  print('┌─────────────────────────────────────────────────┐');
  print('│  GRID SEARCH MODE - running for year ' + GRID_SEARCH_YEAR  + ' only  │');
  print('│  Algorithm: ' + ALGORITHM + '                                │');
  print('│  Set GRID_SEARCH_ENABLED = false after tuning   │');
  print('└─────────────────────────────────────────────────┘');
  processSummerYear(GRID_SEARCH_YEAR);
} else {
  print('Algorithm: ' + ALGORITHM);
  print('Ablation mode: ' + (ABLATION_OPTICAL_ONLY ? 'Optical only (no SAR)' : 'Full (optical + SAR)'));
  YEARS.forEach(function(y) {
    processSummerYear(y);
  });
}

print('ℹ️ If "No valid training data were found" appears for certain years,');
print('  those years contained too few clear Landsat pixels after cloud and dry-day filtering.');
print('  → First, try increasing PRECIP_THRESHOLD_PREV, then PRECIP_THRESHOLD or CLOUD_COVER_MAX.');
print('  Additionally, visual verification of Landsat LST 30 m coverage in each year');
print('  is recommended to ensure the study area is fully covered.');


// ======================== SECTION 12: MAP LEGEND =============================

function addLSTLegend() {
  var legend = ui.Panel({
    style: {position: 'bottom-left', padding: '8px 15px'}
  });

  legend.add(ui.Label({
    value: 'LST (°C) - ' + ALGORITHM,
    style: {fontWeight: 'bold', fontSize: '16px', margin: '0 0 4px 0'}
  }));

  // Gradient bar
  var lon = ee.Image.pixelLonLat().select('longitude');
  var gradient = lon.multiply((VIS_LST.max - VIS_LST.min) / 1)
    .add(VIS_LST.min).rename('LST');

  var gradientThumb = ui.Thumbnail({
    image: gradient.visualize({
      min: VIS_LST.min, max: VIS_LST.max, palette: VIS_LST.palette
    }),
    params: {
      dimensions: '200x15',
      region: ee.Geometry.Rectangle([0, 0, 1, 0.1]),
      format: 'png'
    }
  });
  legend.add(gradientThumb);

  // Min/max labels
  var labels = ui.Panel({
    widgets: [
      ui.Label(VIS_LST.min + '', {margin: '0', fontSize: '11px'}),
      ui.Label(((VIS_LST.min + VIS_LST.max) / 2) + '',
        {margin: '0 0 0 70px', fontSize: '11px'}),
      ui.Label(VIS_LST.max + '', {margin: '0 0 0 70px', fontSize: '11px'})
    ],
    layout: ui.Panel.Layout.flow('horizontal')
  });
  legend.add(labels);
  Map.add(legend);
}

addLSTLegend();
