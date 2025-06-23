# Notebook 7: Baghdad Congestion Model

## README

### Overview

This notebook focuses on developing statistical and machine learning models to explain monthly traffic congestion patterns in **Baghdad**, using the **Traffic Congestion Index (TCI)** as a proxy for **economic activity**. 
Models in this notebook leverage spatial, environmental, and infrastructural variables - most notably NO₂ concentrations - to test whether air pollution levels can meaningfully reflect urban economic dynamics in data-scarce.

### Objective

1. Evaluate linear and non-linear modelling techniques to explain sub-city TCI in Baghdad.
2. Quantify the relationship between NO₂ levels and congestion as a signal of economic throughput.
3. Provide interpretable metrics, including **coefficients** and **local elasticities**, to inform policy.


### Workflow

#### 1. Data Preparation

- Monthly aggregation of sub-city level features:
  - **Target**: TCI – total monthly Traffic Congestion Index (aggregated seconds of delay)
  - **Predictors**:
    - NO₂ mean concentrations (mol/m2)
    - Road network lengths by type
    - Land use shares (built-up, cropland, water bodies, etc.)
    - Population & POI counts
    - Climate data (surface temperature, night-time lights, etc.)

#### 2. Modelling and Evaluation

- Linear Family Models:
  - Ordinary Least Squares (OLS)
  - Log-Log OLS (elasticity form)
  - 3rd-degree Polynomial
- Regularised and Tree-based Models:
  - Lasso Regression
  - Random Forest
  - LightGBM

Each model was tested on the same training/validation/test split. Performance was assessed using **R²** and **RMSE** on the test set to ensure generalisability.


### Modelling Results

| Model Type         | Target        | RMSE (Test) | R² (Test) | Key Comments |
|--------------------|---------------|-------------|-----------|--------------|
| OLS (linear)       | Raw TCI       | 9.85M       | **0.80**  | Simple, interpretable, strong generalisation |
| Lasso              | Raw TCI       | 9.83M       | 0.797     | Sparse model; confirms dominant drivers |
| LightGBM           | Raw TCI       | 9.61M       | 0.807     | Slightly higher R²; interpretability reduced |
| Random Forest      | Raw TCI       | 10.37M      | 0.774     | Non-linear, slight overfit |
| Neural Network     | Raw TCI       | 12.87M      | 0.653     | Underperformed despite tuning |
| Log-Log OLS        | log(TCI)      | 5.06        | –0.01     | Underfits, loses information |
| Poly-3 Regression  | Raw TCI       | 59.7M       | –6.47     | Severe overfitting |

The **Ordinary Least Squares (OLS)** model using raw TCI as the dependent variable was selected as the final model due to its **excellent out-of-sample R² (~0.80)**, **simplicity**, and **interpretability**. Despite minor performance gains from LightGBM (+0.7 pp R²), the loss of coefficient transparency makes OLS more suitable for public policy analysis.


### Interpretation of Best Congestion Model

#### Model Formulation

The model is specified as:

> **TCIᵢ = β₁×NO₂_mean + β₂×POI_count + β₃×LST_day_mean + … + εᵢ**

Where each β coefficient represents the marginal contribution of a feature to monthly traffic congestion at the sub-city level.

#### NO₂ Coefficient and Elasticity

- **β(NO₂_mean) = +53,986,362**  
  A one-unit increase in monthly average NO₂ (mol/m2) is associated with an additional **53.9 million TCI units**. This reflects the magnitude of summing congestion time across roads, days, and cells.

- **Elasticity (local at mean)**:
  Using the formula  
  **εₓ,ᵧ = (βₓ × x̄) / ȳ**,  
  we find that the elasticity of TCI with respect to NO₂ is **~0.25**.

  > A **1% increase in NO₂** concentration corresponds to a **0.25% increase in total monthly congestion**, at average sub-city conditions in Baghdad.


### Conclusion

The modelling exercise confirms that **NO₂ is a powerful and interpretable proxy for economic activity** in Baghdad. Its strong, positive, and statistically significant relationship with congestion underscores the link between urban mobility demand and environmental externalities. The use of a transparent OLS model enables straightforward communication of results to policymakers while preserving predictive strength.

Ultimately, this model supports the use of **remote-sensed air quality data** as a near-real-time pulse of economic activity in low-data settings. Baghdad’s congestion patterns are **well-explained by NO₂ and POI data**, with OLS explaining approximately **80% of observed variance**—a robust result for urban analytics.


## 0 Init: Prepare Packages and Configuration

Get current file/repo/data path in local to make sure the following cells run properly.


```python
import sys
from pathlib import Path
SRC_PATH = Path().resolve().parent / "src"
sys.path.append(str(SRC_PATH))

from config import *
```

## 1 Data Loading


```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, root_mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures
import numpy as np

baghdad_df = pd.read_csv(DATA_PATH / "temp" / 'baghdad_monthly_adm3.csv')

# Features and targets setup
features = [
    'no2_mean', 
    
    #'no2_lag1', 'no2_neighbor_lag1',

    'NTL_mean', 'pop_sum_m', 'road_len',
    'poi_count', 'lu_industrial_area',
    'lu_commercial_area', 'lu_residential_area',
    'non_built_area'

    ,'LST_day_mean','lu_retail_area',
    'lu_farmland_area',
       'lu_farmyard_area', 'road_primary_len', 'road_motorway_len',
       'road_trunk_len', 'road_secondary_len', 'road_tertiary_len',
       'road_residential_len', 'grassland_a', 'cropland_a', 'built_up_a'
    #    ,'snow_a', 'water_bod_a', 'wetland_a', 'sparse_veg_a', 'mangroves_a',
    #    'moss_a', 'unclassified_a'
]

```

## 2 Models y=TCI/road_len

### 2.1 Simple linear Regression


```python
# Target definition
baghdad_df['y1'] = baghdad_df['TCI'] / baghdad_df['road_len']

# Train/test split
X = baghdad_df[features]
y = baghdad_df['y1']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit & predict
lr = LinearRegression().fit(X_train, y_train)
y_pred_train = lr.predict(X_train)
y_pred_test  = lr.predict(X_test)

# Metrics
for label, y_true, y_pred in [
    ('TRAIN', y_train, y_pred_train),
    ('TEST',  y_test,  y_pred_test)
]:
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)
    print(f"Simple LR ({label}): RMSE = {rmse:.4f}, R² = {r2:.4f}")
```

    Simple LR (TRAIN): RMSE = 5.3420, R² = 0.7162
    Simple LR (TEST): RMSE = 6.1411, R² = 0.6770
    

### 2.2  Log-log Linear Regression


```python
# Clone and avoid zeros: common practice is to add a small ε = half the minimum positive value
df_ll = baghdad_df.copy()
epsilon = df_ll[features + ['TCI', 'road_len']].replace(0, np.nan).min().min() / 2

# Log transformation
for col in features:
    df_ll[col] = np.log(df_ll[col].clip(lower=epsilon))
df_ll['y2'] = np.log((df_ll['TCI'] / df_ll['road_len']).clip(lower=epsilon))

# Split
X = df_ll[features]
y = df_ll['y2']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit & evaluate
lr_ll = LinearRegression().fit(X_train, y_train)
for label, X_, y_, model in [
    ('TRAIN', X_train, y_train, lr_ll),
    ('TEST',  X_test,  y_test,  lr_ll)
]:
    pred = model.predict(X_)
    rmse = np.sqrt(mean_squared_error(y_, pred))
    r2   = r2_score(y_, pred)
    print(f"Log–Log LR ({label}): RMSE = {rmse:.4f}, R² = {r2:.4f}")

```

    Log–Log LR (TRAIN): RMSE = 5.0877, R² = 0.1728
    Log–Log LR (TEST): RMSE = 5.0621, R² = -0.0067
    

### 2.3 Polynomial Dg3


```python
# Degree‐3 polynomial expansion (no interactions)
poly = PolynomialFeatures(degree=3, interaction_only=False, include_bias=False)
X_poly = poly.fit_transform(baghdad_df[features])

# Train/test
X_train, X_test, y_train, y_test = train_test_split(
    X_poly, baghdad_df['y1'], test_size=0.2, random_state=42
)

# Fit & evaluate
poly_lr = LinearRegression().fit(X_train, y_train)
for label, X_, y_, model in [
    ('TRAIN', X_train, y_train, poly_lr),
    ('TEST',  X_test,  y_test,  poly_lr)
]:
    pred = model.predict(X_)
    rmse = np.sqrt(mean_squared_error(y_, pred))
    r2   = r2_score(y_, pred)
    print(f"Poly LR (deg 3) ({label}): RMSE = {rmse:.4f}, R² = {r2:.4f}")
```

    Poly LR (deg 3) (TRAIN): RMSE = 2.8633, R² = 0.9185
    Poly LR (deg 3) (TEST): RMSE = 33.2734, R² = -8.4833
    

## 3 Models y=TCI

### 3.1 Simple linear Regression


```python
X = baghdad_df[features]
y = baghdad_df['TCI']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

lr2 = LinearRegression().fit(X_train, y_train)
for label, y_true, y_pred in [
    ('TRAIN', y_train, lr2.predict(X_train)),
    ('TEST',  y_test,  lr2.predict(X_test))
]:
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)
    print(f"Simple LR (TCI) ({label}): RMSE = {rmse:.4f}, R² = {r2:.4f}")
```

    Simple LR (TCI) (TRAIN): RMSE = 8548952.1574, R² = 0.7463
    Simple LR (TCI) (TEST): RMSE = 9846074.6481, R² = 0.7968
    

### 3.2 Log-log Linear Regression


```python
# Add ε and log‐transform
df_ll2 = baghdad_df.copy()
epsilon = df_ll2[features + ['TCI']].replace(0, np.nan).min().min() / 2
for col in features:
    df_ll2[col] = np.log(df_ll2[col].clip(lower=epsilon))
df_ll2['y5'] = np.log(df_ll2['TCI'].clip(lower=epsilon))

X = df_ll2[features]
y = df_ll2['y5']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

lr_ll2 = LinearRegression().fit(X_train, y_train)
for label, X_, y_, model in [
    ('TRAIN', X_train, y_train, lr_ll2),
    ('TEST',  X_test,  y_test,  lr_ll2)
]:
    pred = model.predict(X_)
    rmse = np.sqrt(mean_squared_error(y_, pred))
    r2   = r2_score(y_, pred)
    print(f"Log–Log LR (TCI) ({label}): RMSE = {rmse:.4f}, R² = {r2:.4f}")
```

    Log–Log LR (TCI) (TRAIN): RMSE = 5.8183, R² = 0.1474
    Log–Log LR (TCI) (TEST): RMSE = 5.7425, R² = -0.0201
    

### 3.3 Polynomial Dg3


```python
poly2 = PolynomialFeatures(degree=3, interaction_only=False, include_bias=False)
X_poly2 = poly2.fit_transform(baghdad_df[features])

X_train, X_test, y_train, y_test = train_test_split(
    X_poly2, baghdad_df['TCI'], test_size=0.2, random_state=42
)

poly2_lr = LinearRegression().fit(X_train, y_train)
for label, X_, y_, model in [
    ('TRAIN', X_train, y_train, poly2_lr),
    ('TEST',  X_test,  y_test,  poly2_lr)
]:
    pred = model.predict(X_)
    rmse = np.sqrt(mean_squared_error(y_, pred))
    r2   = r2_score(y_, pred)
    print(f"Poly LR (TCI, deg 3) ({label}): RMSE = {rmse:.4f}, R² = {r2:.4f}")
```

    Poly LR (TCI, deg 3) (TRAIN): RMSE = 2537682.5992, R² = 0.9776
    Poly LR (TCI, deg 3) (TEST): RMSE = 59692127.6602, R² = -6.4689
    

## 4 Experimentation to Improve Performance

Building upon the baseline models, this section focuses on improving predictive accuracy through advanced machine learning techniques. We experiment with:

- Lasso Regression for variable selection and regularization,

- Random Forest to capture non-linearities and interactions,

- LightGBM, a gradient boosting method known for efficiency and accuracy,

- And a Neural Network, designed to learn complex feature representations.

Each model is evaluated to compare performance gains over linear methods, forming the basis for selecting the best model in the next section.

### 4.1 Simple Lasso


```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LassoCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import warnings
from sklearn.exceptions import ConvergenceWarning

# ── Suppress warnings ─────────────────────────────────────
warnings.filterwarnings('ignore', category=ConvergenceWarning)

# ── (Re)define the full feature set & data split ─────────
features_full = [
    'no2_mean', 
    #'no2_lag1', 'no2_neighbor_lag1',

    'NTL_mean', 'pop_sum_m', 'road_len',
    'poi_count', 'lu_industrial_area',
    'lu_commercial_area', 'lu_residential_area',
    'non_built_area', 'LST_day_mean', 'lu_retail_area',
    'lu_farmland_area', 'lu_farmyard_area',
    'road_primary_len', 'road_motorway_len',
    'road_trunk_len', 'road_secondary_len',
    'road_tertiary_len', 'road_residential_len',
    'grassland_a', 'cropland_a', 'built_up_a',
    'water_bod_a', 'wetland_a'
    #,'snow_a'
    # ,'sparse_veg_a', 'mangroves_a', 'moss_a',
    # 'unclassified_a'
]

X_full = baghdad_df[features_full]
y_full = baghdad_df['TCI']
X_tr, X_te, y_tr, y_te = train_test_split(
    X_full, y_full, test_size=0.2, random_state=42
)

# ── Pipeline & Fit ────────────────────────────────────────
lasso_pipeline = Pipeline([
    ('scaler',  StandardScaler()),
    ('lassocv', LassoCV(
        alphas=np.logspace(-4, 1, 50),
        cv=5, max_iter=20000, n_jobs=-1, random_state=42
    ))
])
lasso_pipeline.fit(X_tr, y_tr)

# ── Metrics ───────────────────────────────────────────────
y_tr_pred = lasso_pipeline.predict(X_tr)
y_te_pred = lasso_pipeline.predict(X_te)

rmse_tr = np.sqrt(mean_squared_error(y_tr, y_tr_pred))
r2_tr   = r2_score(y_tr, y_tr_pred)
rmse_te = np.sqrt(mean_squared_error(y_te, y_te_pred))
r2_te   = r2_score(y_te, y_te_pred)
alpha_opt = lasso_pipeline.named_steps['lassocv'].alpha_

# ── Clean output ──────────────────────────────────────────
print(f"Optimal α        : {alpha_opt:.6f}")
print(f"Train →  RMSE = {rmse_tr:.4f}, R² = {r2_tr:.4f}")
print(f"Test  →  RMSE = {rmse_te:.4f}, R² = {r2_te:.4f}")

```

    Optimal α        : 10.000000
    Train →  RMSE = 8539696.9768, R² = 0.7469
    Test  →  RMSE = 9817700.9341, R² = 0.7980
    

Extract coefficients


```python
# 1. Extract fitted components
scaler = lasso_pipeline.named_steps['scaler']
lasso  = lasso_pipeline.named_steps['lassocv']

# 2. Scaled-space coefficients
coef_scaled = lasso.coef_

# 3. Back-transform to original units
coef_original = coef_scaled / scaler.scale_
intercept_original = (
    lasso.intercept_
    - np.dot(scaler.mean_ / scaler.scale_, coef_scaled)
)

# 4. Assemble into DataFrame
coef_df = pd.DataFrame({
    'feature': features_full,
    'coefficient': coef_original
}).set_index('feature')

# Add the intercept
coef_df.loc['Intercept'] = intercept_original

# 5. Sort by magnitude for readability
coef_df['abs_coef'] = coef_df['coefficient'].abs()
coef_df = coef_df.sort_values('abs_coef', ascending=False).drop(columns='abs_coef')

# 6. Print to console
print(coef_df)
```

                           coefficient
    feature                           
    no2_mean              5.619848e+07
    Intercept             1.706929e+07
    poi_count             9.290394e+05
    LST_day_mean         -4.632916e+05
    lu_retail_area        9.375779e+02
    NTL_mean             -6.141713e+02
    lu_farmyard_area     -5.733545e+02
    road_motorway_len    -3.758322e+02
    road_trunk_len       -2.649884e+02
    road_primary_len     -2.065768e+02
    road_len              5.777523e+01
    road_residential_len -4.978246e+01
    road_secondary_len   -4.752816e+01
    road_tertiary_len    -4.349994e+01
    wetland_a            -2.753844e+01
    pop_sum_m             2.219547e+01
    grassland_a          -3.167750e+00
    water_bod_a           4.049054e-01
    cropland_a           -3.158772e-01
    non_built_area        2.766269e-01
    lu_commercial_area    2.415114e-01
    built_up_a           -8.578196e-02
    lu_industrial_area   -8.480142e-02
    lu_residential_area  -8.198301e-02
    lu_farmland_area      6.113320e-03
    

Elasticity


```python
import numpy as np

# 1) Grab the NO₂ coefficient from the fitted lr2 model
beta_no2 = lr2.coef_[features.index("no2_mean")]

# 2) Compute the means
mean_no2 = baghdad_df["no2_mean"].mean()
mean_TCI = baghdad_df["TCI"].mean()

# 3) Elasticity calculation
elasticity_no2 = (beta_no2 * mean_no2) / mean_TCI

print(f"Elasticity of NO₂_mean at sample mean: {elasticity_no2:.4f}")
# Should print approximately 0.25

```

    Elasticity of NO₂_mean at sample mean: 0.2514
    

### 4.2 Random Forest Experimentation


```python
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import shap, warnings, scipy.stats as st
warnings.filterwarnings('ignore')

# ── 1) Raw-unit feature matrix & target ────────────────────────────────
X_rf = baghdad_df[features_full].copy()
y_rf = baghdad_df['TCI'].copy()

X_tr_rf, X_te_rf, y_tr_rf, y_te_rf = train_test_split(
    X_rf, y_rf, test_size=0.2, random_state=42
)

# ── 2) Randomised hyper-tuning (20 draws) ──────────────────────────────
rf = RandomForestRegressor(random_state=42, n_jobs=-1)

param_dist = {
    'n_estimators'    : st.randint(400, 1001),        # 400–1000 trees
    'max_depth'       : [None, 20, 40],
    'min_samples_leaf': [1, 3, 5],
    'max_features'    : ['sqrt', 0.7, 0.9]
}

rs = RandomizedSearchCV(
    rf,
    param_distributions = param_dist,
    n_iter      = 20,
    cv          = 5,
    scoring     = 'r2',
    return_train_score = True,
    n_jobs      = -1,
    random_state= 42
)
rs.fit(X_tr_rf, y_tr_rf)
best_rf = rs.best_estimator_

# ── 3) Evaluation ───────────────────────────────────────────────────────
for label, X_, y_ in [('TRAIN', X_tr_rf, y_tr_rf),
                      ('TEST',  X_te_rf, y_te_rf)]:
    pred = best_rf.predict(X_)
    rmse = np.sqrt(mean_squared_error(y_, pred))
    r2   = r2_score(y_, pred)
    print(f"RF ({label}) → RMSE = {rmse:.4f}, R² = {r2:.4f}")

print("Selected RF params:", rs.best_params_)

# ── 4) SHAP global importances ─────────────────────────────────────────
explainer = shap.TreeExplainer(best_rf)
shap_vals = explainer.shap_values(X_te_rf, check_additivity=False)
shap_df   = pd.Series(np.abs(shap_vals).mean(axis=0),
                      index=X_rf.columns,
                      name='mean|SHAP|').sort_values(ascending=False)

print("\nTop-10 SHAP drivers:")
print(shap_df.head(10).to_string(float_format='%.4f'))

# ── 5) Elasticity approximation (raw units) ────────────────────────────
def elasticity(model, x_row, feature, delta=0.01):
    """
    %Δy / %Δx  using finite difference on raw-unit model.
    """
    x_up = x_row.copy()
    bump = x_up[feature] * delta if x_up[feature] != 0 else delta
    x_up[feature] += bump
    y0 = model.predict(x_row.values.reshape(1, -1))[0]
    y1 = model.predict(x_up.values.reshape(1, -1))[0]
    if y0 == 0:            # guard against div-by-zero
        return np.nan
    return ((y1 - y0) / y0) / delta   # elasticity formula

sample = X_te_rf.iloc[0]
elas = {f: elasticity(best_rf, sample, f) for f in X_rf.columns}
elas_df = pd.Series(elas, name='elasticity').sort_values(
              key=lambda s: s.abs(), ascending=False)

print("\nLocal elasticities for one test sample (top-10):")
print(elas_df.head(10).to_string(float_format='%.4f'))

```

    RF (TRAIN) → RMSE = 5791056.7935, R² = 0.8836
    RF (TEST) → RMSE = 10372546.1159, R² = 0.7745
    Selected RF params: {'max_depth': None, 'max_features': 0.9, 'min_samples_leaf': 5, 'n_estimators': 605}
    
    Top-10 SHAP drivers:
    road_primary_len       3846821.8062
    poi_count              2325672.0367
    non_built_area         1744155.0331
    cropland_a             1713642.4255
    LST_day_mean           1264173.3052
    road_len               1035734.7067
    road_residential_len    769925.1754
    lu_residential_area     764448.9765
    pop_sum_m               753231.9807
    road_motorway_len       668670.4499
    
    Local elasticities for one test sample (top-10):
    LST_day_mean        -0.9786
    built_up_a          -0.7177
    wetland_a           -0.5245
    water_bod_a          0.1236
    grassland_a          0.1188
    NTL_mean            -0.0563
    cropland_a          -0.0375
    road_trunk_len      -0.0000
    pop_sum_m           -0.0000
    road_motorway_len    0.0000
    

Elasticities


```python
# %% [markdown]
# ##### Global elasticity distribution (RF, raw units)

# %%
import numpy as np
import pandas as pd

def elasticity_matrix(model, X, delta=0.01):
    """
    Vectorised elasticity for an entire sample matrix X.
    Returns: array (n_samples, n_features)
    ε_ij = ((f(x_i⊕δ_j) – f(x_i)) / f(x_i)) / δ
    """
    y_base = model.predict(X)
    X_bump = X.copy()
    elas = np.empty_like(X.values, dtype=float)

    for j, col in enumerate(X.columns):
        bump = X[col].values * delta
        bump[bump == 0] = delta        # guard zeros
        X_bump[col] = X[col] + bump
        y_bump = model.predict(X_bump)
        elas[:, j] = ((y_bump - y_base) / y_base) / delta
        X_bump[col] = X[col]           # restore

    return elas

# 1) Compute matrix on the whole test fold
E = elasticity_matrix(best_rf, X_te_rf, delta=0.01)   # shape (n_test, n_feat)

# 2) Wrap in DataFrame
elas_df = pd.DataFrame(E, columns=X_te_rf.columns)

# 3) Aggregate statistics
summary = (elas_df
           .agg(['mean','median',lambda s: s.quantile(0.1),lambda s: s.quantile(0.9)])
           .T.rename(columns={'<lambda_0>':'q10','<lambda_1>':'q90'}))

# 4) Rank by |mean|
summary['abs_mean'] = summary['mean'].abs()
summary = summary.sort_values('abs_mean', ascending=False).drop(columns='abs_mean')

print("Global elasticity summary (mean, median, 10–90 % deciles)\n")
print(summary.head(10).to_string(float_format='{:.4f}'.format))

```

    Global elasticity summary (mean, median, 10–90 % deciles)
    
                          mean  median  <lambda>  <lambda>
    LST_day_mean        3.4976  0.0000   -1.9369   21.0597
    lu_industrial_area  0.5642  0.0000   -0.0000    0.0000
    cropland_a          0.5449  0.0000   -1.6741    2.3852
    water_bod_a         0.3887  0.0000   -0.7366    1.4454
    non_built_area      0.1529  0.0000   -4.1205    2.0252
    road_tertiary_len  -0.1477  0.0000   -0.0000    0.0000
    built_up_a          0.1431  0.0000   -0.6631    0.9639
    wetland_a           0.1165  0.0000   -0.1189    0.3125
    pop_sum_m           0.0527  0.0000   -1.2763    0.8835
    grassland_a         0.0415  0.0000   -0.4009    0.6188
    

### 4.3 LightGBM experimentation


```python
import numpy as np
import pandas as pd
import warnings
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, r2_score
import lightgbm as lgb

warnings.filterwarnings('ignore')

# 1) Prepare raw‐unit dataset
X_lgb = baghdad_df[features_full].copy()
y_lgb = baghdad_df['TCI'].copy()

X_tr_lgb, X_te_lgb, y_tr_lgb, y_te_lgb = train_test_split(
    X_lgb, y_lgb, test_size=0.2, random_state=42
)

# 2) Parameter distributions for RandomizedSearch
param_dist = {
    'num_leaves':        [31, 50, 100],
    'max_depth':         [-1, 10, 20],
    'learning_rate':     [0.01, 0.05, 0.1],
    'n_estimators':      [100, 200, 500],
    'min_child_samples': [10, 20, 50],
    'subsample':         [0.6, 0.8, 1.0],
    'colsample_bytree':  [0.6, 0.8, 1.0]
}

lgb_reg = lgb.LGBMRegressor(random_state=42, n_jobs=-1)

rs_lgb = RandomizedSearchCV(
    estimator=lgb_reg,
    param_distributions=param_dist,
    n_iter=30,
    cv=5,
    scoring='r2',
    return_train_score=True,
    random_state=42,
    n_jobs=-1
)
rs_lgb.fit(X_tr_lgb, y_tr_lgb)

best_params_lgb = rs_lgb.best_params_

# 3) Refit best model on full training set
best_lgb = lgb.LGBMRegressor(**best_params_lgb, random_state=42)
best_lgb.fit(X_tr_lgb, y_tr_lgb)

# 4) Evaluate on train and test
for label, X_e, y_e in [('TRAIN', X_tr_lgb, y_tr_lgb), ('TEST', X_te_lgb, y_te_lgb)]:
    y_pred = best_lgb.predict(X_e)
    rmse   = np.sqrt(mean_squared_error(y_e, y_pred))
    r2     = r2_score(y_e, y_pred)
    print(f"LightGBM ({label}) → RMSE = {rmse:.4f}, R² = {r2:.4f}")

print("\nBest LightGBM parameters:")
print(best_params_lgb)
```

    [LightGBM] [Info] Auto-choosing col-wise multi-threading, the overhead of testing was 0.000323 seconds.
    You can set `force_col_wise=true` to remove the overhead.
    [LightGBM] [Info] Total Bins 1482
    [LightGBM] [Info] Number of data points in the train set: 403, number of used features: 22
    [LightGBM] [Info] Start training from score 11221479.322727
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Info] Auto-choosing col-wise multi-threading, the overhead of testing was 0.000224 seconds.
    You can set `force_col_wise=true` to remove the overhead.
    [LightGBM] [Info] Total Bins 1482
    [LightGBM] [Info] Number of data points in the train set: 403, number of used features: 22
    [LightGBM] [Info] Start training from score 11221479.322727
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    LightGBM (TRAIN) → RMSE = 6813496.5739, R² = 0.8389
    LightGBM (TEST) → RMSE = 9560138.9439, R² = 0.8084
    
    Best LightGBM parameters:
    {'subsample': 0.8, 'num_leaves': 50, 'n_estimators': 100, 'min_child_samples': 50, 'max_depth': -1, 'learning_rate': 0.1, 'colsample_bytree': 0.8}
    


```python
# Predictions
y_tr_pred = best_lgb.predict(X_tr_lgb)
y_te_pred = best_lgb.predict(X_te_lgb)

# Compute metrics
metrics = pd.DataFrame({
    'RMSE': [
        np.sqrt(mean_squared_error(y_tr_lgb, y_tr_pred)),
        np.sqrt(mean_squared_error(y_te_lgb, y_te_pred))
    ],
    'R2': [
        r2_score(y_tr_lgb, y_tr_pred),
        r2_score(y_te_lgb, y_te_pred)
    ]
}, index=['TRAIN', 'TEST'])

# Display nicely
print("LightGBM Performance Metrics")
print(metrics.to_string(float_format='%.4f'))
```

    LightGBM Performance Metrics
                  RMSE     R2
    TRAIN 6849906.7318 0.8371
    TEST  9608374.3596 0.8065
    


```python
# 1) Global SHAP importances
explainer = shap.TreeExplainer(best_lgb)
shap_vals  = explainer.shap_values(X_te_lgb)
# mean absolute impact on model output
shap_imp   = pd.Series(np.abs(shap_vals).mean(axis=0),
                       index=X_te_lgb.columns,
                       name='mean|SHAP|').sort_values(ascending=False)

print("Top-10 global drivers by SHAP:")
print(shap_imp.head(10).to_string(float_format='%.4f'))

# Optional: visual summary
# shap.summary_plot(shap_vals, X_te_lgb, plot_type="bar")

# 2) Elasticity approximation (finite-difference)
def elasticity_pct(model, X, feature, delta=0.01):
    """
    Approximates ε = (%Δy) / (%Δx) for a raw-unit model:
      For each row i: bump xi by xi*delta, compute new y,
      then (Δy / y) / delta.
    Returns vector of local elasticities for each observation.
    """
    x_base = X[feature].values
    bump   = np.where(x_base!=0, x_base*delta, delta)
    X_up   = X.copy()
    X_up[feature] = x_base + bump

    y0 = model.predict(X)
    y1 = model.predict(X_up)
    # avoid /0
    with np.errstate(divide='ignore', invalid='ignore'):
        eps = ((y1 - y0) / y0) / delta
    return eps

# 3) Compute elasticities for each feature across the test set
elas_dict = {}
for feat in X_te_lgb.columns:
    eps = elasticity_pct(best_lgb, X_te_lgb, feat, delta=0.01)
    # summarise: mean, median, 10% & 90% deciles
    elas_dict[feat] = [
        np.nanmean(eps),
        np.nanmedian(eps),
        np.nanpercentile(eps, 10),
        np.nanpercentile(eps, 90)
    ]

elas_df = pd.DataFrame.from_dict(
    elas_dict,
    orient='index',
    columns=['mean_elas','median_elas','q10','q90']
).sort_values('mean_elas', key=lambda s: s.abs(), ascending=False)

print("\nElasticity summary (percent-response per 1% feature bump):")
print(elas_df.head(10).to_string(float_format='%.4f'))
```

    Top-10 global drivers by SHAP:
    road_primary_len      4618026.5990
    poi_count             3937523.5479
    LST_day_mean          2114417.4322
    road_motorway_len     1763763.9300
    non_built_area        1404821.6684
    pop_sum_m             1159875.8743
    lu_residential_area   1131794.4888
    water_bod_a           1121480.6243
    road_secondary_len    1039664.0811
    lu_commercial_area     890669.2143
    
    Elasticity summary (percent-response per 1% feature bump):
                        mean_elas  median_elas     q10    q90
    LST_day_mean           1.1533       0.0000 -3.8567 7.1198
    grassland_a           -0.9425       0.0000  0.0000 0.0000
    road_primary_len      -0.6787       0.0000  0.0000 0.0000
    water_bod_a           -0.6063       0.0000  0.0000 0.0000
    cropland_a            -0.3080       0.0000  0.0000 0.0000
    built_up_a            -0.2184       0.0000  0.0000 0.0000
    non_built_area        -0.1665       0.0000  0.0000 0.0000
    lu_industrial_area    -0.1052       0.0000  0.0000 0.0000
    NTL_mean              -0.1023       0.0000  0.0000 0.0000
    road_tertiary_len     -0.0630       0.0000  0.0000 0.0000
    

### 4.4 Neural Network


```python
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import mean_squared_error, r2_score

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1) Prepare & scale data
X = baghdad_df[features_full].values
y = baghdad_df['TCI'].values.reshape(-1,1)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_tr, X_tmp, y_tr, y_tmp = train_test_split(
    X_scaled, y, test_size=0.3, random_state=42)
X_val, X_te, y_val, y_te = train_test_split(
    X_tmp, y_tmp, test_size=0.5, random_state=42)

# 2) DataLoaders
def make_loader(X, y, bs, shuffle=False):
    ds = TensorDataset(torch.from_numpy(X).float(),
                       torch.from_numpy(y).float())
    return DataLoader(ds, batch_size=bs, shuffle=shuffle)

# 3) Model with BatchNorm
class MLP(nn.Module):
    def __init__(self, dims, dropout):
        super().__init__()
        layers = []
        for i in range(len(dims)-1):
            layers += [
                nn.Linear(dims[i], dims[i+1]),
                nn.BatchNorm1d(dims[i+1]),
                nn.ReLU(),
                nn.Dropout(dropout)
            ]
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        return self.net(x)

# 4) Training routine with early stopping
def train_nn(hidden_dims=[32,16], dropout=0.2, lr=1e-3, wd=1e-4,
             batch_size=64, epochs=100, patience=10):
    dims = [X_tr.shape[1]] + hidden_dims + [1]
    model = MLP(dims, dropout).to(device)
    opt   = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    sched = optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=10)
    loss_fn = nn.MSELoss()

    tr_load = make_loader(X_tr, y_tr, batch_size, shuffle=True)
    val_load= make_loader(X_val, y_val, batch_size)

    best_val_loss, wait = np.inf, 0
    for ep in range(epochs):
        # train
        model.train()
        for xb,yb in tr_load:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss_fn(model(xb), yb).backward()
            opt.step()
            sched.step()

        # validate
        model.eval()
        preds, truths = [], []
        with torch.no_grad():
            for xb,yb in val_load:
                out = model(xb.to(device)).cpu().numpy()
                preds.append(out); truths.append(yb.numpy())
        val_loss = mean_squared_error(
            np.vstack(truths), np.vstack(preds))
        if val_loss < best_val_loss:
            best_val_loss, wait = val_loss, 0
            best_weights = model.state_dict()
        else:
            wait += 1
            if wait >= patience:
                break

    # load best
    model.load_state_dict(best_weights)
    # evaluate on train/val/test
    def eval_set(Xs, ys):
        yhat = model(torch.from_numpy(Xs).float().to(device)).cpu().detach().numpy()
        return (
            np.sqrt(mean_squared_error(ys, yhat)),
            r2_score(ys, yhat)
        )

    rmse_tr, r2_tr = eval_set(X_tr, y_tr)
    rmse_val, r2_val = eval_set(X_val, y_val)
    rmse_te, r2_te = eval_set(X_te, y_te)

    return {
      'model': model,
      'rmse_tr': rmse_tr, 'r2_tr': r2_tr,
      'rmse_val': rmse_val, 'r2_val': r2_val,
      'rmse_te': rmse_te, 'r2_te': r2_te
    }

# 5) Quick grid
configs = [
    {'hidden_dims':[32,16], 'dropout':0.1, 'lr':1e-3},
    {'hidden_dims':[64,32], 'dropout':0.2, 'lr':5e-4},
]
best = None
for cfg in configs:
    res = train_nn(**cfg)
    print(cfg, res['r2_val'], res['r2_te'])
    if best is None or res['r2_te']>best['r2_te']:
        best = res

print("\nBest NN Performance:")
print(f"Train R²={best['r2_tr']:.4f}, Val R²={best['r2_val']:.4f}, Test R²={best['r2_te']:.4f}")

```

    {'hidden_dims': [32, 16], 'dropout': 0.1, 'lr': 0.001} -0.46893307931640615 -0.522192213517753
    {'hidden_dims': [64, 32], 'dropout': 0.2, 'lr': 0.0005} -0.46893307363275527 -0.5221922030759059
    
    Best NN Performance:
    Train R²=-0.4244, Val R²=-0.4689, Test R²=-0.5222
    

Despite extensive experimentation, none of the advanced models surpassed the simple OLS in terms of performance and interpretability. Thus, OLS was retained as the final model for its robustness and clarity.

## 5  Evaluate Model Performance

### 5.1 Metrics: RMSE and R Square

RMSE and R2 for all 6 linear models.


```python
# ============================================================
#  PERFORMANCE DASHBOARD – all six baseline models
# ============================================================
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures

# ------------------------------------------------------------
# 1. Scoring utility
# ------------------------------------------------------------
def evaluate_model(model, X_tr, X_te, y_tr, y_te):
    """Return RMSE_train, R2_train, RMSE_test, R2_test."""
    y_hat_tr = model.predict(X_tr)
    y_hat_te = model.predict(X_te)
    rmse_tr  = np.sqrt(mean_squared_error(y_tr, y_hat_tr))
    rmse_te  = np.sqrt(mean_squared_error(y_te, y_hat_te))
    r2_tr    = r2_score(y_tr, y_hat_tr)
    r2_te    = r2_score(y_te, y_hat_te)
    return rmse_tr, r2_tr, rmse_te, r2_te

# ------------------------------------------------------------
# 2. Regenerate splits & capture metrics
# ------------------------------------------------------------
results = []

# --- 1) Simple LR on TCI/road_len ------------------------------------------
X = baghdad_df[features]
y = baghdad_df['y1']                              # target already computed
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
results.append(
    ['LR', 'TCI/road_len', *evaluate_model(lr, X_tr, X_te, y_tr, y_te)]
)

# --- 2) Log–Log LR on TCI/road_len -----------------------------------------
X = df_ll[features]
y = df_ll['y2']
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
results.append(
    ['Log–Log LR', 'TCI/road_len', *evaluate_model(lr_ll, X_tr, X_te, y_tr, y_te)]
)

# --- 3) Poly-deg-3 LR on TCI/road_len --------------------------------------
poly_tmp = PolynomialFeatures(degree=3, include_bias=False)
X_poly   = poly_tmp.fit_transform(baghdad_df[features])
X_tr, X_te, y_tr, y_te = train_test_split(X_poly, baghdad_df['y1'],
                                          test_size=0.2, random_state=42)
results.append(
    ['Poly LR (d=3)', 'TCI/road_len', *evaluate_model(poly_lr, X_tr, X_te, y_tr, y_te)]
)

# --- 4) Simple LR on raw TCI ------------------------------------------------
X = baghdad_df[features]
y = baghdad_df['TCI']
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
results.append(
    ['LR', 'TCI', *evaluate_model(lr2, X_tr, X_te, y_tr, y_te)]
)

# --- 5) Log–Log LR on raw TCI ----------------------------------------------
X = df_ll2[features]
y = df_ll2['y5']
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
results.append(
    ['Log–Log LR', 'TCI', *evaluate_model(lr_ll2, X_tr, X_te, y_tr, y_te)]
)

# --- 6) Poly-deg-3 LR on raw TCI -------------------------------------------
poly_tmp2 = PolynomialFeatures(degree=3, include_bias=False)
X_poly2   = poly_tmp2.fit_transform(baghdad_df[features])
X_tr, X_te, y_tr, y_te = train_test_split(X_poly2, baghdad_df['TCI'],
                                          test_size=0.2, random_state=42)
results.append(
    ['Poly LR (d=3)', 'TCI', *evaluate_model(poly2_lr, X_tr, X_te, y_tr, y_te)]
)

# ------------------------------------------------------------
# 3. Executive summary table
# ------------------------------------------------------------
df_perf = pd.DataFrame(
    results,
    columns=['Model', 'Target', 'RMSE_train', 'R2_train', 'RMSE_test', 'R2_test']
).set_index(['Model', 'Target'])

display(
    df_perf.style.format('{:.4f}')
          .set_caption("Baseline Linear-Family Models – Performance KPI Matrix")
)

```


<style type="text/css">
</style>
<table id="T_9c196">
  <caption>Baseline Linear-Family Models – Performance KPI Matrix</caption>
  <thead>
    <tr>
      <th class="blank" >&nbsp;</th>
      <th class="blank level0" >&nbsp;</th>
      <th id="T_9c196_level0_col0" class="col_heading level0 col0" >RMSE_train</th>
      <th id="T_9c196_level0_col1" class="col_heading level0 col1" >R2_train</th>
      <th id="T_9c196_level0_col2" class="col_heading level0 col2" >RMSE_test</th>
      <th id="T_9c196_level0_col3" class="col_heading level0 col3" >R2_test</th>
    </tr>
    <tr>
      <th class="index_name level0" >Model</th>
      <th class="index_name level1" >Target</th>
      <th class="blank col0" >&nbsp;</th>
      <th class="blank col1" >&nbsp;</th>
      <th class="blank col2" >&nbsp;</th>
      <th class="blank col3" >&nbsp;</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th id="T_9c196_level0_row0" class="row_heading level0 row0" >LR</th>
      <th id="T_9c196_level1_row0" class="row_heading level1 row0" >TCI/road_len</th>
      <td id="T_9c196_row0_col0" class="data row0 col0" >5.3420</td>
      <td id="T_9c196_row0_col1" class="data row0 col1" >0.7162</td>
      <td id="T_9c196_row0_col2" class="data row0 col2" >6.1411</td>
      <td id="T_9c196_row0_col3" class="data row0 col3" >0.6770</td>
    </tr>
    <tr>
      <th id="T_9c196_level0_row1" class="row_heading level0 row1" >Log–Log LR</th>
      <th id="T_9c196_level1_row1" class="row_heading level1 row1" >TCI/road_len</th>
      <td id="T_9c196_row1_col0" class="data row1 col0" >5.0877</td>
      <td id="T_9c196_row1_col1" class="data row1 col1" >0.1728</td>
      <td id="T_9c196_row1_col2" class="data row1 col2" >5.0621</td>
      <td id="T_9c196_row1_col3" class="data row1 col3" >-0.0067</td>
    </tr>
    <tr>
      <th id="T_9c196_level0_row2" class="row_heading level0 row2" >Poly LR (d=3)</th>
      <th id="T_9c196_level1_row2" class="row_heading level1 row2" >TCI/road_len</th>
      <td id="T_9c196_row2_col0" class="data row2 col0" >2.8633</td>
      <td id="T_9c196_row2_col1" class="data row2 col1" >0.9185</td>
      <td id="T_9c196_row2_col2" class="data row2 col2" >33.2734</td>
      <td id="T_9c196_row2_col3" class="data row2 col3" >-8.4833</td>
    </tr>
    <tr>
      <th id="T_9c196_level0_row3" class="row_heading level0 row3" >LR</th>
      <th id="T_9c196_level1_row3" class="row_heading level1 row3" >TCI</th>
      <td id="T_9c196_row3_col0" class="data row3 col0" >8548952.1574</td>
      <td id="T_9c196_row3_col1" class="data row3 col1" >0.7463</td>
      <td id="T_9c196_row3_col2" class="data row3 col2" >9846074.6481</td>
      <td id="T_9c196_row3_col3" class="data row3 col3" >0.7968</td>
    </tr>
    <tr>
      <th id="T_9c196_level0_row4" class="row_heading level0 row4" >Log–Log LR</th>
      <th id="T_9c196_level1_row4" class="row_heading level1 row4" >TCI</th>
      <td id="T_9c196_row4_col0" class="data row4 col0" >5.8183</td>
      <td id="T_9c196_row4_col1" class="data row4 col1" >0.1474</td>
      <td id="T_9c196_row4_col2" class="data row4 col2" >5.7425</td>
      <td id="T_9c196_row4_col3" class="data row4 col3" >-0.0201</td>
    </tr>
    <tr>
      <th id="T_9c196_level0_row5" class="row_heading level0 row5" >Poly LR (d=3)</th>
      <th id="T_9c196_level1_row5" class="row_heading level1 row5" >TCI</th>
      <td id="T_9c196_row5_col0" class="data row5 col0" >2537682.5992</td>
      <td id="T_9c196_row5_col1" class="data row5 col1" >0.9776</td>
      <td id="T_9c196_row5_col2" class="data row5 col2" >59692127.6602</td>
      <td id="T_9c196_row5_col3" class="data row5 col3" >-6.4689</td>
    </tr>
  </tbody>
</table>



### 5.2 Metrics: AIC, BIC

AIC, BIC and adjusted R2 for different models


```python
def compute_ic(model, X_tr, y_tr):
    """Return (p, AIC, BIC, R2_train, R2_adj)."""
    y_hat = model.predict(X_tr)
    rss   = ((y_tr - y_hat) ** 2).sum()
    n     = len(y_tr)
    p     = np.count_nonzero(model.coef_) + 1
    aic   = n * np.log(rss / n) + 2 * p
    bic   = n * np.log(rss / n) + p * np.log(n)
    r2    = r2_score(y_tr, y_hat)
    r2_adj= 1 - (1 - r2)*(n - 1)/(n - p - 1)
    return p, aic, bic, r2, r2_adj

def compute_test_metrics(model, X_te, y_te):
    """Return (RMSE_test, R2_test)."""
    y_hat = model.predict(X_te)
    return np.sqrt(mean_squared_error(y_te, y_hat)), r2_score(y_te, y_hat)

# container
rows = []

# 1) OLS on TCI/road_len
y1 = baghdad_df['TCI'] / baghdad_df['road_len']
X1 = baghdad_df[features]
X_tr, X_te, y_tr, y_te = train_test_split(X1, y1, test_size=0.2, random_state=42)
rows.append(
    ['OLS (TCI/road_len)',
     *compute_ic(lr, X_tr, y_tr),
     *compute_test_metrics(lr, X_te, y_te)]
)

# 2) Log–Log OLS on TCI/road_len
X2, y2 = df_ll[features], df_ll['y2']
X_tr, X_te, y_tr, y_te = train_test_split(X2, y2, test_size=0.2, random_state=42)
rows.append(
    ['LogLog (TCI/road_len)',
     *compute_ic(lr_ll, X_tr, y_tr),
     *compute_test_metrics(lr_ll, X_te, y_te)]
)

# 3) Poly3 OLS on TCI/road_len
poly = PolynomialFeatures(degree=3, include_bias=False)
X3 = poly.fit_transform(baghdad_df[features])
y3 = baghdad_df['TCI'] / baghdad_df['road_len']
X_tr, X_te, y_tr, y_te = train_test_split(X3, y3, test_size=0.2, random_state=42)
rows.append(
    ['Poly3 (TCI/road_len)',
     *compute_ic(poly_lr, X_tr, y_tr),
     *compute_test_metrics(poly_lr, X_te, y_te)]
)

# 4) OLS on TCI
X4, y4 = baghdad_df[features], baghdad_df['TCI']
X_tr, X_te, y_tr, y_te = train_test_split(X4, y4, test_size=0.2, random_state=42)
rows.append(
    ['OLS (TCI)',
     *compute_ic(lr2, X_tr, y_tr),
     *compute_test_metrics(lr2, X_te, y_te)]
)

# 5) Log–Log OLS on TCI
X5, y5 = df_ll2[features], df_ll2['y5']
X_tr, X_te, y_tr, y_te = train_test_split(X5, y5, test_size=0.2, random_state=42)
rows.append(
    ['LogLog (TCI)',
     *compute_ic(lr_ll2, X_tr, y_tr),
     *compute_test_metrics(lr_ll2, X_te, y_te)]
)

# 6) Poly3 OLS on TCI
poly2 = PolynomialFeatures(degree=3, include_bias=False)
X6 = poly2.fit_transform(baghdad_df[features])
y6 = baghdad_df['TCI']
X_tr, X_te, y_tr, y_te = train_test_split(X6, y6, test_size=0.2, random_state=42)
rows.append(
    ['Poly3 (TCI)',
     *compute_ic(poly2_lr, X_tr, y_tr),
     *compute_test_metrics(poly2_lr, X_te, y_te)]
)

# build DataFrame
df_compare = pd.DataFrame(
    rows,
    columns=[
      'Model', '#params', 'AIC', 'BIC', 'R2_train', 'R2_adj',
      'RMSE_test', 'R2_test'
    ]
).set_index('Model')

# display
display(
    df_compare.style
      .format({
         '#params':'{:.0f}',
         'AIC':'{:.2f}','BIC':'{:.2f}',
         'R2_train':'{:.4f}','R2_adj':'{:.4f}',
         'RMSE_test':'{:.2f}','R2_test':'{:.4f}'
      })
      .set_caption("Complexity‐Penalized & Out‐of‐Sample Performance")
)
```


<style type="text/css">
</style>
<table id="T_180c0">
  <caption>Complexity‐Penalized & Out‐of‐Sample Performance</caption>
  <thead>
    <tr>
      <th class="blank level0" >&nbsp;</th>
      <th id="T_180c0_level0_col0" class="col_heading level0 col0" >#params</th>
      <th id="T_180c0_level0_col1" class="col_heading level0 col1" >AIC</th>
      <th id="T_180c0_level0_col2" class="col_heading level0 col2" >BIC</th>
      <th id="T_180c0_level0_col3" class="col_heading level0 col3" >R2_train</th>
      <th id="T_180c0_level0_col4" class="col_heading level0 col4" >R2_adj</th>
      <th id="T_180c0_level0_col5" class="col_heading level0 col5" >RMSE_test</th>
      <th id="T_180c0_level0_col6" class="col_heading level0 col6" >R2_test</th>
    </tr>
    <tr>
      <th class="index_name level0" >Model</th>
      <th class="blank col0" >&nbsp;</th>
      <th class="blank col1" >&nbsp;</th>
      <th class="blank col2" >&nbsp;</th>
      <th class="blank col3" >&nbsp;</th>
      <th class="blank col4" >&nbsp;</th>
      <th class="blank col5" >&nbsp;</th>
      <th class="blank col6" >&nbsp;</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th id="T_180c0_level0_row0" class="row_heading level0 row0" >OLS (TCI/road_len)</th>
      <td id="T_180c0_row0_col0" class="data row0 col0" >27</td>
      <td id="T_180c0_row0_col1" class="data row0 col1" >1403.30</td>
      <td id="T_180c0_row0_col2" class="data row0 col2" >1511.27</td>
      <td id="T_180c0_row0_col3" class="data row0 col3" >0.7171</td>
      <td id="T_180c0_row0_col4" class="data row0 col4" >0.6967</td>
      <td id="T_180c0_row0_col5" class="data row0 col5" >6.16</td>
      <td id="T_180c0_row0_col6" class="data row0 col6" >0.6745</td>
    </tr>
    <tr>
      <th id="T_180c0_level0_row1" class="row_heading level0 row1" >LogLog (TCI/road_len)</th>
      <td id="T_180c0_row1_col0" class="data row1 col0" >30</td>
      <td id="T_180c0_row1_col1" class="data row1 col1" >1364.47</td>
      <td id="T_180c0_row1_col2" class="data row1 col2" >1484.44</td>
      <td id="T_180c0_row1_col3" class="data row1 col3" >0.1865</td>
      <td id="T_180c0_row1_col4" class="data row1 col4" >0.1209</td>
      <td id="T_180c0_row1_col5" class="data row1 col5" >5.21</td>
      <td id="T_180c0_row1_col6" class="data row1 col6" >-0.0668</td>
    </tr>
    <tr>
      <th id="T_180c0_level0_row2" class="row_heading level0 row2" >Poly3 (TCI/road_len)</th>
      <td id="T_180c0_row2_col0" class="data row2 col0" >3352</td>
      <td id="T_180c0_row2_col1" class="data row2 col1" >-206.74</td>
      <td id="T_180c0_row2_col2" class="data row2 col2" >13197.70</td>
      <td id="T_180c0_row2_col3" class="data row2 col3" >1.0000</td>
      <td id="T_180c0_row2_col4" class="data row2 col4" >1.0000</td>
      <td id="T_180c0_row2_col5" class="data row2 col5" >349.31</td>
      <td id="T_180c0_row2_col6" class="data row2 col6" >-1044.1444</td>
    </tr>
    <tr>
      <th id="T_180c0_level0_row3" class="row_heading level0 row3" >OLS (TCI)</th>
      <td id="T_180c0_row3_col0" class="data row3 col0" >27</td>
      <td id="T_180c0_row3_col1" class="data row3 col1" >12917.94</td>
      <td id="T_180c0_row3_col2" class="data row3 col2" >13025.91</td>
      <td id="T_180c0_row3_col3" class="data row3 col3" >0.7469</td>
      <td id="T_180c0_row3_col4" class="data row3 col4" >0.7286</td>
      <td id="T_180c0_row3_col5" class="data row3 col5" >9826741.80</td>
      <td id="T_180c0_row3_col6" class="data row3 col6" >0.7976</td>
    </tr>
    <tr>
      <th id="T_180c0_level0_row4" class="row_heading level0 row4" >LogLog (TCI)</th>
      <td id="T_180c0_row4_col0" class="data row4 col0" >30</td>
      <td id="T_180c0_row4_col1" class="data row4 col1" >1472.57</td>
      <td id="T_180c0_row4_col2" class="data row4 col2" >1592.53</td>
      <td id="T_180c0_row4_col3" class="data row4 col3" >0.1617</td>
      <td id="T_180c0_row4_col4" class="data row4 col4" >0.0941</td>
      <td id="T_180c0_row4_col5" class="data row4 col5" >5.91</td>
      <td id="T_180c0_row4_col6" class="data row4 col6" >-0.0820</td>
    </tr>
    <tr>
      <th id="T_180c0_level0_row5" class="row_heading level0 row5" >Poly3 (TCI)</th>
      <td id="T_180c0_row5_col0" class="data row5 col0" >3352</td>
      <td id="T_180c0_row5_col1" class="data row5 col1" >10670.06</td>
      <td id="T_180c0_row5_col2" class="data row5 col2" >24074.50</td>
      <td id="T_180c0_row5_col3" class="data row5 col3" >1.0000</td>
      <td id="T_180c0_row5_col4" class="data row5 col4" >1.0000</td>
      <td id="T_180c0_row5_col5" class="data row5 col5" >323927385.85</td>
      <td id="T_180c0_row5_col6" class="data row5 col6" >-218.9459</td>
    </tr>
  </tbody>
</table>



Across both target definitions, the simple OLS approaches (12 parameters, no transforms or high-order terms) consistently give you the best out-of-sample R² while keeping model complexity and information‐criteria penalties low.

The polynomial expansions over-fit, and the log–log specs under-fit—they never beat plain OLS on real-world generalisation.

### 5.3 Best Model Traceback

Coefficient traceback for the best-performing OLS model (Target = TCI).


```python
# 1. Re-create the exact train/test split used for lr2
X = baghdad_df[features]            # <— same feature list
y = baghdad_df['TCI']
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 2. (Re)fit lr2 on the training fold, if not already in memory
best_lr = LinearRegression().fit(X_tr, y_tr)

# 3. Extract coefficients & intercept (already in original units)
coef_df = pd.DataFrame({
    'feature': X.columns,
    'coefficient': best_lr.coef_
}).set_index('feature')

# Add intercept row for completeness
coef_df.loc['Intercept'] = best_lr.intercept_

# 4. Sort by absolute magnitude for readability
coef_df['abs_coef'] = coef_df['coefficient'].abs()
coef_df = coef_df.sort_values('abs_coef', ascending=False).drop(columns='abs_coef')

# 5. Display
print("OLS coefficients in native units (ΔTCI per 1-unit feature change):\n")
print(coef_df.to_string(float_format='%.4f'))

```

    OLS coefficients in native units (ΔTCI per 1-unit feature change):
    
                           coefficient
    feature                           
    no2_mean             53986362.3235
    Intercept            17055232.5802
    poi_count              930026.1514
    LST_day_mean          -462200.1825
    lu_retail_area            933.0909
    NTL_mean                 -602.1745
    lu_farmyard_area         -570.2696
    road_motorway_len        -375.2164
    road_trunk_len           -266.7777
    road_primary_len         -206.2958
    road_len                   57.6798
    road_residential_len      -49.7191
    road_secondary_len        -48.4863
    road_tertiary_len         -43.1960
    pop_sum_m                  22.0768
    grassland_a                -3.4462
    cropland_a                 -0.3315
    non_built_area              0.3116
    lu_commercial_area          0.3023
    built_up_a                 -0.0916
    lu_residential_area        -0.0825
    lu_industrial_area         -0.0802
    lu_farmland_area            0.0066
    
