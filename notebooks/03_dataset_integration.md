# Notebook 3: Dataset Integration

## README

### Overview

This notebook integrates spatial NO₂ data with multiple auxiliary datasets to generate a unified, mesh-based dataset for modeling and analysis in two case study cities: **Addis Ababa, Ethiopia** and **Baghdad, Iraq**.  

It contains two part: (1) multi-dataset integration & new feature generation, and (2) data temporal & spacial aggregation.

It consolidates outputs from:

- **NO₂ pollution data** (processed in Notebook 1)  
- **Human activity and other environmental feature datasets** (processed in Notebook 2)

### Objective

- **Merge**: Join NO₂ pollution values with auxiliary features (e.g., population, roads, night lights) by mesh ID and date.  
- **New Feature**: Generate derived features such as NO2 lag feature and No2 neighor lag featrue.
- **Aggregation**: Get prepared dataset with different temporal-spatial resolution.


### Workflow

1. **Feature Merging**  
   - Load cleaned NO₂ and feature datasets from previous notebooks  
   - Merge all spatial datasets by mesh cell (`geom_id`)  
   - Ensure alignment in spatial extent and resolution  

2. **Feature Engineering**  
   - Compute new features (e.g., spatial & temporal-based new feature)  

3.  **Aggregation and Export**  
   - Aggregate in time: from daily value to monthly level.
   - Aggregate in space: from cell level to sub-administrative region.
   - Export all the data in progress as GeoPackage or CSV for downstream tasks  


### Outputs

- `city_mesh_data.gpkg`: GeoDataFrame with all integrated features per mesh cell, one file per day (731 files for each city).
- `full_city_df.parquet`: Concatenated DataFrame including all dates in one file, convenient for downstream analysis (one file for each city).
- `city_timeResolution_spaceResolution.csv`: Aggregated DataFrame including all dates in one file, in differen data resolution.


## 0 Init: Prepare Packages and Configuration

Get current file/repo/data path in local to make sure the following cells run properly.


```python
import sys
from pathlib import Path
SRC_PATH = Path().resolve().parent / "src"
sys.path.append(str(SRC_PATH))

from config import *
```

## 1 Merge Multiple Features

Combine features from different GeoPackage files into one GeoPackage file. 

Then save the data as parquet format for better reading efficiency.

### Addis Ababa


```python
# Addis Ababa
from helpercollections import merge_multiple_gpkgs, convert_gpkgs_to_parquet
import geopandas as gpd

addis_feature_mesh_paths = [DATA_PATH / "addis-no2-mesh-data",      # NO2 data: numeric
                            DATA_PATH / "addis-temp-mesh-data",     # Daily Average Temperature 
                            DATA_PATH / "addis-pop-mesh-data",      # Population data: numeric
                            DATA_PATH / "addis-NTL-mesh-data",      # Night Time Light data: numeric
                            DATA_PATH / "addis-CC-mesh-data",       # Cloud Category data: category
                            DATA_PATH / "addis-LST-mesh-data",      # Land Surface Temperature data: numeric
                            DATA_PATH / "addis-lc-mesh-data",       # Land Cover data: category
                            DATA_PATH / "addis-ESA-mesh-data",      # Land Cover ESA (area) data: numeric
                            DATA_PATH / "addis-OSM-mesh-data",      # Point Of Interest data: numeric
                            DATA_PATH / "addis-OSM2-mesh-data",     # New Land Use Data, combined with ESA and OSM
                            ]
output_folder = DATA_PATH / "addis-mesh-data"
merge_multiple_gpkgs(addis_feature_mesh_paths, output_folder)
convert_gpkgs_to_parquet(mesh_folder=output_folder, output_path=DATA_PATH / "temp", file_name="full_addis_df") 
```


```python
gdf_addis = gpd.read_file(DATA_PATH / "addis-mesh-data" / "addis-ababa-2023-01-02.gpkg")
gdf_addis.head(3)
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>geom_id</th>
      <th>no2_mean</th>
      <th>temp_mean</th>
      <th>pop_sum_m</th>
      <th>NTL_mean</th>
      <th>cloud_category</th>
      <th>LST_day_mean</th>
      <th>landcover_2023</th>
      <th>tree_cover_a</th>
      <th>shrubland_a</th>
      <th>...</th>
      <th>lu_farmyard_share</th>
      <th>road_motorway_len</th>
      <th>road_trunk_len</th>
      <th>road_primary_len</th>
      <th>road_secondary_len</th>
      <th>road_tertiary_len</th>
      <th>road_residential_len</th>
      <th>fossil_pp_count</th>
      <th>non_built_area</th>
      <th>geometry</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0.00005</td>
      <td>25.13</td>
      <td>969.683960</td>
      <td>10.372897</td>
      <td>0.0</td>
      <td>27.732857</td>
      <td>12.0</td>
      <td>9900</td>
      <td>113700</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1351.258983</td>
      <td>3079.516097</td>
      <td>0</td>
      <td>198200</td>
      <td>POLYGON ((38.78925 8.83942, 38.78925 8.84841, ...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>0.00005</td>
      <td>22.73</td>
      <td>1574.853149</td>
      <td>10.460668</td>
      <td>1.0</td>
      <td>27.150000</td>
      <td>12.0</td>
      <td>200</td>
      <td>80600</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>3690.168402</td>
      <td>0</td>
      <td>149900</td>
      <td>POLYGON ((38.79824 8.83942, 38.79824 8.84841, ...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>0.00005</td>
      <td>21.05</td>
      <td>1859.865723</td>
      <td>10.903213</td>
      <td>1.0</td>
      <td>27.150000</td>
      <td>13.0</td>
      <td>33300</td>
      <td>40500</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>846.632771</td>
      <td>2949.800017</td>
      <td>0</td>
      <td>187500</td>
      <td>POLYGON ((38.80722 8.83942, 38.80722 8.84841, ...</td>
    </tr>
  </tbody>
</table>
<p>3 rows × 45 columns</p>
</div>


### Iraq - Baghdad


```python
# # Baghdad
from helpercollections import merge_multiple_gpkgs, convert_gpkgs_to_parquet
import geopandas as gpd

baghdad_feature_mesh_paths = [DATA_PATH / "baghdad-no2-mesh-data",      # NO2 data: numeric
                              DATA_PATH / "baghdad-pop-mesh-data",      # Population data: numeric
                              DATA_PATH / "baghdad-NTL-mesh-data",      # Night Time Light data: numeric
                              DATA_PATH / "baghdad-CC-mesh-data",       # Cloud Category data: category
                              DATA_PATH / "baghdad-LST-mesh-data",      # Land Surface Temperature data: numeric
                              DATA_PATH / "baghdad-temp-mesh-data",     # Daily Average Temperature 
                              DATA_PATH / "baghdad-lc-mesh-data",       # Land Cover data: category
                              DATA_PATH / "baghdad-ESA-mesh-data",      # Land Cover ESA (area) data: numeric
                              DATA_PATH / "baghdad-OSM-mesh-data",      # Point Of Interest data: numeric
                              DATA_PATH / "baghdad-TCI-mesh-data",      # Traffic Congestion Index data: numeric
                              DATA_PATH / "baghdad-OSM2-mesh-data",     # New Land Use Data, combined with ESA and OSM
                              ]
  
output_folder = DATA_PATH / "baghdad-mesh-data"
merge_multiple_gpkgs(baghdad_feature_mesh_paths, output_folder)
convert_gpkgs_to_parquet(mesh_folder=output_folder, output_path=DATA_PATH / "temp", file_name="full_baghdad_df") 
```


```python
gdf_baghdad = gpd.read_file(DATA_PATH / "baghdad-mesh-data" / "baghdad-2023-01-01.gpkg")
gdf_baghdad.head(3)
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>geom_id</th>
      <th>no2_mean</th>
      <th>pop_sum_m</th>
      <th>NTL_mean</th>
      <th>cloud_category</th>
      <th>LST_day_mean</th>
      <th>temp_mean</th>
      <th>landcover_2023</th>
      <th>tree_cover_a</th>
      <th>shrubland_a</th>
      <th>...</th>
      <th>road_motorway_len</th>
      <th>road_trunk_len</th>
      <th>road_primary_len</th>
      <th>road_secondary_len</th>
      <th>road_tertiary_len</th>
      <th>road_residential_len</th>
      <th>fossil_pp_count</th>
      <th>TCI</th>
      <th>non_built_area</th>
      <th>geometry</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0.000111</td>
      <td>44.653709</td>
      <td>2.088287</td>
      <td>0.0</td>
      <td>18.09</td>
      <td>11.831877</td>
      <td>12.0</td>
      <td>1400</td>
      <td>35500</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>333800</td>
      <td>POLYGON ((44.56542 32.81546, 44.56542 32.82444...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>0.000111</td>
      <td>45.872368</td>
      <td>2.345411</td>
      <td>0.0</td>
      <td>17.79</td>
      <td>11.828498</td>
      <td>12.0</td>
      <td>3300</td>
      <td>43200</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>613300</td>
      <td>POLYGON ((44.5744 32.81546, 44.5744 32.82444, ...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>0.000111</td>
      <td>40.733757</td>
      <td>2.524289</td>
      <td>0.0</td>
      <td>17.75</td>
      <td>11.824177</td>
      <td>10.0</td>
      <td>21700</td>
      <td>65400</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>788800</td>
      <td>POLYGON ((44.58339 32.81546, 44.58339 32.82444...</td>
    </tr>
  </tbody>
</table>
<p>3 rows × 46 columns</p>
</div>


## 2 Generate New Features 

Based on the exploratory data analysis below, generate new features related to NO₂ spatial and temporal autocorrelation.

The following cell made the pre-processing of the data, including:

1) Integrate all the gpkg into a large file with date.

2) Add new features:

    - Yesterday NO₂ (1 day lag)

    - Yesterday average NO₂ level.

3) Export as parquet file for faster read in the future.

### Addis Ababa


```python
import pandas as pd
from analysis import add_lag_features
full_addis_df = pd.read_parquet(DATA_PATH / "temp" / "full_addis_df.parquet", engine="pyarrow")

add_lag_features(full_addis_df, output_path=DATA_PATH / "temp", file_name="full_addis_df")
```

### Baghdad


```python
import pandas as pd
from analysis import add_lag_features
full_baghdad_df = pd.read_parquet(DATA_PATH / "temp" / "full_baghdad_df.parquet", engine="pyarrow")

add_lag_features(full_baghdad_df, output_path=DATA_PATH / "temp", file_name="full_baghdad_df")
```

## 3 Aggregation: Temporal & Spatial Aggregation

- Initial data resolution:
    - Time: daily
    - Space: cell (mesh grid)

- Total four types of data with different aggregation level:
    1. Cell level & Daily Data (original parquet file)
    2. Cell level & Monthly Data
    3. Sub-region level & Daily Data
    4. Sub-region level & Monthly Data

### Addis Ababa


```python
import numpy as np
import pandas as pd
# Read the Addis Ababa data
full_df = pd.read_parquet(DATA_PATH / "temp" / "full_addis_df.parquet", engine="pyarrow")
full_df['month'] = full_df['date'].dt.to_period('M')

# Original dataset: daily / cell resolution
daily_cell_ori = full_df
```


```python
# type(full_df)
full_df.columns
```


    Index(['geom_id', 'no2_mean', 'pop_sum_m', 'NTL_mean', 'road_len',
           'road_share', 'poi_count', 'poi_share', 'lu_industrial_area',
           'lu_industrial_share', 'lu_commercial_area', 'lu_commercial_share',
           'lu_residential_area', 'lu_residential_share', 'lu_retail_area',
           'lu_retail_share', 'lu_farmland_area', 'lu_farmland_share',
           'lu_farmyard_area', 'lu_farmyard_share', 'road_motorway_len',
           'road_trunk_len', 'road_primary_len', 'road_secondary_len',
           'road_tertiary_len', 'road_residential_len', 'fossil_pp_count',
           'geometry_x', 'date', 'no2_lag1', 'no2_neighbor_lag1', 'cloud_category',
           'LST_day_mean', 'landcover_2023', 'Shape_Leng', 'Shape_Area', 'ADM3_EN',
           'ADM3_PCODE', 'month'],
          dtype='object')



```python
full_df.head()
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>geom_id</th>
      <th>no2_mean</th>
      <th>pop_sum_m</th>
      <th>NTL_mean</th>
      <th>road_len</th>
      <th>road_share</th>
      <th>poi_count</th>
      <th>poi_share</th>
      <th>lu_industrial_area</th>
      <th>lu_industrial_share</th>
      <th>...</th>
      <th>no2_lag1</th>
      <th>no2_neighbor_lag1</th>
      <th>cloud_category</th>
      <th>LST_day_mean</th>
      <th>landcover_2023</th>
      <th>Shape_Leng</th>
      <th>Shape_Area</th>
      <th>ADM3_EN</th>
      <th>ADM3_PCODE</th>
      <th>month</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0.000051</td>
      <td>969.68396</td>
      <td>7.266073</td>
      <td>5860.59401</td>
      <td>0.000745</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>12.0</td>
      <td>0.538101</td>
      <td>0.010351</td>
      <td>Akaki Kality</td>
      <td>ET140101</td>
      <td>2023-01</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>0.000050</td>
      <td>969.68396</td>
      <td>10.372897</td>
      <td>5860.59401</td>
      <td>0.000745</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.000051</td>
      <td>0.000038</td>
      <td>0.0</td>
      <td>25.130</td>
      <td>12.0</td>
      <td>0.538101</td>
      <td>0.010351</td>
      <td>Akaki Kality</td>
      <td>ET140101</td>
      <td>2023-01</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>0.000047</td>
      <td>969.68396</td>
      <td>1.124154</td>
      <td>5860.59401</td>
      <td>0.000745</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.000050</td>
      <td>0.000052</td>
      <td>0.0</td>
      <td>15.588</td>
      <td>12.0</td>
      <td>0.538101</td>
      <td>0.010351</td>
      <td>Akaki Kality</td>
      <td>ET140101</td>
      <td>2023-01</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>0.000047</td>
      <td>969.68396</td>
      <td>0.727840</td>
      <td>5860.59401</td>
      <td>0.000745</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.000047</td>
      <td>0.000050</td>
      <td>0.0</td>
      <td>30.670</td>
      <td>12.0</td>
      <td>0.538101</td>
      <td>0.010351</td>
      <td>Akaki Kality</td>
      <td>ET140101</td>
      <td>2023-01</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>0.000058</td>
      <td>969.68396</td>
      <td>3.964316</td>
      <td>5860.59401</td>
      <td>0.000745</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.000047</td>
      <td>0.000044</td>
      <td>0.0</td>
      <td>30.990</td>
      <td>12.0</td>
      <td>0.538101</td>
      <td>0.010351</td>
      <td>Akaki Kality</td>
      <td>ET140101</td>
      <td>2023-01</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 39 columns</p>
</div>


#### Cell - Monthly Level

First, generate the cell level, monthly data and save as csv file.


```python
# features that need mean when aggregate from day to month level
time_agg_mean_feature = [
        'no2_mean', 'no2_lag1', 'no2_neighbor_lag1',
        'NTL_mean',
        # 'cloud_category', 
        'LST_day_mean', 
        # 'landcover_2023',
        'pop_sum_m',  
        'road_len', 
        'poi_count', 'lu_industrial_area',
        'lu_commercial_area',  'lu_residential_area', 'lu_retail_area', 'lu_farmland_area', 
        'lu_farmyard_area', 
        'road_primary_len',
        'road_motorway_len', 'road_trunk_len',  'road_secondary_len', 'road_tertiary_len', 'road_residential_len',
         
]

monthly_cell = daily_cell_ori.groupby(['geom_id', 'month', 'ADM3_EN'])[time_agg_mean_feature].mean().reset_index()
monthly_cell.to_csv(DATA_PATH / "temp" / 'addis_monthly_cell.csv', index=False)
```


```python
# monthly_cell.head()
```

#### Sub-district - Daily Level

Second, generate the sub-district level, daily data and save as csv file.


```python
spatial_agg_sum_feature = [
        'no2_mean', 'no2_lag1', 'no2_neighbor_lag1',
        'NTL_mean',
        'pop_sum_m',  
        'road_len', 
        'poi_count', 'lu_industrial_area',
        'lu_commercial_area',  'lu_residential_area', 'lu_retail_area', 'lu_farmland_area', 
        'lu_farmyard_area', 
        'road_primary_len',
        'road_motorway_len', 'road_trunk_len',  'road_secondary_len', 'road_tertiary_len', 'road_residential_len',
         
]
space_agg_mean_feature = [
        # 'cloud_category', 
        'LST_day_mean', 
        # 'landcover_2023',       
]

daily_adm3_sum = daily_cell_ori.groupby(['date', 'ADM3_EN'])[spatial_agg_sum_feature].sum().reset_index()
daily_adm3_avg = daily_cell_ori.groupby(['date', 'ADM3_EN'])[space_agg_mean_feature].mean().reset_index()
daily_adm3 = daily_adm3_avg.merge(daily_adm3_sum, on=['date', 'ADM3_EN'], how='left')
daily_adm3.to_csv(DATA_PATH / "temp" / 'addis_daily_adm3.csv', index=False)
```


```python
# daily_adm3.head()
```

#### Sub-district - Monthly Level

Third, generate the sub-district level, monthly data and save as csv file.


```python
space_agg_sum_feature = [
        'no2_mean', 'no2_lag1', 'no2_neighbor_lag1',
        'NTL_mean',
        'pop_sum_m',  
        'road_len', 
        'poi_count', 'lu_industrial_area',
        'lu_commercial_area',  'lu_residential_area', 'lu_retail_area', 'lu_farmland_area', 
        'lu_farmyard_area', 
        'road_primary_len',
        'road_motorway_len', 'road_trunk_len',  'road_secondary_len', 'road_tertiary_len', 'road_residential_len',
         
]

space_agg_mean_feature = [
        # 'cloud_category', 
        'LST_day_mean', 
        # 'landcover_2023',       
]

monthly_adm3_sum = monthly_cell.groupby(['month', 'ADM3_EN'])[space_agg_sum_feature].sum().reset_index()
monthly_adm3_avg = monthly_cell.groupby(['month', 'ADM3_EN'])[space_agg_mean_feature].mean().reset_index()
```


```python
monthly_adm3 = monthly_adm3_avg.merge(monthly_adm3_sum, on=['month', 'ADM3_EN'], how='left')
monthly_adm3.to_csv(DATA_PATH / "temp" / 'addis_monthly_adm3.csv', index=False)
```


```python
# monthly_adm3
```

### Baghdad


```python
import numpy as np
import pandas as pd
# Read the data
full_df = pd.read_parquet(DATA_PATH / "temp" / "full_baghdad_df.parquet", engine="pyarrow")
full_df['month'] = full_df['date'].dt.to_period('M')
full_df = full_df.rename(columns={'temp_mean': 'LST_day_mean'})   # unify to LST_day_mean

# Original dataset: daily / cell resolution
daily_cell_ori = full_df
```


```python
full_df.columns
```


    Index(['geom_id', 'no2_mean', 'pop_sum_m', 'NTL_mean', 'road_len',
           'road_share', 'poi_count', 'poi_share', 'lu_industrial_area',
           'lu_industrial_share', 'lu_commercial_area', 'lu_commercial_share',
           'lu_residential_area', 'lu_residential_share', 'lu_retail_area',
           'lu_retail_share', 'lu_farmland_area', 'lu_farmland_share',
           'lu_farmyard_area', 'lu_farmyard_share', 'road_motorway_len',
           'road_trunk_len', 'road_primary_len', 'road_secondary_len',
           'road_tertiary_len', 'road_residential_len', 'fossil_pp_count', 'TCI',
           'geometry_x', 'date', 'no2_lag1', 'no2_neighbor_lag1', 'cloud_category',
           'LST_day_mean', 'landcover_2023', 'Shape_Leng', 'Shape_Area', 'ADM3_EN',
           'ADM3_PCODE', 'month'],
          dtype='object')


#### Cell - Monthly Level

First, generate the cell level, monthly data and save as csv file.


```python
# features that need mean when aggregate from day to month level
time_agg_mean_feature = [
        'no2_mean', 'no2_lag1', 'no2_neighbor_lag1',
        'NTL_mean',
        'TCI',
        # 'cloud_category', 
        'LST_day_mean', 
        # 'landcover_2023',
        'pop_sum_m',  
        'road_len', 
        'poi_count', 'lu_industrial_area',
        'lu_commercial_area',  'lu_residential_area', 'lu_retail_area', 'lu_farmland_area', 
        'lu_farmyard_area', 
        'road_primary_len',
        'road_motorway_len', 'road_trunk_len',  'road_secondary_len', 'road_tertiary_len', 'road_residential_len',
         
]

monthly_cell = daily_cell_ori.groupby(['geom_id', 'month', 'ADM3_EN'])[time_agg_mean_feature].mean().reset_index()
monthly_cell.to_csv(DATA_PATH / "temp" / 'baghdad_monthly_cell.csv', index=False)
```


```python
# monthly_cell.head()
```

#### Sub-district - Daily Level
Second, generate the sub-district level, daily data and save as csv file.


```python
spatial_agg_sum_feature = [
        'no2_mean', 'no2_lag1', 'no2_neighbor_lag1',
        'NTL_mean',
        'TCI',
        'pop_sum_m',  
        'road_len', 
        'poi_count', 'lu_industrial_area',
        'lu_commercial_area',  'lu_residential_area', 'lu_retail_area', 'lu_farmland_area', 
        'lu_farmyard_area', 
        'road_primary_len',
        'road_motorway_len', 'road_trunk_len',  'road_secondary_len', 'road_tertiary_len', 'road_residential_len',
         
]
space_agg_mean_feature = [
        # 'cloud_category', 
        'LST_day_mean', 
        # 'landcover_2023',       
]

daily_adm3_sum = daily_cell_ori.groupby(['date', 'ADM3_EN'])[spatial_agg_sum_feature].sum().reset_index()
daily_adm3_avg = daily_cell_ori.groupby(['date', 'ADM3_EN'])[space_agg_mean_feature].mean().reset_index()
daily_adm3 = daily_adm3_avg.merge(daily_adm3_sum, on=['date', 'ADM3_EN'], how='left')
daily_adm3.to_csv(DATA_PATH / "temp" / 'baghdad_daily_adm3.csv', index=False)
```


```python
daily_adm3
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>date</th>
      <th>ADM3_EN</th>
      <th>LST_day_mean</th>
      <th>no2_mean</th>
      <th>no2_lag1</th>
      <th>no2_neighbor_lag1</th>
      <th>NTL_mean</th>
      <th>TCI</th>
      <th>pop_sum_m</th>
      <th>road_len</th>
      <th>...</th>
      <th>lu_residential_area</th>
      <th>lu_retail_area</th>
      <th>lu_farmland_area</th>
      <th>lu_farmyard_area</th>
      <th>road_primary_len</th>
      <th>road_motorway_len</th>
      <th>road_trunk_len</th>
      <th>road_secondary_len</th>
      <th>road_tertiary_len</th>
      <th>road_residential_len</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2023-01-01</td>
      <td>Abu Ghraib</td>
      <td>12.420119</td>
      <td>0.095418</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>16517.924950</td>
      <td>2.796893e+06</td>
      <td>5.111909e+05</td>
      <td>2.712699e+06</td>
      <td>...</td>
      <td>2.436915e+07</td>
      <td>0.0</td>
      <td>3.297929e+07</td>
      <td>8734.319149</td>
      <td>72805.034518</td>
      <td>134296.278767</td>
      <td>0.000000</td>
      <td>42553.429081</td>
      <td>267884.154210</td>
      <td>1.157555e+06</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2023-01-01</td>
      <td>Al-Fahama</td>
      <td>12.771666</td>
      <td>0.037629</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>7831.303788</td>
      <td>3.661187e+07</td>
      <td>1.019493e+06</td>
      <td>1.771264e+06</td>
      <td>...</td>
      <td>3.925814e+07</td>
      <td>0.0</td>
      <td>1.072887e+06</td>
      <td>0.000000</td>
      <td>9374.771374</td>
      <td>23504.337300</td>
      <td>31207.293767</td>
      <td>4882.810604</td>
      <td>165507.886467</td>
      <td>1.322054e+06</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2023-01-01</td>
      <td>Al-Jisr</td>
      <td>13.350142</td>
      <td>0.066570</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>6524.999007</td>
      <td>5.499618e+06</td>
      <td>2.537896e+05</td>
      <td>6.784165e+05</td>
      <td>...</td>
      <td>9.367158e+06</td>
      <td>0.0</td>
      <td>1.893582e+07</td>
      <td>0.000000</td>
      <td>29253.292783</td>
      <td>0.000000</td>
      <td>31548.343818</td>
      <td>14049.948014</td>
      <td>52184.422521</td>
      <td>3.835369e+05</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2023-01-01</td>
      <td>Al-Karrada Al-Sharqia</td>
      <td>13.822724</td>
      <td>0.043889</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>10284.304605</td>
      <td>2.778583e+07</td>
      <td>6.804654e+05</td>
      <td>1.657440e+06</td>
      <td>...</td>
      <td>2.692580e+07</td>
      <td>0.0</td>
      <td>6.301526e+06</td>
      <td>0.000000</td>
      <td>86620.551798</td>
      <td>32598.567966</td>
      <td>3730.427599</td>
      <td>40736.417691</td>
      <td>118433.362443</td>
      <td>8.482135e+05</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2023-01-01</td>
      <td>Al-Latifya</td>
      <td>12.399308</td>
      <td>0.072134</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>5038.490275</td>
      <td>3.488123e+06</td>
      <td>1.598628e+05</td>
      <td>8.827485e+05</td>
      <td>...</td>
      <td>2.360183e+06</td>
      <td>0.0</td>
      <td>2.155193e+08</td>
      <td>0.000000</td>
      <td>7899.550833</td>
      <td>58368.588112</td>
      <td>39685.810012</td>
      <td>40019.461397</td>
      <td>59683.649226</td>
      <td>2.846482e+05</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>15346</th>
      <td>2024-12-31</td>
      <td>Markaz Al-Karkh</td>
      <td>13.878697</td>
      <td>0.000000</td>
      <td>0.021264</td>
      <td>0.021154</td>
      <td>8148.203962</td>
      <td>2.332277e+05</td>
      <td>1.772346e+05</td>
      <td>7.375371e+05</td>
      <td>...</td>
      <td>1.005766e+07</td>
      <td>0.0</td>
      <td>0.000000e+00</td>
      <td>0.000000</td>
      <td>71372.128640</td>
      <td>14098.977033</td>
      <td>0.000000</td>
      <td>50811.351119</td>
      <td>31200.153262</td>
      <td>2.529159e+05</td>
    </tr>
    <tr>
      <th>15347</th>
      <td>2024-12-31</td>
      <td>Markaz Al-Mada'in</td>
      <td>13.722056</td>
      <td>0.000000</td>
      <td>0.406985</td>
      <td>0.404960</td>
      <td>7029.665967</td>
      <td>1.069760e+05</td>
      <td>1.636998e+05</td>
      <td>7.846149e+05</td>
      <td>...</td>
      <td>1.549490e+07</td>
      <td>0.0</td>
      <td>1.479552e+08</td>
      <td>0.000000</td>
      <td>22978.465260</td>
      <td>0.000000</td>
      <td>42400.937644</td>
      <td>0.000000</td>
      <td>42946.439662</td>
      <td>4.669922e+05</td>
    </tr>
    <tr>
      <th>15348</th>
      <td>2024-12-31</td>
      <td>Markaz Al-Mahmoudiya</td>
      <td>13.583168</td>
      <td>0.000000</td>
      <td>0.034868</td>
      <td>0.034775</td>
      <td>1850.581081</td>
      <td>1.531018e+05</td>
      <td>1.072385e+05</td>
      <td>4.749349e+05</td>
      <td>...</td>
      <td>8.522331e+03</td>
      <td>0.0</td>
      <td>3.965280e+07</td>
      <td>0.000000</td>
      <td>9762.885545</td>
      <td>7638.868219</td>
      <td>9296.516086</td>
      <td>2541.127138</td>
      <td>57374.549910</td>
      <td>2.398848e+05</td>
    </tr>
    <tr>
      <th>15349</th>
      <td>2024-12-31</td>
      <td>Markaz Al-Thawra</td>
      <td>13.638146</td>
      <td>0.000000</td>
      <td>0.046052</td>
      <td>0.046124</td>
      <td>10100.558853</td>
      <td>3.099584e+05</td>
      <td>1.156935e+06</td>
      <td>1.571223e+06</td>
      <td>...</td>
      <td>3.415564e+07</td>
      <td>0.0</td>
      <td>0.000000e+00</td>
      <td>0.000000</td>
      <td>109349.280838</td>
      <td>12861.668398</td>
      <td>0.000000</td>
      <td>49075.556344</td>
      <td>167318.888287</td>
      <td>1.085157e+06</td>
    </tr>
    <tr>
      <th>15350</th>
      <td>2024-12-31</td>
      <td>That al Salasil</td>
      <td>12.653585</td>
      <td>0.000000</td>
      <td>0.036308</td>
      <td>0.036559</td>
      <td>5702.589884</td>
      <td>4.826648e+04</td>
      <td>2.476259e+05</td>
      <td>4.983867e+05</td>
      <td>...</td>
      <td>6.333917e+06</td>
      <td>0.0</td>
      <td>3.258954e+05</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>3718.209313</td>
      <td>14451.666038</td>
      <td>2326.787823</td>
      <td>18441.066309</td>
      <td>3.578889e+05</td>
    </tr>
  </tbody>
</table>
<p>15351 rows × 23 columns</p>
</div>


#### Sub-district - Monthly Level

Third, generate the sub-district level, monthly data and save as csv file.


```python
space_agg_sum_feature = [
        'no2_mean', 'no2_lag1', 'no2_neighbor_lag1',
        'NTL_mean',
        'TCI',
        'pop_sum_m',  
        'road_len', 
        'poi_count', 'lu_industrial_area',
        'lu_commercial_area',  'lu_residential_area', 'lu_retail_area', 'lu_farmland_area', 
        'lu_farmyard_area', 
        'road_primary_len',
        'road_motorway_len', 'road_trunk_len',  'road_secondary_len', 'road_tertiary_len', 'road_residential_len',
         
]

space_agg_mean_feature = [
        # 'cloud_category', 
        'LST_day_mean', 
        # 'landcover_2023',       
]

monthly_adm3_sum = monthly_cell.groupby(['month', 'ADM3_EN'])[space_agg_sum_feature].sum().reset_index()
monthly_adm3_avg = monthly_cell.groupby(['month', 'ADM3_EN'])[space_agg_mean_feature].mean().reset_index()
```


```python
monthly_adm3 = monthly_adm3_avg.merge(monthly_adm3_sum, on=['month', 'ADM3_EN'], how='left')
monthly_adm3.to_csv(DATA_PATH / "temp" / 'baghdad_monthly_adm3.csv', index=False)
```


```python
# monthly_adm3
```
