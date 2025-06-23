# Notebook 1: Air Pollution Data Preparation

## README

### Overview
This notebook conducts the data processing and comprehensive analysis of nitrogen dioxide (NO₂) pollution using [Sentinel-5P data](https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S5P_NRTI_L3_NO2), with a focus on **Ethiopia (Addis Ababa)** and **Iraq (Baghdad)**. It covers the full workflow, including data retrieval, preprocessing, aggregation, and visualisation.

In addition to NO₂, other key pollutants - **sulfur dioxide (SO₂)**, **carbon monoxide (CO)**, and **ozone (O₃)** - are also processed and briefly explored for inter-pollutant relationships.

### Objective
- **Clean & aggregate** raw Sentinel-5P NO₂ to mesh grids.  
- **Visualize** spatial–temporal patterns (static maps & GIFs).  
- **Analyse** pollutant dynamics with  
  1. **Time-series plots** (daily & monthly trends), and  
  2. **Correlation matrix** across NO₂, SO₂, CO, and O₃.  

### Workflow
1. **Data Retrieval**  
   - Download NO₂, SO₂, CO, O₃ via Earth Engine Python API.

2. **Data Processing**  
   - Gap-fill (e.g., cloud masking) and clip to city boundaries.  
   - Aggregate NO₂ to mesh grids; optionally resample to monthly means.

3. **Exploratory Analysis**  
   - **Time-Series Analysis**: plot daily & monthly pollutant trends.  
   - **Correlation Analysis**: compute Pearson/Spearman matrix and visualise as heat-map.

4. **Visualisation**  
   - Compare raw vs. filled rasters.  
   - Overlay NO₂ on urban grids.  
   - Export animated GIFs for NO₂ evolution.

### Data Process Pipeline

This notebook processes the air pollution data downloaded in *appendix_preparation.ipynb* through the following steps:

- **(1) Filling Missing Value**: Spot the missing values in raster and replenish them using iterative filling, using **mean** of the neighbour raster as the replenish value.

- **(2) Clipping to Region**: Clipping the data to the interested area, and output the filled raster.

- **(3) Aggregation**: Import the generated mesh and aggregate the raster to the mesh level.

Step 2 and 3 are realised by selecting and aggregating the data within the mesh grid. 

### Outputs
| Type | File | Description |
|------|------|-------------|
| Raster | `no2_filled_rasters.tif` | Gap-filled NO₂ rasters |
| Vector/Tabular | `no2_mesh_data.gpkg`| Grid-level NO₂ statistics |
| Figures | `visual_comparison.png` | Raw vs. filled snapshot |
| Figures | `no2_timelapse.gif` | Time-lapse of NO₂ |
| Figures | `pollutant_timeseries.png` | Multi-pollutant time-series plots |
| Figures | `pollution_corr_heatmap.png` | Correlation matrix heat-map |

*Note: no2 above can be changed to other types of air pollution*

### Notes
- The NO₂ filling step is computationally heavy and may take 8+ hours per city for full-year data.

- Data quality is significantly improved after gap-filling, especially in cloud-obscured areas.


## 0 Init: Prepare Packages and Configuration

Get current file/repo/data path in local to make sure the following cells run properly.


```python
import sys
from pathlib import Path
SRC_PATH = Path().resolve().parent / "src"
sys.path.append(str(SRC_PATH))

from config import *
from missingvalue import*
from visualization import*
```

## 1  Nitrogen Dioxide NO₂

### 1.1 Fill Missing Data


```python
eth_tiff_path = DATA_PATH / 'Ethiopia-no2'
fill_missing_data('Ethiopia', data_tiff_path=eth_tiff_path, output_path=DATA_PATH)
```

**Visualisation**

Now demonstrate the raster before and after the missing value. Use *Ethiopia_no2_2023-01-01.tif* file as an example to show what this missing data process loop does.


```python
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
# plot original image
src, band, profile, nodata_value = read_tiff(DEMO_PATH / 'Ethiopia_NO2_2023-01-01.tif')
plot_raster(band, percent_clip=0.5, ax=axes[0], title="Original Satellite Image")

# filled image
src, band_filled, profile, nodata_value = read_tiff(DEMO_PATH / 'Ethiopia_NO2_2023-01-01_filled.tif')
plot_raster(band_filled, percent_clip=0.5, ax=axes[1], title="Filled Values Satellite Image")

plt.show()
```


    
![png](01_air_pollution_data_preparation_files/01_air_pollution_data_preparation_8_0.png)
    


From the figure on the left, there are missing values in the plot, represented by the small white regions.  It is clear that the small missing parts in the original figure are filled perfectly.

Iraq - Baghdad


```python
from missingvalue import fill_missing_data

iraq_tiff_path = DATA_PATH / 'Iraq-no2'
fill_missing_data('Iraq', data_tiff_path=iraq_tiff_path, output_path=DATA_PATH)
```

**Visualisation**

Use *Iraq_NO2_2023-01-01.tif* file as an example to show the results of filling missing value.


```python
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
# plot original image
src, band, profile, nodata_value = read_tiff(DEMO_PATH / 'Iraq_NO2_2023-01-01.tif')
plot_raster(band, percent_clip=0.5, ax=axes[0], title="Original Satellite Image")

# filled image
src, band_filled, profile, nodata_value = read_tiff(DEMO_PATH / 'Iraq_NO2_2023-01-01_filled.tif')
plot_raster(band_filled, percent_clip=0.5, ax=axes[1], title="Filled Values Satellite Image")

plt.show()
```


    
![png](01_air_pollution_data_preparation_files/01_air_pollution_data_preparation_14_0.png)
    


Similarly, the small missing value areas on the upper right are filled perfectly. Thus, the iterative filling method can significantly enhance the data quality.

### 1.2 Aggregate Based on Mesh Grid


```python
from aggregation import*

addis_meshes_path = DATA_PATH / 'addis-mesh-data'
baghdad_meshes_path = DATA_PATH / 'baghdad-mesh-data'

mesh_addis = DATA_PATH / "mesh-grid" / "grid_addis_ababa.gpkg"
mesh_baghdad = DATA_PATH / "mesh-grid" / "grid_baghdad.gpkg"

lyr_addis_name = fiona.listlayers(mesh_addis)[0]         # control layer number = 1 
lyr_baghdad_name = fiona.listlayers(mesh_baghdad)[0]
```

Ethiopia - Addis Ababa


```python
# Aggregate Ethiopia - Addis Ababa
eth_no2_filled_path = DATA_PATH / 'Ethiopia-no2-filled'
aggregate_data(
    data_tiff_path=eth_no2_filled_path, 
    mesh_path=addis_meshes_path, 
    layer_name=lyr_addis_name,
    feature_name="no2_mean"
    )
```

**Visualisation**

Show aggregated result in 2023-01-01 in Addis Ababa.


```python
demo_mesh = gpd.read_file(DEMO_PATH / 'addis-ababa-2023-01-01.gpkg')
plot_mesh(mesh=demo_mesh, feature='no2_mean', title="Addis Ababa NO$_2$ Concentration", show_edges=False)
plt.show()
```


    
![png](01_air_pollution_data_preparation_files/01_air_pollution_data_preparation_21_0.png)
    


Iraq - Baghdad


```python
# Aggregate Iraq - Baghdad
iraq_no2_filled_path = DATA_PATH / 'Iraq-no2-filled'
aggregate_data(
    data_tiff_path=iraq_no2_filled_path, 
    mesh_path=baghdad_meshes_path, 
    layer_name=lyr_baghdad_name,
    feature_name="no2_mean"
    )
```

**Visualisation**

Show aggregated result in 2023-01-01 in Baghdad.


```python
demo_mesh = gpd.read_file(DEMO_PATH / 'baghdad-2023-01-01.gpkg')
plot_mesh(mesh=demo_mesh, feature='no2_mean', title="Baghdad NO$_2$ Concentration", show_edges=False)
plt.show()
```


    
![png](01_air_pollution_data_preparation_files/01_air_pollution_data_preparation_25_0.png)
    


### 1.3 Data Visualisation

This chapter is used to generate a dynamic figure, to show how the NO₂ distribution changes over time.

Note:

- In the coloration system, percentile clipping and contrast stretching method is used to improve the visual effects of the image.

- In this chapter, the dynamic distribution of NO₂ is generated, in format of GIF. 

#### Dynamic NO₂ Distribution - Country Level

**Ethiopia - Addis Ababa**


```python
no2_eth_tif_dir = DATA_PATH / 'Ethiopia-no2-filled'  
tiff_2_gif(no2_eth_tif_dir, output_path=DATA_PATH, output_name="ethiopia-no2-animation", fps = 8)
```

    Scanning percentiles: 100%|██████████| 131/131 [00:13<00:00,  9.96it/s]
    


    
![png](01_air_pollution_data_preparation_files/01_air_pollution_data_preparation_29_1.png)
    


    Animation saved to: D:\Projects\MSc_Group_Project\air-pollution-mobility-research-project\data\animation-output\ethiopia-no2-animation.gif
    

**Iraq - Baghdad**


```python
no2_iraq_tif_dir= DATA_PATH / 'Iraq-no2-filled'  
tiff_2_gif(no2_iraq_tif_dir, output_path=DATA_PATH, output_name="iraq-no2-animation", fps = 8)
```

    Scanning percentiles: 100%|██████████| 720/720 [00:38<00:00, 18.75it/s]
    


    
![png](01_air_pollution_data_preparation_files/01_air_pollution_data_preparation_31_1.png)
    


    Animation saved to: D:\Projects\MSc_Group_Project\air-pollution-mobility-research-project\data\animation-output\iraq-no2-animation.gif
    

#### Dynamic NO₂ Mesh - City Level

**Ethiopia - Addis Ababa**


```python
addis_gpkg_path = DATA_PATH / 'addis-no2-mesh-data'

mesh_2_gif(
    gpkg_path=addis_gpkg_path, 
    output_path=DATA_PATH,
    output_name= "addis-ababa-no2-animation", 
    feature='no2_mean',
   )
plt.show()
```

    Scanning percentiles: 100%|██████████| 731/731 [00:07<00:00, 93.50it/s] 
    


    
![png](01_air_pollution_data_preparation_files/01_air_pollution_data_preparation_34_1.png)
    


    Animation saved to: D:\Projects\MSc_Group_Project\air-pollution-mobility-research-project\data\animation-output\addis-ababa-no2-animation.gif
    

**Iraq - Baghdad**


```python
baghdad_gpkg_path = DATA_PATH / 'baghdad-no2-mesh-data'

mesh_2_gif(
    gpkg_path=baghdad_gpkg_path, 
    output_path=DATA_PATH,
    output_name= "baghdad-no2-animation", 
    feature='no2_mean',
   )
plt.show()
```

    Scanning percentiles: 100%|██████████| 731/731 [00:24<00:00, 30.02it/s]
    


    
![png](01_air_pollution_data_preparation_files/01_air_pollution_data_preparation_36_1.png)
    


    Animation saved to: D:\Projects\MSc_Group_Project\air-pollution-mobility-research-project\data\animation-output\baghdad-no2-animation.gif
    

## 2 Ozone

### 2.1 Fill Missing Data

Ethiopia - Addis Ababa


```python
eth_tiff_path = DATA_PATH / 'Ethiopia-Ozone'
fill_missing_data('Ethiopia', data_tiff_path=eth_tiff_path, output_path=DATA_PATH, feature='Ozone')
```

Iraq - Baghdad


```python
from missingvalue import fill_missing_data

iraq_tiff_path = DATA_PATH / 'Iraq-Ozone'
fill_missing_data('Iraq', data_tiff_path=iraq_tiff_path, output_path=DATA_PATH, feature='Ozone')
```

Visualisation before and after data filling

### 2.2 Aggregate Based on Mesh Grid


```python
from aggregation import*

addis_meshes_path = DATA_PATH / 'addis-empty-mesh-data'
baghdad_meshes_path = DATA_PATH / 'baghdad-empty-mesh-data'

mesh_addis = DATA_PATH / "mesh-grid" / "grid_addis_ababa.gpkg"
mesh_baghdad = DATA_PATH / "mesh-grid" / "grid_baghdad.gpkg"

lyr_addis_name = fiona.listlayers(mesh_addis)[0]         # control layer number = 1 
lyr_baghdad_name = fiona.listlayers(mesh_baghdad)[0]
```

Ethiopia - Addis Ababa


```python
# Aggregate Ethiopia - Addis Ababa
eth_ozone_filled_path = DATA_PATH / 'Ethiopia-Ozone-filled'
aggregate_data(
    data_tiff_path=eth_ozone_filled_path, 
    mesh_path=addis_meshes_path, 
    layer_name=lyr_addis_name,
    feature_name="ozone_mean"
    )
```

Visualisation


```python
demo_mesh = gpd.read_file( DATA_PATH / 'addis-empty-mesh-data' / 'addis-ababa-2024-11-01.gpkg')
plot_mesh(mesh=demo_mesh, feature='ozone_mean', title="Addis Ababa Ozone Concentration", show_edges=False)
plt.show()
```


    
![png](01_air_pollution_data_preparation_files/01_air_pollution_data_preparation_49_0.png)
    


Iraq - Baghdad


```python
iraq_ozone_filled_path = DATA_PATH / 'Iraq-Ozone-filled'
aggregate_data(
    data_tiff_path=iraq_ozone_filled_path, 
    mesh_path=baghdad_meshes_path, 
    layer_name=lyr_baghdad_name,
    feature_name="ozone_mean"
    )
```

Visualisation


```python
demo_mesh = gpd.read_file(DATA_PATH / 'Baghdad-empty-mesh-data'/'baghdad-2023-01-02.gpkg')
plot_mesh(mesh=demo_mesh, feature='ozone_mean', title="Baghdad O$_3$ Concentration 2023-01-01", show_edges=False)
plt.show()
```


    
![png](01_air_pollution_data_preparation_files/01_air_pollution_data_preparation_53_0.png)
    


## 3 Carbon Monoxide CO

### 3.1 Filling Missing Data

Ethiopia - Addis Ababa


```python
eth_tiff_path = DATA_PATH / 'Ethiopia-CO'
fill_missing_data('Ethiopia', data_tiff_path=eth_tiff_path, output_path=DATA_PATH, feature='CO')
```

Iraq - Baghdad


```python
iraq_tiff_path = DATA_PATH / 'Iraq-CO'
fill_missing_data('Iraq', data_tiff_path=iraq_tiff_path, output_path=DATA_PATH, feature='CO')
```

### 3.2 Aggregate Based on Mesh Grid


```python
from aggregation import*

addis_meshes_path = DATA_PATH / 'addis-empty-mesh-data'
baghdad_meshes_path = DATA_PATH / 'baghdad-empty-mesh-data'

mesh_addis = DATA_PATH / "mesh-grid" / "grid_addis_ababa.gpkg"
mesh_baghdad = DATA_PATH / "mesh-grid" / "grid_baghdad.gpkg"

lyr_addis_name = fiona.listlayers(mesh_addis)[0]         # control layer number = 1 
lyr_baghdad_name = fiona.listlayers(mesh_baghdad)[0]
```

Ethiopia - Addis Ababa


```python
# Aggregate Ethiopia - Addis Ababa
eth_CO_filled_path = DATA_PATH / 'Ethiopia-CO-filled'
aggregate_data(
    data_tiff_path=eth_CO_filled_path, 
    mesh_path=addis_meshes_path, 
    layer_name=lyr_addis_name,
    feature_name="CO_mean"
    )
```


```python
demo_mesh = gpd.read_file(DATA_PATH / 'addis-empty-mesh-data'/'addis-ababa-2023-01-01.gpkg')
plot_mesh(mesh=demo_mesh, feature='CO_mean', title="Addis Ababa CO Concentration 2023-01-01 ", show_edges=False)
plt.show()
```


    
![png](01_air_pollution_data_preparation_files/01_air_pollution_data_preparation_64_0.png)
    


Iraq - Baghdad


```python
iraq_CO_filled_path = DATA_PATH / 'Iraq-CO-filled'
aggregate_data(
    data_tiff_path=iraq_CO_filled_path, 
    mesh_path=baghdad_meshes_path, 
    layer_name=lyr_baghdad_name,
    feature_name="CO_mean"
    )
```


```python
demo_mesh = gpd.read_file(DATA_PATH / 'Baghdad-empty-mesh-data'/'baghdad-2023-01-01.gpkg')
plot_mesh(mesh=demo_mesh, feature='CO_mean', title="Baghdad CO Concentration 2023-01-01 ", show_edges=False)
plt.show()
```


    
![png](01_air_pollution_data_preparation_files/01_air_pollution_data_preparation_67_0.png)
    


## 4 Sulfur Dioxide SO₂

### 4.1 Filling Missing Data

Ethiopia - Addis Ababa


```python
eth_tiff_path = DATA_PATH / 'Ethiopia-SO2'
fill_missing_data('Ethiopia', data_tiff_path=eth_tiff_path, output_path=DATA_PATH, feature='so2')
```

Iraq - Baghdad


```python
iraq_tiff_path = DATA_PATH / 'Iraq-SO2'
fill_missing_data('Iraq', data_tiff_path=iraq_tiff_path, output_path=DATA_PATH, feature='so2')
```

### 4.2 Aggregate Based on Mesh Grid


```python
from aggregation import*

addis_meshes_path = DATA_PATH / 'addis-empty-mesh-data'
baghdad_meshes_path = DATA_PATH / 'baghdad-empty-mesh-data'

mesh_addis = DATA_PATH / "mesh-grid" / "grid_addis_ababa.gpkg"
mesh_baghdad = DATA_PATH / "mesh-grid" / "grid_baghdad.gpkg"

lyr_addis_name = fiona.listlayers(mesh_addis)[0]         # control layer number = 1 
lyr_baghdad_name = fiona.listlayers(mesh_baghdad)[0]
```

Ethiopia - Addis Ababa


```python
# Aggregate Ethiopia - Addis Ababa
eth_so2_filled_path = DATA_PATH / 'Ethiopia-so2-filled'
aggregate_data(
    data_tiff_path=eth_so2_filled_path, 
    mesh_path=addis_meshes_path, 
    layer_name=lyr_addis_name,
    feature_name="so2_mean"
    )
```

Visualisation


```python
demo_mesh = gpd.read_file(DATA_PATH / 'addis-empty-mesh-data'/'addis-ababa-2023-01-01.gpkg')
plot_mesh(mesh=demo_mesh, feature='so2_mean', title="Addis Ababa SO$_2$ Concentration 2023-01-01 ", show_edges=False)
plt.show()
```


    
![png](01_air_pollution_data_preparation_files/01_air_pollution_data_preparation_79_0.png)
    


Iraq - Baghdad


```python
iraq_so2_filled_path = DATA_PATH / 'Iraq-so2-filled'
aggregate_data(
    data_tiff_path=iraq_so2_filled_path, 
    mesh_path=baghdad_meshes_path, 
    layer_name=lyr_baghdad_name,
    feature_name="so2_mean"
    )
```

Visualisation


```python
demo_mesh = gpd.read_file(DATA_PATH / 'Baghdad-empty-mesh-data'/'baghdad-2023-01-02.gpkg')
plot_mesh(mesh=demo_mesh, feature='so2_mean', title="Baghdad SO$_2$ Concentration 2023-01-01", show_edges=False)
plt.show()
```


    
![png](01_air_pollution_data_preparation_files/01_air_pollution_data_preparation_83_0.png)
    


## 5 Air Pollution Trend Analysis

Analysis all the air pollution types in both area to see if any changes happened in the last two years.

#### 5.1 Addis Ababa


```python
# Addis Ababa
from helpercollections import merge_multiple_gpkgs, convert_gpkgs_to_parquet

FILE_PATH = DATA_PATH / "additional-air-pollution"
addis_feature_mesh_paths = [DATA_PATH / "addis-no2-mesh-data",          # NO2 data
                            FILE_PATH / "addis-SO2-mesh-data",          # SO2 data 
                            FILE_PATH / "addis-CO-mesh-data",           # SO2 data 
                            FILE_PATH / "addis-ozone-mesh-data",        # O3 data 
                            ]
output_folder = FILE_PATH / "addis-air-pollutant-mesh-data"
merge_multiple_gpkgs(addis_feature_mesh_paths, output_folder)
convert_gpkgs_to_parquet(mesh_folder=output_folder, output_path=FILE_PATH, file_name="addis_air_pollutant_df") 
```

Visualize the correlation matrix & changing trend


```python
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

FILE_PATH = DATA_PATH / "additional-air-pollution"
addis_air_pollution = pd.read_parquet(path=FILE_PATH / "addis_air_pollutant_df.parquet")
addis_air_df = addis_air_pollution.fillna(addis_air_pollution.mean(numeric_only=True))  # fill NA

addis_air_df = addis_air_df.groupby(['date'])[['no2_mean',	'so2_mean',	'CO_mean',	'ozone_mean',]].sum().rolling(window=60, center=True).mean() # Smooth
addis_air_plt = addis_air_df.rename(columns={'no2_mean':"NO2 level",	'so2_mean':"SO2 level",	'CO_mean':"CO level", 'ozone_mean':"Ozone level"})
```

##### Time Series Visualization


```python
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

plt.style.use('default')

scaler = MinMaxScaler(feature_range=(1,100))
addis_scaled_data = pd.DataFrame(scaler.fit_transform(addis_air_plt.dropna()), columns=addis_air_plt.columns, index=addis_air_plt.dropna().index)

colors = cm.viridis(np.linspace(0, 1, len(addis_scaled_data.columns)))

addis_scaled_data.plot(color=colors, figsize=(12, 6))
plt.title('Temporal Trend of Air Pollution in Addis Ababa')
plt.xlabel('Date')
plt.ylabel('Air Pollution Level (scaled 0-100)')
plt.grid(True)
plt.legend(loc='lower left')
plt.tight_layout()
plt.savefig(DEMO_PATH / 'Temporal Trend of Air Pollution in Addis Ababa.png', dpi=700)
plt.show()
```


    
![png](01_air_pollution_data_preparation_files/01_air_pollution_data_preparation_91_0.png)
    


##### Correlation Visualization


```python
from visualization import plot_corr_matrix
plot_corr_matrix(addis_air_plt, cols_of_interest=addis_air_plt.columns,output_path=DEMO_PATH,  plot_name='Air Pollution Level in Addis Ababa', figsize=(8, 6))
```


    
![png](01_air_pollution_data_preparation_files/01_air_pollution_data_preparation_93_0.png)
    


    Heatmap saved to D:\Projects\MSc_Group_Project\air-pollution-mobility-research-project\data\demo-data\Air Pollution Level in Addis Ababa.png
    

#### 5.2 Baghdad


```python
# Baghdad
from helpercollections import merge_multiple_gpkgs, convert_gpkgs_to_parquet
import geopandas as gpd

FILE_PATH = DATA_PATH / "additional-air-pollution"
baghdad_feature_mesh_paths = [DATA_PATH / "baghdad-no2-mesh-data",        # NO2 data
                            FILE_PATH / "baghdad-SO2-mesh-data",          # SO2 data 
                            FILE_PATH / "baghdad-CO-mesh-data",           # SO2 data 
                            FILE_PATH / "baghdad-ozone-mesh-data",        # O3 data 
                            ]
output_folder = FILE_PATH / "baghdad-air-pollutant-mesh-data"
merge_multiple_gpkgs(baghdad_feature_mesh_paths, output_folder)
convert_gpkgs_to_parquet(mesh_folder=output_folder, output_path=FILE_PATH, file_name="baghdad_air_pollutant_df.parquet") 
```

    Progress: 100%|██████████| 731/731 [03:10<00:00,  3.84it/s]
    100%|██████████| 731/731 [00:25<00:00, 28.30it/s]
    D:\Projects\MSc_Group_Project\air-pollution-mobility-research-project\src\helpercollections.py:925: FutureWarning: The behavior of DataFrame concatenation with empty or all-NA entries is deprecated. In a future version, this will no longer exclude empty or all-NA columns when determining the result dtypes. To retain the old behavior, exclude the relevant entries before the concat operation.
      full_df = pd.concat(all_data, ignore_index=True)
    

Visualize the correlation matrix & changing trend


```python
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

FILE_PATH = DATA_PATH / "additional-air-pollution"
baghdad_air_pollution = pd.read_parquet(path=FILE_PATH / "baghdad_air_pollutant_df.parquet")
baghdad_air_df = baghdad_air_pollution.fillna(baghdad_air_pollution.mean(numeric_only=True))  # fill NA

baghdad_air_df = baghdad_air_df.groupby(['date'])[['no2_mean',	'so2_mean',	'CO_mean',	'ozone_mean',]].sum().rolling(window=60, center=True).mean() # Smooth
baghdad_air_plt = baghdad_air_df.rename(columns={'no2_mean':"NO2 level",	'so2_mean':"SO2 level",	'CO_mean':"CO level", 'ozone_mean':"Ozone level"})
```

##### Time Series Visualization


```python
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

plt.style.use('default')

scaler = MinMaxScaler(feature_range=(1,100))
baghdad_scaled_data = pd.DataFrame(scaler.fit_transform(baghdad_air_plt.dropna()), columns=baghdad_air_plt.columns, index=baghdad_air_plt.dropna().index)

colors = cm.viridis(np.linspace(0, 1, len(baghdad_scaled_data.columns)))

baghdad_scaled_data.plot(color=colors, figsize=(12, 6))
plt.title('Temporal Trend of Air Pollution in Baghdad')
plt.xlabel('Date')
plt.ylabel('Air Pollution Level (scaled 0-100)')
plt.grid(True)
plt.legend(loc='lower left')
plt.tight_layout()
plt.savefig(DEMO_PATH / 'Temporal Trend of Air Pollution in Baghdad', dpi=700)
plt.show()
```


    
![png](01_air_pollution_data_preparation_files/01_air_pollution_data_preparation_99_0.png)
    


##### Correlation Visualization


```python
from visualization import plot_corr_matrix
plot_corr_matrix(baghdad_air_plt, cols_of_interest=baghdad_air_plt.columns,output_path=DEMO_PATH,  plot_name='Air Pollution Level in Baghdad', figsize=(8, 6))
```


    
![png](01_air_pollution_data_preparation_files/01_air_pollution_data_preparation_101_0.png)
    


    Heatmap saved to D:\Projects\MSc_Group_Project\air-pollution-mobility-research-project\data\demo-data\Air Pollution Level in Baghdad.png
    
