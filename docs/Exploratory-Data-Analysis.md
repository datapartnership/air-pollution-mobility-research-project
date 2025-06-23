# 4. Exploratory Data Analysis

## 4.1 Addis Ababa

### 4.1.1 NO<sub>2</sub>

![](/docs/images/EDA/fig1.png)

For both weekdays and weekends, there is a prominent concentration of NO<sub>2</sub> in the northwestern region. This region, highlighted in pink in Figure 4.3, is the heart of Addis Ababa. It is where most of the roads converge. Most of the Addis Ababa Light Rail System is located here as evidenced by Figure 4.2. The Addis Ababa Bole International (ADD) Airport is located towards the southeast of this region as shown in Figure 4.3. The figure also shows the major highway corridors of A1 and A2. The A1 highway connects Addis Ababa to Dire Dawa and Djibouti, and it is a part of the Trans African Highway network. These mobility systems may be major contributing factors to the NO<sub>2</sub> levels in this region. The southeast and eastern parts show markedly lower NO<sub>2</sub> levels. This area is mostly marked by extensive cropland, indicating higher agricultural activity rather than industrial production. The extreme northern and northeastern border regions also show lower NO<sub>2</sub> levels which is in line with the region being forest land.

![](/docs/images/EDA/fig2.png)

On closer inspection of Figure 4.1, we also notice a variation in NO<sub>2</sub> levels on weekends as compared to weekdays. This may be associated with lower economic activity on the weekends such as less people travelling to work and school, businesses being shut, and increased home-based activities. 

![](/docs/images/EDA/fig3.png)

A comparison is made for NO<sub>2</sub> levels in 2023 and 2024 in Figure 4.4. As expected, weekday NO<sub>2</sub> levels remain higher due to commuter traffic, especially in the northwest core region. On weekends, the NO<sub>2</sub> levels are lower with a smoother spatial distribution. The NO<sub>2</sub> level for both workdays and weekends rose in 2024, this is likely from expanding road networks and vehicle numbers. Weekend levels have also increased, suggesting growing leisure, delivery, and industrial activities outside workdays due to the expansion.

The daily subregional NO<sub>2</sub> levels have been attached in Appendix Figure 3. All the subregions consistently show a spike in NO<sub>2</sub> levels in February-March and November-December, with a marked decrease in August. In Ethiopia, February-March and November-December fall in the dry season; with dry, stable atmospheric conditions and low wind which can trap pollutants near the surface due to poor vertical mixing. The period also coincides with pre-holiday commerce and traffic activity due to festivals and the peak tourist season. On the other hand, August sees the highest rainfall in Ethiopia, which may be the reason for the drop in NO<sub>2</sub> levels due to washing away of NO<sub>2</sub>, as highlighted in section 2.2. 

The NO<sub>2</sub> levels in the city are also analysed using autocorrelation. Temporal autocorrelation and spatial correlation are computed to investigate the correlation of NO<sub>2</sub> with itself over time and over the regions of interest. The NO₂ levels in Addis Ababa exhibit short-term autocorrelation, supporting the use of low-order autoregressive models for prediction.

![](/docs/images/EDA/fig4.png)

As evidenced by Figure 4.5, there is a strong positive partial autocorrelation at lag 1, indicating significant short-term persistence, but the influence drops off rapidly after a lag of one day. This is in line with NO<sub>2</sub>’s short lifetime of less than a day. Minor peaks at lags 5, 10, and 15 suggest possible weekly traffic cycles, while the confidence interval (±1 standard deviation of PACF values across all cells) reflects spatial heterogeneity across grid cells. 

![](/docs/images/EDA/fig5.png)

The spatial correlation in Figure 4.6 shows the correlation of NO₂ level with its neighbouring locations over two years between 2023 and 2024. The pollution is spatially dependent and concentrated, where high local spatial autocorrelation is concentrated in the western and central areas. High Local Moran’s I values cluster around the city centre and in the outer region, indicated by dark red and black zones. The high Local Moran’s I value at the centre, around 3.0, indicates strong local clustering properties around the area. The spatial gradient of NO₂ clustering indicates the potential for local spillover effects, where pollution in one area affects nearby regions due to consistent localised resources, such as daily traffic emissions. 

![](/docs/images/EDA/fig6.png)

In contrast to 2023, several regions in the northeast, southeast outskirts and eastern suburban zones transitioned from low/no autocorrelation (yellow/white) to moderate–high autocorrelation (orange/red) in 2024. This is shown by Figure 4.7. In 2023, the high-local Moran's I values are slightly more concentrated. On the other hand, in 2024, the high autocorrelation zone appears to diffuse slightly outward.

### 4.1.2 Population

The population distribution for 2023 and 2024 also shows a similar trend to NO<sub>2</sub>, i.e., a possibly high correlation to the transport network and industrial production regions. Particularly, two offshoots are observed spreading from the northwest horizontally to the east and diagonally to the south. For the first case, the presence of the East-West line of the LRT corridor and A1 highway may be a possible reason. For the second case, higher population is observed along the North-South line of the LRT followed by the A1 highway corridor. 

Similar to NO<sub>2</sub>, lower population numbers are observed in croplands and forested regions. Between 2023 and 2024, the total population in Addis Ababa increased by 2.17% to about 4,281,917. The boxplot in Appendix Figure 1 shows a mean of 7336 in 2023 and 7842 in 2024. The histogram in Appendix Figure 2 shows that a considerable number of cells have population values below 5000.

![](/docs/images/EDA/fig7.png)

### 4.1.3 Land Use (Industrial)

The industrial land use area distribution as shown in Figure 4.9 also shows higher values along the A1 highway corridor. Furthermore, higher industrial area towards the east can be attributed to the Special Economic Zone (SEZ) of Bole Lemi as shown in Figure 4.10.

![](/docs/images/EDA/fig8.png)

### 4.1.4 Points of Interest

The points of interest data was processed to obtain the total percentage of POI expected to produce considerable amounts of NO<sub>2</sub> due to industrial activity and/or mobility. These POI included supermarkets, hospitals, car dealerships, car rentals, carwash, marketplaces, malls, and universities. Comparing Figures 4.3 and 4.11, a higher concentration of POI is observed in the heart of Addis Ababa.

![](/docs/images/EDA/fig9.png)

### 4.1.5 Road Length

The total road length per cell was calculated in metres per cell as shown in Figure 4.12. Comparing this with Figures 4.2, 4.3, and 4.8, we observe a correlation between the persistence of roads, LRT, and population distribution.

![](/docs/images/EDA/fig10.png)

### 4.1.6 Distance to Primary Road

To investigate the effects of distance from the closest primary road, the cell with the primary road is categorised as 1, while the ones that do not contain primary roads are categorised as 0. 

Figure 4.13 shows that the cells in the northwest part of Addis Ababa have the most access to the primary road. The ones in the southeast part of the city, in Akaki Kality, tend to have a larger distance (around 6000 m) from the nearest primary road.

![](/docs/images/EDA/fig11.png)

### 4.1.7 Land Surface Temperature

The maps in Figures 4.14 and 4.15 illustrate the spatial distribution of average land surface temperature (LST) in Addis Ababa for the years 2023 and 2024, derived from MODIS Terra (measured at 10:30 AM) and Aqua (measured at 1:30 PM) satellites. As expected, Aqua-based maps (Figure 4.14) show higher temperatures than Terra (Figure 4.15), due to increased solar heating by early afternoon. 

![](/docs/images/EDA/fig12.png)

![](/docs/images/EDA/fig13.png)

A consistent pattern is observed across both satellite products: surface temperatures in 2024 are generally higher than in 2023, particularly during weekends. The southern and southeastern zones of the city consistently exhibit elevated temperatures, likely corresponding to lower-elevation and less vegetated areas. Meanwhile, the northern highland areas of Addis Ababa remain relatively cooler throughout all scenarios, possibly due to the existing tree cover.

A subtle yet noticeable difference appears between workday and weekend temperatures, especially in the Aqua maps. Weekend heating is more intense, particularly over the central region and peripheral urban districts along the highway corridors. 

The observed increase in temperature distribution across the city is largely attributed to anthropogenic factors (McCarthy, 2001). Migration, political centralisation, and economic opportunities within the city have led to rapid urbanisation and population growth, resulting in a decrease in vegetation and green spaces alongside increased human and vehicular activity, including commercial and transportation emissions. The high congestion in the city centre reflects the key urban challenges facing Ethiopia, where a combination of inadequate public transport systems, insufficient infrastructure for non-motorised transport, and unplanned spatial expansion causes the traffic congestio (UN-Habitat, 2025).

### 4.1.8 Cloud Cover Category

As illustrated in Figure 4.17, in both years, the southern and southeastern parts of the city consistently experience clearer skies, while the northern and central zones tend to be cloudier.

Cloud cover towards the north is influenced by a combination of factors related to the city's high altitude and the surrounding topography. The interaction of warm, moist air rising over the Ethiopian highlands can lead to condensation and cloud formation as warm, humid air is forced upwards by the terrain. The hills and mountains in the northern part of the city create a natural barrier for airflows, forcing warm, moist air upwards, causing it to cool and condense into clouds (Figure 4.16).

![](/docs/images/EDA/fig14.png)

![](/docs/images/EDA/fig15.png)

### 4.1.9 Land Cover (ESA) 

As shown in Figure 4.18, most built-up regions are centred around the central and northeast part of the city, and the southeast part of the region is mostly cropland. Tree cover, indicated in dark green, is mainly concentrated in the north and scattered across the central-west areas, while the grassland is spread around the city. 

Shrubland and grassland are primarily found in less densely developed urban areas, and bare or sparsely vegetated zones are located along the southern and eastern edges of the city. The areas of water bodies and wetlands are limited, with only small and isolated patches of blue and teal barely visible on the map.

![](/docs/images/EDA/fig16.png)

## 4.2 Baghdad

### 4.2.1 NO<sub>2</sub>

The weekend in Iraq is Friday-Saturday. For the purpose of subsequent traffic congestion analysis, the week has been divided into three parts: workday (Monday-Wednesday), extended workday (Thursday, Sunday), and weekend (Friday-Saturday). As mentioned in section 2.1.2, the area under consideration is the Baghdad Governorate. 

The spatial distribution of NO<sub>2</sub> concentrations in Baghdad for 2023 and 2024 highlights persistently elevated emissions in the urban core during workdays and extended workdays, correlating with traffic density and continuous industrial activity. 

Workday averages consistently show higher NO₂ levels around the city centre, reflecting the increase in vehicular and factory operations. Workday emissions in 2024, as indicated in Figure 4.19, contrasted with a modest increase in weekend pollutant levels. The increasing central-city pollution is mostly attributed to ongoing activities in vehicular traffic and industrial facilities, reinforcing the effects of daily urban mobility and extended working hours on NO<sub>2</sub> pollution.

![](/docs/images/EDA/fig17.png)

As highlighted in Figure 4.20, the central region is the city of Baghdad. It is the capital and largest city of Iraq and thus, the nation’s economic hub. The green spot in the figure indicates Baghdad International Airport (BGW). As shown in Figure 4.21, moving towards the south from the city centre is a major mobility corridor formed by the Baghdad–Basra Railway Line, and Highway 8 and Motorway 1 (M1) which connect Baghdad to Kuwait. The figure also shows Highway 6 which goes from the city centre towards the southeast to Al Kut and connects Baghdad to Tehran. 

![](/docs/images/EDA/fig18.png)

As shown in Figure 4.19, the highest NO<sub>2</sub> concentrations are observed in Baghdad city centre; where most of the urban activity is concentrated. Considerable emissions are also observed as we move towards the southeast, along the Highway 6 corridor. Similar to Addis Ababa, there is a slight variation in emissions for weekdays and weekends. The reasons for this variation may also be similar. 

The daily subregional NO<sub>2</sub> levels attached in Appendix Figure 4 show that emission levels in the outer regions are fairly consistent throughout the year. On the other hand, for the central regions, the emissions profile is more peaky, with spikes often seen between July-September. These months are marked by high temperatures and clear skies along with power cuts. This leads to increased electricity demand for air conditioning and usage of diesel generators. The period also coincides with Muharram, which often involves communal events and processions drawing large crowds.

![](/docs/images/EDA/fig19.png)

Similar to Addis Ababa, the NO<sub>2</sub> levels in Baghdad show short-term correlation. The average PACF value of Baghdad shows low autocorrelation at lag 1, with all subsequent lags hovering near zero. The narrower confidence interval in Baghdad suggests more homogeneous and random temporal behaviour. The smooth PACF at later lags imply the NO₂ concentrations may change independently from one day to the next, indicating that episodic factors, such as weather, likely influence NO<sub>2</sub> levels in Baghdad. 

Figure 4.23 shows a much sharper and more concentrated hotspot in the spatial autocorrelation of Baghdad.  Outside this zone, spatial autocorrelation is nearly zero. The local Moran’s I value at the centre of Baghdad is much higher than that of Addis Ababa. With a value of 8 at the city centre, Baghdad has a very strong local hotspot compared to Addis Ababa, which has more moderate clustering spread over a larger area.

![](/docs/images/EDA/fig20.png)

Figure 4.23 above shows the highest autocorrelation in central-eastern Baghdad, with the highest local Moran’s I value in the city centre being higher than 6.0. The orange/yellow zones (Local Moran’s I ≈ 2–4) in the northwest and southwest parts of the city appear to have widened in 2024, indicating a stronger consistency in NO<sub>2</sub> concentrations. On the other hand, there is less extreme clustering within the center in 2024 compared to the previous year. This shift suggests that NO<sub>2</sub> pollution has become more spatially diffuse, likely due to the redistribution of traffic as well as meteorological factors (e.g., temperature inversions) that trap pollutants more broadly (iData, 2024).

### 4.2.2 Population

![](/docs/images/EDA/fig21.png)

In Baghdad, the population distribution shows a similar profile to NO<sub>2</sub> emissions distribution, with a higher concentration in the city centre. The population is particularly high in Sadr City and Al Baladiyat to the north of the Army Canal, parallel to the Tigris River. Closer inspection shows considerable population concentration along the Tigris and Highway 6 to the southeast and along the rail line in the south.

The population distribution is much lower in the outer regions, which are mainly defined by rangeland and cropland. The population increased by 199,125 between 2023 and 2024, a 2.10% increase. The boxplot in Appendix Figure 1 shows a mean value of 1547 in 2023 and 1579 in 2024. However, a large number of cells lie beyond the range. This, coupled with the histogram in Appendix Figure 2, indicates a few cells have very large population values compared to others.

### 4.2.3 Industrial Area and Points of Interest

he obtained industrial area data shows sparseness, as does the points of interest data. The trend of higher human activity concentrations in the city centre and along the highway corridors con-tinues to prevail. 

![](/docs/images/EDA/fig22.png)

### 4.2.4 Road Length

The total road length per cell in the grid shows a stark resemblance to the population distribu-tion and NO<sub>2</sub> emissions plot. This shows a strong correlation between human settlement, mobil-ity, and air pollution.

![](/docs/images/EDA/fig23.png)

### 4.2.5 Traffic Congestion Intensity (TCI)

Similar to NO<sub>2</sub>, the average Traffic Congestion Intensity (TCI) in Baghdad is analysed across three temporal categories: regular workdays, extended workdays, and weekends. 

On regular workdays, the congestion index is highest in the city centre and along major arterial roads, indicating significant commuting activity concentrated in core business zones. During extended workdays, the spatial extent of congestion expands slightly, suggesting the potential of longer working hours or late-day activities. In contrast, the congestion index is considerably lower on weekends in both intensity and spatial coverage, although the city centre still exhibits moderate levels of TCI. 

![](/docs/images/EDA/fig24.png)

### 4.2.6 Land Surface Temperature

Similar to Addis Ababa, the spatial distributions of average land surface temperatures (LST) from MODIS Terra (10:30 AM overpass) and Aqua (1:30 PM overpass) satellites are considered. A clear seasonal and temporal gradient is visible for both satellites, with Figure 4.30 showing that the average temperature in 2024 is slightly higher across all day types. 

Higher temperatures tend to concentrate in the city centre and industrial zones, while peripheral areas remain relatively cooler. Weekday maps display intensified heating in the western, central, and eastern Baghdad. Weekend maps show a modest temperature increase, particularly in the city centre, indicating the potential of enhanced anthropogenic heat emissions from commercial, traffic, and industrial activity in the city centre. The higher temperature in the eastern part of the city can be attributed to sparse vegetation.

The results of the temperature distribution within the city center are supported by Alshammari et al. (2025), who used Sentinel-3 satellite Land Surface Temperature (LST) data and OpenStreetMap urban infrastructure data to confirm a notable rise in temperatures of up to 1.3°C in Baghdad, attributed to the urban heat island effect. 

![](/docs/images/EDA/fig25.png)

![](/docs/images/EDA/fig26.png)

### 4.2.7 Cloud Cover Category

In this section, the most frequently occurring cloud cover category within each cell is analysed. Fig. 4.31 shows consistent clear-sky conditions (category 0) across all days of the week and the entire city for each year. The result reflects the weather condition in Baghdad, where the country experiences hot, dry summers characterised by clear skies. In winter, Baghdad sees some rainfall, although most of the year the sky remains clear. With an almost entirely flat and low-lying terrain, the city is less likely to form clouds.

![](/docs/images/EDA/fig27.png)

### 4.2.8 Land Cover (ESA)

Figure 4.32 shows that the developed area in Baghdad is concentrated along the Tigris River and in the city’s central region, where there are extensive vegetation and water-related land cover areas compared to Addis Ababa. 

Extensive areas of tree cover and shrubland are particularly concentrated along the Tigris River and in the northern and eastern parts of the city. Cropland is widespread in all directions, especially east and south, which is consistent with the city's location within the Mesopotamian agricultural plain. Grassland and bare or sparsely vegetated areas are also prominent, especially toward the outer city limits. Wetlands are especially visible in the southern and eastern sections, which aligns with the seasonal flooding patterns and marshland ecosystems characteristic of the Tigris-Euphrates river basin.

![](/docs/images/EDA/fig28.png)

### 4.3 Multivariate Correlation Analysis

To explore interdependencies between different features, correlation matrices were computed at the spatial grid-cell level. This analysis helps identify which factors are most strongly associated with NO<sub>2</sub> concentrations, providing insight into potential drivers of urban air pollution. 

We calculated pairwise Pearson correlation coefficients among the selected variables. The correlation matrix quantifies linear relationships, where values range from -1 (perfect negative) to +1 (perfect positive). Strong positive correlations suggest shared spatial patterns or common underlying causes. Weak or negative correlations may point to inverse relationships or independent dynamics.

The resulting matrix is visualised using heatmaps to facilitate interpretation. This allows us to:

1.	Identify variables with the strongest associations to NO2.

2.	Detect multicollinearity, which informs feature selection in modeling.

3.	Understand cross-variable interactions that shape pollution distribution.

This correlation analysis supports subsequent explanatory modeling and helps prioritise key predictors for intervention or policy consideration. The results are attached in the Appendix as Figures 5 and 6.

### 4.4 Air Pollutant Analysis

An analysis of different air pollutants, namely NO<sub>2</sub>, SO<sub>2</sub>, CO, and ozone, was done for both cities in 2023 and 2024. This was done to ascertain consistency between air pollutant trends.

### 4.4.1 Air Pollution Level in Addis Ababa

The correlation matrix for air pollution levels in Addis Ababa reveals significant interdependencies among key atmospheric pollutants. Notably, nitrogen dioxide shows a strong negative correlation with sulfur dioxide (SO<sub>2</sub>) (r = –0.71) and ozone (O<sub>3</sub>) (r = –0.62), indicating that periods of elevated NO<sub>2</sub> concentrations are typically associated with decreased levels of these two pollutants. Conversely, NO<sub>2</sub> exhibits a moderate positive correlation with carbon monoxide (CO) (r = 0.59), suggesting shared anthropogenic sources such as vehicular emissions. 

SO<sub>2</sub> and O<sub>3</sub> are positively correlated (r = 0.77), as are SO<sub>2</sub> and CO (r = –0.71), although the latter indicates a potential source divergence. The strongest negative correlation in the matrix is observed between CO and O<sub>3</sub> (r = –0.72), suggesting the hypothesis of contrasting seasonal behaviours and atmospheric dynamics between primary and secondary pollutants. 

![](/docs/images/EDA/fig29.png)

### 4.4.2 Temporal Trend of Air Pollution in Addis Ababa

The temporal trend of air pollution in Addis Ababa demonstrates variations across different pollutants, as illustrated in Figure 4.34. According to the result of smoothed moving average modelling, nitrogen dioxide and sulfur dioxide levels exhibit relatively high variability throughout the year. NO<sub>2</sub> peaks in early 2024 and late 2023, indicating the possibility of heightened vehicular or industrial activity during those periods. 

Conversely, SO<sub>2</sub> shows a major rise during the late months of 2023, followed by a decline toward the year's end. Carbon monoxide displays a pronounced seasonal cycle, reaching its highest concentrations around March 2023. Ozone levels present an inverse pattern compared to other pollutants, peaking around June and July, when photochemical reactions are enhanced by higher solar radiation. 

The SO<sub>2</sub> concentrations exhibit periods of partial synchrony with NO<sub>2</sub> concentration, particularly during early 2023 and early 2024, suggesting overlapping emission sources such as vehicular traffic and industrial combustion. However, this relationship is not consistent year-round. Ozone concentration, in contrast, displays a distinct inverse relationship with NO<sub>2</sub> concentrations, with O<sub>3</sub> concentrations peaking during the summer months when NO<sub>2</sub> levels are at their lowest. This pattern reflects the photochemical processes in which NO<sub>2</sub> participates in ozone formation but is also depleted by it, especially under high solar radiation. Carbon monoxide concentration generally trends with that of NO<sub>2</sub> during colder months, likely due to common sources such as incomplete combustion from heating and transport.

![](/docs/images/EDA/fig30.png)

### 4.4.3 Air Pollution Level in Baghdad

The correlation matrix of air pollution levels in Baghdad presents a different interaction profile compared to that of Addis Ababa, suggesting distinct atmospheric dynamics and emission patterns. 

Nitrogen dioxide displays a weak positive correlation with sulfur dioxide (r = 0.27) and carbon monoxide (r = 0.25), indicating some degree of co-emission, likely from transport and combustion sources such as diesel generators. Ozone, however, remains negatively correlated with NO<sub>2</sub> (r = –0.53), consistent with the well-documented photochemical relationship in which NO<sub>2</sub> is consumed during ozone formation. SO<sub>2</sub> however exhibits negligible correlation with CO (r = 0.07), suggesting they originate from different sources or are influenced by divergent meteorological factors.

![](/docs/images/EDA/fig31.png)

### 4.4.4 Temporal Trend of Air Pollution in Baghdad

According to smoothed moving average modelling, the temporal dynamics of air pollution in Baghdad reveal distinctive seasonal relationships between nitrogen dioxide and the other pollutants. SO<sub>2</sub> exhibits a strong positive correlation with NO<sub>2</sub> during the first half of 2023, as both pollutants peak concurrently in summer and decline afterward, suggesting shared emission sources such as traffic and industrial activity. However, this alignment weakens toward late 2023 and early 2024.  

The carbon monoxide concentration shows a moderately synchronised trend with NO<sub>2</sub> concentration in the latter part of 2023 and early 2024, particularly during periods of heightened winter emissions, reinforcing the likelihood of overlapping anthropogenic sources such as vehicular exhaust and heating-related combustion. 

In contrast, the ozone concentration demonstrates a strong inverse relationship with NO<sub>2</sub>, particularly during the summer. This decrease in ozone may have been due to an increase in NO<sub>x</sub> produced by diesel generators during heat waves, which suppressed the production of ozone. The inverse trend in April, just before summer, may have resulted from the photochemical processes in which sunlight-driven ozone formation consumes NO<sub>2</sub>, especially under intense solar radiation.

![](/docs/images/EDA/fig32.png)