# Exploratory Data Analysis

## Addis Ababa

### 1. NO<sub>2</sub> :

![](/docs/images/EDA/fig1.png)

For both weekdays and weekends, there is a prominent concentration of NO<sub>2</sub> in the northwestern region. This region, highlighted in pink in Figure 3, is the heart of Addis Ababa. It is where most of the roads converge. Most of the Addis Ababa Light Rail System is located here as evidenced by Figure 2. The Addis Ababa Bole International (ADD) Airport is located towards the southeast of this region as shown in Figure 3. The figure also shows the major highway corridors of A1 and A2. The A1 highway connects Addis Ababa to Dire Dawa and Djibouti, and it is a part of the Trans African Highway network. These mobility systems may be major contributing factors to the NO<sub>2</sub> levels in this region. The southeast and eastern parts show markedly lower NO<sub>2</sub> levels. This area is mostly marked by extensive cropland, indicating higher agricultural activity rather than industrial production. The extreme northern and northeastern border regions also show lower NO<sub>2</sub> levels which is in line with the region being forest land.

![](/docs/images/EDA/fig2.png)

On closer inspection, we also notice a variation in NO<sub>2</sub> levels on weekends as compared to weekdays. This may be associated with lower economic activity on the weekends (less people travelling to work, businesses being shut). On the other hand, in some places, there may be increased NO<sub>2</sub> emissions due to higher usage of private vehicles for commercial activity involving recreation (visits to malls, marketplaces) and worship (visits to Church on Sundays). 

The daily subregional NO<sub>2</sub> levels have been attached in Appendix Figure 3. All the subregions consistently show a spike in NO<sub>2</sub> levels in February-March and November-December, with a marked decrease in August. In Ethiopia, February-March and November-December fall in the dry season; with dry, stable atmospheric conditions and low wind which can trap pollutants near the surface due to poor vertical mixing. The period also coincides with pre-holiday commerce and traffic activity due to festivals and the peak tourist season. On the other hand, August sees the highest rainfall in Ethiopia, which may be the reason for the drop in NO<sub>2</sub> levels due to washing away of NO<sub>2</sub>, as highlighted in the Literature Review.

### 2. Population :

![](/docs/images/EDA/fig3.png)

The population distribution for 2023 and 2024 also shows a similar trend to NO<sub>2</sub>, i.e., a possibly high correlation to the transport network and industrial production regions. Particularly, two offshoots are observed spreading from the northwest horizontally to the east and diagonally to the south. For the first case, the presence of the East-West line of the LRT corridor and A1 highway may be a possible reason. For the second case, higher population is observed along the North-South line of the LRT followed by the A1 highway corridor. 

Similar to NO<sub>2</sub>, lower population numbers are observed in croplands and forested regions. Between 2023 and 2024, the total population in Addis Ababa increased by 2.17% to about 4,281,917. The boxplot in Appendix Figure 1 shows a mean of 7336 in 2023 and 7842 in 2024. The histogram in Appendix Figure 2 shows that a considerable number of cells have population values below 5000.

### 3. Industrial Area :

The industrial land use area distribution as shown in Figure 5 also shows higher values along the A1 highway corridor. Furthermore, higher industrial area towards the east can be attributed to the Special Economic Zone (SEZ) of Bole Lemi as shown in Figure 6.

![](/docs/images/EDA/fig4.png)

### 4. Points of Interest :

The points of interest data was processed to obtain the total percentage of POI expected to produce considerable amounts of NO<sub>2</sub> due to industrial activity and/or mobility. These POI included supermarkets, hospitals, car dealerships, car rentals, carwash, marketplaces, malls, and universities. Comparing Figure 3 and Figure 7, a higher concentration of POI is observed in the heart of Addis Ababa.

![](/docs/images/EDA/fig5.png)

### 5. Road Length :

The total road length per cell was calculated in metres per cell as shown in Figure 8. Comparing this with figures 2, 3, and 4, we observe a correlation between the persistence of roads, LRT, and population distribution.

![](/docs/images/EDA/fig6.png)

## Baghdad

### 1. NO<sub>2</sub> :

![](/docs/images/EDA/fig7.png)

The weekend in Iraq is Friday-Saturday. For the purpose of subsequent traffic congestion analysis, the week has been divided into three parts: workday (Monday-Wednesday), extended workday (Thursday, Sunday), and weekend (Friday-Saturday). The area under consideration is the Baghdad Governorate (or province). 

![](/docs/images/EDA/fig8.png)

As highlighted in Figure 10, the central region is the city of Baghdad. It is the capital and largest city of Iraq and thus, the nation’s economic hub. The green spot in the figure indicates Baghdad International Airport (BGW). As shown in Figure 11, moving towards the south from the city centre is a major mobility corridor formed by the Baghdad–Basra Railway Line, and Highway 8 and Motorway 1 (M1) which connect Baghdad to Kuwait. The figure also shows Highway 6 which goes from the city centre towards the southeast to Al Kut and connects Baghdad to Tehran, Iran. 

As shown in Figure 9, the highest NO<sub>2</sub> concentrations are observed in Baghdad city centre; where most of the urban activity is concentrated. Considerable emissions are also observed as we move towards the southeast, along the Highway 6 corridor. Similar to Addis Ababa, there is a slight variation in emissions for weekdays and weekends. The reasons for this variation may be similar, except for worship (visits to Mosque) taking place on Fridays instead of Sundays. The daily subregional NO<sub>2</sub> levels attached in Appendix Figure 4 show that emission levels in the outer regions are fairly consistent throughout the year. On the other hand, for the central regions, the emissions profile is more peaky, with spikes often seen between July-September. These months are marked by high temperatures and clear skies along with power cuts. This leads to increased electricity demand for air conditioning and usage of diesel generators. The period also coincides with Muharram, which often involves communal events and processions drawing large crowds.

### 2. Population :

![](/docs/images/EDA/fig9.png)

In Baghdad, the population distribution shows a similar profile to NO<sub>2</sub> emissions distribution, with a higher concentration in the city centre. The population is particularly high in Sadr City and Al Baladiyat to the north of the Army Canal, parallel to the Tigris River. Closer inspection shows considerable population concentration along the Tigris and Highway 6 to the southeast and along the rail line in the south.

The population distribution is much lower in the outer regions, which are mainly defined by rangeland and cropland. The population increased by 199125 between 2023 and 2024, a 2.10% increase. The boxplot in Appendix Figure 1 shows a mean value of 1547 in 2023 and 1579 in 2024. However, a large number of cells lie beyond the range. This, coupled with the histogram in Appendix Figure 2, indicates a few cells have very large population values compared to others.

### 3. Industrial Area and Points of Interest :

The obtained industrial area data shows sparseness, as does the points of interest data. The trend of higher human activity concentrations in the city centre and along the highway corridors continues to prevail. 

![](/docs/images/EDA/fig10.png)

### 4. Road Length :

The total road length per cell in the grid shows a stark resemblance to the population distribution and NO<sub>2</sub> emissions plot. This shows a strong correlation between human settlement, mobility, and air pollution.

![](/docs/images/EDA/fig11.png)

## Appendix

![](/docs/images/EDA/fig12.png)
