# 6. Results & Discussion

## 6.1 Model Results 

### 6.1.1 NO<sub>2</sub> Explanatory Model

To gain a deeper understanding of the key drivers influencing NO<sub>2</sub> concentration dynamics in the interested region, we performed NO<sub>2</sub> level explanatory analysis using Random Forest (RF) and XGBoost (XGB) models. Model interpretation was conducted via SHAP values.

The target variable is grid-level NO<sub>2</sub> concentration, with values on the order of 10⁻⁵. For feature preprocessing, the RF model utilised raw input data, while all features in the XGB model were normalised to the [0, 1] range. This decision was based on comparative performance evaluation across different preprocessing strategies, including unscaled inputs, scaling only the features, and scaling both features and the target variable.

i.	Addis Ababa

The SHAP value violin plots of the best two models for Addis Ababa are shown in Figure 6.1 and 6.2. Both models consistently identified the lagged NO<sub>2</sub> concentration in neighboring grids as the most influential predictor, highlighting the significant role of spatial diffusion in pollutant dynamics. Additionally, features reflecting human activity levels, such as cloud cover, population, and nightlights ranked highly in both models, underscoring the non-negligible contribution of anthropogenic factors to air pollution levels.

Of particular interest is the strong influence of nighttime light intensity observed in both models. This variable commonly serves as a proxy for night-time economic activity and population aggregation, capturing composite effects of commercial vibrancy, traffic density, and industrial lighting. SHAP analysis reveals that areas with higher NTL values (the reddish regions in the SHAP plots) are positively associated with elevated NO<sub>2</sub> concentrations, indicating that mobile pollution sources such as night-time traffic and industrial emissions may play a critical role in the spatio-temporal distribution of NO<sub>2</sub>.

When it comes to meteorological factors, the land surface temperature stands out as an important variable in the XGB model, but it shows little influence in the RF model. This difference may be due to the effect of feature normalisation in XGB, which can make small changes in temperature appear more impactful. It may also reflect the more complex and nonlinear role temperature plays in NO₂ behaviour. On one hand, higher temperatures can boost air circulation and speed up chemical reactions that involve NO<sub>2</sub>. On the other hand, in some situations, high temperatures can increase the formation of ground-level ozone, which may reduce NO<sub>2</sub> levels. This kind of two-sided effect might be better captured by the XGB model, which is more sensitive to subtle patterns in the data.

In addition, road-related variables closely linked to transportation activity (e.g., residential road length, primary road length, total road length) demonstrate medium-to-high importance in both models. Road length not only reflects the density of transportation infrastructure but also indirectly indicates the frequency of vehicular movement and emission sources. Particularly in the RF model, where no normalisation was applied, road-related features with larger value scales exhibit stronger SHAP responses, suggesting a stable contribution to NO<sub>2</sub> levels.

In summary, both models reveal the multifactorial drivers of NO<sub>2</sub> concentration variability, including spatial lag effects, night-time economic activity, meteorological conditions, and transportation infrastructure. While the exact rankings of feature importance differ slightly between models, the core influential variables remain consistent. These findings offer actionable insights for urban air pollution mitigation, suggesting that policy efforts should focus on controlling emissions in high-NTL areas, regulating night-time economic activities, and fostering regional coordination in response to spatial diffusion of pollutants.

![](/docs/images/Model-Results/fig1.png)

![](/docs/images/Model-Results/fig2.png)

ii.	Baghdad

The SHAP plots in Figures 6.3 and 6.4 show that similar to Addis Ababa, both models for Baghdad consistently highlight spatial lag effects, confirming the critical role of regional pollutant spillover. Indicators of human activity, such as nighttime light intensity, also show strong and consistent influence across the models, reflecting the contribution of nocturnal economic and transportation activity to urban NO<sub>2</sub> emissions. These shared findings underscore a common underlying structure in both models, where spatial dependence and anthropogenic factors are central to explaining pollutant variation.

![](/docs/images/Model-Results/fig3.png)

![](/docs/images/Model-Results/fig4.png)

In the case of Baghdad, the XGBoost model further emphasises the role of dynamic environmental factors. In particular, land surface temperature (10:30 AM) emerges as a highly influential feature, likely due to its role in modulating vertical mixing, atmospheric stability, and the photochemical transformation of NO<sub>2</sub>. Elevated temperatures can reduce pollutant dispersion and intensify local accumulation of NO<sub>2</sub>, especially under stagnant meteorological conditions. The model also ranks Traffic Congestion Intensity (TCI) among the top predictors, capturing the real-time impact of mobility bottlenecks on transport emissions. These results suggest that the XGBoost model is especially sensitive to temporally variable features, which aligns with its ability to model complex nonlinear interactions when inputs are normalised.

By contrast, the Random Forest model shows a tendency to prioritise structural and infrastructural features, such as residential road length and total road length, over meteorological or dynamic urban variables. This is likely influenced by the use of raw feature scales, which may bias the model toward variables with inherently larger numeric scales. As a result, Random Forest may underrepresent the relative impact of high-frequency or small-scale fluctuations (e.g., TCI or temperature), and instead overemphasise more stable, cumulative spatial attributes. Nevertheless, RF still captures the importance of key variables such as nightlights and lagged NO<sub>2</sub> value for neighbouring cells, suggesting broad alignment in the most essential predictors.

The divergence in feature prioritisation between the two models reveals their complementary strengths: XGBoost appears better suited for capturing short-term, high-variability drivers of NO<sub>2</sub> (e.g., meteorology and traffic dynamics), while Random Forest offers a more stable representation of long-term or structural determinants (e.g., built environment and infrastructure). These differences are not only methodological, influenced by model architecture and preprocessing pipelines, but also conceptual, highlighting how different modeling approaches may uncover distinct but meaningful layers of insight in urban air pollution dynamics.

Overall, the results from both models illustrate a multifactorial landscape driving NO<sub>2</sub> variability in Baghdad, shaped by spatial spillover, human mobility, and atmospheric regulation.

### 6.1.2 Industrial Production Model

The System GMM model estimates the relationship between exports from sub-cities in Addis Ababa and a range of environmental and urban activity variables. 

We first built an XGBoost model using NO<sub>2</sub> along with other economic proxies (e.g., nightlights and land use features), and identified the best-performing hyperparameter configuration. Using this optimised setup, we then re-trained the model excluding NO<sub>2</sub> to assess its marginal contribution. The results are as outlined in Table 6.1. As shown in this table, removing NO<sub>2</sub> leads to a substantial decline in model performance: RMSE increases, while R² drops from 0.468 to 0.213. This clearly highlights NO<sub>2</sub>’s role in enhancing model accuracy and supports its value as a complementary proxy for economic activity.

![](/docs/images/Model-Results/fig6.png)

The SHAP plot obtained through XGBoost is as shown in Figure 6.5. Both NO<sub>2</sub> and nightlights are found to be strong predictors of industrial production with NO<sub>2</sub> ranking as the most important feature. Points of interest have a substantial impact, followed by road length, population, and industrial area. For commercial area, the plot shows that larger areas have negative SHAP values while residential areas have the least effect. Different land use types (industrial, commercial, residential) have relatively small effects, suggesting that single land use types have limited predictive power for economic activity and need to be combined with other features.

![](/docs/images/Model-Results/fig5.png)

The system GMM model was tested by excluding NO<sub>2</sub>, but the model failed, indicating inadequateness of data variables. The results of the system GMM industrial production model with NO<sub>2</sub> are as shown in Table 6.2. The results indicate a nuanced relationship. On its own, NO<sub>2</sub> has a positive effect on exports (coefficient = 1.78), suggesting that higher economic activity, which is often associated with greater pollution, may correlate with increased exports. The model suggests that a 1% increase in NO<sub>2</sub> is associated with a 1.78% increase in export values. 

However, when NO<sub>2</sub> is combined with commercial and industrial land use, the interaction terms become negative. This implies that in areas with substantial commercial or industrial land, higher NO<sub>2</sub> concentrations are associated with lower exports, potentially due to congestion, reduced productivity, or adverse health impacts affecting labour productivity. It could also indicate that industries may prefer having their operations away from highly urbanised areas. Furthermore, this pattern suggests a threshold or diminishing returns effect, implying a non-linear relationship between NO<sub>2</sub> and industrial production. This is also evidenced by the SHAP plot. At lower levels of air pollution, NO<sub>2</sub> may simply be a byproduct of growing economic activity. However, beyond a certain point, especially in already dense or industrialised areas, the negative externalities of pollution, such as health impacts, reduced worker productivity, or logistical inefficiencies, begin to outweigh the economic benefits.

![](/docs/images/Model-Results/fig7.png)

Nighttime lights and points of interest show positive associations with exports, acting as proxies for economic vibrancy and urban density. Accordingly, a 1% increase in nightlights intensity across a sub-city is associated with a 2.96% increase in export values, and this value increases to 6.55% for points of interest.

The model diagnostics confirm its statistical reliability. The Sargan test supports instrument validity (p = 0.96), while the absence of second-order autocorrelation (p = 0.08) suggests consistent moment conditions. Overall, while modest pollution levels may accompany economic growth, their interaction with dense land use can counteract potential gains, highlighting a trade-off between urban expansion and environmental burden. 

In conclusion, NO<sub>2</sub> can be a good measure of industrial production in a region. The findings suggest that NO<sub>2</sub> can be effectively used not only as an input feature within predictive models but also as a standalone spatial proxy for economic activity intensity in similar urban environments. However, the relationship between the NO<sub>2</sub> and industrial production may not necessarily always be linear, other variables and the specific context of the region must be considered as well. 

### 6.1.3 Mobility Model

The results obtained from the OLS model for the top contributors to TCI are as outlined in Table 6.3. Both NO<sub>2</sub> and TCI are aggregated spatial-temporal quantities. TCI aggregates congestion minutes for every road cell across all days of the month. The NO<sub>2</sub> levels variable averages daily values for each grid cell, then sums across all cells in a district. Thus, the coefficient captures total congestion response to total pollution, naturally leading to a large absolute number as shown in the results table.

To make this interpretable in relative terms, we compute a point elasticity:

$\varepsilon_{NO<sub>2</sub>, TCI} = \beta_{NO} \frac{\bar{x}}{\bar{y}} \approx 0.25$

For a typical Baghdad district and month, a 1% increase in average NO<sub>2</sub> concentration is associated with roughly a 0.25% increase in total monthly congestion (TCI). Because the regression is in levels, this 0.25 figure is a local effect, it holds near current mean conditions. Elasticity would vary in areas with very different baseline pollution or congestion levels.

NO<sub>2</sub> is a by-product of combustion (traffic, power generation, industrial boilers). Its strong positive coefficient indicates that air-quality deterioration is tightly linked to higher mobility demand and hence economic throughput in Baghdad. Simpler proxies like lags or neighbouring NO<sub>2</sub> add little once the mean level is in the regression; they were therefore removed to avoid collinearity.

In summary, a transparent, district-month OLS confirms that NO<sub>2</sub> is a reliable real-time pulse of economic activity, explaining ~80 % of congestion variance while remaining fully interpretable for policy use.

![](/docs/images/Model-Results/fig8.png)

### 6.2 Limitations and Future Directions

This study offers promising insights into how satellite-based indicators, particularly NO<sub>2</sub>, can help estimate economic activity in cities where conventional data is limited. While all the tested models are validated to ensure reliability and robustness, there are some limitations that should be acknowledged to guide future work. These can be summarised as follows:

1.	Data availability and resolution remain major hurdles. While global satellite datasets are designed to cover all regions, cities like Baghdad and Addis Ababa still suffer from patchy observations. Factors such as persistent cloud cover and retrieval issues led to missing values in variables like NO₂, which had to be estimated through spatial and temporal interpolation. While these methods were applied carefully, they introduce a layer of uncertainty that could affect model outcomes. This highlights a broader challenge; the need for more consistent and high-quality satellite data in underrepresented regions.

2.	The analysis covers a relatively short time period of just under two years (2023–2024). This was sufficient for model training and testing, but it limits the ability to detect longer-term patterns or seasonal effects. Expanding the analysis to at least 3 to 5 years of continuous data would help improve model stability and provide a more comprehensive view of economic cycles and environmental dynamics, although it may increase computational demand.

3.	Ground-truth economic data is often missing at the sub-city level, especially in lower-income contexts. This study used practical proxies, exports for Addis Ababa and traffic congestion intensity (TCI) for Baghdad, but more accurate calibration would require matching satellite indicators with standard economic measures like GDP, industrial output, or electricity consumption where those are available. Furthermore, the limited nature of the exports data leads to fewer time steps for running a dynamic panel model. While system GMM is built to withstand this, it does interfere with the reproducibility of the results due to a lack of assurance on whether heterogeneity is captured satisfactorily by the data. The use of city level variables may resolve this issue, subject to data availability.

4.	While system GMM is a good fit for our industrial production data, the model using R (or Stata) is not directly reproducible in Python. This may be an issue for researchers looking for reproducibility using different software. This can be resolved through the use of Machine Learning models such as XGBoost, as attempted in this project.

A practical recommendation going forward is to train models in regions where both the explanatory variables (e.g. NO<sub>2</sub>, NTL, land use) and the economic indicators are reliable, and then apply these models in similar settings where only the satellite features are available. This transfer approach allows for broader applicability while maintaining rigour. To ensure that such models generalise well, they should be tested against alternative benchmarks, such as mobility trends or TCI, grid usage, or trade flows, in the target region. At the same time, producing outputs such as elasticities or SHAP-based feature importance can help keep the models interpretable and policy-relevant.

### 6.3 Conclusion

The extensive literature review, exploratory data analysis, and models all point to the applicability and importance of NO<sub>2</sub> in measuring economic activity resulting from industrial production and mobility. Across two very different case studies, industrial performance in Addis Ababa and urban mobility in Baghdad, NO<sub>2</sub> consistently emerged as one of the most informative features.

Chapter 2 addresses the production and measurement of NO<sub>2</sub> and factors influencing it, as evidenced by the literature. These findings are validated in Chapter 4 through the exploratory data analysis as relationships are drawn and correlations are estimated. The testing of the modelling methods with different combinations of data variables in Chapter 3 and 5 help build the relationship between NO<sub>2</sub> and economic activity. The results of the models in Chapter 6 confirm our hypothesis; the variability in congestion and industrial production can be reliably predicted using NO<sub>2</sub>, and in some cases, NO<sub>2</sub> may even be better than traditionally used predictors such as nightlights. Using a combination of predictors to support each other can produce even better results, especially since contextual variability can introduce unexpected deviations.

Machine learning models delivered strong results where detailed spatial data was available, particularly in the NO<sub>2</sub> explanatory model. However, when data was more aggregated, as in the economic models, simpler methods like ordinary least squares (OLS) performed nearly as well and offered a clearer view of how different variables influenced the outcome. In the Addis Ababa model, NO<sub>2</sub> had an estimated elasticity of +1.78, meaning a 1% increase in NO<sub>2</sub> levels was associated with a 1.78% increase in monthly exports. In Baghdad, the NO<sub>2</sub> elasticity with respect to traffic congestion intensity (TCI) was +0.25%, reinforcing its relevance even in settings where economic activity is more mobility-driven. This suggests that for many real-world applications, especially those involving public or institutional stakeholders, interpretable models remain a critical asset.

Given its advantages of global coverage, frequent updates, and low acquisition costs, satellite-derived NO<sub>2</sub> data holds particular promise for application in low- and middle-income countries, especially in urban areas where economic data may be sparse, outdated, or unreliable. This includes regions characterised by informal economies or affected by conflict, where conventional economic statistics are often insufficient. Thus, NO<sub>2</sub> offers a valuable and practical tool for monitoring and analysing economic activity in data-scarce contexts.