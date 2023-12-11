# What Drives Water Potability? 
- What is potability? Potable water, is water that is suitable for drinking and is safe for ingestion.


# Project Description:

This project contains findings from reshearch based on classification machine learning to determine the drivers of potability.
Data was acquired from this website: https://www.kaggle.com/datasets/adityakadiwal/water-potability


# Project Goal:
My goal is to identify drivers of potability, why some drivers are more important than others, predict potability using machine learning, and make recomendations based on the information that I find to potentially increase safe water potability. 

# Executive Summary: 


1. Goal: Show what water metrics drive potability and what can improve potability
2. Big Idea: pH levels within a safe range of human consumption are potable
3. Key Findings: 
    - No feature alone drives potability
    - Machine learning feature selection found 3 top features, but they didn’t improve the model.
    - However, analyzing features under safe levels for humans shows ph is a driver of water potability.
4. Recomendations: 
    - Water quality features should not be analyzed on single features.
    - When deciding on potable water, metrics should be analyzed by EPA and WHO approved levels. 



# Project Replication:

1. You will need to dowload the data from: https://www.kaggle.com/datasets/adityakadiwal/water-potability
2. Make sure your enviormnet matches the specifications listed on the environment.yml file in this repository. If you environment does meet these requirements you can set up your conda environment using the file. 
3. You then can download this repository to your local device. Each process of the data science pipeline has thier own file. For understanding: 
    - A: Acquire: How the data was acquired
    - B: Prepare: How the data was prepared
    - C: Explore: Initial exploration using no feature engineering
    - C2: Explore 2: Second phase of exploraiont using featuring enginnering
    - C3. Clustering: Analyzing if there are potential clusters in the data
    - D. Modeling: Creating the models. **Note**, There are two process in this data. 
        - # Phases: 
            1. Run phase one using all features. Record Data.
                - To do this change the list on X vairbales to be:
                    - train[all_features]
                    - validate[all_features]
                    - test[all_features]
                - Run the entire notebook.
            2. Run phase two using set features. Record Data.
                - To do this, change the list on X variables to be: 
                    - train[set_features]
                    - validate[set_features]
                    - test[set_features]
                - Run the entire notebook.

# Initial Questions: 

1. What does each feature mean and what could it mean for human health?
2. What are safe ranges for features for humans? 
3. Could there be potential clusters in the data? 
4. Is this data accurate? 
5. What are the drivers of potability?
6. If I bin the data, could this lead to even better features?

Questions listed on the final notebook: 
1. Could chloramines be a big influence? They are used to treat water to make it drinkable.
2. Could Turbidity be important? How much clarity in the water could be a major impact on drinkable water.
3. Could ph be important? Is the water to acidic or too basic? Humans can safley consume water on a ph range of 6.5 to 8.5.


# Proget Planning:
1. Data Acquisition: 
    - Acquire the data from: https://www.kaggle.com/datasets/adityakadiwal/water-potability
2. Data Peperations: 
    - Get an overhead look at the data
    - Investigate null values
    - Investigate outliers
    - Clean up column names
3. Data Exploration: 
    - Analyze Univariate variables
    - Analyze Bivariate variables
    - Analyze Multivariate variables
    - Statistical Testing: 
        - Chi2
        - Spearman's R
4. Modeling: 
    - Decision Tree
    - Random Forest
    - KNN
    - Logistic Regression
5. Presentation: 
    - Create a 5 minute storytelling presentation
    
    
    
# Data Dictionary

| **Quality Index** | **Detailed Definition** | **Simplified** | **Mesaurement** |  **Safe Range** | 
|-------------------|-----------------|----------------| ----------------|-------------|
|pH | A quantitative measure of the acidity or basicity of an aqueous or other liquid solution | The measure of the acidity or basicity of water | pH | 6.5 to 8.5 |
|Hardness | the total calcium and magnesium ion concentration in a water sample | How hard or soft the water is | mg/L <br> **Levels** <br> 0 - 17.1 : Soft <br> 17.1 - 60 : Slightly Hard <br> 60 - 120 : Moderately Hard <br> 120 - 180 : Hard <br> 180+ : Very Hard | 120 - 170 mg/L |
| Total dissolved solids (TDS) | Dissovled inoragnic and organic compounds. Can be: Salts, Potassium, Calcium, Sodium, Bicarbonates etc. | How many minerals are in the water | ppm | 500 ppm - 1000 ppm |
|Chloramines | Chlorine and Chloramine used to treat water | Disinfectants used to public water systems to treat the water | ppm |Less than 4 mg/L |
|Sulfate | Naturally occurring minerals in minerals, soil, and rocks | Large amounts of sulfates can cause laxative effects | mg/L |Less than 250 mg/L |
|Conductivity | Measure the ionic process of a solution that enables it to transmit current | Amount of ion concentraions in the water that result in electrial conductivity | μS/cm |Less than 400  μS/cm |
|Organic Carbon | (TOC) Total Organic Carbon is the measure of the total amount of carbon in organic compouns in pure water| TOC can come from decaying natural organic mattter | ppm |Less than 3 mg/L | 
| Trihalomethanes | Formed when natural organic mateials react with chlorine | Chemicals found in water treated with chlorine| μS/cm  | Less than 80 ppm |
| Turbidity | Meaure of light penetraion of water impacted by suspened materials | How much supspended materias in the water impacting clarity | NTU |Lower than 5.00 NTU |
| Potability | potable water is suitable for drinking | Is the water safe for human consumption | 1 indicates potable, 0 indicates not potable |


# Exploration:
<ins>Exploration Summary of features without bins:</ins>


1. pH = min: 0.23 / max: 14.00
- Data looks normal, outliers don't look like and issue
2. hardness = min: 73.49 / max: 317.34
- Minor skew to the right; outliers don't look like and issue
3. solids = min: 320.94 / max: 56488.67
- Positive Skew, Outliers don't look like and issue.
4. chloramines = min: 1.39 / max: 13.13
- data looks normal, outliers don't look like an issue
5. sulfate = min: 129.00 / max: 481.03
- data looks negatively skewed, outlires dont look noraml, could remove one
6. conductivity = min: 201.62 / max: 753.34
- data looks normal, outliers don't look bad
7. organic_carbon = min: 2.20 / max: 27.01
- data looks normal, outliers don't look like an issue
8. trihalomethanes = min: 8.58 / max: 124.00
- looks skewed right, outlires don't look like an issue
9. trubidity = min: 1.45 / max = 6.49
- looks nomral, outliers don't seem to be an issue


- Single features on their own do not impact portability. After completing the statistical test comparison of means, no feature was statistically significant, and the null hypothesis could not be rejected.
- 59 percent of observations are not potable or safe to drink. 

<ins>Exploration Summary with feature bins:</ins> 


- To explore the features more to find what could potentially drive portability, I binned features on safe levels for human consumption. These levels were chosen based on the documentation provided by the data source, the WHO, the EPA, and online sources. 
 - Once the features had been binned, I ran statistical tests, $Chi^2$, to find if the potability was dependent on these new features. One feature, the ph range, did reject the null hypothesis. 
 
<ins>Machine Learning Selection:</ins> Select K Best, RFE, and Sequential Feature Selector:

- Due to not finding statistical significance for features on their own, I inquired as to what might be found by machine learning selection. I used Select K best, RFE, and Sequential Feature selector. 
- Phase one features that were selected were: 
    - ph
    - hardness
    - chloramines
    - conductivity
- Phase two features that were selected were
    - ph
    - hardness
    - ph_range_safe
    - hardness_range_moderately_hard


# Hypothesis Testing: 

1. Does Potability differ across ph?
2. Does potability differ across chloramines?
3. If we organize ph levels by ranges that are safe for humans does it impact potability?
4. If we organize hardness by a water hardness scale, does it impact potability?

Results are listed in the final notebook.


# Modeling:
### How did our Models Do?
Models were compared on thier train and validate accuracy, as well as the difference between these two scores. I used accuracy beacuse the predictions revolve around safe or unsafe drinking water. If there are any mistakes it could lead to harm.


Legend: <br>
The table bellow shows the results of the modeling tests: <br>
- DT(#): Indicates the depth of the decision tree
 - RF(#): Indicates the depth of the random forest
 - LR(#): Indicates the C of logistic regression
 - KNN(#): Indicates the number of neighbors
<center> Basline Accuracy = 0.60 <center>
    
| **Model** | **Train Accuracy** | **Validate Accuracy** |  **Difference** | 
| ------------|-----------------|-----------------| ------| 
| DT(3): All Features | 0.643 | 0.612 | 0.030 |
| DT(3): Select Features | 0.631 | 0.590 | 0.041 |
| RF(4) : All Features | 0.679 | 0.629 | 0.049 |
| RF(3) : Select Features | 0.631 | 0.604 | 0.026 |
| LR(0.0491) : All Features | 0.596 | 0.596 | 0.000 |
| LR(0.0491) : Select Features | 0.608 | 0.596 | 0.011 | 
| KNN(19) : All Features | 0.615 | 0.565 | 0.049 |
| KNN(20): Select Features | 0.630 | 0.546 | 0.080 |



# Conclusion: 
1. Summary:
    - Without feature engineering, each feature on its own does not provide much information about what causes potability. This raises concerns about potential bias in the data. Some features that were thought to drive potability, based on background knowledge, were found not to be drivers in reality.
    - This study highlights that scientists or regulators evaluating water potability should not rely solely on prior knowledge or individual features/metrics. Water potability cannot be assumed based on these factors alone.
    - I would not recommend using this machine learning model to predict water potability until a higher accuracy can be achieved with new or better data.

2. Recommendations:
    - When analyzing bodies of water to determine potability, do not rely on individual features alone. They do not appear to be reliable indicators of water quality. If these features were used as predictors, they could lead to inaccurate predictions and unsafe drinking water.
    - Analyze all features based on recommended safe levels proposed by the World Health Organization (WHO) or the Environmental Protection Agency (EPA). Do not rely on a single feature, such as pH, as it could lead to inaccurate predictions.

3. Next Steps:
    - I would like to apply these machine learning methods to more water quality datasets to identify clearer drivers of water quality.
    - I have examined the data three times in search of drivers. I would like to spend more time on a new dataset to identify any drivers present.
    - I am interested in examining marine biology data, specifically water quality around coral, to determine if water quality is causing harm