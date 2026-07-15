# Traffic Congestion Insights Report

Created dynamically based on historical traffic data training and EDA analysis.

## 1. Dataset Overview
- **Total Records Analyzed**: 48,120 hourly readings
- **Time Horizon**: 2015-11-01 00:00:00 to 2017-06-30 23:00:00
- **Represented Junctions**: [1, 2, 3, 4]

## 2. Traffic Flow Characteristics
Across the observed junctions, traffic volumes vary significantly, indicating differing capacities and roles in the transit network:

- **Overall Network Peak Hour**: Hour 19:00 (typically evening commute block)
- **Junction Stats**:
  - **Junction 1**: Mean Traffic = 45.1 vehicles/hr, Peak Max Traffic = 156 vehicles/hr, Total Readings = 14,592
  - **Junction 2**: Mean Traffic = 14.3 vehicles/hr, Peak Max Traffic = 48 vehicles/hr, Total Readings = 14,592
  - **Junction 3**: Mean Traffic = 13.7 vehicles/hr, Peak Max Traffic = 180 vehicles/hr, Total Readings = 14,592
  - **Junction 4**: Mean Traffic = 7.3 vehicles/hr, Peak Max Traffic = 36 vehicles/hr, Total Readings = 4,344


### Peak Congestion Periods
- **Weekday vs. Weekend**: Traffic volumes are higher on average during weekdays. Commutes are highly concentrated around two standard rush hour bands: Morning (07:00 - 09:00) and Evening (17:00 - 19:00).
- **Weekly Patterns**: Traffic typically peaks on mid-week days (Tuesday through Thursday) and declines during weekends (Saturday and Sunday).

## 3. Modeling and Prediction Results
To solve the traffic volume regression task, we tested and evaluated multiple modeling algorithms.

### Model Comparison Table
| Model | MAE | RMSE | R² Score |
| :--- | :--- | :--- | :--- |
| Linear Regression | 8.605 | 11.993 | 0.806 |
| Decision Tree | 6.446 | 11.685 | 0.816 |
| Random Forest | 5.629 | 9.035 | 0.890 |
| Gradient Boosting | 5.610 | 8.617 | 0.900 |
| XGBoost | 5.260 | 8.221 | 0.909 |


The selected best model is **XGBoost**, achieving an R² score of **0.9088** on the chronological holdout dataset.

*Note: Holdout test set spans 2017-03-22 18:00:00 to 2017-06-30 23:00:00.*

### Model Explainability
Global feature importances from the best model show that the top features driving the traffic predictions are:
**Hist_Traffic_J_D_H, Year, Month, Junction_4, Junction_1**

This indicates that cyclical hour shapes combined with historical traffic levels provide the strongest signals for identifying congestion peaks.
