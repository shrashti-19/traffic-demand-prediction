# Traffic Demand Prediction - Flipkart Gridlock 2.0

## Overview

This project was developed for the Flipkart Gridlock 2.0 Traffic Demand Prediction challenge.

The objective is to predict traffic demand using road, weather, location, and temporal features. The solution uses CatBoost Regression with feature engineering and cross-validation to improve prediction performance.

---

## Dataset

### Training Data

* 77,299 rows
* 11 columns
* Target Variable: `demand`

### Test Data

* 41,778 rows
* 10 columns

### Features

| Feature       | Description                 |
| ------------- | --------------------------- |
| Index         | Unique record identifier    |
| geohash       | Encoded geographic location |
| day           | Day identifier              |
| timestamp     | Time of observation         |
| RoadType      | Type of road                |
| NumberofLanes | Number of lanes             |
| LargeVehicles | Presence of large vehicles  |
| Landmarks     | Nearby landmarks            |
| Temperature   | Temperature reading         |
| Weather       | Weather condition           |
| demand        | Target traffic demand       |

---

## Project Structure

```text
traffic-demand-prediction/
│
├── data/
│   ├── train.csv
│   ├── test.csv
│   └── sample_submission.csv
│
├── src/
│   ├── train_baseline.py
│   ├── generate_submission.py
│   ├── generate_submission_v2.py
│   ├── generate_submission_v3.py
│   ├── generate_submission_v4.py
│   └── generate_submission_v5_cv.py
│
├── submissions/
│   ├── submission_catboost.csv
│   ├── submission_v4_timeslot.csv
│   └── submission_v5_cv.csv
│
├── notebooks/
│
├── requirements.txt
└── README.md
```

---

## Feature Engineering

### Time-Based Features

The original timestamp column was transformed into:

* hour
* minute
* time_slot (96 intervals per day)
* hour_sin
* hour_cos
* is_morning_peak
* is_evening_peak

These features help the model capture daily traffic patterns and rush-hour behavior.

---

## Model

### CatBoost Regressor

Configuration:

```python
CatBoostRegressor(
    iterations=1500,
    learning_rate=0.03,
    depth=10,
    loss_function="RMSE",
    eval_metric="R2"
)
```

### Cross Validation

* 5-Fold K-Fold Cross Validation
* Averaged predictions across folds
* Reduced variance and improved generalization

---

## Results

| Version            |    Score |
| ------------------ | -------: |
| Baseline CatBoost  | 88.15381 |
| Timestamp Features | 88.22686 |
| Time Slot Features | 89.00124 |
| 5-Fold CV Ensemble | 89.11174 |

Best Public Score:

**89.11174**

---

## Installation

```bash
pip install -r requirements.txt
```

---

## Run

Generate final submission:

```bash
cd src
python generate_submission_v5_cv.py
```

Output:

```text
submissions/submission_v5_cv.csv
```

---

## Future Improvements

* Advanced geohash feature extraction
* Location-time interaction features
* Target encoding
* CatBoost hyperparameter tuning
* Ensemble of multiple gradient boosting models

---

## Author

Built for Flipkart Gridlock 2.0 Traffic Demand Prediction Challenge.
