# Kaggle Housing price prediction
## Description
This repo contains my solution to the Kaggle Housing price prediction task based on the AMES dataset. Four models have been trained on the data and the one with the best CV score, a Random Forest Regressor was used for the test prediction and the submission of the solution.

## Files
Main file: Kaggle_Housing.ipynb - Jupyter notebook containing the training of the four models.

EDA: Kaggle_Housing_EDA.ipynb - Exploratory data analysis. Each feature is inspected and evaluated. The final feature selection is implemented in the main file.

## Scoring
Best CV score (mean absolute error) was achieved by the Random Forest Classifier, using GridSearchCV for hyperparameter tuning. 

CV score: 0.103 (before hyperparameter tuning)

Public score (Test score): 0.150

Ranking: 2678
