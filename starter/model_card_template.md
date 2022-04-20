# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
Luiz Otávio Matias created the model. It is Random Forest using the default hyperparameters in scikit-learn 1.0.2.
## Intended Use
This model should be used to predict Predict whether income exceeds $50K/yr based on census data. The associated tasks is classification.

## Training Data
The data was obtained from the UCI Machine Learning Repository (https://archive.ics.uci.edu/ml/datasets/census+income).

The target class was "salary" with >50K or <=50K.

The data set has 32561 rows (it was modified and stored in data folder in the starter repository), and a 80-20 split was used to break this into a train and test set. No stratification was done. To use the data for training a One Hot Encoder was used on the categorical features and a label binarizer was used on the labels.

Categorical Features:

"workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country"

Continuos Features:

'age',
 'capital-gain',
 'capital-loss',
 'education-num',
 'fnlgt',
 'hours-per-week'
## Evaluation Data

The data used to evaluate was the 20% test set.
## Metrics
The model was evaluated using Precision, Recall and F-beta. The values was 0.7540, 0.6251 and 0.6835 respectively.


## Ethical Considerations
Census Income Data Set contains sensitive attribute information like sex and race. The project was made for educational purposes. Therefore, there was no bias test.
## Caveats and Recommendations

There were no hyperparameters tuning or robust feature engineering. The project was made focused on MLOps.