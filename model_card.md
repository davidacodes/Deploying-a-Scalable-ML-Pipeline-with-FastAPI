# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details

This model is a random forest classifier (`sklearn.ensemble.RandomForestClassifier`) built with scikit-learn 1.5.1 in Python 3.10. It was trained with the library's default hyperparameters and a fixed random seed (`random_state=42`) so that training is reproducible. The model predicts a binary label indicating whether a person's annual income exceeds $50,000. 

## Intended Use

The model is intended to predict whether an individual's income is above or below $50,000 per year based on demographic and employment attributes from census data.

## Training Data

The model was trained on the publicly available Census Income (Adult) dataset from the UCI Machine Learning Repository, provided in this repository as `data/census.csv`. Each row contains 14 features, including six continuous features (such as age, capital-gain, and hours-per-week) and eight categorical features (workclass, education, marital-status, occupation, relationship, race, sex, and native-country), along with the binary salary label (`<=50K` or `>50K`).

## Evaluation Data

The model was evaluated on the held-out 20% test split that was not used during training. The test data was processed with the same one-hot encoder and label binarizer that were fitted on the training data, ensuring that evaluation reflects how the model will behave on unseen data at inference time. In addition to overall performance, metrics were computed on slices of the test data for every unique value of each categorical feature, with results written to `slice_output.txt`.

## Metrics

On the held-out test set the model achieves a precision of 0.7391, a recall of 0.6384, and an F1 score of 0.6851. This means that when the model predicts an income above $50,000 it is correct about 74% of the time, and it successfully identifies about 64% of all high earners in the test set.

Slice-based evaluation shows that performance is not uniform across groups. For example, recall is 0.6607 for males but only 0.5107 for females, and F1 ranges from 0.6000 for the Amer-Indian-Eskimo race category to 0.8000 for the Other category.

## Ethical Considerations

The dataset comes from the 1994 U.S. Census and encodes historical and societal biases in income distribution across race, sex, and national origin. Because the model learns from these patterns, its predictions reflect and could amplify those biases; the slice metrics confirm that performance differs across demographic groups, with notably lower recall for women than for men. The model should therefore never be used to make or support decisions about real individuals, such as creditworthiness, employment, or access to services. The dataset contains no personally identifiable information, but the demographic attributes it uses (race, sex, native-country) are sensitive, and any downstream use should carefully consider fairness implications.

## Caveats and Recommendations

The training data is more than three decades old, so the relationships it captures between demographics, occupation, and income do not reflect the current economy, and the model should not be expected to generalize to present-day data. The dataset is also imbalanced (roughly 76% of rows are labeled `<=50K`), which contributes to the lower recall on the positive class. Metrics computed on very small slices should not be trusted, as a handful of rows can produce degenerate perfect or zero scores.
