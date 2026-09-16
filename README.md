# Tamil-English Code-Mixed Offensive Language Detection

## Dataset Loading, Preprocessing and Classical Baseline

This module implements the complete classical baseline pipeline for detecting offensive language in Tamil-English code-mixed social-media text.

The pipeline starts from the official DravidianCodeMix dataset and performs data loading, preprocessing, exploratory data analysis, TF-IDF feature extraction, classical model training, hyperparameter tuning, model selection, final test evaluation, error analysis, and model saving.

## 1. Project Objective

The objective is to develop a baseline Natural Language Processing system that classifies Tamil-English social-media text into different offensive-language categories.

The classical baseline provides a reference point for comparison with the Transformer-based model implemented in the next stage of the project.

The complete pipeline is:

```text
Raw Dataset
     ↓
Dataset Loading
     ↓
Data Inspection
     ↓
Data Cleaning
     ↓
Duplicate and Overlap Analysis
     ↓
Exploratory Data Analysis
     ↓
TF-IDF Feature Extraction
     ↓
Logistic Regression + Linear SVM
     ↓
Hyperparameter Tuning using Development Set
     ↓
Model Selection using Macro F1
     ↓
Final Test Evaluation
     ↓
Error Analysis
     ↓
Model Saving
```

## 2. Dataset

The project uses the Tamil offensive-language portion of the DravidianCodeMix dataset.

The dataset contains Tamil social-media text with different forms of language usage, including:

- Tamil script
- Romanized Tamil
- English
- Tamil-English code-mixed text
- Informal language and slang
- Social-media style expressions

The dataset provides predefined training, development, and test files.

### Dataset Files

```text
DravidianCodeMix/
│
├── tamil_offensive_full_train.csv
├── tamil_offensive_full_dev.csv
└── tamil_offensive_full_test.csv
```

The files are tab-separated and contain the text and corresponding offensive-language label.

### Dataset Sizes

| Split | Samples |
|---|---:|
| Train | 35,139 |
| Development | 4,388 |
| Test | 4,392 |
| Total | 43,919 |

The official train, development, and test partitions are retained instead of creating a new random split.

## 3. Classification Labels

The dataset contains six classes:

```text
1. Not_offensive
2. Offensive_Untargetede
3. Offensive_Targeted_Insult_Individual
4. Offensive_Targeted_Insult_Group
5. Offensive_Targeted_Insult_Other
6. not-Tamil
```

`Offensive_Untargetede` is retained exactly as it appears in the original dataset.

The six classes are not equally represented, resulting in class imbalance. Therefore, Macro F1 is used as the primary metric for model selection.

## 4. Environment and Dependencies

The notebook uses Python and the following libraries:

```text
pandas
numpy
matplotlib
seaborn
scikit-learn
joblib
```

Install the dependencies using:

```python
%pip install pandas numpy matplotlib seaborn scikit-learn joblib
```

## 5. Reproducibility

A fixed random seed is used throughout the experiment:

```python
SEED = 42
```

This ensures reproducible results wherever random operations are involved.

## 6. Dataset Loading

The dataset files are loaded using Pandas.

Since the original files are tab-separated and do not contain a standard header, the loader uses:

```python
pd.read_csv(path, sep="\t", header=None, names=["text", "label", "extra"], encoding="utf-8")
```

The columns are:

- `text` – input social-media comment
- `label` – offensive-language category
- `extra` – unused column present in the file format

The unused `extra` column is subsequently removed.

## 7. Initial Dataset Inspection

The loaded data is inspected by displaying:

- Sample records
- Dataset dimensions
- Column names
- Label distributions
- Missing values
- Duplicate samples

This verifies that the dataset has been loaded correctly before preprocessing.

## 8. Duplicate Analysis

Exact duplicate text-label pairs are checked independently within the training, development, and test sets.

Duplicates are removed within each individual split.

The official train, development, and test partitions are not randomly recreated.

## 9. Cross-Split Overlap Analysis

The notebook checks whether the same text occurs in multiple official partitions.

The following pairs are examined:

```text
Train ↔ Development
Train ↔ Test
Development ↔ Test
```

Both exact text overlap and label conflicts are identified.

A small number of overlapping instances contain different labels in different partitions.

These instances are retained without manual relabeling or removal in order to preserve the original dataset annotations and official partitions.

The identified inconsistencies can also be considered during error analysis.

## 10. Exploratory Data Analysis

The notebook performs exploratory analysis to understand the dataset.

### 10.1 Class Distribution

The number of samples belonging to each offensive-language category is examined.

The training data is imbalanced, with `Not_offensive` representing the largest class and some targeted offensive categories containing considerably fewer samples.

The class distribution is visualized using count and percentage plots.

### 10.2 Text Length Analysis

Text length is analyzed using:

```text
Character length
Word count
```

These statistics are calculated for the training, development, and test sets.

The distribution of word counts is visualized using histograms.

### 10.3 Word Count by Class

Word-count distributions across offensive-language categories are visualized using box plots.

This helps examine whether different categories exhibit different text-length patterns.

### 10.4 Representative Samples

Representative examples from each class are displayed for manual inspection.

These examples demonstrate:

- Tamil script
- Romanized Tamil
- English
- Code-mixing
- Slang
- Offensive expressions
- Informal social-media language

## 11. Data Cleaning

The preprocessing strategy is intentionally lightweight.

Aggressive preprocessing is avoided because Tamil-English code-mixed text contains useful linguistic and stylistic information.

The cleaning process performs:

- Removal of missing text or labels
- Removal of empty texts
- Normalization of whitespace
- Replacement of carriage returns, tabs, and newlines with spaces
- Reduction of excessive punctuation
- Removal of spaces before punctuation
- Final whitespace normalization

Examples of punctuation normalization include:

```text
!!!!!!!!    → !!!
????????    → ???
........    → ...
```

The preprocessing preserves:

- Tamil characters
- English words
- Code-mixing
- Slang
- Punctuation
- Meaningful character elongation

## 12. Train, Development and Test Variables

The cleaned datasets are separated into input and target variables:

```text
X_train → Training text
y_train → Training labels

X_dev   → Development text
y_dev   → Development labels

X_test  → Test text
y_test  → Test labels
```

The development set is used for hyperparameter tuning and model selection.

The test set is reserved for final evaluation.

## 13. TF-IDF Feature Extraction

TF-IDF is used as the classical text representation.

The vectorizer uses:

```text
Unigrams + Bigrams
```

Configuration:

```python
TfidfVectorizer(lowercase=True, ngram_range=(1, 2), min_df=2, max_df=0.95, sublinear_tf=True)
```

The TF-IDF vectorizer is fitted only on the training data:

```python
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
```

The development and test sets are only transformed:

```python
X_dev_tfidf = tfidf_vectorizer.transform(X_dev)
X_test_tfidf = tfidf_vectorizer.transform(X_test)
```

This prevents information from the development or test sets from influencing the learned vocabulary and IDF values.

## 14. Classical Baseline Models

Two classical machine-learning classifiers are evaluated using the same TF-IDF representation.

### Logistic Regression

Logistic Regression is used because it:

- Works well with sparse text features
- Is computationally efficient
- Provides a linear decision boundary
- Provides class probabilities using `predict_proba()`

### Linear SVM

`LinearSVC` is used because it:

- Works effectively with high-dimensional sparse text features
- Uses a linear decision boundary
- Provides a strong classical alternative to Logistic Regression
- Is computationally efficient for TF-IDF representations

Both models use the same TF-IDF features.

## 15. Hyperparameter Tuning

The regularization parameter `C` is tuned for both models.

The candidate values are:

```python
C_values = [0.01, 0.1, 1, 10, 100]
```

Each candidate model is trained on the training set and evaluated on the development set.

The following metrics are recorded:

- Development Accuracy
- Development Macro F1
- Development Weighted F1

## 16. Model Selection Metric

Macro F1 is used as the primary model-selection metric.

Macro F1 calculates the F1 score independently for each class and then gives every class equal importance.

This is appropriate because the dataset contains six classes with class imbalance.

The selection process is:

```text
Training Set
     ↓
Train model with candidate C
     ↓
Development Set
     ↓
Calculate Macro F1
     ↓
Repeat for all C values
     ↓
Select highest Development Macro F1
```

The test set is not used during hyperparameter tuning.

## 17. Logistic Regression Tuning

Logistic Regression is trained for each candidate `C` value.

For every configuration:

```text
Training Set
     ↓
Logistic Regression
     ↓
Development Set
     ↓
Macro F1
```

The best configuration is stored as:

```text
best_logreg_model
best_C_logreg
best_macro_f1_logreg
```

The complete tuning results are stored in:

```text
logreg_dev_results_df
```

## 18. Linear SVM Tuning

`LinearSVC` is trained for each candidate `C` value.

For every configuration:

```text
Training Set
     ↓
Linear SVM
     ↓
Development Set
     ↓
Macro F1
```

The best configuration is stored as:

```text
best_svm_model
best_C_svm
best_macro_f1_svm
```

The complete tuning results are stored in:

```text
svm_dev_results_df
```

## 19. Classical Baseline Selection

The best Logistic Regression and best Linear SVM configurations are compared using their Development Macro F1 scores.

The model with the higher Development Macro F1 is selected as the final classical baseline.

```text
Best Logistic Regression Macro F1
              │
              ├──── Compare ─────┐
              │                   │
Best Linear SVM Macro F1          │
              │                   │
              └──── Select ───────┘
                       ↓
              Final Classical Baseline
```

Only the selected model is used for detailed final test evaluation.

## 20. Final Test Prediction

After selecting the final baseline model, predictions are generated on the test set:

```python
y_test_pred = final_baseline_model.predict(X_test_tfidf)
```

The test set is used only at this final stage.

## 21. Final Evaluation Metrics

The selected baseline is evaluated using:

- Accuracy
- Macro Precision
- Macro Recall
- Macro F1
- Weighted Precision
- Weighted Recall
- Weighted F1

### Macro Metrics

Macro averaging gives equal importance to every class.

It is useful for evaluating performance across both majority and minority classes.

### Weighted Metrics

Weighted averaging takes the number of samples in each class into account.

It provides an overall performance measure that reflects the actual class distribution.

Both Macro F1 and Weighted F1 are reported to provide complementary views of model performance.

## 22. Classification Report

A classification report is generated for the selected baseline model.

The report provides:

```text
Precision
Recall
F1-score
Support
```

for each class.

This helps identify which offensive-language categories are easier or more difficult for the model to classify.

## 23. Confusion Matrix

A confusion matrix is generated using the final test predictions.

```text
Rows    → Actual labels
Columns → Predicted labels
```

The matrix helps identify specific class confusions, particularly between different targeted offensive-language categories.

## 24. Normalized Confusion Matrix

A row-normalized confusion matrix is also generated.

Each row represents the distribution of predictions for one actual class.

This makes it easier to compare classification behavior across classes with different numbers of samples.

## 25. Final Baseline Performance Summary

A compact DataFrame is created containing:

```text
Model
Best C
Accuracy
Macro Precision
Macro Recall
Macro F1
Weighted F1
```

This provides a single summary of the selected classical baseline.

## 26. Feature Importance and Model Interpretation

The selected classical model is interpreted using its learned TF-IDF coefficients.

For each class, the strongest TF-IDF terms are identified.

These terms provide an indication of which words or n-grams contribute strongly toward each offensive-language category.

The interpretation is based on the coefficients of the linear model.

## 27. Test Error Analysis

Misclassified test samples are extracted by comparing:

```text
Actual Label
      vs.
Predicted Label
```

Only incorrectly classified samples are retained.

The error-analysis table contains:

```text
text
actual_label
predicted_label
```

These examples are manually inspected to identify common failure patterns.

Potential sources of errors include:

- Code-mixing
- Ambiguous language
- Slang
- Short comments
- Context-dependent expressions
- Similar offensive categories
- Fine-grained distinctions between targeted categories
- Annotation inconsistencies

## 28. Prediction Confidence

The method used for confidence analysis depends on the selected model.

### Logistic Regression

Logistic Regression provides class probabilities using:

```python
predict_proba()
```

The maximum predicted class probability is used as the confidence value.

```text
Confidence = Maximum predicted-class probability
```

### Linear SVM

`LinearSVC` does not provide `predict_proba()` by default.

Therefore, its `decision_function()` output is used instead.

The resulting value is explicitly treated as a:

```text
Decision Score
```

and not as a probability.

## 29. Saved Test Predictions

The test predictions and error-analysis results are saved for further analysis.

```text
results/
│
├── baseline_test_predictions.csv
└── baseline_errors.csv
```

### baseline_test_predictions.csv

Contains predictions for all test samples.

### baseline_errors.csv

Contains only the misclassified test samples.

## 30. Saved Model

The TF-IDF vectorizer and selected baseline model are serialized using `joblib`.

```text
models/
│
├── tfidf_vectorizer.pkl
└── best_tuned_baseline.pkl
```

### tfidf_vectorizer.pkl

Stores the fitted TF-IDF vocabulary and transformation configuration.

### best_tuned_baseline.pkl

Stores the selected Logistic Regression or Linear SVM model.

These files can later be loaded for inference without retraining.

## 31. Experiment Configuration

The experiment configuration is stored for reproducibility.

The configuration records:

- Random seed
- Dataset sizes
- TF-IDF parameters
- Selected model
- Selected `C`
- Selected Development Macro F1
- Best Logistic Regression `C`
- Best Logistic Regression Development Macro F1
- Best Linear SVM `C`
- Best Linear SVM Development Macro F1

## 32. Project Output Structure

After running the notebook, the project can contain:

```text
Project/
│
├── DravidianCodeMix/
│   ├── tamil_offensive_full_train.csv
│   ├── tamil_offensive_full_dev.csv
│   └── tamil_offensive_full_test.csv
│
├── data/
│   └── processed/
│       ├── train.csv
│       ├── dev.csv
│       └── test.csv
│
├── models/
│   ├── tfidf_vectorizer.pkl
│   └── best_tuned_baseline.pkl
│
├── results/
│   ├── baseline_test_predictions.csv
│   ├── baseline_errors.csv
│   └── baseline_experiment_config.csv
│
└── Tamil-English_Code-Mixed_Offensive_Language_Detection.ipynb
```

## 33. Reproducibility Workflow

To reproduce the classical baseline:

```text
1. Install dependencies
        ↓
2. Place the DravidianCodeMix dataset
        ↓
3. Load official train/dev/test files
        ↓
4. Inspect and clean the data
        ↓
5. Remove exact duplicates within each split
        ↓
6. Analyze cross-split overlap and label conflicts
        ↓
7. Perform exploratory data analysis
        ↓
8. Fit TF-IDF on training data
        ↓
9. Transform development and test data
        ↓
10. Tune Logistic Regression on Dev
        ↓
11. Tune Linear SVM on Dev
        ↓
12. Compare best Macro F1 values
        ↓
13. Select the final classical baseline
        ↓
14. Evaluate once on the Test set
        ↓
15. Generate classification report
        ↓
16. Generate confusion matrices
        ↓
17. Perform error analysis
        ↓
18. Save predictions and model
```

## 34. Key Design Decisions

### Official Data Splits

The official train, development, and test partitions are retained.

This allows the classical baseline and Transformer model to use the same data partitions.

### Minimal Preprocessing

Aggressive text normalization is avoided because Tamil-English code-mixed text contains useful linguistic and stylistic information.

### TF-IDF Representation

TF-IDF with unigrams and bigrams provides a simple, interpretable, and computationally efficient classical representation.

### Two Classical Models

Logistic Regression and Linear SVM provide two different linear classification approaches using exactly the same features.

### Development-Based Tuning

Hyperparameters are selected using the development set rather than the test set.

### Macro F1 for Model Selection

Macro F1 is used as the primary selection criterion because all six classes should receive equal importance despite class imbalance.

### Test Set Isolation

The test set is reserved for final evaluation after model selection.

## 35. Classical Baseline Summary

The implemented classical baseline consists of:

```text
Input
  ↓
Tamil-English Social-Media Text
  ↓
Lightweight Preprocessing
  ↓
TF-IDF Unigrams + Bigrams
  ↓
┌──────────────────────┐
│ Logistic Regression  │
│        OR            │
│ Linear SVM           │
└──────────────────────┘
  ↓
Development Macro F1
  ↓
Select Best Baseline
  ↓
Final Test Evaluation
  ↓
Offensive-Language Category
```

The resulting classical baseline establishes a reference performance level for comparison with the Transformer-based model.

## 36. Next Stage

The classical baseline serves as the foundation for the next stage of the project:

```text
Classical Baseline
        ↓
Transformer Model
        ↓
Performance Comparison
        ↓
Error Analysis
        ↓
End-to-End NLP Application
```

The Transformer component will use the same processed train, development, and test partitions to ensure a consistent comparison between the classical and Transformer-based approaches.
