# Dataset Notes

> **Note (added later):** these are raw exploratory notes from the
> project's first pass — written before the chronological-split leakage
> bugs were found and fixed, using a default 0.50 threshold that was
> later replaced. Left as-is intentionally, as a genuine record of the
> early process. See `paper_draft/case_study.md` for what was actually
> found on rigorous re-validation, including a specific correction to
> the geographic findings below — the "Geographic Supply Stress
> Exploration" section reports a store-volume confound as if it were a
> genuine effect, which is exactly the flaw an external reviewer later
> caught and this project fixed.

## Dataset

M5 Forecasting Accuracy Dataset

## Initial Questions

-   What defines a shortage event?
-   Can demand spikes be used as a proxy?
-   Which products exhibit the highest volatility?

## Evaluation Ideas

-   Temporal split
-   Missing data robustness
-   Demand shock simulation
-   Cross-category generalization

## Initial Stress Event Definition

A stress event occurs when daily sales exceed the item's historical 90th
percentile.

### Rationale

-   Identifies unusually high demand periods.
-   Acts as a proxy for inventory pressure.
-   Can be computed consistently across products.
-   Creates a reproducible target variable for experimentation.

## Lessons Learned

### Data Representation

The original M5 dataset is stored in a wide format, with one column per
day.

To support machine learning workflows, the data was reshaped into a long
format using `pandas.melt()`, resulting in one row per product-day
observation.

### Scalability Observation

The full transformed dataset consumed approximately 22.8 GB of memory.

To enable rapid experimentation, an initial subset of 100 products was
selected, reducing memory usage to approximately 0.75 GB.

### Stress Event Definition

Because the dataset does not contain inventory levels or explicit
stockout events, a proxy target is required.

The initial definition of a stress event is:

-   Daily sales exceed the item's historical 90th percentile.

This identifies unusually high demand periods that may correspond to
inventory pressure.

## Initial Class Balance

After defining stress events as:

sales \> item-specific 90th percentile

the resulting class distribution was:

-   Normal days (0): 93.9%
-   Stress days (1): 6.1%

Observation:

The positive class is relatively rare but still sufficiently represented
for experimentation. This suggests the initial threshold produces a
meaningful distinction between typical demand and unusually high demand.

## Baseline Logistic Regression Results

Unweighted logistic regression achieved high overall accuracy (94%) but
failed to detect stress events (recall ≈ 0).

Applying class balancing reduced overall accuracy to 78% but improved
stress-event recall to 30%.

Interpretation:

This highlights the importance of class imbalance handling in
supply-chain stress prediction. Traditional accuracy metrics may obscure
poor rare-event performance.

## Model Comparison --- Baseline Experiments

### Logistic Regression (Unweighted)

Results:

-   Accuracy: 94%
-   Stress-event recall: \~0%

Observation:

Although overall accuracy was high, the model failed to detect rare
stress events. This demonstrates that accuracy alone is not sufficient
for evaluating imbalanced supply-chain risk problems.

------------------------------------------------------------------------

### Logistic Regression (Balanced)

Results:

-   Accuracy: 78%
-   Stress-event recall: 30%
-   Stress-event precision: 10%

Observation:

Applying class weighting significantly improved rare-event detection at
the cost of lower overall accuracy. This confirms the importance of
explicitly handling class imbalance.

------------------------------------------------------------------------

### Random Forest (Balanced)

Results:

-   Accuracy: 38%
-   Stress-event recall: 87%
-   Stress-event precision: 8%

Observation:

Random Forest dramatically improved recall, capturing most stress
events, but at the expense of many false positives. This indicates a
strong preference toward aggressive shortage detection.

------------------------------------------------------------------------

## Business Tradeoff Interpretation

In supply-chain settings:

-   False Positive → unnecessary extra inventory
-   False Negative → missed shortage / stockout

False negatives are typically more costly than false positives.

This suggests recall may be a more important optimization target than
raw accuracy.

------------------------------------------------------------------------

## Feature Importance Findings

Random Forest feature importance:

-   rolling_mean_7 → 69.4%
-   sales_lag_1 → 18.9%
-   sales_lag_7 → 11.7%

Interpretation:

Recent rolling demand trends appear to be substantially more predictive
of stress events than isolated single-day observations.

This suggests that sustained elevated demand may be a stronger precursor
to supply stress than abrupt one-day spikes.

## Volatility Feature Experiment

Added:

-   rolling_std_7 (7-day rolling standard deviation)

Purpose:

To measure short-term sales volatility and test whether instability
contributes to supply stress.

Results:

Random Forest performance remained largely unchanged:

-   Stress recall: 86%
-   Stress precision: 8%

However, feature importance changed substantially:

-   rolling_mean_7 → 48.0%
-   rolling_std_7 → 40.8%
-   sales_lag_1 → 9.3%
-   sales_lag_7 → 1.9%

Interpretation:

Volatility appears to be a major explanatory feature for stress-event
prediction, nearly as important as recent average demand.

This suggests that supply stress may be driven by both sustained demand
pressure and demand instability.

## Calendar Feature Enrichment

Added:

-   is_event_day
-   is_weekend
-   SNAP indicators (CA, TX, WI)

Results:

Random Forest improved significantly:

-   Accuracy: 61%
-   Stress recall: 60%
-   Stress precision: 9%

Key findings:

-   Weekend effect became a major predictor (12.3% importance)
-   Event days contributed minimally
-   SNAP indicators had low influence

Interpretation:

This suggests that supply stress may follow regular weekly behavioral
patterns more strongly than formal holiday events or benefit cycles.

This improves both predictive quality and business interpretability.

## Price Enrichment Experiment

Added:

-   sell_price
-   price_change_1

Results:

Random Forest improved:

-   Accuracy: 65%
-   Stress recall: 46%
-   Stress precision: 10%

Feature importance:

-   sell_price → 44.2%
-   is_weekend → 19.8%
-   rolling_mean_7 → 12.4%
-   rolling_std_7 → 12.2%

Key finding:

Price became the strongest predictive feature.

Interpretation:

Supply stress appears to be highly associated with price level,
suggesting pricing may act as a proxy for scarcity, demand intensity, or
promotional behavior.

## Discount Feature Exploration

### Objective

Investigate whether price discounts are associated with an increased
likelihood of supply stress events.

### Methodology

-   Created a binary feature `is_discount` using the weekly percentage
    price change:

    ``` python
    is_discount = (price_pct_change_1 < 0).astype(int)
    ```

-   A value of `1` indicates the product's price decreased compared to
    the previous week, while `0` indicates no discount.

### Findings

-   Discounts were extremely rare, occurring in only **0.04%** of
    observations.
-   Despite their rarity, discounted observations exhibited a higher
    stress-event rate:
    -   **No Discount:** 6.08%
    -   **Discount:** 8.03%

### Interpretation

-   Discounted products were approximately **32% more likely** to
    experience a stress event than non-discounted products.
-   This suggests an association between promotional pricing and supply
    stress, although it does **not** establish a causal relationship.
-   One possible explanation is that promotions increase customer
    demand, which may contribute to temporary supply shortages.

### Conclusion

-   The `is_discount` feature was retained for model training as a
    business-relevant feature.
-   Although discounts occur infrequently, they may provide useful
    predictive information when combined with demand history, calendar
    features, and pricing variables.

## Price Direction Analysis

### Objective

Investigate whether different types of price movements are associated
with supply stress events.

### Methodology

-   Created a categorical feature `price_direction` from the weekly
    percentage price change (`price_pct_change_1`):
    -   **Decrease**: `price_pct_change_1 < 0`
    -   **Increase**: `price_pct_change_1 > 0`
    -   **No Change**: `price_pct_change_1 == 0` or missing

### Findings

  Price Direction     Stress Event Rate
  ----------------- -------------------
  No Change                       6.08%
  Decrease                        8.03%
  Increase                        9.89%

### Interpretation

-   Products experiencing **price increases** exhibited the highest
    stress-event rate (9.89%).
-   Products with **price decreases** also showed an elevated
    stress-event rate (8.03%) compared to products with no price changes
    (6.08%).
-   This suggests that **price movements, regardless of direction, are
    associated with periods of increased supply stress**.
-   A possible explanation is that pricing decisions often occur in
    response to changing market conditions such as promotions, demand
    surges, or inventory constraints.

### Conclusion

-   Price movement appears to be a more informative signal than price
    level alone.
-   Future models should evaluate whether `price_direction` provides
    greater predictive value than separate discount or price-increase
    indicators.

## Price Feature Evaluation

### Objective

Evaluate whether pricing information improves supply stress prediction.

### Methodology

Added two pricing-related features:

-   `sell_price`
-   `price_change_1`

These were incorporated into the Random Forest classifier alongside
historical demand, volatility, and calendar features.

### Results

Compared with the previous Random Forest model:

-   Stress-event recall improved from **46% to 60%**.
-   Stress-event F1-score increased from **0.16 to 0.18**.
-   Overall accuracy decreased from **65% to 59%**, reflecting the
    model's increased focus on detecting rare stress events.

### Feature Importance

  Feature            Relative Importance
  ---------------- ---------------------
  sell_price                       0.494
  is_weekend                       0.175
  rolling_std_7                    0.113
  rolling_mean_7                   0.106
  sales_lag_1                      0.053
  sales_lag_7                      0.024
  price_change_1                   0.004

### Interpretation

The current selling price emerged as the strongest predictor of supply
stress, accounting for nearly half of the model's feature importance. In
contrast, short-term price changes contributed very little additional
predictive value.

These findings suggest that the absolute price level contains
substantially more predictive information than recent price movements.

## Geographic Supply Stress Exploration

### Objective

Evaluate whether geographic information (state and store) is associated
with supply-chain stress events.

### Results

#### State-Level Analysis

  State     Stress Event Rate
  ------- -------------------
  CA                    7.53%
  TX                    5.61%
  WI                    4.61%

California exhibited the highest stress-event rate, while Wisconsin
exhibited the lowest. The approximately 63% difference between the
highest and lowest states suggests that geographic location captures
meaningful variation in supply-chain behavior.

#### Store-Level Analysis

  Store     Stress Event Rate
  ------- -------------------
  CA_3                 10.65%
  CA_1                  8.97%
  TX_2                  7.43%
  WI_1                  6.48%
  CA_2                  5.88%
  TX_3                  5.79%
  CA_4                  4.63%
  WI_3                  4.29%
  TX_1                  3.62%
  WI_2                  3.04%

Stress-event rates varied substantially across stores, ranging from
approximately 3% to over 10%. The highest-stress store (CA_3)
experienced supply stress more than three times as frequently as the
lowest-stress store (WI_2).

### Interpretation

These findings indicate that geographic variables contain meaningful
predictive information. Differences between stores and states may
reflect regional demand patterns, customer behavior, replenishment
schedules, inventory policies, or distribution network characteristics.

Although this exploratory analysis does not establish causality, it
demonstrates that location-based features are likely to improve
predictive performance.

> **Correction (added later):** this interpretation turned out to be
> wrong in an important way. The store-level rate spread seen here
> (3%–10%+) was later shown to correlate with store sales *volume* at
> Pearson r=0.85 — high-volume stores were crossing the stress threshold
> more often simply because the threshold was computed by pooling sales
> across all 10 stores per item, not because of any genuine geographic
> effect. Grouping the threshold by `(item_id, store_id)` instead of
> `item_id` alone dropped that correlation to r=0.03. Location features
> may still carry some real signal, but the effect size reported here is
> substantially a labeling artifact, not the geographic finding it's
> described as. See `paper_draft/case_study.md`, "Scaling past 100 items
> — and a critique that held up."

### Modeling Decision

Based on these findings, `state_id` and `store_id` were selected for
inclusion in the next iteration of the Random Forest model. Because both
variables have relatively low cardinality, one-hot encoding was chosen
as the encoding strategy.

## Geographic Feature Engineering

To capture regional differences in supply stress, geographic information
was incorporated into the feature set using one-hot encoding. The
categorical variables `state_id` and `store_id` were transformed into
binary indicator variables using `pd.get_dummies()` with
`drop_first=True` to avoid redundant dummy variables.

The resulting model included:

-   2 state-level indicator features (`TX`, `WI`; `CA` used as the
    baseline)
-   9 store-level indicator features (with `CA_1` used as the baseline)

This expanded the final feature matrix to 22 predictor variables while
maintaining a fully numeric dataset suitable for machine learning.

Model validation confirmed:

-   1,592,450 observations
-   22 engineered features
-   No missing values
-   Fully numeric feature matrix (float64 and int64 only)

This preprocessing step prepared the dataset for production-style
machine learning workflows.

------------------------------------------------------------------------

## Model Performance After Adding Geographic Features

After incorporating geographic features, the Random Forest model
demonstrated its strongest performance to date.

Classification Results:

-   Accuracy: **67%**
-   Precision (Stress Event): **13%**
-   Recall (Stress Event): **58%**
-   F1 Score (Stress Event): **0.21**

Compared with previous iterations, the inclusion of location information
improved precision and overall F1 score while maintaining strong recall.
This suggests that geographical context contributes meaningful
predictive information beyond historical sales, calendar effects, and
pricing variables.

------------------------------------------------------------------------

## Geographic Feature Importance

Feature importance analysis revealed that geographic variables became
some of the strongest predictors of supply stress.

Top features included:

1.  sell_price
2.  store_id_CA_3
3.  is_weekend
4.  rolling_mean_7
5.  store_id_WI_2
6.  rolling_std_7
7.  store_id_TX_1

Several store-specific variables ranked above traditional lag features,
indicating that certain stores consistently exhibit different supply
stress behavior.

This suggests that operational factors associated with individual store
locations---such as local demand patterns, replenishment policies,
distribution logistics, or regional purchasing behavior---contain
valuable predictive signal.

While the model cannot determine the underlying business cause, it
successfully identifies location as an important contributor to
predicting supply stress.

------------------------------------------------------------------------

## Key Observations

Several notable patterns emerged during feature engineering:

-   Current selling price remained the single most informative feature.
-   Historical demand (rolling averages and rolling standard deviation)
    continued to provide strong predictive value.
-   Store-level indicators contributed more predictive information than
    state-level indicators.
-   Weekly price changes contributed relatively little additional
    predictive power compared with absolute selling price.
-   Geographic context substantially improved the model's ability to
    distinguish between high-risk and low-risk observations.

These findings reinforce that supply shortages are influenced by a
combination of historical demand, pricing strategy, calendar effects,
and local operational characteristics rather than any single factor
alone.

## Random Forest Hyperparameter Optimization

To improve model performance, hyperparameters were optimized using
`GridSearchCV` with 3-fold cross-validation. Because the target variable
was highly imbalanced, model selection was based on the F1 score rather
than overall accuracy.

Parameter grid:

-   `n_estimators`: \[100, 200\]
-   `max_depth`: \[10, 15, None\]
-   `min_samples_leaf`: \[1, 5\]
-   `max_features`: \["sqrt"\]

The best-performing model selected the following configuration:

-   `n_estimators = 200`
-   `max_depth = 15`
-   `min_samples_leaf = 5`
-   `max_features = "sqrt"`

Best cross-validation F1 score:

-   **0.220**

Evaluation on the held-out test set produced:

-   Accuracy: **71%**
-   Precision (Stress Event): **14%**
-   Recall (Stress Event): **56%**
-   F1 Score (Stress Event): **0.22**

Compared with the baseline Random Forest, hyperparameter tuning improved
overall accuracy, precision, and F1 score while maintaining strong
recall. Increasing the number of trees and introducing a minimum leaf
size produced a model that generalized better to unseen data without
substantially sacrificing sensitivity to supply stress events.

## SHAP Explainability Analysis

To better understand the Random Forest model beyond traditional feature
importance scores, SHAP (SHapley Additive exPlanations) was used to
quantify how each feature influenced individual predictions.

### Why SHAP?

Traditional Random Forest feature importance indicates which variables
are important but does not explain:

-   whether high feature values increase or decrease risk,
-   how features influence individual predictions,
-   or whether learned relationships are consistent with business
    intuition.

SHAP addresses these limitations by assigning each feature a
contribution toward the model's prediction.

------------------------------------------------------------------------

## Key Findings

### Sell Price

Sell price remained the most influential predictor.

SHAP analysis showed that:

-   Lower selling prices generally pushed predictions toward higher
    supply stress.
-   Higher selling prices tended to reduce predicted supply stress.

This relationship is likely associative rather than causal. Lower prices
may coincide with promotions, increased customer demand, or faster
inventory depletion rather than directly causing shortages.

------------------------------------------------------------------------

### Weekend Effect

The `is_weekend` feature consistently increased predicted supply stress.

This aligns with expectations that consumer purchasing activity
increases during weekends, placing greater pressure on inventory levels.

------------------------------------------------------------------------

### Demand Features

The rolling demand statistics continued to provide meaningful predictive
information.

-   Higher recent average sales (`rolling_mean_7`) generally increased
    predicted stress.
-   Greater demand volatility (`rolling_std_7`) also contributed
    positively toward stress predictions.

These findings support the original hypothesis that sustained or
volatile demand is associated with higher inventory risk.

------------------------------------------------------------------------

### Geographic Features

Several location-based features demonstrated meaningful predictive
power.

Examples include:

-   Store CA_3 consistently increased predicted stress.
-   Wisconsin stores generally reduced predicted stress.
-   Certain Texas stores also contributed negatively toward stress
    predictions.

These observations agree with earlier exploratory analysis showing that
supply stress varies across geographic regions and individual stores.

------------------------------------------------------------------------

## Overall Interpretation

The SHAP explanations indicate that the model is relying primarily on
interpretable business signals rather than arbitrary statistical
patterns.

The strongest drivers of predicted supply stress include:

-   lower selling prices,
-   weekend periods,
-   higher recent demand,
-   more volatile demand,
-   and specific high-risk store locations.

Conversely, higher prices, stable demand, weekdays, and several
Wisconsin and Texas stores generally reduced predicted supply stress.

The consistency between SHAP explanations and earlier exploratory data
analysis increases confidence that the model is learning meaningful
operational relationships rather than overfitting random noise.

------------------------------------------------------------------------

## Final Project Outcome

The project now represents a complete end-to-end machine learning
workflow.

Completed stages include:

-   exploratory data analysis
-   target engineering
-   feature engineering
-   Random Forest benchmarking
-   hyperparameter tuning
-   SHAP explainability
-   chronological validation
-   XGBoost benchmarking
-   threshold optimization
-   production refactoring
-   reusable prediction pipeline
-   model persistence
-   deployment-ready project structure

Random Forest served as the primary benchmark model throughout
development. After chronological validation and model comparison,
XGBoost was selected as the final production model because it achieved
the strongest balance of minority-class recall and F1 score while
supporting a production-ready inference pipeline.

The repository now separates experimentation from production code,
making it easier to maintain, extend, and deploy.

## Chronological Validation

After hyperparameter tuning, the Random Forest model was re-evaluated
using a chronological train-test split rather than a random split.
Earlier observations were used for training, while later observations
were reserved for testing to better simulate a real forecasting
scenario.

### Results

Random Split

-   Accuracy: 71%
-   Stress Precision: 14%
-   Stress Recall: 56%
-   Stress F1: 0.22

Chronological Split

-   Accuracy: 69%
-   Stress Precision: 14%
-   Stress Recall: 55%
-   Stress F1: 0.22

### Interpretation

Model performance remained remarkably stable under chronological
validation. The nearly identical F1 score indicates that the model
generalizes well to future observations and is not heavily dependent on
information leakage introduced by random train-test splits.

This increases confidence that the engineered demand, pricing, calendar,
and location features capture persistent operational relationships
rather than historical noise.

## XGBoost Model Benchmarking

To determine whether a gradient-boosting algorithm could outperform the
tuned Random Forest model, an XGBoost classifier was evaluated using the
same chronological train-test split. This provided a fair comparison
between the two ensemble methods under identical forecasting conditions.

### Initial Baseline

The first XGBoost model was trained using standard parameters:

-   n_estimators = 200
-   max_depth = 6
-   learning_rate = 0.05
-   subsample = 0.8
-   colsample_bytree = 0.8

### Initial Results

  Metric                 Value
  -------------------- -------
  Accuracy                 92%
  Precision (Stress)      0.62
  Recall (Stress)         0.00
  F1 (Stress)             0.00

Although overall accuracy was very high, the model almost never
predicted the minority (stress) class. This occurred because the dataset
is highly imbalanced, causing the model to favor the majority class in
order to maximize overall accuracy. As a result, the model was
unsuitable for the business objective of detecting supply stress.

### Handling Class Imbalance

To improve minority-class detection, the `scale_pos_weight` parameter
was introduced. The weight was calculated directly from the training
data as the ratio of majority-class to minority-class observations:

``` python
scale_pos_weight = (
    (y_train_time == 0).sum() /
    (y_train_time == 1).sum()
)
```

This increased the penalty for misclassifying stress events during model
training.

### Balanced XGBoost Results

  Metric                 Value
  -------------------- -------
  Accuracy                 66%
  Precision (Stress)      0.14
  Recall (Stress)         0.63
  F1 (Stress)             0.23

Introducing `scale_pos_weight` dramatically improved the model's ability
to identify stress events. Although overall accuracy decreased, recall
increased from 0.00 to 0.63 and the minority-class F1 score improved
from 0.00 to 0.23.

### Comparison with Random Forest

  Model                   Accuracy     Recall         F1
  --------------------- ---------- ---------- ----------
  Tuned Random Forest          69%       0.55       0.22
  Balanced XGBoost             66%   **0.63**   **0.23**

Both models performed similarly; however, the balanced XGBoost model
achieved the highest recall and F1 score for the minority class. Because
identifying supply stress events is the primary business objective,
XGBoost was selected as the preferred model for subsequent analysis.

### Key Takeaway

This experiment demonstrated that overall accuracy can be misleading
when working with highly imbalanced datasets. Explicitly accounting for
class imbalance using `scale_pos_weight` substantially improved
XGBoost's ability to detect rare supply stress events, making it a more
appropriate model for the forecasting task.

## Confusion Matrix Interpretation

A confusion matrix was generated for the final tuned XGBoost model using
the chronological test period.

  Actual / Predicted     No Stress   Stress
  -------------------- ----------- --------
  No Stress                 \~196k    \~97k
  Stress                     9,488   16,069

The model correctly identified over 16,000 supply stress events while
missing approximately 9,500, resulting in a recall of 63% for the
minority class.

The model also produced a substantial number of false positives. This
reflects an intentional trade-off resulting from class balancing
(`scale_pos_weight`) and the optimization objective focused on
minority-class F1 rather than overall accuracy.

For an early-warning forecasting system, prioritizing the detection of
potential supply stress is often preferable to maximizing overall
accuracy, as the operational cost of missing genuine disruptions may
exceed the cost of investigating false alarms.

The confusion matrix confirms that the final model behaves as intended
by favoring higher sensitivity to supply stress events.

## Threshold Optimization and Business Trade-offs

After tuning the XGBoost model using RandomizedSearchCV, model
performance was further evaluated by adjusting the probability
classification threshold. Rather than automatically classifying
observations with probabilities above 0.50 as stress events, the
decision threshold was optimized using the Precision--Recall curve to
maximize the F1 score.

### Precision--Recall Analysis

Because supply stress events are relatively uncommon, the
Precision--Recall curve provides a more informative evaluation than
overall accuracy. The tuned XGBoost model achieved:

-   Average Precision (AP): **0.186**
-   Best F1 threshold: **0.627**

This threshold produced the following performance:

  Metric        Value
  ----------- -------
  Precision      0.20
  Recall         0.36
  F1-score       0.26

Compared with the default threshold (0.50), increasing the threshold
improved precision (14% → 20%) and minority-class F1 score (0.23 →
0.26), while reducing recall (63% → 36%).

### Business Interpretation

Adjusting the classification threshold does not change the underlying
model---it only changes how predicted probabilities are converted into
class labels.

A lower threshold generates more alerts, increasing the likelihood of
detecting supply stress events but also producing more false positives.

A higher threshold makes the model more conservative, reducing false
alarms but increasing the number of missed stress events.

### Final Decision

Although the optimized threshold produced a higher minority-class F1
score, the default threshold of **0.50** was retained as the preferred
operating point for this project.

The primary objective is to identify potential supply stress events as
early as possible. In supply chain monitoring, missing a genuine stress
event can be more costly than investigating additional false positives.
Therefore, prioritizing recall provides greater practical value for an
early-warning system.

This experiment demonstrates that model deployment decisions should be
driven not only by statistical metrics, but also by the underlying
business objectives and operational trade-offs.

# Production Pipeline

After completing model development, the project was refactored into a
production-style machine learning pipeline.

The project was separated into reusable Python modules:

    src/
        load_data.py
        preprocess.py
        features.py
        train.py
        evaluate.py
        predict.py

Benefits:

-   reusable code
-   easier testing
-   consistent feature engineering
-   cleaner notebooks
-   simpler deployment
-   reduced code duplication

# Shared Feature Engineering

Training and inference now use the exact same feature engineering
functions, eliminating training-serving skew.

The shared pipeline creates:

-   lag features
-   rolling statistics
-   calendar features
-   pricing features
-   geographic encoding

# Model Persistence

The final XGBoost model is exported together with:

-   trained model
-   feature schema
-   default threshold
-   alternative F1 threshold

# Production Inference

A reusable prediction pipeline now:

1.  Loads the trained model.
2.  Loads metadata.
3.  Engineers features.
4.  Aligns columns.
5.  Predicts probabilities.
6.  Returns business-friendly risk labels.

# Future Improvements

> **Status update (added later):** several of these have since been
> done. Left the original list intact below for the historical record,
> with status noted inline.

-   Replace the proxy target with verified stockout labels. *(Still
    open — no inventory data exists in M5.)*
-   Add inventory and supplier lead-time features. *(Still open.)*
-   Calibrate probabilities. *(Done — `scripts/deploy_calibrated_model.py`.
    Confirmed genuinely miscalibrated (raw "0.90" was really ~25-28%
    likely), fixed via isotonic calibration, and shipped as the actual
    deployed model. See README "Probability Calibration".)*
-   Monitor model drift. *(Done — `src/monitoring.py`, KS test/PSI,
    `POST /check-drift` endpoint.)*
-   Add scheduled batch scoring. *(Still open.)*
-   Compare with LightGBM and CatBoost. *(Done —
    `scripts/model_comparison.py`. XGBoost wins on average precision
    (0.236 vs 0.230 LightGBM vs 0.220 CatBoost); ROI-optimal net is
    within noise across all three ($0/$0/$16). XGBoost's selection is
    justified, not arbitrary — see README "Model Comparison".)*
-   Build a Streamlit dashboard. *(Done — see `frontend/`.)*
-   Deploy as a REST API. *(Done — FastAPI, containerized, see
    `api/`.)*

Also since completed but not anticipated in this list: two chronological
data-leakage bugs found and fixed with regression tests, dependency
lockfiles pinned and CI-verified, a store-volume confound in the stress
target found (by external review) and fixed, validation extended from
100 items to the complete 30,490-series catalog, and a dbt/Snowflake
data warehouse layer built and run against real data. See
`paper_draft/case_study.md` for the full story.
