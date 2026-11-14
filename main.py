# CapitolRegime
# Forecasting future VIX volatility regimes using congressional stock disclosures,
# historical market data, and machine learning.

# Author: Kyle Lund


# Imports
import sys
from pathlib import Path
from sklearn.pipeline import Pipeline
import alpha_prep
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit, cross_validate
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import logging
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    balanced_accuracy_score,
    confusion_matrix,
    roc_auc_score
)



# ------------------------------------------- Methods-------------------------------------------------------------


# Plot recent VIX history against the model's predicted regime
def plot_vix_prediction(prediction_row, predicted_regime, elevated_probability):
    # Prepare one chronological VIX observation per disclosure date
    vix_history = master_df[["disclosureDate", "VIX_close"]].copy()
    vix_history["disclosureDate"] = pd.to_datetime(vix_history["disclosureDate"])
    vix_history = (
        master_df[["disclosureDate", "VIX_close"]]
        .drop_duplicates("disclosureDate")
        .sort_values("disclosureDate")
        .reset_index(drop=True)
    )

    # Extract transaction's disclosure date and VIX values around that date
    d_date = pd.to_datetime(prediction_row["disclosureDate"].iloc[0])
    current_vix = prediction_row['VIX_close'].iloc[0]
    prior_change = prediction_row['VIX_pct_change_10d'].iloc[0] * 100

    # Assign the predicted regime's label and display color
    if predicted_regime == 0:
        box_color = "steelblue"
        predicted_regime = "Normal"
    else:
        box_color = "firebrick"
        predicted_regime = "Elevated"

    # Find the historical observation closest to the disclosure date
    closest_idx = (
        vix_history["disclosureDate"]
        .sub(d_date)
        .abs()
        .idxmin()
    )

    # Display 10 trading days of VIX history before and 10 days after the closest observation
    start_idx = max(0, closest_idx - 10)
    end_idx = min(len(vix_history), closest_idx + 11)
    plot_df = vix_history.iloc[start_idx:end_idx]

    # Create the chart and plot historical VIX values
    fig, ax = plt.subplots(figsize=(10, 5))

    # Plot historical VIX
    ax.plot(
        plot_df["disclosureDate"],
        plot_df["VIX_close"],
        label=f"VIX at Disclosure: {prediction_row['VIX_close'].iloc[0]}",
        color="salmon"
    )

    # Mark a horizontal threshold separating Normal and Elevated VIX regimes
    ax.axhline(
        18,
        linestyle="--",
        label="Regime threshold (18)",
        color="blue"
    )

    # Mark a vertical line for when the transaction was disclosed
    ax.axvline(
        d_date,
        linestyle=":",
        label=f"Disclosure Date: {d_date.strftime('%Y-%m-%d')}",
        color="darkgray"
    )

    # Display the predicted regime and supporting market information
    ticker = prediction_row['ticker'].iloc[0]
    transaction_size = prediction_row['transaction_size'].iloc[0]
    transaction_type = prediction_row['transactionType'].iloc[0]

    output = (
        f"Transaction: {ticker} - ~${transaction_size:,.0f} | "
        f"Prior 10D VIX Change: {prior_change:.1f}%"
    )
    prediction_text = (
        f"VIX Regime Forecast (10D):   {predicted_regime}    |    "
        f"Confidence: {elevated_probability}%\n"
        f"Transaction: {ticker} - {transaction_type} ~${transaction_size:,.0f} | "
        f"Prior 10D VIX Change: {prior_change:.1f}%"
    )

    ax.text(
        0.98,
        0.95,
        prediction_text,
        transform=ax.transAxes,
        ha="right",
        va="top",
        bbox=dict(boxstyle="round", color=box_color, alpha=0.69)
    )

    ax.set_title("VIX Market Regime Prediction")
    ax.set_xlabel("Date")
    ax.set_ylabel("VIX")

    # Focus window view on the historical VIX observations surrounding disclosure
    ax.set_xlim(
        plot_df["disclosureDate"].min(),  # left edge
        d_date + pd.Timedelta(days=3)  # right edge
    )

    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    # Render graph
    plt.show()


# Log a comparison of evaluation metrics across models
def compare_results(results_list):
    # Create list of metrics to be evaluated
    metric_columns = [
        "accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
        "balanced_accuracy"
    ]

    comparison = (
        pd.DataFrame(results_list)
        .set_index("model")
    )

    logging.info(
        "\nMODEL COMPARISON (Test Set):\n\n%s",
        comparison[metric_columns].round(4).to_string()
    )


# Calculates evaluation metrics and returns a results dictionary
def evaluate_model(model_name, predictions, probabilities, y_test):
    return {
        "model": model_name,
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions),
        "recall": recall_score(y_test, predictions),
        "f1": f1_score(y_test, predictions),
        "roc_auc": roc_auc_score(y_test, probabilities),
        "balanced_accuracy": balanced_accuracy_score(y_test, predictions),
        "confusion_matrix": confusion_matrix(y_test, predictions)
    }


# Logs evaluation metrics from evaluate_model()
def log_results(results):
    output = [
        "\n",
        f"MODEL RESULTS: {results['model']}",
        "-" * 42,
        f"Accuracy:           {results['accuracy']:.4f}",
        f"Precision:          {results['precision']:.4f}",
        f"Recall:             {results['recall']:.4f}",
        f"F1 Score:           {results['f1']:.4f}",
        f"ROC AUC:            {results['roc_auc']:.4f}",
        f"Balanced Accuracy:  {results['balanced_accuracy']:.4f}",
        "\n"
    ]

    logging.info("\n".join(output))


# Log cross-validation metrics and average performance
def log_cv_results(model_name, cv_results, metrics, best_index=None):
    log_message = f"\nCROSS-VALIDATION: {model_name}\n"
    log_message += "-" * 42 + "\n"

    for metric in metrics:
        if best_index is None:
            # Results from cross_validate()
            scores = cv_results[f"test_{metric}"]
        else:
            # Results from GridSearchCV, using the winning parameters
            scores = [
                cv_results[f"split{fold}_test_{metric}"][best_index]
                for fold in range(5)
            ]

        mean_score = sum(scores) / len(scores)

        log_message += f"{metric.replace('_', ' ').title()}: {mean_score:.4f}\n"

    # Log the entire summary with one timestamp.
    logging.info(log_message)


# Logs the hyper-parameters selected for a model based on the mean cross-validation ROC AUC.
def log_tuning_results(model_name, grid_search):
    output = [
        "\n",
        f"HYPERPARAMETER TUNING: {model_name}",
        "-" * 42,
        f"Best Parameters: {grid_search.best_params_}",
        f"Best CV ROC AUC: {grid_search.best_score_:.4f}",
        ""
    ]

    logging.info("\n".join(output))


# Generates a VIX regime prediction for a transaction, displays the graph when available,
# and prints a text summary of the prediction and relevant market data.
def predict_transaction(model_name, model, row):
    # Return None if the row is empty
    if row is None:
        return None

    # Define transaction row X-set for prediction
    X = row.drop(
        columns=[
            "member.name",
            "ticker",
            "transactionDate",
            "disclosureDate",
            "transactionType",
            # Target Feature
            "VIX_regime_10d"
        ]
    )

    # Generate prediction; return none if generation fails
    try:
        prediction = model.predict(X)[0]
        probabilities = model.predict_proba(X)[0]
        confidence = f"{probabilities[prediction] * 100:.2f}%"

    except Exception as e:
        print(f"ERROR: Could not generate prediction: {e}")
        print(f"Failed prediction for transaction: {row.to_string()}")
        return None

    # Attempt to display the graph
    try:
        plot_vix_prediction(
            row,
            prediction,
            confidence,
        )
    except Exception as err:
        # Log detailed traceback for developers/debugging
        logging.warning(
            "Failed to render VIX prediction graph: %s", err, exc_info=True
        )
        # Clean user notification
        print("\n[Notice] Graphical window unavailable. See text summary below.")

    # Print a text output version of predictions to user
    # Extract values from the single-row DataFrame

    current_vix = row["VIX_close"].iloc[0]
    vix_change = row["VIX_pct_change_10d"].iloc[0] * 100
    disclosure_date = pd.to_datetime(row["disclosureDate"].iloc[0]).strftime("%Y-%m-%d")
    transaction_type = row["transactionType"].iloc[0]
    ticker = row["ticker"].iloc[0]
    transaction_size = row["transaction_size"].iloc[0]

    # Print prediction details when graph fails or is closed
    print(f"\nModel: {model_name}")
    print("=" * 30)
    print(f"VIX Regime Forecast (10D): {'Elevated' if prediction == 1 else 'Normal'}")
    print(f"Predicted-Class Probability: {confidence}")
    print(f"VIX at Disclosure: {float(current_vix):.2f}")
    print(f"Prior 10D VIX Change: {vix_change:.2f}%")
    print(f"Disclosure Date: {disclosure_date}")
    print(f"Transaction: {ticker} - {transaction_type}  ~${transaction_size}")

# Checks the training data for substantial class imbalance between Normal and Elevated
# Warns user if the minority class is below 20% of the data.
def distribution_check(X_train_data, y_train_data):
    # Check the distribution of volatility regimes in the training data
    class_counts = y_train_data.value_counts()
    total = len(y_train_data)

    normal_count = class_counts.get(0, 0)
    elevated_count = class_counts.get(1, 0)

    normal_pct = normal_count / total * 100
    elevated_pct = elevated_count / total * 100

    # Flag a substantial class imbalance
    minority_pct = min(normal_pct, elevated_pct)

    # Print / log substantial class imbalance of minority class below 20%
    if minority_pct < 20:
        print(
            "WARNING: Model evaluation training data contains a class imbalance. "
            "Predictive performance may vary between volatility regimes."
        )
        print("\n--- Training Data Diagnostics ---")
        print(f"Normal:   {normal_count:,} ({normal_pct:.1f}%)")
        print(f"Elevated: {elevated_count:,} ({elevated_pct:.1f}%)")

        logging.warning(f"Evaluation training data contains class imbalance minority class below 20%. "
                        f"cutoff_date = {cutoff_date.strftime('%Y-%m-%d')} Consider adjusting cutoff_date for more robust training.\n"
                        f"Normal:   {normal_count:,} ({normal_pct:.1f}%)\n"
                        f"Elevated: {elevated_count:,} ({elevated_pct:.1f}%)"
                        )

    # Print / log single class validation folds
    for fold, (_, val_idx) in enumerate(
            tscv.split(X_train_data), start=1
    ):
        y_val = y_train_data.iloc[val_idx]
        fold_counts = y_val.value_counts()

        if len(fold_counts) < 2:
            missing_class = (
                1 if 0 in fold_counts.index else 0
            )

            print(
                f"WARNING: Validation Fold {fold} contains "
                f"only one class. ROC AUC cannot be calculated "
                f"for this fold."
            )
            logging.warning(f"Validation fold {fold} contains only one class; ROC AUC not calculated for this fold"
                            f"Performance metrics may not fully reflect the model's ability to identify both regimes.")



# -------------------------------------------Configuration--------------------------------------------------------------
# Configure logging parameters
logging.basicConfig(
    filename="log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# region --------------------DO NOT MODIFY ABOVE THIS LINE -----------------------------------
# endregion
# ================================ USER CONFIGURATION ========================================
DATASET_PATH = Path("CSV_data") / "master_dataset_git.csv"  # Configure Datasource Path

# TRANSACTION SELECTION (True: Custom transaction | False: Most recent dataset transaction)
user_transaction = False

# CUSTOM TRANSACTION DETAILS (Active only when user_transaction = True)
member = "Lois Frankel"                 # Member of Congress
ticker = "JPM"                          # Stock ticker symbol
transaction_date = "2023-03-22"         # Transaction date (YYYY-MM-DD)
disclosure_date = "2023-04-28"          # Disclosure date (YYYY-MM-DD)
transaction_type = "Purchase"           # Transaction type: "Purchase" or "Sale"
transaction_range = "$1,001 - $15,000"  # Disclosed trade value range

# MODEL ENGINE SELECTION (Enable either or both models to generate predictions)
use_baseline_model = False  # Enable/Disable Baseline Logistic Regression
use_tuned_model = True      # Enable/Disable Tuned Machine Learning Model

# EVALUATION SETTINGS (Current chronological cutoff range: 2014-01-03 to 2026-04-08)
evaluation_cutoff = [True, "2024-04-01"]
log_testset_evaluation = True  # Log additional details about test-set evaluation

# region --------------------DO NOT MODIFY BELOW THIS LINE -----------------------------------

latest_transaction = not user_transaction
master_df = pd.read_csv(DATASET_PATH)
# endregion


# ------------------------------------------------MAIN------------------------------------------------------------------
if __name__ == "__main__":
    # Stores metric results for all models
    all_results = []

    # Build or retrieve the transaction to predict
    if user_transaction:
        try:
            # Build user transaction
            user_row = alpha_prep.add_transaction(
                member,
                ticker,
                transaction_date,
                disclosure_date,
                transaction_type,
                transaction_range
            )
        # Report error and exit if transaction creation fails
        except Exception as e:
            print(f"Error building user transaction, exiting program. Exception: {e}")
            raise
            sys.exit(1)
    else:
        # Retrieve latest transaction; Report error and exit if unavailable
        if master_df.empty:
            print("Error: The transaction dataset is empty.")
            sys.exit(1)
        user_row = master_df.iloc[[-1]]  # Replace idx with a specific transaction in the dataset you want to predict

    # --- X/y Assignment ---

    # X contains the model input features, excluding identifiers and the target
    X = master_df.drop(
        columns=[
            "member.name",
            "ticker",
            "transactionDate",
            "disclosureDate",
            "transactionType",
            # Target feature
            "VIX_regime_10d"
        ]
    )

    # y contains the target variable to predict
    y = master_df["VIX_regime_10d"]

    # Convert disclosure dates to datetime for chronological splitting
    master_df["disclosureDate"] = pd.to_datetime(master_df["disclosureDate"])

    # Exclude transactions without known 10-trading-day VIX outcomes
    eligible_mask = master_df["VIX_regime_10d"].notna()

    # --- Select the chronological split for evaluation and prediction ---

    # Use the requested evaluation cutoff when enabled; otherwise use the disclosure date from the user row
    if evaluation_cutoff[0]:  # evaluation_cutoff[Bool, "YYYY-MM-DD"]
        try:
            #
            evaluation_date = pd.to_datetime(evaluation_cutoff[1])

            # Check evaluation date is in range of the test-set,
            # disable test-set logging if date is out-of-range
            if (
                    evaluation_date > master_df["disclosureDate"].max()
                    or evaluation_date < master_df["disclosureDate"].min()
            ):
                print(
                    "The evaluation cutoff date is outside the date range available in the main dataset. "
                    "Test-set evaluation was not logged and the baseline model was used for predictions. "
                )
                evaluation_cutoff[0] = False
                use_tuned_model = False
                use_baseline_model = True
                log_testset_evaluation = False
                logging.info(f"Forced Baseline Model fit due to ineligible evaluation cutoff date")

            # Create evaluation train/test sets based on cutoff date
            eval_train_mask = (
                    (master_df["disclosureDate"] < evaluation_date)
                    & eligible_mask
            )

            eval_test_mask = (
                    (master_df["disclosureDate"] >= evaluation_date)
                    & eligible_mask
            )

            # Split features and target using chronological masks
            X_eval_train = X.loc[eval_train_mask].copy()
            X_eval_test = X.loc[eval_test_mask].copy()

            y_eval_train = y.loc[eval_train_mask].copy()
            y_eval_test = y.loc[eval_test_mask].copy()

            # If either test or train data sets are empty, exit the program
            if X_eval_train.empty or y_eval_train.empty:
                raise ValueError("Training set is empty. Check the evaluation cutoff date.")

            if X_eval_test.empty or y_eval_test.empty:
                raise ValueError("Test set is empty. Check the evaluation cutoff date.")

        except (pd.errors.ParserError, ValueError, pd.errors.OutOfBoundsDatetime) as exc:
            print(f"Could not process 'evaluation_cutoff' date: {evaluation_cutoff[1]} \n {exc}")
            print("Aborting Prediction")
            sys.exit(1)

    elif use_tuned_model and not evaluation_cutoff[0]:
        print(
            f"ERROR: An eligible evaluation cutoff date is required to use the tuned model.\n"
            f"Make sure cutoff date is toggled on with 'use_tuned_model' toggle\n"
            f"Continuing prediction process with baseline model and no evaluation logging"
        )
        use_tuned_model = False
        use_baseline_model = True
        log_testset_evaluation = False

        # Log forced baseline model prediction
        logging.info(f"Forced Baseline Model fit due to ineligible evaluation cutoff date")

    #
    cutoff_date = pd.to_datetime(user_row["disclosureDate"].iloc[0])

    # Assign earlier eligible transactions to training and later ones to testing.
    train_mask = (
            (master_df["disclosureDate"] < cutoff_date)
            & eligible_mask
    )

    test_mask = (
            (master_df["disclosureDate"] >= cutoff_date)
            & eligible_mask
    )

    # Split features and target using chronological masks
    X_train = X.loc[train_mask].copy()
    X_test = X.loc[test_mask].copy()

    y_train = y.loc[train_mask].copy()
    y_test = y.loc[test_mask].copy()

    # --- Preprocessing ---

    # Create pipeline so imputation, scaling, and model training can happen inside each fold
    logistic_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=1000, random_state=42))
    ])

    # Confirm chronological order before folding
    assert master_df["disclosureDate"].is_monotonic_increasing

    # Create 5 chronological folds to evaluate model performance across periods of training
    tscv = TimeSeriesSplit(n_splits=5)

    if use_tuned_model:
        # Check distribution of volatility on the model evaluation dataset
        distribution_check(X_eval_train, y_eval_train)

        # --- Hyper-parameter Tuning ---

        # Create a dictionary of parameters to tune
        param_grid = {
            "model__C": [0.005, 0.01, 0.015, 0.025, 0.05, 0.1],
            "model__class_weight": [None, "balanced"]
        }

        # Configure grid search to select the best hyper-parameters based on cross-validated ROC AUC.
        grid_search = GridSearchCV(
            logistic_pipeline,
            param_grid=param_grid,
            cv=tscv,
            scoring=["accuracy", "precision", "recall", "f1", "roc_auc", "balanced_accuracy"],
            refit="roc_auc",  # Select the best model using ROC AUC
            n_jobs=-1
        )

        # Run the search using the training data
        grid_search.fit(X_eval_train, y_eval_train)

        # Debugging
        # Mean metrics for the winning combination.

        # Get the results for the best hyperparameter combination.
        optimized_lr_model = grid_search.best_estimator_

        if log_testset_evaluation:
            logging.info(f"Parameter tested: {param_grid}")
            logging.info(f"Best Parameters: {grid_search.best_params_}")

            scoring = {
                "accuracy": "accuracy",
                "precision": "precision",
                "recall": "recall",
                "f1": "f1",
                "roc_auc": "roc_auc",
                "balanced_accuracy": "balanced_accuracy"
            }

            log_cv_results(
                "Tuned Logistic Regression",
                grid_search.cv_results_,
                scoring.keys(),
                best_index=grid_search.best_index_
            )

            # Generate test-set predictions and elevated regime probabilities
            optimized_predictions = optimized_lr_model.predict(X_eval_test)
            optimized_probabilities = optimized_lr_model.predict_proba(X_eval_test)[:, 1]

            # Store optimized model test-set metrics
            all_results.append(
                evaluate_model(
                    "Optimized-Logistic",
                    optimized_predictions,
                    optimized_probabilities,
                    y_eval_test))

        # Predict the selected transaction
        predict_transaction(
            "Optimized Logistic Regresstion",
            optimized_lr_model,
            user_row)

    if use_baseline_model:
        # --- Baseline Logistic Model ---

        # Fit baseline logistic model to training data
        logistic_pipeline.fit(X_train, y_train)

        if log_testset_evaluation:
            # Check distribution of volatility on the model evaluation dataset
            distribution_check(X_eval_train, y_eval_train)

            # Define the performance metrics used to evaluate each fold
            scoring = {
                "accuracy": "accuracy",
                "precision": "precision",
                "recall": "recall",
                "f1": "f1",
                "roc_auc": "roc_auc",
                "balanced_accuracy": "balanced_accuracy"
            }

            logistic_cv = cross_validate(
                logistic_pipeline,
                X_eval_train,
                y_eval_train,
                cv=tscv,
                scoring=scoring
            )

            log_cv_results(
                "Baseline Logistic",
                logistic_cv,
                scoring.keys(),
            )

            # Create baseline predictions/probabilities
            baseline_predictions = logistic_pipeline.predict(X_eval_test)
            baseline_probabilities = logistic_pipeline.predict_proba(X_eval_test)[:, 1]

            # Append baseline results to the 'all_results' list
            all_results.append(
                evaluate_model(
                    "Baseline-Logistic Full testing set",
                    baseline_predictions,
                    baseline_probabilities,
                    y_eval_test
                )
            )

        # Predict transaction and display graph
        predict_transaction(
            "Baseline Logistic Regression",
            logistic_pipeline,
            user_row
        )

    if log_testset_evaluation:
        # Compare and log all model performance
        compare_results(all_results)
        logging.info(f"evaluation_cutoff settings: {evaluation_cutoff}")
