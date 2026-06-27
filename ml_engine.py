"""
------------------------------------------------------
Supervised Learning App
ml_engine.py

Part 3.1
Imports
Helper Functions
Data Loading
Preprocessing
------------------------------------------------------
"""

import os
import time
import warnings

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

warnings.filterwarnings("ignore")


# ==========================================================
# Create Output Folder
# ==========================================================

OUTPUT_FOLDER = "output"

if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)


# ==========================================================
# Read CSV Files
# ==========================================================

def load_data(train_path, test_path):
    """
    Load training and testing CSV files.

    Returns
    -------
    train_df
    test_df
    """

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    return train_df, test_df


# ==========================================================
# Validate Dataset
# ==========================================================

def validate_dataset(train_df, test_df):
    """
    Validate training and testing datasets.
    """

    if train_df.empty:
        raise Exception("Training CSV is empty.")

    if test_df.empty:
        raise Exception("Test CSV is empty.")

    if train_df.shape[1] < 2:
        raise Exception(
            "Training CSV must contain at least "
            "one feature column and one target column."
        )

    feature_columns = train_df.columns[:-1]

    if len(feature_columns) != len(test_df.columns):
        raise Exception(
            "Feature count mismatch between "
            "training and test datasets."
        )


# ==========================================================
# Split Features and Target
# ==========================================================

def split_dataset(train_df, test_df):
    """
    Assumes last column is target.
    """

    X = train_df.iloc[:, :-1].copy()

    y = train_df.iloc[:, -1].copy()

    X_test = test_df.copy()

    return X, y, X_test


# ==========================================================
# Encode Target Labels
# ==========================================================

def encode_target(y):
    """
    Encode target labels if categorical.
    """

    encoder = None

    if y.dtype == object:

        encoder = LabelEncoder()

        y = encoder.fit_transform(y)

    return y, encoder


# ==========================================================
# Detect Column Types
# ==========================================================

def detect_columns(X):
    """
    Detect numeric and categorical columns.
    """

    numeric_columns = X.select_dtypes(
        include=["int64", "float64"]
    ).columns.tolist()

    categorical_columns = X.select_dtypes(
        exclude=["int64", "float64"]
    ).columns.tolist()

    return numeric_columns, categorical_columns


# ==========================================================
# Build Preprocessor
# ==========================================================

def build_preprocessor(
    numeric_columns,
    categorical_columns
):
    """
    Build preprocessing pipeline.
    """

    from sklearn.preprocessing import (
        StandardScaler,
        OneHotEncoder
    )

    numeric_pipeline = Pipeline(

        steps=[

            (
                "imputer",
                SimpleImputer(strategy="mean")
            ),

            (
                "scaler",
                StandardScaler()
            )

        ]

    )

    categorical_pipeline = Pipeline(

        steps=[

            (
                "imputer",
                SimpleImputer(strategy="most_frequent")
            ),

            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore"
                )
            )

        ]

    )

    preprocessor = ColumnTransformer(

        transformers=[

            (
                "num",
                numeric_pipeline,
                numeric_columns
            ),

            (
                "cat",
                categorical_pipeline,
                categorical_columns
            )

        ]

    )

    return preprocessor


# ==========================================================
# Prepare Dataset
# ==========================================================

def prepare_dataset(
    train_path,
    test_path
):
    """
    Complete preprocessing pipeline.

    Returns
    -------
    X_train
    X_valid
    y_train
    y_valid
    X_test
    preprocessor
    label_encoder
    """

    train_df, test_df = load_data(
        train_path,
        test_path
    )

    validate_dataset(
        train_df,
        test_df
    )

    X, y, X_test = split_dataset(
        train_df,
        test_df
    )

    y, label_encoder = encode_target(y)

    numeric_columns, categorical_columns = detect_columns(X)

    preprocessor = build_preprocessor(
        numeric_columns,
        categorical_columns
    )

    X_train, X_valid, y_train, y_valid = train_test_split(

        X,
        y,

        test_size=0.20,

        random_state=42,

        shuffle=True

    )

    return (

        X_train,
        X_valid,

        y_train,
        y_valid,

        X_test,

        preprocessor,

        label_encoder

    )


# ==========================================================
# Save Predictions
# ==========================================================

def save_predictions(predictions):

    output_path = os.path.join(
        OUTPUT_FOLDER,
        "output.csv"
    )

    df = pd.DataFrame(
        {
            "Prediction": predictions
        }
    )

    df.to_csv(
        output_path,
        index=False
    )

    return output_path


# ==========================================================
# Save Report
# ==========================================================

def save_report(report_text):

    report_path = os.path.join(
        OUTPUT_FOLDER,
        "report.txt"
    )

    with open(
        report_path,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(report_text)

    return report_path

# ==========================================================
# PART 3.2A - CLASSIFICATION PIPELINE
# ==========================================================


from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC

from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

import pandas as pd
from sklearn.preprocessing import LabelEncoder

def label_encoder(df: pd.DataFrame, columns_to_encode: list) -> pd.DataFrame:
    """Converts text categories into numbers using Sklearn's LabelEncoder."""
    df_encoded = df.copy()
    le = LabelEncoder()
    
    for col in columns_to_encode:
        if col in df_encoded.columns:
            df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
            
    return df_encoded




def train_classification_models(
    X_train,
    X_valid,
    y_train,
    y_valid,
    X_test,
    preprocessor,
    laber_encoder=None
):
    """
    Train all classification models and compare performance.

    Returns
    -------
    Dictionary containing all model results.
    """
    if label_encoder is not None:
        output_predictions = label_encoder.inverse_transform(best_predictions)
    else:
        output_predictions = best_predictions

    models = {

        "Logistic Regression":
            LogisticRegression(max_iter=1000),

        "Decision Tree":
            DecisionTreeClassifier(random_state=42),

        "Random Forest":
            RandomForestClassifier(
                n_estimators=200,
                random_state=42
            ),

        "KNN":
            KNeighborsClassifier(n_neighbors=5),

        "Support Vector Machine":
            SVC(probability=True),

        "Gaussian Naive Bayes":
            GaussianNB()

    }

    results = []

    best_model = None

    best_pipeline = None

    best_accuracy = -1

    best_predictions = None

    start_time = time.time()

    for model_name, model in models.items():

        ####################################################
        # Build Pipeline
        ####################################################

        pipeline = Pipeline(

            steps=[

                ("preprocessor", preprocessor),

                ("model", model)

            ]

        )

        ####################################################
        # Fit Model
        ####################################################

        pipeline.fit(
            X_train,
            y_train
        )

        ####################################################
        # Validation Prediction
        ####################################################

        prediction = pipeline.predict(
            X_valid
        )

        ####################################################
        # Metrics
        ####################################################

        accuracy = accuracy_score(
            y_valid,
            prediction
        )

        precision = precision_score(
            y_valid,
            prediction,
            average="weighted",
            zero_division=0
        )

        recall = recall_score(
            y_valid,
            prediction,
            average="weighted",
            zero_division=0
        )

        f1 = f1_score(
            y_valid,
            prediction,
            average="weighted",
            zero_division=0
        )

        matrix = confusion_matrix(
            y_valid,
            prediction
        )

        ####################################################
        # Store Result
        ####################################################

        results.append({

            "Model": model_name,

            "Accuracy": accuracy,

            "Precision": precision,

            "Recall": recall,

            "F1 Score": f1,

            "Confusion Matrix": matrix

        })

        ####################################################
        # Best Model
        ####################################################

        if accuracy > best_accuracy:

            best_accuracy = accuracy

            best_model = model_name

            best_pipeline = pipeline

            best_predictions = pipeline.predict(
                X_test
            )

    execution_time = time.time() - start_time

        ####################################################
    # Decode Predictions (if target was encoded)
    ####################################################

    if 'label_encoder' in globals() and label_encoder is not None:
        try:
            output_predictions = label_encoder.inverse_transform(
                best_predictions
            )
        except Exception:
            output_predictions = best_predictions
    else:
        output_predictions = best_predictions

    ####################################################
    # Save Prediction CSV
    ####################################################

    output_csv = save_predictions(output_predictions)

    ####################################################
    # Build Report
    ####################################################

    report = []
    report.append("=" * 60)
    report.append("SUPERVISED LEARNING REPORT")
    report.append("=" * 60)
    report.append("")
    report.append("Problem Type : Classification")
    report.append("")
    report.append(f"Best Model : {best_model}")
    report.append(f"Validation Accuracy : {best_accuracy:.4f}")
    report.append(f"Execution Time : {execution_time:.2f} seconds")
    report.append("")
    report.append("-" * 60)
    report.append("ALL MODEL RESULTS")
    report.append("-" * 60)
    report.append("")

    for item in results:

        report.append(f"Model : {item['Model']}")
        report.append(f"Accuracy : {item['Accuracy']:.4f}")
        report.append(f"Precision : {item['Precision']:.4f}")
        report.append(f"Recall : {item['Recall']:.4f}")
        report.append(f"F1 Score : {item['F1 Score']:.4f}")
        report.append("Confusion Matrix:")
        report.append(str(item["Confusion Matrix"]))
        report.append("")
        report.append("-" * 40)

    report_text = "\n".join(report)

    ####################################################
    # Save Report
    ####################################################

    report_file = save_report(report_text)

    ####################################################
    # Return Everything
    ####################################################

    return {

        "problem_type": "classification",

        "best_model": best_model,

        "best_pipeline": best_pipeline,

        "best_accuracy": best_accuracy,

        "execution_time": execution_time,

        "predictions": output_predictions,

        "results": results,

        "output_csv": output_csv,

        "report_file": report_file

    }

# ==========================================================
# PART 3.3A - REGRESSION PIPELINE
# ==========================================================

from sklearn.pipeline import Pipeline

from sklearn.linear_model import LinearRegression

from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import RandomForestRegressor

from sklearn.neighbors import KNeighborsRegressor

from sklearn.svm import SVR

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


def train_regression_models(
    X_train,
    X_valid,
    y_train,
    y_valid,
    X_test,
    preprocessor
):
    """
    Train multiple regression models and compare them.

    Returns
    -------
    Dictionary containing model results.
    """

    models = {

        "Linear Regression":
            LinearRegression(),

        "Decision Tree":
            DecisionTreeRegressor(
                random_state=42
            ),

        "Random Forest":
            RandomForestRegressor(
                n_estimators=200,
                random_state=42
            ),

        "KNN":
            KNeighborsRegressor(
                n_neighbors=5
            ),

        "Support Vector Regressor":
            SVR()

    }

    results = []

    best_model = None

    best_pipeline = None

    best_r2 = float("-inf")

    best_predictions = None

    start_time = time.time()

    for model_name, model in models.items():

        ####################################################
        # Create Pipeline
        ####################################################

        pipeline = Pipeline(

            steps=[

                ("preprocessor", preprocessor),

                ("model", model)

            ]

        )

        ####################################################
        # Train Model
        ####################################################

        pipeline.fit(
            X_train,
            y_train
        )

        ####################################################
        # Validation Prediction
        ####################################################

        prediction = pipeline.predict(
            X_valid
        )

        ####################################################
        # Metrics
        ####################################################

        mae = mean_absolute_error(
            y_valid,
            prediction
        )

        mse = mean_squared_error(
            y_valid,
            prediction
        )

        rmse = np.sqrt(mse)

        r2 = r2_score(
            y_valid,
            prediction
        )

        ####################################################
        # Store Results
        ####################################################

        results.append({

            "Model": model_name,

            "MAE": mae,

            "MSE": mse,

            "RMSE": rmse,

            "R2": r2

        })

        ####################################################
        # Best Model (Highest R²)
        ####################################################

        if r2 > best_r2:

            best_r2 = r2

            best_model = model_name

            best_pipeline = pipeline

            best_predictions = pipeline.predict(
                X_test
            )

    execution_time = time.time() - start_time

    ####################################################
    # Save Prediction CSV
    ####################################################

    output_csv = save_predictions(best_predictions)

    ####################################################
    # Generate Report
    ####################################################

    report = []

    report.append("=" * 60)
    report.append("SUPERVISED LEARNING REPORT")
    report.append("=" * 60)
    report.append("")
    report.append("Problem Type : Regression")
    report.append("")
    report.append(f"Best Model : {best_model}")
    report.append(f"Best Validation R² : {best_r2:.4f}")
    report.append(f"Execution Time : {execution_time:.2f} seconds")
    report.append("")
    report.append("-" * 60)
    report.append("ALL MODEL RESULTS")
    report.append("-" * 60)
    report.append("")

    for item in results:

        report.append(f"Model : {item['Model']}")
        report.append(f"MAE  : {item['MAE']:.6f}")
        report.append(f"MSE  : {item['MSE']:.6f}")
        report.append(f"RMSE : {item['RMSE']:.6f}")
        report.append(f"R²   : {item['R2']:.6f}")
        report.append("")
        report.append("-" * 40)

    ####################################################
    # Mathematical Information
    ####################################################

    report.append("")
    report.append("=" * 60)
    report.append("METRIC INFORMATION")
    report.append("=" * 60)
    report.append("")
    report.append("MAE (Mean Absolute Error)")
    report.append("Average absolute difference between actual and predicted values.")
    report.append("Smaller values are better.")
    report.append("")

    report.append("MSE (Mean Squared Error)")
    report.append("Average squared prediction error.")
    report.append("Large errors are penalized more heavily.")
    report.append("")

    report.append("RMSE (Root Mean Squared Error)")
    report.append("Square root of MSE.")
    report.append("Expressed in the same unit as the target variable.")
    report.append("")

    report.append("R² Score (Coefficient of Determination)")
    report.append("Measures how well the model explains the variance.")
    report.append("Closer to 1.0 indicates a better model.")
    report.append("")

    ####################################################
    # Save Report
    ####################################################

    report_text = "\n".join(report)

    report_file = save_report(report_text)

    ####################################################
    # Return Results
    ####################################################

    return {

        "problem_type": "regression",

        "best_model": best_model,

        "best_pipeline": best_pipeline,

        "best_r2": best_r2,

        "execution_time": execution_time,

        "predictions": best_predictions,

        "results": results,

        "output_csv": output_csv,

        "report_file": report_file

    }

# ==========================================================
# PART 3.4
# Main ML Pipeline
# ==========================================================

def run_ml_pipeline(
    train_path,
    test_path,
    problem_type
):
    """
    Main entry point called by gui.py

    Parameters
    ----------
    train_path : str
        Path to training CSV

    test_path : str
        Path to testing CSV

    problem_type : str
        "classification" or "regression"

    Returns
    -------
    dict
        Result dictionary
    """

    (
        X_train,
        X_valid,
        y_train,
        y_valid,
        X_test,
        preprocessor,
        label_encoder

    ) = prepare_dataset(
        train_path,
        test_path
    )

    if problem_type.lower() == "classification":

        result = train_classification_models(

            X_train=X_train,

            X_valid=X_valid,

            y_train=y_train,

            y_valid=y_valid,

            X_test=X_test,

            preprocessor=preprocessor,

            label_encoder=label_encoder

        )

    elif problem_type.lower() == "regression":

        result = train_regression_models(

            X_train=X_train,

            X_valid=X_valid,

            y_train=y_train,

            y_valid=y_valid,

            X_test=X_test,

            preprocessor=preprocessor

        )

    else:

        raise ValueError(
            "Problem type must be "
            "'classification' or 'regression'."
        )

    return result


# ==========================================================
# Standalone Testing
# ==========================================================

if __name__ == "__main__":

    TRAIN_DATA = "data.csv"

    TEST_DATA = "test.csv"

    PROBLEM_TYPE = "classification"
    # Change to "regression" if required

    try:

        result = run_ml_pipeline(

            train_path=TRAIN_DATA,

            test_path=TEST_DATA,

            problem_type=PROBLEM_TYPE

        )

        print("\n===================================")
        print("Training Completed Successfully")
        print("===================================")

        print("Problem Type :", result["problem_type"])
        print("Best Model   :", result["best_model"])

        if result["problem_type"] == "classification":
            print(
                "Accuracy     :",
                round(result["best_accuracy"], 4)
            )

        else:
            print(
                "R² Score     :",
                round(result["best_r2"], 4)
            )

        print(
            "Execution Time :",
            round(result["execution_time"], 2),
            "seconds"
        )

        print(
            "Prediction CSV :",
            result["output_csv"]
        )

        print(
            "Report File    :",
            result["report_file"]
        )

    except Exception as e:

        print("\nPipeline Failed")
        print(str(e))