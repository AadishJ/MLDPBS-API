from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
import os
import subprocess
import tempfile
import json
from flask_cors import CORS
import gc  # Import garbage collector

# Global variable for batch size
BATCH_SIZE = 10  # Changed to 10 to match the smaller dataset size

app = Flask(__name__)
# Enable CORS for all routes and all origins
CORS(
    app,
    resources={
        r"/*": {"origins": "*", "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"]}
    },
)

# Set up gunicorn timeout configuration
app.config["TIMEOUT"] = 300  # 5 minutes instead of default 30 seconds


def prepare_data(data):
    X = data["Day No."].values.reshape(-1, 1)
    y = data["Number of entries"].values
    
    # For small datasets, use a minimum test size of 1 sample
    n_samples = len(X)
    if n_samples <= 5:
        # For very small datasets, use 80% for training, but at least 1 sample for testing
        test_size = 1
        train_size = max(1, n_samples - test_size)
        
        # Manual split to ensure we have at least 1 sample in each set
        train_indices = np.arange(train_size)
        test_indices = np.arange(train_size, n_samples)
        
        X_train, X_test = X[train_indices], X[test_indices] if len(test_indices) > 0 else X[-1:] 
        y_train, y_test = y[train_indices], y[test_indices] if len(test_indices) > 0 else y[-1:]
        
        return X_train, X_test, y_train, y_test
    else:
        # For larger datasets, use the regular train_test_split
        return train_test_split(X, y, test_size=0.2, random_state=42)


def train_and_predict(X_train, X_test, y_train, y_test, future_days=7):
    # Use only two models instead of four to save memory
    models = {
        "XGBoost": XGBRegressor(random_state=42),
        "GBR": GradientBoostingRegressor(random_state=42),
    }

    results = {}
    for name, model in models.items():
        try:
            # Set lower complexity for models to save memory
            if name == "XGBoost":
                model.set_params(max_depth=3, n_estimators=50)
            elif name == "GBR":
                model.set_params(max_depth=3, n_estimators=50)

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            
            # For future prediction, get the maximum day number
            last_day = np.max(np.vstack([X_train, X_test]))
            future_days_array = np.arange(last_day + 1, last_day + 1 + future_days).reshape(-1, 1)
            future_predictions = model.predict(future_days_array)

            # Explicitly clean up memory
            gc.collect()

        except Exception as e:
            print(f"Error training {name} model: {str(e)}")
            print(f"Falling back to LinearRegression for {name}")
            model = LinearRegression()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            
            # For future prediction, get the maximum day number
            last_day = np.max(np.vstack([X_train, X_test]))
            future_days_array = np.arange(last_day + 1, last_day + 1 + future_days).reshape(-1, 1)
            future_predictions = model.predict(future_days_array)

        results[name] = {"mse": float(mse), "predictions": future_predictions.tolist()}

    return results


def run_buddy_allocation(percentages, batch_info=""):
    """Run the BuddyAllocation program with the given percentages."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    buddy_executable = os.path.join(script_dir, "BuddyAllocation")

    # Create a predictions directory if it doesn't exist
    predictions_dir = os.path.join(script_dir, "predictions")
    os.makedirs(predictions_dir, exist_ok=True)

    # Create the percentages file with batch information
    percentages_file = os.path.join(predictions_dir, "percentages.txt")
    
    # Append to file instead of overwriting
    with open(percentages_file, "a") as f:
        f.write(f"\n{batch_info}\n")
        for i, percentage in enumerate(percentages):
            f.write(f"Dataset {i+1}: {percentage:.2f}%\n")
        f.write("-" * 40 + "\n")

    try:
        # Check if the executable exists
        if os.path.exists(buddy_executable):
            # Run the executable with a timeout to prevent hanging
            run_result = subprocess.run(
                [buddy_executable],
                capture_output=True,
                text=True,
                check=True,
                timeout=30,
            )

            # Just return the results section of the output
            output_lines = run_result.stdout.split("\n")
            result_section = []
            results_found = False

            for line in output_lines:
                if line.strip() == "Results:":
                    results_found = True
                    result_section.append(line)
                elif results_found:
                    result_section.append(line)

            # Also append the results to the percentages file
            with open(percentages_file, "a") as f:
                f.write("Results:\n")
                for line in result_section:
                    f.write(f"{line}\n")
                f.write("=" * 50 + "\n")

            return "\n".join(result_section)
        else:
            return "BuddyAllocation executable not found. It should be compiled during the build process."
    except subprocess.TimeoutExpired:
        return "BuddyAllocation execution timed out after 30 seconds."
    except subprocess.CalledProcessError as e:
        return f"Error running BuddyAllocation: {e.stderr}"


@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get JSON data from request
        data = request.json

        if not data or not isinstance(data, list):
            return (
                jsonify(
                    {
                        "error": "Invalid data format. Expected an array of dataset objects"
                    }
                ),
                400,
            )

        # Limit number of datasets to prevent memory issues
        if len(data) > 5:
            return (
                jsonify(
                    {"error": "Too many datasets. Maximum 5 allowed for processing."}
                ),
                400,
            )

        # Clear the percentages file before starting
        script_dir = os.path.dirname(os.path.abspath(__file__))
        predictions_dir = os.path.join(script_dir, "predictions")
        os.makedirs(predictions_dir, exist_ok=True)
        percentages_file = os.path.join(predictions_dir, "percentages.txt")
        with open(percentages_file, "w") as f:
            f.write(f"Predictions started at: {pd.Timestamp.now()}\n")
            f.write("=" * 50 + "\n")

        dataset_names = [dataset_obj["name"] for dataset_obj in data]
        
        # Find the maximum number of entries across all datasets
        max_entries = max([len(dataset_obj["data"]) for dataset_obj in data])
        
        # Calculate number of batches needed
        num_batches = (max_entries + BATCH_SIZE - 1) // BATCH_SIZE
        
        all_results = []
        
        # Process data in batches
        for batch in range(num_batches):
            start_idx = batch * BATCH_SIZE
            end_idx = min((batch + 1) * BATCH_SIZE, max_entries)
            
            batch_info = f"Batch {batch+1}/{num_batches} (Entries {start_idx+1}-{end_idx})"
            print(f"Processing {batch_info}")
            
            batch_datasets = []
            
            # Extract the current batch from each dataset
            for dataset_obj in data:
                df_full = pd.DataFrame(dataset_obj["data"])
                
                # Skip dataset if it doesn't have enough entries for this batch
                if start_idx >= len(df_full):
                    # Create an empty dataframe with the same columns
                    df_batch = pd.DataFrame(columns=df_full.columns)
                else:
                    end_for_this_dataset = min(end_idx, len(df_full))
                    df_batch = df_full.iloc[start_idx:end_for_this_dataset].copy()
                
                # Validate required columns
                if "Day No." not in df_batch.columns or "Number of entries" not in df_batch.columns:
                    if len(df_batch) > 0:  # Only check if we have data
                        return (
                            jsonify(
                                {
                                    "error": f"Dataset {dataset_obj['name']} missing required columns ('Day No.' and 'Number of entries')"
                                }
                            ),
                            400,
                        )
                
                batch_datasets.append(df_batch)
            
            # Process this batch
            batch_results = []
            xgb_averages = []
            
            for df, name in zip(batch_datasets, dataset_names):
                if len(df) > 0:  # Only process if we have data
                    try:
                        X_train, X_test, y_train, y_test = prepare_data(df)
                        results = train_and_predict(X_train, X_test, y_train, y_test)
                        xgb_avg = np.mean(results["XGBoost"]["predictions"])
                    except Exception as e:
                        print(f"Error processing dataset {name}: {str(e)}")
                        # Fallback to a simple average if ML fails
                        xgb_avg = df["Number of entries"].mean() if len(df) > 0 else 0
                        results = {
                            "XGBoost": {"mse": 0, "predictions": [float(xgb_avg)] * 7},
                            "GBR": {"mse": 0, "predictions": [float(xgb_avg)] * 7}
                        }
                else:
                    # If no data for this dataset in this batch, set average to 0
                    results = {"XGBoost": {"mse": 0, "predictions": [0]}, "GBR": {"mse": 0, "predictions": [0]}}
                    xgb_avg = 0
                
                batch_results.append(results)
                xgb_averages.append(xgb_avg)
                
                # Force garbage collection after each dataset
                gc.collect()
            
            # Calculate percentages for this batch
            total_sum = sum(xgb_averages)
            percentages = [
                (avg / total_sum) * 100 if total_sum > 0 else 0 for avg in xgb_averages
            ]
            
            # Run BuddyAllocation for this batch and append to percentages file
            buddy_output = run_buddy_allocation(percentages, batch_info)
            
            # Collect results for the final response
            all_results.append({
                "batch": batch_info,
                "predictions": [
                    {
                        "dataset": name,
                        "results": result,
                        "xgb_average": float(xgb_avg),
                        "percentage": float(pct),
                    }
                    for name, result, xgb_avg, pct in zip(
                        dataset_names, batch_results, xgb_averages, percentages
                    )
                ],
                "buddy_allocation_output": buddy_output,
            })
            
            # Force garbage collection before next batch
            gc.collect()

        # Prepare final response
        response = {
            "datasets": dataset_names,
            "batch_results": all_results,
            "percentages_file": percentages_file
        }

        return jsonify(response)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# Add at the very end of your app.py file
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)