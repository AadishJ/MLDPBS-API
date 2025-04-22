import pandas as pd
import requests
import os
import argparse
import json
import math
import time

def main():
    # Set up command line arguments
    parser = argparse.ArgumentParser(description='Client for ML prediction API with chunking')
    parser.add_argument('--data-dir', type=str, default="/home/apaarraina/Implementation/MLDPBS-API/data/",
                        help='Directory containing CSV files')
    parser.add_argument('--server', type=str, default="http://127.0.0.1:5000",
                        help='Server URL')
    parser.add_argument('--chunk-size', type=int, default=1000,
                        help='Size of data chunks to process (default: 1000)')
    parser.add_argument('--output-dir', type=str, default="./results/",
                        help='Directory to save results')
    args = parser.parse_args()

    # Set up directories
    DATA_DIR = args.data_dir
    OUTPUT_DIR = args.output_dir
    CHUNK_SIZE = args.chunk_size
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # API endpoint
    API_URL = f"{args.server}/predict"

    # Specific CSV files to process
    csv_files = [
        "simplified_calories.csv",
        "simplified_heart_rate_data.csv",
        "simplified_sleep_data.csv",
    ]

    # Load all datasets
    print(f"Loading datasets from {DATA_DIR}...")
    all_dataframes = {}
    
    for filename in csv_files:
        file_path = os.path.join(DATA_DIR, filename)
        
        try:
            if not os.path.exists(file_path):
                print(f"Warning: File {file_path} does not exist. Skipping.")
                continue
                
            df = pd.read_csv(file_path)
            
            # Validate required columns
            if "Day No." not in df.columns or "Number of entries" not in df.columns:
                print(f"Skipping {filename}: Missing required columns ('Day No.' and 'Number of entries').")
                continue
            
            # Sort by Day No. to ensure chronological order
            df = df.sort_values(by="Day No.").reset_index(drop=True)
            
            dataset_name = os.path.splitext(filename)[0]
            all_dataframes[dataset_name] = df
            
            total_chunks = math.ceil(len(df) / CHUNK_SIZE)
            print(f"Loaded {filename} with {len(df)} rows - will be processed in {total_chunks} chunks")
            
        except Exception as e:
            print(f"Error loading {filename}: {str(e)}")

    if not all_dataframes:
        print("No valid datasets found. Exiting.")
        return

    # Create datasets array to send to the server
    # The server will now handle the chunking internally
    datasets_to_send = []
    for name, df in all_dataframes.items():
        datasets_to_send.append({
            "name": name,
            "data": df.to_dict(orient="records")
        })
    
    if not datasets_to_send:
        print("No data to send. Exiting.")
        return
    
    # Send request to API
    try:
        print(f"Sending request to {API_URL} with {len(datasets_to_send)} datasets...")
        start_time = time.time()
        response = requests.post(API_URL, json=datasets_to_send, timeout=300)  # Increased timeout to 5 minutes
        end_time = time.time()
        
        if response.status_code == 200:
            data = response.json()
            
            # Save full JSON response
            json_path = os.path.join(OUTPUT_DIR, "all_prediction_results.json")
            with open(json_path, "w") as f:
                json.dump(data, f, indent=2)
            
            # Process each batch result
            for i, batch_result in enumerate(data["batch_results"]):
                batch_info = batch_result["batch"]
                batch_dir = os.path.join(OUTPUT_DIR, f"batch_{i+1}")
                os.makedirs(batch_dir, exist_ok=True)
                
                # Save batch-specific results
                batch_json_path = os.path.join(batch_dir, "prediction_results.json")
                with open(batch_json_path, "w") as f:
                    json.dump(batch_result, f, indent=2)
                
                # Save percentages to txt file
                percentages_path = os.path.join(batch_dir, "percentages.txt")
                with open(percentages_path, "w") as f:
                    f.write(f"Batch: {batch_info}\n")
                    f.write("-" * 40 + "\n")
                    for prediction in batch_result["predictions"]:
                        dataset_name = prediction["dataset"]
                        percentage = prediction["percentage"]
                        f.write(f"{dataset_name}: {percentage:.2f}%\n")
                
                # Print summary
                print(f"\nResults for {batch_info}:")
                for prediction in batch_result["predictions"]:
                    dataset_name = prediction["dataset"]
                    xgb_avg = prediction["xgb_average"]
                    percentage = prediction["percentage"]
                    print(f"  {dataset_name}: XGB avg={xgb_avg:.2f}, Allocation={percentage:.2f}%")
                
                # Save buddy allocation output
                buddy_output_path = os.path.join(batch_dir, "buddy_allocation.txt")
                with open(buddy_output_path, "w") as f:
                    f.write(batch_result["buddy_allocation_output"])
                    
                print(f"Batch results saved to {batch_dir}/")
            
            # Note the location of the complete percentages file created by the server
            if "percentages_file" in data:
                print(f"\nComplete percentages file: {data['percentages_file']}")
                
            print(f"\nProcessing complete! Processing time: {end_time - start_time:.2f} seconds")
            print(f"All results are available in the {OUTPUT_DIR} directory.")
                
        else:
            print(f"Error from API: {response.status_code}")
            print(response.text)
                
    except Exception as e:
        print(f"Request failed: {str(e)}")

if __name__ == "__main__":
    main()