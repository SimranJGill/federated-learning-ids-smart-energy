import os
import shutil
import glob
import kagglehub

# Define target paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data", "raw")

CICIOT_TARGET_DIR = os.path.join(DATA_DIR, "CICIoT2023", "wataiData", "csv", "CICIoT2023")
EDGE_TARGET_FILE = os.path.join(DATA_DIR, "EdgeIIoTset", "Edge-IIoTset dataset", "Selected dataset for ML and DL", "DNN-EdgeIIoT-dataset.csv")

def check_existing_files():
    # Check if edge file exists
    edge_exists = os.path.isfile(EDGE_TARGET_FILE)
    
    # Check if cic directory exists and has csv files
    cic_exists = False
    if os.path.isdir(CICIOT_TARGET_DIR):
        csv_files = glob.glob(os.path.join(CICIOT_TARGET_DIR, "*.csv"))
        if len(csv_files) > 0:
            cic_exists = True
            
    return edge_exists, cic_exists

def download_and_setup():
    print("Checking existing dataset files...")
    edge_exists, cic_exists = check_existing_files()
    
    if edge_exists and cic_exists:
        print("✅ Both datasets are already downloaded and configured in data/raw/.")
        return

    # Authenticate (kagglehub login)
    print("Authenticating with Kaggle. If you are not logged in, you will be prompted for your username and API key.")
    try:
        kagglehub.login()
    except Exception as e:
        print(f"Warning/Info: kagglehub login check returned: {e}")

    # Create target directories
    os.makedirs(os.path.dirname(EDGE_TARGET_FILE), exist_ok=True)
    os.makedirs(CICIOT_TARGET_DIR, exist_ok=True)

    if not edge_exists:
        print("\n📥 Downloading EdgeIIoTset dataset from Kaggle...")
        try:
            download_path = kagglehub.dataset_download("mohamedamineferrag/edgeiiotset-cyber-security-dataset-of-iot-iiot")
            print(f"Downloaded to: {download_path}")
            
            # Find DNN-EdgeIIoT-dataset.csv recursively
            found_file = None
            for root, dirs, files in os.walk(download_path):
                for file in files:
                    if file.lower() == "dnn-edgeiiot-dataset.csv":
                        found_file = os.path.join(root, file)
                        break
                if found_file:
                    break
            
            if found_file:
                print(f"Copying dataset file to target destination: {EDGE_TARGET_FILE}")
                shutil.copy(found_file, EDGE_TARGET_FILE)
                print("✅ EdgeIIoTset configured successfully.")
            else:
                print("❌ Could not find 'DNN-EdgeIIoT-dataset.csv' in the downloaded files.")
        except Exception as e:
            print(f"❌ Failed to download EdgeIIoTset: {e}")

    if not cic_exists:
        print("\n📥 Downloading CICIoT2023 dataset from Kaggle...")
        try:
            download_path = kagglehub.dataset_download("madhavmalhotra/unb-cic-iot-dataset")
            print(f"Downloaded to: {download_path}")
            
            # Find all CSV files recursively
            csv_files = []
            for root, dirs, files in os.walk(download_path):
                for file in files:
                    if file.lower().endswith(".csv"):
                        csv_files.append(os.path.join(root, file))
            
            if csv_files:
                print(f"Copying {len(csv_files)} CSV files to target destination: {CICIOT_TARGET_DIR}")
                for f in csv_files:
                    dest = os.path.join(CICIOT_TARGET_DIR, os.path.basename(f))
                    shutil.copy(f, dest)
                print("✅ CICIoT2023 configured successfully.")
            else:
                print("❌ Could not find any CSV files in the downloaded CICIoT2023 dataset.")
        except Exception as e:
            print(f"❌ Failed to download CICIoT2023: {e}")

if __name__ == "__main__":
    download_and_setup()
