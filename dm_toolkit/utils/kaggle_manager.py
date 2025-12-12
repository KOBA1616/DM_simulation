import os
import json
import shutil
from pathlib import Path

class KaggleManager:
    def __init__(self):
        self.api = None
        self.is_authenticated = False
        self.error_message = ""

        try:
            from kaggle.api.kaggle_api_extended import KaggleApi
            self.api = KaggleApi()
            self.api.authenticate()
            self.is_authenticated = True
        except ImportError:
            self.error_message = "Kaggle library not installed. Run `pip install kaggle`."
        except Exception as e:
            self.error_message = f"Authentication failed: {str(e)}"

    def download_dataset(self, dataset_slug, target_dir, unzip=True):
        """
        Download a dataset from Kaggle.
        dataset_slug: 'username/dataset-name'
        target_dir: Local path to save files
        """
        if not self.is_authenticated:
            raise Exception(f"Not authenticated: {self.error_message}")

        print(f"Downloading {dataset_slug} to {target_dir}...")
        self.api.dataset_download_files(dataset_slug, path=target_dir, unzip=unzip, quiet=False)
        return True

    def create_dataset_version(self, folder_path, message, delete_old_versions=False):
        """
        Upload a new version of the dataset.
        folder_path: Directory containing the model/data files AND dataset-metadata.json
        message: Commit message
        """
        if not self.is_authenticated:
            raise Exception(f"Not authenticated: {self.error_message}")

        if not os.path.exists(os.path.join(folder_path, "dataset-metadata.json")):
            raise Exception("dataset-metadata.json not found in the folder.")

        print(f"Uploading new version from {folder_path}...")
        self.api.dataset_create_version(folder_path, message, delete_old_versions=delete_old_versions, quiet=False)
        return True

    def init_dataset_metadata(self, folder_path, title, slug):
        """
        Initialize dataset-metadata.json in the target folder.
        slug: 'username/dataset-name'
        """
        meta = {
            "title": title,
            "id": slug,
            "licenses": [{"name": "CC0-1.0"}]
        }

        # Ensure directory exists
        os.makedirs(folder_path, exist_ok=True)

        meta_path = os.path.join(folder_path, "dataset-metadata.json")
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=4)

        return meta_path

    def create_new_dataset(self, folder_path, public=False):
        """
        Create a brand new dataset on Kaggle.
        """
        if not self.is_authenticated:
            raise Exception(f"Not authenticated: {self.error_message}")

        if not os.path.exists(os.path.join(folder_path, "dataset-metadata.json")):
            raise Exception("dataset-metadata.json not found in the folder.")

        self.api.dataset_create_new(folder_path, public=public, quiet=False)
        return True
