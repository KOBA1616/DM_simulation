import os
import shutil
import glob
import re

class GenerationManager:
    def __init__(self, data_root="data"):
        self.data_root = data_root
        self.models_dir = os.path.join(data_root, "models")
        self.production_dir = os.path.join(self.models_dir, "production")
        self.candidates_dir = os.path.join(self.models_dir, "candidates")
        self.archive_dir = os.path.join(self.models_dir, "archive")
        self.training_data_dir = os.path.join(data_root, "training_data")

        self.setup_directories()

    def setup_directories(self):
        for d in [self.production_dir, self.candidates_dir, self.archive_dir, self.training_data_dir]:
            os.makedirs(d, exist_ok=True)

    def get_production_model_path(self):
        # Look for *.pth in production dir
        files = glob.glob(os.path.join(self.production_dir, "*.pth"))
        if not files:
            return None
        # Return the most recent one or the first one
        return files[0] # Should be only one

    def get_latest_generation_number(self):
        # Check archive and production for highest gen number
        # Format: model_gen_XXX.pth
        max_gen = 0

        all_files = glob.glob(os.path.join(self.production_dir, "*.pth")) + \
                    glob.glob(os.path.join(self.archive_dir, "*.pth"))

        for f in all_files:
            basename = os.path.basename(f)
            match = re.search(r"gen_(\d+)", basename)
            if match:
                gen = int(match.group(1))
                if gen > max_gen:
                    max_gen = gen

        return max_gen

    def create_candidate_path(self, generation):
        filename = f"model_gen_{generation:04d}.pth"
        return os.path.join(self.candidates_dir, filename)

    def promote_candidate(self, candidate_path):
        if not os.path.exists(candidate_path):
            raise FileNotFoundError(f"Candidate not found: {candidate_path}")

        # Archive current production
        current_prod = self.get_production_model_path()
        if current_prod:
            shutil.move(current_prod, self.archive_dir)
            print(f"Archived production model to {self.archive_dir}")

        # Move candidate to production
        filename = os.path.basename(candidate_path)
        dest = os.path.join(self.production_dir, filename)
        shutil.move(candidate_path, dest)
        print(f"Promoted {candidate_path} to production: {dest}")
        return dest

    def get_training_data_path(self, generation):
        filename = f"data_gen_{generation:04d}.npz"
        return os.path.join(self.training_data_dir, filename)

    def cleanup_candidates(self):
        # Remove all files in candidates dir
        files = glob.glob(os.path.join(self.candidates_dir, "*"))
        for f in files:
            os.remove(f)
