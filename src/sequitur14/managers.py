import yaml
import hashlib
from pathlib import Path
from datetime import datetime
import sys

class JobManager:
    def __init__(self, config: dict, base_data_path="../data", base_results_path="../results", mode="analysis"):
        self.original_config = config
        self.mode = mode
        self.base_data_path = Path(base_data_path)
        self.base_results_path = Path(base_results_path)

        if mode == "preprocess":
            self.job_name = config.get("data_name", "unnamed-data")
            self.run_dir = self.base_data_path / self.job_name
            self.config_path = self.run_dir / "config.yaml"
            self.config = config

        elif mode == "analysis":
            self.job_name = config.get("job_name", "unnamed-job")
            data_name = config.get("data_name")
            if not data_name:
                raise ValueError("Analysis config must specify 'data_name'")

            # Load data config
            data_config_path = self.base_data_path / data_name / "config.yaml"
            if not data_config_path.exists():
                print(f"❌ Error: Could not find preprocessed dataset config at {data_config_path}")
                print("   Make sure you've run preprocess.py and the data_name is correct.")
                sys.exit(1)

            with open(data_config_path, "r") as f:
                data_config = yaml.safe_load(f)

            # Merge with analysis config
            merged_config = {**data_config, **config, "job_type": "analysis"}
            self.config = merged_config
            self.hash_id = self._hash_config(merged_config)
            self.run_dir = self.base_results_path / self.job_name / self.hash_id
            self.config_path = self.run_dir / "config.yaml"

        else:
            raise ValueError("Mode must be either 'preprocess' or 'analysis'")

        self._check_or_create()

    def _hash_config(self, config):
        serialized = yaml.dump(config, sort_keys=True)
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:10]

    def _check_or_create(self):
        self.run_dir.mkdir(parents=True, exist_ok=True)
        if self.config_path.exists():
            with open(self.config_path, "r") as f:
                existing = yaml.safe_load(f)
            if existing != self.config:
                raise ValueError(f"Mismatch with existing config at: {self.config_path}")
        else:
            with open(self.config_path, "w") as f:
                yaml.dump(self.config, f, sort_keys=False)
            print(f"✓ Created new {self.mode} run: {self.run_dir}")

    def get_path(self, name: str) -> Path:
        path = self.run_dir / name
        path.mkdir(parents=True, exist_ok=True)
        return path

    def get_file(self, key: str) -> Path:
        return Path(self.config[key])

    @property
    def base_dir(self):
        return self.run_dir

    def print_header(self):
        print("\n" + "=" * 60)
        label = "Data pull" if self.mode == "preprocess" else "Experiment"
        print(f"🔬  {label}: {self.job_name}")

        if self.mode == "analysis":
            tags = self.config.get("tags")
            if tags:
                print(f"🏷   Tags: {', '.join(tags)}")
            notes = self.config.get("notes")
            if notes:
                print(f"📝  Notes: {notes}")

        print(f"📁  Job directory: {self.run_dir}")
        print("=" * 60 + "\n")