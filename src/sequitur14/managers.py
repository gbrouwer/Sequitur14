# === FILE: managers.py ===
# === PURPOSE: Manage experiment/job configuration and directory layout ===

import hashlib
import json
import os
from pathlib import Path
import yaml


class JobManager:
    def __init__(self, config, mode="analyze", base_results_path="../results", force=False):
        self.config = config
        self.mode = mode
        self.force = force

        self.base_results_path = Path(base_results_path)
        self.data_name = config["data_name"]
        self.job_name = config.get("job_name", self.data_name)
        self.config_hash = self._get_hash(config)

        self.data_dir = Path("../data") / self.data_name
        self.job_dir = self.base_results_path / self.job_name / self.config_hash
        self.config_path = self.data_dir / "config.yaml"
        self.base_dir = self.job_dir

        self._check_or_create()

    def _get_hash(self, config):
        config_str = json.dumps(config, sort_keys=True)
        return hashlib.sha1(config_str.encode()).hexdigest()[:8]

    def _check_or_create(self):
        self.job_dir.mkdir(parents=True, exist_ok=True)

        if self.config_path.exists():
            if self.force:
                with open(self.config_path, "w") as f:
                    yaml.dump(self.config, f)
                print(f"⚠️ Overwrote existing config at {self.config_path}")
                return

            with open(self.config_path) as f:
                existing_config = yaml.safe_load(f)
            if existing_config != self.config:
                raise ValueError(f"Mismatch with existing config at: {self.config_path}")
        else:
            self.data_dir.mkdir(parents=True, exist_ok=True)
            with open(self.config_path, "w") as f:
                yaml.dump(self.config, f)

    def print_header(self):
        print(f"Job: {self.job_name} ({self.mode})")
        print(f"Hash: {self.config_hash}")
        print(f"Output directory: {self.job_dir}")
