# === FILE: managers.py ===
# === PURPOSE: Manage experiment/job configuration and directory layout ===

import hashlib
import json
import yaml
from pathlib import Path

class JobManager:
    def __init__(self, config, mode="analyze", force=False):
        self.config = config
        self.mode = mode
        self.force = force

        self.job_name = config["job_name"]
        self.data_dir = Path("../data") / self.job_name
        self.results_dir = Path("../results") / self.job_name
        self.config_path = self.data_dir / "config.yaml"
        self.status_path = self.data_dir / "status.yaml"

        self._ensure_dirs()
        self._load_or_write_config()

    def _ensure_dirs(self):
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def _load_or_write_config(self):
        if self.config_path.exists():
            with open(self.config_path) as f:
                existing = yaml.safe_load(f)
            if existing != self.config and not self.force:
                raise ValueError("Config mismatch. Use force=True to override.")
        with open(self.config_path, "w") as f:
            yaml.dump(self.config, f)

    def get_data_path(self, key):
        if key not in {"raw", "staging", "processed", "corpus"}:
            raise ValueError(f"Unexpected data subdirectory: {key}")
        path = self.data_dir / key
        path.mkdir(parents=True, exist_ok=True)
        return path

    def get_results_path(self, method):
        method_dir = self.results_dir / method
        method_dir.mkdir(parents=True, exist_ok=True)
        return method_dir

    def update_status(self, step):
        status = {}
        if self.status_path.exists():
            with open(self.status_path, "r") as f:
                status = yaml.safe_load(f) or {}
        status[step] = True
        with open(self.status_path, "w") as f:
            yaml.dump(status, f)

    def check_pipeline_ready(self, steps=("scraped", "staged", "preprocessed","corpus")):
        if not self.status_path.exists():
            return False
        with open(self.status_path, "r") as f:
            status = yaml.safe_load(f) or {}
        missing = [s for s in steps if not status.get(s)]
        if missing:
            print(f"⚠️ Missing pipeline steps: {missing}")
            return False
        return True

    def snapshot_to_results(self):
        config_out = self.results_dir / "config.yaml"
        with open(config_out, "w") as f:
            yaml.dump(self.config, f)
        if self.status_path.exists():
            status_out = self.results_dir / "status.yaml"
            with open(status_out, "w") as out, open(self.status_path, "r") as inp:
                out.write(inp.read())

    def print_header(self):
        print(f"Job: {self.job_name} ({self.mode})")
        print(f"Data directory: {self.data_dir}")
        print(f"Results directory: {self.results_dir}")