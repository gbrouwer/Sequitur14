# === FILE: server.py ===
# === LOCATION: ROOT/src/server.py ===
# === PURPOSE: Serve the dashboard from ROOT/viz/ using Flask ===

from flask import Flask, send_from_directory
from pathlib import Path

app = Flask(__name__)

# Absolute path to the viz/ directory containing index.html and results/
VIZ_DIR = Path(__file__).resolve().parent.parent / "viz"

@app.route("/")
def index():
    """
    Serve the dashboard homepage (index.html).
    """
    return send_from_directory(VIZ_DIR, "index.html")

@app.route("/<path:filename>")
def static_files(filename):
    """
    Serve any file located inside the viz/ directory.
    This includes: dashboard.js, style.css, results/, etc.
    """
    return send_from_directory(VIZ_DIR, filename)

if __name__ == "__main__":
    print(f"✓ Serving dashboard from: {VIZ_DIR}")
    app.run(debug=True, port=5000)