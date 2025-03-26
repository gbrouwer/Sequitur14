from pathlib import Path

# Path to the visualization results directory (relative to main.py)
VIZ_ROOT = Path("viz/results")
OUTPUT_DIR = Path("viz")

# Output files
INDEX_HTML = OUTPUT_DIR / "index.html"
JS_FILE = OUTPUT_DIR / "dashboard.js"

TEMPLATE_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Pulse AI Dashboard</title>
  <script src="https://d3js.org/d3.v7.min.js"></script>
  <script src="dashboard.js" defer></script>
  <link rel="stylesheet" href="style.css">
</head>
<body>
  <div id="app">
    <h1 style="text-align:center;">Pulse AI Dashboard</h1>
    <!-- Charts will render here -->
  </div>
</body>
</html>
"""

def find_projects():
    """Find all project result folders and return list of (path, label) tuples."""
    paths = []
    for exp_dir in VIZ_ROOT.glob("*/*/*"):
        if exp_dir.is_dir():
            data = exp_dir.parts[-3]
            exp = exp_dir.parts[-2]
            job = exp_dir.parts[-1]
            label = f"{data} / {exp} / {job}"
            paths.append((exp_dir, label))
    return paths

def generate_js_loader():
    """Create JS stub that dynamically loads charts from config/data."""
    lines = ["// === AUTO-GENERATED ==="]
    lines.append("document.addEventListener('DOMContentLoaded', () => {")
    lines.append("  const app = document.getElementById('app');")

    for exp_dir, label in find_projects():
        web_path = exp_dir.relative_to(VIZ_ROOT)
        lines.append(f"  const section = document.createElement('section');")
        lines.append(f"  section.innerHTML = `<h2>{label}</h2>`;")
        lines.append(f"  app.appendChild(section);")

        for config_path in exp_dir.glob("*_config.json"):
            chart_stem = config_path.stem.replace("_config", "")
            data_file = config_path.with_name(f"{chart_stem}.csv")
            rel_data = data_file.relative_to(VIZ_ROOT).as_posix()
            rel_config = config_path.relative_to(VIZ_ROOT).as_posix()

            lines.append(f"  fetch('{rel_data}').then(res => res.text()).then(csv => {{")
            lines.append(f"    fetch('{rel_config}').then(r => r.json()).then(config => {{")
            lines.append(f"      renderChart(section, csv, config);")
            lines.append(f"    }});")
            lines.append(f"  }});")

    lines.append("});")

    # Placeholder renderChart function
    lines.append("\nfunction renderChart(container, csvText, config) {")
    lines.append("  const div = document.createElement('div');")
    lines.append("  div.className = 'viz-block';")
    lines.append("  div.innerHTML = `<pre>${config.caption || 'Untitled chart'}</pre>`;")
    lines.append("  container.appendChild(div);")
    lines.append("  // TODO: Add chart logic here...")
    lines.append("}")

    return "\n".join(lines)

def main():
    OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

    # Write clean HTML shell
    INDEX_HTML.write_text(TEMPLATE_HTML, encoding="utf-8")
    print(f"✓ Wrote HTML shell to {INDEX_HTML}")

    # Write dynamic JS loader stub
    JS_FILE.write_text(generate_js_loader(), encoding="utf-8")
    print(f"✓ Wrote JS logic to {JS_FILE}")

if __name__ == "__main__":
    main()
