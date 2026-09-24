from pathlib import Path
import subprocess
import sys
import os
from html import escape
from datetime import datetime

# ============================================================
# CIC-IDS2017 FULL PIPELINE — STAGE 01 -> 29 + 39 + 40
# ============================================================
# รัน Stage 01 ถึง 29 และ Stage 39 ตามลำดับ
# ถ้า Stage ใด ERROR / ไม่พบไฟล์ จะหยุดทันที
# และบันทึก output ลง cic_pipeline_results.html
# ============================================================

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_HTML = BASE_DIR / "cic_pipeline_results.html"

SCRIPTS = [
    "01_check_data.py",
    "02_inspect_data.py",
    "03_clean_data.py",
    "04_feature_analysis.py",
    "05_create_behavior_features.py",
    "06_create_behavior_dataset.py",

    "07_split_normal_attack.py",
    "08_train_models.py",
    "09_create_test_dataset.py",
    "10_check_n_estimators.py",
    "11_check_samples_features.py",
    "12_final_models.py",
    "13_anomaly_score.py",
    "14_network_emotion.py",
    "15_final_analysis.py",

    "16_one_class_svm_behavior.py",
    "17_tune_ocsvm_threshold.py",
    "18_tune_ocsvm_hyperparameters.py",
    "19_final_ocsvm_model.py",
    "20_ocsvm_anomaly_score.py",
    "21_ocsvm_risk_emotion.py",
    "22_ocsvm_analysis.py",

    "23_lof_baseline.py",
    "24_tune_lof_threshold.py",
    "25_tune_lof_hyperparameters.py",
    "26_final_lof_model.py",
    "27_lof_anomaly_score.py",
    "28_lof_risk_emotion.py",
    "29_lof_analysis.py",

    "39_compare_three_models.py",
    "40_statistical_threshold_cic.py",
]


def build_html(sections, status):
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>CIC-IDS2017 Pipeline — Stage 01-29 + 39 + 40</title>
<style>
body {{
    font-family: Arial, sans-serif;
    margin: 0;
    padding: 24px;
    background: #f4f4f4;
    color: #222;
}}
.container {{
    max-width: 1500px;
    margin: auto;
}}
header {{
    background: #111;
    color: #fff;
    padding: 20px;
    border-radius: 12px;
    margin-bottom: 20px;
}}
section {{
    background: #fff;
    margin-bottom: 18px;
    border-radius: 12px;
    overflow: hidden;
    box-shadow: 0 2px 8px rgba(0,0,0,.08);
}}
h2 {{
    margin: 0;
    padding: 14px 18px;
    font-size: 17px;
}}
.ok {{
    background: #e9f7ec;
    color: #1f6b2b;
}}
.fail {{
    background: #fdeaea;
    color: #9b1c1c;
}}
.pre {{
    margin: 0;
    padding: 16px;
    background: #0f1115;
    color: #e8e8e8;
    white-space: pre-wrap;
    overflow-x: auto;
    line-height: 1.4;
    font-family: Consolas, monospace;
    font-size: 13px;
}}
.badge {{
    float: right;
    font-size: 12px;
    padding: 4px 8px;
    border-radius: 999px;
    background: #fff;
}}
</style>
</head>
<body>
<div class="container">
<header>
<h1>CIC-IDS2017 Pipeline — Stage 01-29 + 39 + 40</h1>
<div>Status: {escape(status)}</div>
<div>Generated: {escape(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))}</div>
</header>
{''.join(sections)}
</div>
</body>
</html>"""


def save_html(sections, status):
    OUTPUT_HTML.write_text(
        build_html(sections, status),
        encoding="utf-8"
    )


def run_script(script_name: str):
    script_path = BASE_DIR / script_name

    if not script_path.exists():
        return False, f"File not found: {script_path}", 1

    process = subprocess.Popen(
        ["uv", "run", "python", "-u", script_name],
        cwd=BASE_DIR,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
        env={
            **os.environ,
            "PYTHONIOENCODING": "utf-8",
        },
    )

    output_lines = []

    assert process.stdout is not None

    for line in process.stdout:
        print(line, end="")
        output_lines.append(line)

    code = process.wait()

    return code == 0, "".join(output_lines), code


def main():
    print("=" * 110)
    print("CIC-IDS2017 FULL PIPELINE — STAGE 01 -> 29 + 39 + 40")
    print("=" * 110)
    print(f"Total scripts: {len(SCRIPTS)}")
    print(f"Working directory: {BASE_DIR}")
    print("=" * 110)

    sections = []

    for i, script_name in enumerate(SCRIPTS, start=1):
        print()
        print("=" * 110)
        print(f"[{i}/{len(SCRIPTS)}] RUNNING: {script_name}")
        print("=" * 110)

        ok, output_text, return_code = run_script(script_name)

        label = "OK" if ok else f"FAILED (code {return_code})"
        css = "ok" if ok else "fail"

        sections.append(
            f'<section><h2 class="{css}">[{i}/{len(SCRIPTS)}] '
            f'{escape(script_name)} '
            f'<span class="badge">{escape(label)}</span></h2>'
            f'<div class="pre">{escape(output_text)}</div></section>'
        )

        save_html(sections, "RUNNING" if ok else "FAILED")

        if not ok:
            print()
            print("=" * 110)
            print(f"STOPPED: {script_name}")
            print(f"Return code: {return_code}")
            print(f"HTML saved: {OUTPUT_HTML.resolve()}")
            print("=" * 110)
            sys.exit(return_code)

    save_html(sections, "COMPLETED")

    print()
    print("=" * 110)
    print("ALL PIPELINE COMPLETED — STAGE 01-29 + 39 + 40 + 40")
    print("=" * 110)
    print(f"Total completed: {len(SCRIPTS)}")
    print(f"HTML saved to: {OUTPUT_HTML.resolve()}")
    print("=" * 110)


if __name__ == "__main__":
    main()
