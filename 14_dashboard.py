import html
import json
import webbrowser
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go


# =========================================================
# FILES
# =========================================================

DATA_FILE = Path("network_emotion_result.csv")
OUTPUT_FILE = Path("network_emotion_dashboard_replay.html")

if not DATA_FILE.exists():
    raise FileNotFoundError(
        f"ไม่พบ {DATA_FILE}. "
        "วางไฟล์ network_emotion_result.csv ไว้โฟลเดอร์เดียวกับไฟล์นี้ก่อนรัน"
    )

data = pd.read_csv(DATA_FILE)


# =========================================================
# SETTINGS
# =========================================================

emotion_order = [
    "Happy",
    "Alert",
    "Anxious",
    "Stressed",
    "Angry",
]

emotion_emoji = {
    "Happy": "😄",
    "Alert": "😮",
    "Anxious": "😰",
    "Stressed": "😣",
    "Angry": "😡",
}

emotion_colors = {
    "Happy": "#22c55e",
    "Alert": "#f59e0b",
    "Anxious": "#eab308",
    "Stressed": "#f97316",
    "Angry": "#ef4444",
}

risk_names = {
    1: "Happy",
    2: "Alert",
    3: "Anxious",
    4: "Stressed",
    5: "Angry",
}

# รุ่นนี้เป็น "Offline Replay"
# ไม่อ่าน packet ใหม่จาก network
# ใช้ข้อมูลที่ประมวลผลเสร็จแล้วจาก CSV มาเล่นทีละ record
PLAYBACK_POINTS = 360


# =========================================================
# CLEAN DISPLAY
# =========================================================

def display_name(value):
    return str(value).replace("�", "–")


data["display_attack_type"] = data["attack_type"].map(display_name)
data["emotion"] = data["emotion"].astype(str)
data["risk_level"] = data["risk_level"].astype(int)
data["anomaly_score"] = pd.to_numeric(
    data["anomaly_score"], errors="coerce"
).fillna(0.0)


# =========================================================
# SUMMARY
# =========================================================

total_records = len(data)
traffic_types = data["attack_type"].nunique()

emotion_count = (
    data["emotion"]
    .value_counts()
    .reindex(emotion_order, fill_value=0)
)

dominant_emotion = emotion_count.idxmax()
dominant_percent = emotion_count[dominant_emotion] / total_records * 100

final_precision = 0.5995
final_recall = 0.9573
final_f1 = 0.7373


# =========================================================
# TRAFFIC SUMMARY
# =========================================================

summary = (
    data.groupby("attack_type")
    .agg(
        Records=("attack_type", "size"),
        MeanScore=("anomaly_score", "mean"),
        MainRisk=("risk_level", lambda x: int(x.mode().iloc[0])),
        MainEmotion=("emotion", lambda x: x.mode().iloc[0]),
    )
    .reset_index()
)

summary["DisplayType"] = summary["attack_type"].map(display_name)
summary["Emoji"] = summary["MainEmotion"].map(emotion_emoji)
summary["RiskText"] = summary["MainRisk"].map(lambda x: f"Risk {x}")
summary = summary.sort_values("MeanScore").reset_index(drop=True)


# =========================================================
# OFFLINE PLAYBACK DATA
# =========================================================

# เลือกตัวอย่างแบบกระจายตลอดทั้ง dataset
# เพื่อให้การเล่นครอบคลุมข้อมูลหลายช่วง
if total_records <= PLAYBACK_POINTS:
    playback = data.copy()
else:
    positions = np.linspace(
        0,
        total_records - 1,
        PLAYBACK_POINTS,
        dtype=int,
    )
    playback = data.iloc[positions].copy()

playback_rows = []

for idx, row in playback.reset_index(drop=True).iterrows():
    playback_rows.append(
        {
            "seq": int(idx + 1),
            "attack_type": display_name(row["attack_type"]),
            "emotion": str(row["emotion"]),
            "risk_level": int(row["risk_level"]),
            "anomaly_score": float(row["anomaly_score"]),
        }
    )


# =========================================================
# STATIC DONUT
# =========================================================

fig_donut = go.Figure(
    go.Pie(
        labels=[f"{emotion_emoji[e]} {e}" for e in emotion_order],
        values=emotion_count.values,
        hole=0.60,
        marker=dict(
            colors=[emotion_colors[e] for e in emotion_order]
        ),
        textinfo="percent",
        hovertemplate=(
            "<b>%{label}</b><br>"
            "Records: %{value:,}<br>"
            "Share: %{percent}<extra></extra>"
        ),
    )
)

fig_donut.update_layout(
    height=320,
    margin=dict(l=5, r=5, t=5, b=30),
    showlegend=True,
    legend=dict(
        orientation="h",
        y=-0.08,
    ),
    paper_bgcolor="white",
)


# =========================================================
# STATIC EMOTION BY TRAFFIC
# =========================================================

emotion_table = pd.crosstab(
    data["display_attack_type"],
    data["emotion"],
    normalize="index",
) * 100

emotion_table = emotion_table.reindex(
    columns=emotion_order,
    fill_value=0,
)

emotion_table = emotion_table.sort_values(
    by=["Angry", "Stressed", "Anxious"],
    ascending=False,
)

fig_stack = go.Figure()

for emotion in emotion_order:
    fig_stack.add_trace(
        go.Bar(
            x=emotion_table.index,
            y=emotion_table[emotion],
            name=f"{emotion_emoji[emotion]} {emotion}",
            marker_color=emotion_colors[emotion],
            hovertemplate=(
                f"<b>{emotion_emoji[emotion]} {emotion}</b><br>"
                "%{x}<br>%{y:.2f}%<extra></extra>"
            ),
        )
    )

fig_stack.update_layout(
    height=370,
    barmode="stack",
    margin=dict(l=20, r=20, t=20, b=100),
    xaxis=dict(tickangle=-55),
    yaxis_title="Percentage (%)",
    xaxis_title="",
    paper_bgcolor="white",
    plot_bgcolor="white",
    legend=dict(orientation="h", y=1.08),
)


# =========================================================
# STATIC TRAFFIC SUMMARY TABLE
# =========================================================

traffic_table_rows = ""

for _, row in summary.iterrows():
    risk = int(row["MainRisk"])
    emotion = row["MainEmotion"]

    traffic_table_rows += f"""
    <tr>
        <td><span class="emoji-big">{emotion_emoji[emotion]}</span></td>
        <td><b>{html.escape(row["DisplayType"])}</b></td>
        <td><span class="risk risk-{risk}">Risk {risk}</span></td>
        <td>{emotion}</td>
        <td class="mono">{row["MeanScore"]:.6f}</td>
        <td>{int(row["Records"]):,}</td>
    </tr>
    """


# =========================================================
# EMOTION CARDS
# =========================================================

emotion_cards = ""

for risk, emotion in enumerate(emotion_order, start=1):
    count = int(emotion_count[emotion])
    percent = count / total_records * 100

    emotion_cards += f"""
    <div class="emotion-card" id="emotion-card-{emotion}">
        <div class="emotion-face">{emotion_emoji[emotion]}</div>
        <div class="emotion-name">{emotion}</div>
        <div class="emotion-count">{count:,}</div>
        <div class="emotion-percent">{percent:.2f}%</div>
        <div class="emotion-risk">Risk {risk}</div>
    </div>
    """


# =========================================================
# PLOTLY EMBED
# =========================================================

donut_html = fig_donut.to_html(
    full_html=False,
    include_plotlyjs=True,
    config={
        "responsive": True,
        "displayModeBar": False,
    },
)

stack_html = fig_stack.to_html(
    full_html=False,
    include_plotlyjs=False,
    config={
        "responsive": True,
        "displayModeBar": False,
    },
)

playback_json = json.dumps(
    playback_rows,
    ensure_ascii=False,
)


# =========================================================
# DASHBOARD HTML
# =========================================================

html_doc = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">

<title>Network Emotion Dashboard — Offline Replay</title>

<style>
* {{
    box-sizing: border-box;
}}

body {{
    margin: 0;
    background: #f3f6fa;
    color: #172033;
    font-family: "Segoe UI", Arial, sans-serif;
}}

.topbar {{
    height: 6px;
    background: linear-gradient(90deg, #0ea5e9, #2563eb, #7c3aed);
}}

.header {{
    height: 70px;
    background: #ffffff;
    border-bottom: 1px solid #dce3eb;
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0 22px;
}}

.brand {{
    display: flex;
    gap: 11px;
    align-items: center;
}}

.brand-icon {{
    width: 39px;
    height: 39px;
    border-radius: 10px;
    background: #e8f5ff;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 23px;
}}

.brand-title {{
    font-size: 19px;
    font-weight: 800;
}}

.brand-sub {{
    color: #64748b;
    font-size: 11px;
    margin-top: 2px;
}}

.header-right {{
    display: flex;
    align-items: center;
    gap: 8px;
}}

.status-pill {{
    display: flex;
    align-items: center;
    gap: 7px;
    padding: 7px 12px;
    border-radius: 999px;
    background: #eef6ff;
    border: 1px solid #bfdbfe;
    color: #1d4ed8;
    font-size: 11px;
    font-weight: 800;
}}

.status-dot {{
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: #22c55e;
    box-shadow: 0 0 0 4px #dcfce7;
}}

.container {{
    max-width: 1500px;
    margin: 0 auto;
    padding: 15px;
}}

.stats {{
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 12px;
    margin-bottom: 14px;
}}

.stat {{
    background: #fff;
    border: 1px solid #dce3eb;
    border-radius: 12px;
    padding: 13px 15px;
}}

.stat-label {{
    font-size: 10px;
    color: #64748b;
    font-weight: 800;
    text-transform: uppercase;
    letter-spacing: .45px;
}}

.stat-value {{
    margin-top: 5px;
    font-size: 22px;
    font-weight: 850;
}}

.stat-sub {{
    font-size: 10px;
    color: #94a3b8;
    margin-top: 3px;
}}

.panel {{
    background: #fff;
    border: 1px solid #dce3eb;
    border-radius: 12px;
    overflow: hidden;
}}

.panel-title {{
    min-height: 44px;
    padding: 0 14px;
    display: flex;
    align-items: center;
    gap: 8px;
    border-bottom: 1px solid #e6ebf0;
    font-size: 13px;
    font-weight: 800;
}}

.panel-body {{
    padding: 12px;
}}

.playback-layout {{
    display: grid;
    grid-template-columns: 1.65fr 1fr;
    gap: 14px;
}}

.emotion-row {{
    display: grid;
    grid-template-columns: repeat(5, 1fr);
    gap: 8px;
    margin-bottom: 12px;
}}

.emotion-card {{
    background: #f8fafc;
    border: 1px solid #e5eaf0;
    border-radius: 10px;
    text-align: center;
    padding: 9px 6px;
    transition: .18s ease;
}}

.emotion-card.active {{
    transform: translateY(-2px);
    background: #ffffff;
    box-shadow: 0 7px 20px rgba(15, 23, 42, .10);
    border-width: 2px;
}}

.emotion-face {{
    font-size: 28px;
    line-height: 1;
}}

.emotion-name {{
    font-size: 11px;
    font-weight: 800;
    margin-top: 5px;
}}

.emotion-count {{
    margin-top: 5px;
    font-size: 13px;
    font-weight: 850;
}}

.emotion-percent {{
    font-size: 10px;
    color: #64748b;
    margin-top: 2px;
}}

.emotion-risk {{
    font-size: 9px;
    color: #94a3b8;
    margin-top: 2px;
}}

.monitor {{
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 16px;
    background:
        radial-gradient(circle at 78% 18%, rgba(59,130,246,.09), transparent 30%),
        linear-gradient(180deg, #fbfdff, #f8fafc);
}}

.monitor-top {{
    display: flex;
    justify-content: space-between;
    align-items: center;
}}

.live-label {{
    font-size: 10px;
    color: #64748b;
    font-weight: 800;
    text-transform: uppercase;
    letter-spacing: .6px;
}}

.replay-badge {{
    font-size: 10px;
    font-weight: 850;
    color: #1d4ed8;
    background: #dbeafe;
    padding: 5px 8px;
    border-radius: 999px;
}}

.current {{
    display: grid;
    grid-template-columns: 165px 1fr;
    gap: 17px;
    align-items: center;
    margin-top: 14px;
}}

.current-face {{
    height: 165px;
    border-radius: 22px;
    background: #fff;
    border: 1px solid #e2e8f0;
    display: flex;
    justify-content: center;
    align-items: center;
    font-size: 82px;
    box-shadow: 0 10px 25px rgba(15, 23, 42, .06);
}}

.current-emotion {{
    font-size: 30px;
    font-weight: 900;
    margin-bottom: 7px;
}}

.current-traffic {{
    font-size: 14px;
    color: #475569;
    font-weight: 700;
    margin-bottom: 13px;
}}

.detail-grid {{
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 8px;
}}

.detail {{
    background: #fff;
    border: 1px solid #e5eaf0;
    border-radius: 9px;
    padding: 9px 10px;
}}

.detail-label {{
    color: #94a3b8;
    font-size: 9px;
    font-weight: 800;
    text-transform: uppercase;
}}

.detail-value {{
    margin-top: 3px;
    font-size: 14px;
    font-weight: 850;
}}

.progress-wrap {{
    margin-top: 15px;
}}

.progress-head {{
    display: flex;
    justify-content: space-between;
    font-size: 10px;
    color: #64748b;
    font-weight: 750;
    margin-bottom: 6px;
}}

.progress-bar {{
    width: 100%;
    height: 8px;
    background: #e2e8f0;
    border-radius: 999px;
    overflow: hidden;
}}

.progress-fill {{
    height: 100%;
    width: 0%;
    background: linear-gradient(90deg, #22c55e, #3b82f6, #ef4444);
    border-radius: 999px;
    transition: width .15s linear;
}}

.controls {{
    display: flex;
    flex-wrap: wrap;
    gap: 7px;
    margin-top: 13px;
}}

.btn {{
    border: 1px solid #d5dce5;
    background: #fff;
    color: #1f2937;
    padding: 8px 12px;
    border-radius: 8px;
    cursor: pointer;
    font-size: 11px;
    font-weight: 800;
}}

.btn:hover {{
    background: #f8fafc;
}}

.btn.primary {{
    background: #2563eb;
    color: white;
    border-color: #2563eb;
}}

.speed.active {{
    background: #eef2ff;
    color: #3730a3;
    border-color: #a5b4fc;
}}

.small-note {{
    margin-top: 9px;
    color: #94a3b8;
    font-size: 10px;
    line-height: 1.5;
}}

.section-grid {{
    display: grid;
    grid-template-columns: 1.05fr .95fr;
    gap: 14px;
    margin-top: 14px;
}}

.feed-wrap {{
    max-height: 292px;
    overflow: auto;
}}

.feed {{
    display: grid;
    gap: 6px;
}}

.feed-item {{
    display: grid;
    grid-template-columns: 36px 1fr auto;
    gap: 8px;
    align-items: center;
    border: 1px solid #edf1f5;
    background: #fbfcfe;
    border-radius: 8px;
    padding: 8px;
}}

.feed-face {{
    font-size: 22px;
    text-align: center;
}}

.feed-main {{
    min-width: 0;
}}

.feed-title {{
    font-size: 11px;
    font-weight: 800;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}}

.feed-sub {{
    color: #94a3b8;
    font-size: 9px;
    margin-top: 2px;
}}

.feed-score {{
    font-family: Consolas, monospace;
    font-size: 9px;
    color: #475569;
}}

svg#scoreChart {{
    width: 100%;
    height: 250px;
    background: #fbfcfe;
    border: 1px solid #edf1f5;
    border-radius: 9px;
}}

.chart-scale {{
    display: flex;
    justify-content: space-between;
    gap: 10px;
    margin-top: 6px;
    color: #94a3b8;
    font-size: 9px;
    font-family: Consolas, monospace;
}}

table {{
    width: 100%;
    border-collapse: collapse;
    font-size: 11px;
}}

.table-wrap {{
    overflow: auto;
    max-height: 370px;
}}

th {{
    position: sticky;
    top: 0;
    background: #f7f9fb;
    color: #64748b;
    text-align: left;
    font-weight: 800;
    padding: 9px 8px;
    border-bottom: 1px solid #e5eaf0;
}}

td {{
    padding: 8px;
    border-bottom: 1px solid #eef2f6;
    vertical-align: middle;
}}

.mono {{
    font-family: Consolas, monospace;
    font-size: 10px;
}}

.emoji-big {{
    font-size: 19px;
}}

.risk {{
    display: inline-block;
    padding: 3px 7px;
    border-radius: 999px;
    font-size: 10px;
    font-weight: 800;
}}

.risk-1 {{ background:#dcfce7; color:#166534; }}
.risk-2 {{ background:#fef3c7; color:#92400e; }}
.risk-3 {{ background:#fef9c3; color:#854d0e; }}
.risk-4 {{ background:#ffedd5; color:#9a3412; }}
.risk-5 {{ background:#fee2e2; color:#991b1b; }}

.legend-note {{
    padding: 10px;
    border: 1px solid #e5eaf0;
    border-radius: 9px;
    background: #f8fafc;
    font-size: 10px;
    color: #64748b;
    line-height: 1.55;
}}

.footer {{
    padding: 14px 0 3px;
    text-align: center;
    color: #94a3b8;
    font-size: 10px;
}}

@media (max-width: 1100px) {{
    .playback-layout,
    .section-grid {{
        grid-template-columns: 1fr;
    }}
}}

@media (max-width: 800px) {{
    .stats {{
        grid-template-columns: repeat(2, 1fr);
    }}

    .current {{
        grid-template-columns: 1fr;
    }}

    .current-face {{
        height: 130px;
        font-size: 65px;
    }}
}}

@media (max-width: 600px) {{
    .stats {{
        grid-template-columns: 1fr;
    }}

    .emotion-row {{
        grid-template-columns: repeat(2, 1fr);
    }}

    .header {{
        height: auto;
        padding: 12px;
        gap: 10px;
        flex-direction: column;
        align-items: flex-start;
    }}
}}
</style>
</head>

<body>

<div class="topbar"></div>

<div class="header">
    <div class="brand">
        <div class="brand-icon">🧠</div>
        <div>
            <div class="brand-title">Network Emotion Dashboard</div>
            <div class="brand-sub">
                CIC-IDS2017 • Offline Network Traffic Analysis • Replay Mode
            </div>
        </div>
    </div>

    <div class="header-right">
        <div class="status-pill">
            <span class="status-dot" id="statusDot"></span>
            <span id="statusText">OFFLINE REPLAY</span>
        </div>
    </div>
</div>

<div class="container">

    <div class="stats">
        <div class="stat">
            <div class="stat-label">Total Traffic</div>
            <div class="stat-value">{total_records:,}</div>
            <div class="stat-sub">Records in saved test data</div>
        </div>

        <div class="stat">
            <div class="stat-label">Traffic Types</div>
            <div class="stat-value">{traffic_types}</div>
            <div class="stat-sub">BENIGN + attack classes</div>
        </div>

        <div class="stat">
            <div class="stat-label">Dominant Emotion</div>
            <div class="stat-value">{emotion_emoji[dominant_emotion]} {dominant_emotion}</div>
            <div class="stat-sub">{dominant_percent:.2f}% of all records</div>
        </div>

        <div class="stat">
            <div class="stat-label">Model F1-score</div>
            <div class="stat-value">{final_f1:.4f}</div>
            <div class="stat-sub">
                Recall {final_recall:.4f} • Precision {final_precision:.4f}
            </div>
        </div>
    </div>


    <!-- =====================================================
         PLAYBACK MONITOR
    ====================================================== -->

    <div class="panel">
        <div class="panel-title">
            ▶️ Offline Traffic Playback
        </div>

        <div class="panel-body">

            <div class="emotion-row">
                {emotion_cards}
            </div>

            <div class="playback-layout">

                <div class="monitor">

                    <div class="monitor-top">
                        <div class="live-label">
                            SIMULATED LIVE VIEW
                        </div>

                        <div class="replay-badge">
                            DATA REPLAY • NOT REAL-TIME
                        </div>
                    </div>

                    <div class="current">

                        <div class="current-face" id="currentFace">
                            💤
                        </div>

                        <div>

                            <div class="current-emotion" id="currentEmotion">
                                Ready
                            </div>

                            <div class="current-traffic" id="currentTraffic">
                                กด Start เพื่อเริ่มเล่นข้อมูล
                            </div>

                            <div class="detail-grid">

                                <div class="detail">
                                    <div class="detail-label">Risk</div>
                                    <div class="detail-value" id="currentRisk">
                                        —
                                    </div>
                                </div>

                                <div class="detail">
                                    <div class="detail-label">Anomaly Score</div>
                                    <div class="detail-value mono" id="currentScore">
                                        —
                                    </div>
                                </div>

                                <div class="detail">
                                    <div class="detail-label">Replay Record</div>
                                    <div class="detail-value" id="currentSeq">
                                        0 / {len(playback_rows)}
                                    </div>
                                </div>

                            </div>

                        </div>

                    </div>

                    <div class="progress-wrap">

                        <div class="progress-head">
                            <span id="progressLeft">
                                Replay progress
                            </span>

                            <span id="progressRight">
                                0%
                            </span>
                        </div>

                        <div class="progress-bar">
                            <div class="progress-fill" id="progressFill"></div>
                        </div>

                    </div>

                    <div class="controls">

                        <button class="btn primary" id="startBtn">
                            ▶ Start
                        </button>

                        <button class="btn" id="pauseBtn">
                            ⏸ Pause
                        </button>

                        <button class="btn" id="resetBtn">
                            ↺ Reset
                        </button>

                        <button class="btn speed active" data-speed="1">
                            1×
                        </button>

                        <button class="btn speed" data-speed="2">
                            2×
                        </button>

                        <button class="btn speed" data-speed="4">
                            4×
                        </button>

                    </div>

                    <div class="small-note">
                        ระบบนี้ไม่ได้รับ packet ใหม่จาก network
                        แต่จำลองการไหลของข้อมูลโดยนำผลจาก
                        <b>network_emotion_result.csv</b>
                        มาเล่นทีละรายการ เพื่อให้หน้าจอดูเหมือนกำลัง monitor แบบต่อเนื่อง
                    </div>

                </div>


                <div class="panel" style="border-radius:12px;">

                    <div class="panel-title">
                        📈 Current Anomaly Score
                    </div>

                    <div class="panel-body">
                        <svg id="scoreChart"
                             viewBox="0 0 720 250"
                             preserveAspectRatio="none">
                            <polyline
                                id="scoreLine"
                                fill="none"
                                stroke="#2563eb"
                                stroke-width="4"
                                stroke-linecap="round"
                                stroke-linejoin="round"
                                points="">
                            </polyline>
                        </svg>

                        <div class="chart-scale">
                            <span>Min: <b id="chartMinLabel"></b></span>
                            <span>Fixed Y-axis</span>
                            <span>Max: <b id="chartMaxLabel"></b></span>
                        </div>

                        <div class="legend-note" style="margin-top:8px;">
                            <b>การอ่าน Score:</b>
                            ค่า anomaly score ที่ต่ำลงหมายถึงข้อมูลมีความผิดปกติมากขึ้น
                            ใน Isolation Forest ที่ใช้ในงานนี้
                        </div>
                    </div>

                </div>

            </div>
        </div>
    </div>


    <!-- =====================================================
         FEED + DONUT
    ====================================================== -->

    <div class="section-grid">

        <section class="panel">
            <div class="panel-title">
                🧾 Replay Feed
            </div>

            <div class="panel-body">
                <div class="feed-wrap">
                    <div class="feed" id="feed"></div>
                </div>
            </div>
        </section>

        <section class="panel">
            <div class="panel-title">
                🎭 Emotion Overview
            </div>

            <div class="panel-body">
                {donut_html}
            </div>
        </section>

    </div>


    <!-- =====================================================
         TRAFFIC SUMMARY
    ====================================================== -->

    <section class="panel" style="margin-top:14px;">

        <div class="panel-title">
            📋 Traffic Summary
        </div>

        <div class="panel-body">

            <div class="table-wrap">

                <table>

                    <thead>
                        <tr>
                            <th>Emotion</th>
                            <th>Traffic Type</th>
                            <th>Risk</th>
                            <th>Main Emotion</th>
                            <th>Mean Score</th>
                            <th>Records</th>
                        </tr>
                    </thead>

                    <tbody>
                        {traffic_table_rows}
                    </tbody>

                </table>

            </div>

        </div>

    </section>


    <!-- =====================================================
         EMOTION BY TRAFFIC
    ====================================================== -->

    <section class="panel" style="margin-top:14px;">

        <div class="panel-title">
            🎭 Emotion by Traffic Type
        </div>

        <div class="panel-body">
            {stack_html}
        </div>

    </section>


    <div class="footer">
        CIC-IDS2017 • Isolation Forest • Network Emotion Analysis • Offline Replay Dashboard
    </div>

</div>


<script>
const playbackData = {playback_json};

const emotionMeta = {{
    "Happy":    {{ emoji: "😄", color: "#22c55e", risk: 1 }},
    "Alert":    {{ emoji: "😮", color: "#f59e0b", risk: 2 }},
    "Anxious":  {{ emoji: "😰", color: "#eab308", risk: 3 }},
    "Stressed": {{ emoji: "😣", color: "#f97316", risk: 4 }},
    "Angry":    {{ emoji: "😡", color: "#ef4444", risk: 5 }}
}};

let currentIndex = -1;
let running = false;
let timer = null;
let speed = 1;

let scoreHistory = [];
let recentFeed = [];


function setStatus(text, runningState) {{
    document.getElementById("statusText").textContent = text;

    const dot = document.getElementById("statusDot");

    if (runningState) {{
        dot.style.background = "#22c55e";
        dot.style.boxShadow = "0 0 0 4px #dcfce7";
    }} else {{
        dot.style.background = "#94a3b8";
        dot.style.boxShadow = "0 0 0 4px #e2e8f0";
    }}
}}


function clearActiveEmotionCards() {{
    document.querySelectorAll(".emotion-card").forEach(card => {{
        card.classList.remove("active");
        card.style.borderColor = "#e5eaf0";
    }});
}}


function activateEmotionCard(emotion) {{
    clearActiveEmotionCards();

    const card = document.getElementById("emotion-card-" + emotion);

    if (card) {{
        card.classList.add("active");
        card.style.borderColor = emotionMeta[emotion].color;
    }}
}}


// =========================================================
// FIXED-SCALE ANOMALY SCORE CHART
// ใช้ Min/Max ของข้อมูล Replay ทั้งชุดเป็นแกน Y เดียว
// สเกลจึงไม่เปลี่ยนไปมาระหว่างการเล่น
// =========================================================

const chartScores = playbackData.map(row => Number(row.anomaly_score));

const chartDataMin = Math.min(...chartScores);
const chartDataMax = Math.max(...chartScores);

const chartPadding = Math.max(
    (chartDataMax - chartDataMin) * 0.08,
    0.005
);

const fixedChartMin = chartDataMin - chartPadding;
const fixedChartMax = chartDataMax + chartPadding;


function updateScoreChart() {{
    const line = document.getElementById("scoreLine");

    if (scoreHistory.length < 2) {{
        line.setAttribute("points", "");
        return;
    }}

    const width = 720;
    const height = 250;
    const padX = 20;
    const padY = 20;

    const range = fixedChartMax - fixedChartMin;

    const points = scoreHistory.map((score, i) => {{
        const x =
            padX +
            (i / Math.max(scoreHistory.length - 1, 1)) *
            (width - padX * 2);

        let normalized =
            (score - fixedChartMin) / range;

        normalized = Math.max(0, Math.min(1, normalized));

        const y =
            height -
            padY -
            normalized *
            (height - padY * 2);

        return `${{x.toFixed(2)}},${{y.toFixed(2)}}`;
    }}).join(" ");

    line.setAttribute("points", points);
}}


function addFeed(row) {{
    recentFeed.unshift(row);

    if (recentFeed.length > 8) {{
        recentFeed.pop();
    }}

    const feed = document.getElementById("feed");

    feed.innerHTML = recentFeed.map(item => {{
        const meta = emotionMeta[item.emotion] || emotionMeta["Alert"];

        return `
            <div class="feed-item">
                <div class="feed-face">${{meta.emoji}}</div>

                <div class="feed-main">
                    <div class="feed-title">
                        ${{item.attack_type}}
                    </div>

                    <div class="feed-sub">
                        Risk ${{item.risk_level}} • ${{item.emotion}} • Record ${{item.seq}}
                    </div>
                </div>

                <div class="feed-score">
                    ${{Number(item.anomaly_score).toFixed(6)}}
                </div>
            </div>
        `;
    }}).join("");
}}


function showRow(index) {{
    if (index < 0 || index >= playbackData.length) {{
        return;
    }}

    const row = playbackData[index];
    const meta = emotionMeta[row.emotion] || emotionMeta["Alert"];

    currentIndex = index;

    document.getElementById("currentFace").textContent = meta.emoji;
    document.getElementById("currentEmotion").textContent =
        `${{meta.emoji}} ${{row.emotion}}`;

    document.getElementById("currentEmotion").style.color = meta.color;

    document.getElementById("currentTraffic").textContent =
        row.attack_type;

    document.getElementById("currentRisk").textContent =
        `Risk ${{row.risk_level}}`;

    document.getElementById("currentScore").textContent =
        Number(row.anomaly_score).toFixed(6);

    document.getElementById("currentSeq").textContent =
        `${{row.seq}} / ${{playbackData.length}}`;

    const percent =
        ((index + 1) / playbackData.length) * 100;

    document.getElementById("progressFill").style.width =
        percent + "%";

    document.getElementById("progressRight").textContent =
        percent.toFixed(1) + "%";

    document.getElementById("progressLeft").textContent =
        `Record ${{row.seq}} / ${{playbackData.length}}`;

    activateEmotionCard(row.emotion);

    scoreHistory.push(Number(row.anomaly_score));

    if (scoreHistory.length > 35) {{
        scoreHistory.shift();
    }}

    updateScoreChart();
    addFeed(row);
}}


function tick() {{
    if (!running) {{
        return;
    }}

    if (currentIndex >= playbackData.length - 1) {{
        running = false;
        clearInterval(timer);
        timer = null;

        setStatus("REPLAY COMPLETE", false);

        document.getElementById("startBtn").textContent =
            "▶ Replay Again";

        return;
    }}

    showRow(currentIndex + 1);
}}


function startReplay() {{
    if (playbackData.length === 0) {{
        return;
    }}

    if (currentIndex >= playbackData.length - 1) {{
        resetReplay();
    }}

    if (running) {{
        return;
    }}

    running = true;

    setStatus("PLAYING OFFLINE DATA", true);

    if (currentIndex < 0) {{
        showRow(0);
    }}

    if (timer !== null) {{
        clearInterval(timer);
    }}

    const interval = Math.max(120, 900 / speed);

    timer = setInterval(tick, interval);

    document.getElementById("startBtn").textContent =
        "▶ Playing...";
}}


function pauseReplay() {{
    running = false;

    if (timer !== null) {{
        clearInterval(timer);
        timer = null;
    }}

    setStatus("PAUSED", false);

    document.getElementById("startBtn").textContent =
        "▶ Continue";
}}


function resetReplay() {{
    running = false;

    if (timer !== null) {{
        clearInterval(timer);
        timer = null;
    }}

    currentIndex = -1;
    scoreHistory = [];
    recentFeed = [];

    document.getElementById("currentFace").textContent = "💤";
    document.getElementById("currentEmotion").textContent = "Ready";
    document.getElementById("currentEmotion").style.color = "#172033";

    document.getElementById("currentTraffic").textContent =
        "กด Start เพื่อเริ่มเล่นข้อมูล";

    document.getElementById("currentRisk").textContent = "—";
    document.getElementById("currentScore").textContent = "—";
    document.getElementById("currentSeq").textContent =
        `0 / ${{playbackData.length}}`;

    document.getElementById("progressFill").style.width = "0%";
    document.getElementById("progressRight").textContent = "0%";
    document.getElementById("progressLeft").textContent =
        "Replay progress";

    document.getElementById("feed").innerHTML = "";

    clearActiveEmotionCards();
    updateScoreChart();

    setStatus("OFFLINE REPLAY", false);

    document.getElementById("startBtn").textContent =
        "▶ Start";
}}


document.getElementById("startBtn").addEventListener(
    "click",
    startReplay
);

document.getElementById("pauseBtn").addEventListener(
    "click",
    pauseReplay
);

document.getElementById("resetBtn").addEventListener(
    "click",
    resetReplay
);

document.querySelectorAll(".speed").forEach(button => {{
    button.addEventListener("click", () => {{
        document.querySelectorAll(".speed").forEach(btn => {{
            btn.classList.remove("active");
        }});

        button.classList.add("active");

        speed = Number(button.dataset.speed);

        if (running) {{
            clearInterval(timer);

            const interval = Math.max(120, 900 / speed);

            timer = setInterval(tick, interval);
        }}
    }});
}});


// แสดงแกน Y แบบคงที่ตลอดการ Replay
document.getElementById("chartMinLabel").textContent =
    fixedChartMin.toFixed(6);

document.getElementById("chartMaxLabel").textContent =
    fixedChartMax.toFixed(6);


// เริ่มอัตโนมัติหลังเปิดหน้า
setTimeout(() => {{
    startReplay();
}}, 1000);

</script>

</body>
</html>
"""


# =========================================================
# SAVE + OPEN
# =========================================================

OUTPUT_FILE.write_text(html_doc, encoding="utf-8")

print("=" * 72)
print("OFFLINE REPLAY DASHBOARD CREATED")
print("=" * 72)
print(f"File : {{OUTPUT_FILE.resolve()}}")
print(f"Rows : {{total_records:,}}")
print(f"Replay points : {{len(playback_rows):,}}")
print("Mode : OFFLINE REPLAY (NOT REAL-TIME)")
print("=" * 72)

webbrowser.open(OUTPUT_FILE.resolve().as_uri())
