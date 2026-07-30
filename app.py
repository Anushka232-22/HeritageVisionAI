"""
HeritageVisionAI — Structural Crack Inspection GUI
====================================================
Streamlit front-end for the HeritagePipeline (YOLO crack detector +
EfficientNet damage classifier + severity scoring).

Run from the project root:
    streamlit run app.py

Expects the same relative paths your pipeline already uses:
    models/detector/best_detector.pt
    models/classifier/best_model_v2.pth
    models/classifier/classes.json
"""

import io
import json
import time
import uuid
from pathlib import Path

import streamlit as st
from PIL import Image, ImageDraw, ImageFont

from src.inference.pipeline import HeritagePipeline
from src.severity.report_generator import generate_report

try:
    from src.visualization.visualize import Visualizer
    HAS_VISUALIZER = True
except Exception:
    HAS_VISUALIZER = False


# ------------------------------------------------------------------
# Page config
# ------------------------------------------------------------------

st.set_page_config(
    page_title="HeritageVisionAI — Crack Inspection",
    page_icon="🧱",
    layout="wide",
    initial_sidebar_state="expanded",
)

UPLOAD_DIR = Path("data/gui_uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

RISK_COLORS = {
    "LOW": "#4CAF7D",
    "MODERATE": "#D9A441",
    "HIGH": "#E0793E",
    "CRITICAL": "#C0392B",
}


# ------------------------------------------------------------------
# Styling — graphite / blueprint theme, grounded in masonry subject
# ------------------------------------------------------------------

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,500;9..144,600;9..144,700&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500;600&display=swap');

    html, body, [class*="css"]  {
        font-family: 'IBM Plex Sans', sans-serif;
    }

    .stApp {
        background-color: #14181C;
        color: #E8ECEF;
    }

    h1, h2, h3 {
        font-family: 'Fraunces', serif !important;
        color: #EDE8E0 !important;
        letter-spacing: 0.2px;
    }

    .hv-eyebrow {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.72rem;
        letter-spacing: 2.5px;
        text-transform: uppercase;
        color: #A8562E;
        margin-bottom: 0.2rem;
    }

    .hv-subtitle {
        color: #8B96A0;
        font-size: 0.95rem;
        margin-top: -0.4rem;
        margin-bottom: 1.4rem;
    }

    hr.hv-rule {
        border: none;
        border-top: 1px solid #2C343B;
        margin: 1.4rem 0;
    }

    /* Stat cards */
    .stat-card {
        background: #1D2329;
        border: 1px solid #2C343B;
        border-radius: 4px;
        padding: 14px 16px;
        height: 100%;
    }
    .stat-label {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.68rem;
        letter-spacing: 1.5px;
        text-transform: uppercase;
        color: #8B96A0;
        margin-bottom: 4px;
    }
    .stat-value {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 1.6rem;
        font-weight: 600;
        color: #E8ECEF;
    }
    .stat-unit {
        font-size: 0.85rem;
        color: #8B96A0;
        margin-left: 4px;
    }

    /* Risk badge */
    .risk-badge {
        display: inline-block;
        font-family: 'IBM Plex Mono', monospace;
        font-weight: 600;
        font-size: 0.85rem;
        letter-spacing: 1.5px;
        padding: 5px 14px;
        border-radius: 3px;
        color: #14181C;
    }

    /* Brick severity meter */
    .brick-meter-wrap {
        display: flex;
        align-items: center;
        gap: 14px;
        margin: 6px 0 2px 0;
        flex-wrap: wrap;
    }
    .brick-meter {
        display: flex;
        gap: 3px;
    }
    .brick {
        width: 13px;
        height: 24px;
        border-radius: 1px;
        border: 1px solid #2C343B;
        background: #1D2329;
    }
    .score-num {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 1.3rem;
        font-weight: 600;
        color: #E8ECEF;
    }
    .score-max {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.85rem;
        color: #8B96A0;
    }

    /* Recommendation box */
    .rec-box {
        border-left: 3px solid #A8562E;
        background: #1D2329;
        padding: 10px 16px;
        color: #D8DEE3;
        font-size: 0.92rem;
        margin-top: 10px;
    }

    /* Crack card */
    .crack-card {
        background: #1D2329;
        border: 1px solid #2C343B;
        border-radius: 4px;
        padding: 12px 14px;
    }
    .crack-title {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.8rem;
        letter-spacing: 1px;
        color: #A8562E;
        text-transform: uppercase;
        margin-bottom: 6px;
    }
    .crack-row {
        display: flex;
        justify-content: space-between;
        font-size: 0.85rem;
        color: #C6CCD1;
        padding: 2px 0;
        border-bottom: 1px dashed #2C343B;
    }
    .crack-row:last-child { border-bottom: none; }
    .crack-row span.k { color: #8B96A0; }
    .crack-row span.v { font-family: 'IBM Plex Mono', monospace; }

    section[data-testid="stSidebar"] {
        background-color: #171B1F;
        border-right: 1px solid #2C343B;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ------------------------------------------------------------------
# Model loading (cached — heavy: loads YOLO + EfficientNet once)
# ------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading detector and classifier models…")
def load_pipeline():
    pipe = HeritagePipeline()
    viz = Visualizer() if HAS_VISUALIZER else None
    return pipe, viz


def render_brick_meter(score: float, risk_level: str) -> str:
    total_bricks = 20
    filled = round((score / 100) * total_bricks)
    color = RISK_COLORS.get(risk_level, "#8B96A0")

    bricks_html = ""
    for i in range(total_bricks):
        style = f"background:{color};border-color:{color};" if i < filled else ""
        bricks_html += f'<div class="brick" style="{style}"></div>'

    return f"""
    <div class="brick-meter-wrap">
        <div class="brick-meter">{bricks_html}</div>
        <div><span class="score-num">{score}</span><span class="score-max">&nbsp;/ 100</span></div>
    </div>
    """


def stat_card(label: str, value, unit: str = "") -> str:
    return f"""
    <div class="stat-card">
        <div class="stat-label">{label}</div>
        <div class="stat-value">{value}<span class="stat-unit">{unit}</span></div>
    </div>
    """


def fallback_annotate(image: Image.Image, cracks: list) -> Image.Image:
    """Draw bounding boxes + labels ourselves if Visualizer is unavailable
    or raises, so the GUI still shows something useful."""
    img = image.convert("RGB").copy()
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    for crack in cracks:
        x1, y1, x2, y2 = crack["bbox"]
        color = "#E0793E"
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        label = f'{crack["damage_type"]} ({crack["damage_confidence"]:.2f})'
        draw.rectangle([x1, max(0, y1 - 16), x1 + 8 * len(label), y1], fill=color)
        draw.text((x1 + 2, max(0, y1 - 15)), label, fill="#14181C", font=font)

    return img


def crop_from_bbox(image: Image.Image, bbox: list) -> Image.Image:
    x1, y1, x2, y2 = bbox
    return image.crop((x1, y1, x2, y2))


# ------------------------------------------------------------------
# Sidebar
# ------------------------------------------------------------------

with st.sidebar:
    st.markdown('<div class="hv-eyebrow">HeritageVisionAI</div>', unsafe_allow_html=True)
    st.markdown("### Inspection Console")
    st.markdown(
        '<p class="hv-subtitle">YOLO crack detector · EfficientNet damage '
        "classifier · automated severity scoring for heritage masonry.</p>",
        unsafe_allow_html=True,
    )
    st.markdown('<hr class="hv-rule">', unsafe_allow_html=True)
    st.markdown("**Pipeline stages**")
    st.markdown(
        "1. Detect cracks (YOLO)\n"
        "2. Classify damage type per crack\n"
        "3. Extract length / area / density\n"
        "4. Score overall severity & risk"
    )
    st.markdown('<hr class="hv-rule">', unsafe_allow_html=True)
    st.caption("Models load once and stay cached for the session.")


# ------------------------------------------------------------------
# Header
# ------------------------------------------------------------------

st.markdown('<div class="hv-eyebrow">Structural Survey</div>', unsafe_allow_html=True)
st.markdown("# Heritage Crack Inspection")
st.markdown(
    '<p class="hv-subtitle">Upload a photo of a masonry surface to run detection, '
    "damage classification, and severity scoring.</p>",
    unsafe_allow_html=True,
)

# Load pipeline (shows spinner on first run only)
try:
    pipeline, visualizer = load_pipeline()
except Exception as e:
    st.error(
        "Couldn't load the models. Check that these files exist relative to "
        "the folder you launched Streamlit from:\n\n"
        "- `models/detector/best_detector.pt`\n"
        "- `models/classifier/best_model_v2.pth`\n"
        "- `models/classifier/classes.json`"
    )
    st.exception(e)
    st.stop()

uploaded_file = st.file_uploader(
    "Upload an image (JPG or PNG)", type=["jpg", "jpeg", "png"]
)

run_col, _ = st.columns([1, 4])
run_clicked = run_col.button("Run Inspection", type="primary", disabled=uploaded_file is None)


# ------------------------------------------------------------------
# Run inspection
# ------------------------------------------------------------------

if run_clicked and uploaded_file is not None:
    # Save upload to disk — the pipeline needs a real file path
    suffix = Path(uploaded_file.name).suffix or ".jpg"
    save_path = UPLOAD_DIR / f"{uuid.uuid4().hex}{suffix}"
    with open(save_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    original_image = Image.open(save_path).convert("RGB")

    with st.spinner("Running detection, classification, and severity scoring…"):
        t0 = time.time()
        report = pipeline.analyze(str(save_path))
        elapsed = time.time() - t0

        json_path = None
        try:
            json_path = generate_report(report)
        except Exception:
            pass

        annotated_image = None
        if visualizer is not None:
            try:
                viz_path = visualizer.draw_predictions(str(save_path), report)
                annotated_image = Image.open(viz_path).convert("RGB")
            except Exception:
                annotated_image = None

        if annotated_image is None:
            annotated_image = fallback_annotate(original_image, report["cracks"])

    st.session_state["report"] = report
    st.session_state["original_image"] = original_image
    st.session_state["annotated_image"] = annotated_image
    st.session_state["json_path"] = json_path
    st.session_state["elapsed"] = elapsed


# ------------------------------------------------------------------
# Results
# ------------------------------------------------------------------

if "report" in st.session_state:
    report = st.session_state["report"]
    original_image = st.session_state["original_image"]
    annotated_image = st.session_state["annotated_image"]
    elapsed = st.session_state["elapsed"]
    severity = report["severity"]
    risk_level = severity["risk_level"]
    risk_color = RISK_COLORS.get(risk_level, "#8B96A0")

    st.markdown('<hr class="hv-rule">', unsafe_allow_html=True)

    img_col1, img_col2 = st.columns(2)
    with img_col1:
        st.markdown("**Original**")
        st.image(original_image, use_container_width=True)
    with img_col2:
        st.markdown("**Detections**")
        st.image(annotated_image, use_container_width=True)

    st.caption(f"Analyzed `{report['image_name']}` in {elapsed:.2f}s")

    st.markdown('<hr class="hv-rule">', unsafe_allow_html=True)

    # Stat row
    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(stat_card("Cracks Found", report["num_cracks"]), unsafe_allow_html=True)
    c2.markdown(stat_card("Total Length", severity["crack_length"], "px"), unsafe_allow_html=True)
    c3.markdown(stat_card("Total Area", severity["crack_area"], "px²"), unsafe_allow_html=True)
    c4.markdown(stat_card("Avg. Density", severity["crack_density"]), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Severity meter + risk badge
    sev_col, badge_col = st.columns([3, 1])
    with sev_col:
        st.markdown("**Severity Score**")
        st.markdown(render_brick_meter(severity["severity_score"], risk_level), unsafe_allow_html=True)
    with badge_col:
        st.markdown("**Risk Level**")
        st.markdown(
            f'<span class="risk-badge" style="background:{risk_color};">{risk_level}</span>',
            unsafe_allow_html=True,
        )

    st.markdown(
        f'<div class="rec-box">{severity["recommendation"]}</div>',
        unsafe_allow_html=True,
    )

    # Per-crack breakdown
    if report["num_cracks"] > 0:
        st.markdown('<hr class="hv-rule">', unsafe_allow_html=True)
        st.markdown("### Crack Detail")

        cols_per_row = 3
        cracks = report["cracks"]
        for row_start in range(0, len(cracks), cols_per_row):
            row_cracks = cracks[row_start: row_start + cols_per_row]
            cols = st.columns(cols_per_row)
            for col, crack in zip(cols, row_cracks):
                idx = row_start + row_cracks.index(crack) + 1
                with col:
                    crop = crop_from_bbox(original_image, crack["bbox"])
                    st.image(crop, use_container_width=True)
                    m = crack["metrics"]
                    st.markdown(
                        f"""
                        <div class="crack-card">
                            <div class="crack-title">Crack {idx}</div>
                            <div class="crack-row"><span class="k">Damage type</span><span class="v">{crack['damage_type']}</span></div>
                            <div class="crack-row"><span class="k">Detection conf.</span><span class="v">{crack['detection_confidence']}</span></div>
                            <div class="crack-row"><span class="k">Class conf.</span><span class="v">{crack['damage_confidence']}</span></div>
                            <div class="crack-row"><span class="k">Length</span><span class="v">{m['crack_length']} px</span></div>
                            <div class="crack-row"><span class="k">Area</span><span class="v">{m['crack_area']} px²</span></div>
                            <div class="crack-row"><span class="k">Density</span><span class="v">{m['crack_density']}</span></div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
    else:
        st.info("No cracks detected in this image.")

    # Downloads
    st.markdown('<hr class="hv-rule">', unsafe_allow_html=True)
    dl_col1, dl_col2 = st.columns(2)
    with dl_col1:
        st.download_button(
            "Download JSON Report",
            data=json.dumps(report, indent=4),
            file_name=f"{Path(report['image_name']).stem}_report.json",
            mime="application/json",
        )
    with dl_col2:
        buf = io.BytesIO()
        annotated_image.save(buf, format="PNG")
        st.download_button(
            "Download Annotated Image",
            data=buf.getvalue(),
            file_name=f"{Path(report['image_name']).stem}_annotated.png",
            mime="image/png",
        )

else:
    st.markdown('<hr class="hv-rule">', unsafe_allow_html=True)
    st.info("Upload an image and click **Run Inspection** to begin.")