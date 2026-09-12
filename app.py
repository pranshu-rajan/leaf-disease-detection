import streamlit as st
import torch
import torch.nn as nn
import timm
import numpy as np
import cv2
import json
import os
import re
import datetime
from io import BytesIO
from xml.sax.saxutils import escape

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.units import inch, mm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable,
)
from reportlab.lib import colors
from PIL import Image
import torchvision.transforms as T

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="Plant Leaf Disease Detector",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# THEME / MODERN CSS
# ============================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;600;700;800&display=swap');
@import url('https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.3/font/bootstrap-icons.css');

html, body, [class*="css"]  {
    font-family: 'Manrope', sans-serif;
}

/* ---- Icon system (replaces emoji throughout the app) ---- */
.icon { margin-right: 0.45rem; }
.icon-success { color: #2E7D32; }
.icon-warning { color: #F57F17; }
.icon-error { color: #C62828; }
.icon-muted { color: #8A8A8A; }
.icon-white { color: #ffffff; }

.status-row { display: flex; align-items: center; gap: 0.55rem; margin: 0.3rem 0; font-size: 0.92rem; }
.status-dot {
    width: 9px; height: 9px; border-radius: 50%; flex-shrink: 0;
    box-shadow: 0 0 0 3px rgba(0,0,0,0.03);
}
.status-dot-ok { background: #2E7D32; }
.status-dot-warn { background: #F57F17; }
.status-dot-error { background: #C62828; }

.section-heading {
    display: flex; align-items: center; gap: 0.55rem;
    font-size: 1.25rem; font-weight: 800; color: #1B5E20;
    margin: 0.2rem 0 0.6rem 0;
}
.section-heading i { font-size: 1.1rem; }

.pipeline-step { display: flex; align-items: center; gap: 0.6rem; font-size: 0.92rem; margin: 0.15rem 0; }
.pipeline-step i { color: #2E7D32; width: 1.1rem; text-align: center; }

.brand-mark {
    display: inline-flex; align-items: center; justify-content: center;
    width: 40px; height: 40px; border-radius: 12px;
    background: rgba(255,255,255,0.18);
    border: 1px solid rgba(255,255,255,0.35);
    margin-right: 0.8rem; vertical-align: middle;
}

/* ---- Hero header ---- */
.hero {
    background: linear-gradient(135deg, #1B5E20 0%, #2E7D32 45%, #66BB6A 100%);
    padding: 2.2rem 2rem;
    border-radius: 18px;
    margin-bottom: 1.6rem;
    box-shadow: 0 10px 30px rgba(27, 94, 32, 0.25);
}
.hero-title {
    font-size: 2.1rem;
    font-weight: 800;
    color: white;
    margin: 0;
}
.hero-subtitle {
    color: rgba(255,255,255,0.9);
    font-size: 1rem;
    margin-top: 0.35rem;
}
.hero-badges { margin-top: 0.9rem; }
.hero-pill {
    display: inline-block;
    background: rgba(255,255,255,0.18);
    color: white;
    border: 1px solid rgba(255,255,255,0.35);
    padding: 0.25rem 0.75rem;
    border-radius: 999px;
    font-size: 0.78rem;
    margin-right: 0.5rem;
    backdrop-filter: blur(4px);
}

/* ---- Cards ---- */
.card {
    background: var(--background-color, #ffffff);
    border: 1px solid rgba(46, 125, 50, 0.15);
    border-radius: 16px;
    padding: 1.3rem 1.5rem;
    box-shadow: 0 4px 18px rgba(0,0,0,0.05);
}
.result-card {
    background: linear-gradient(135deg, #F1F8E9 0%, #E8F5E9 100%);
    border-left: 6px solid #2E7D32;
    border-radius: 14px;
    padding: 1.3rem 1.6rem;
    margin-bottom: 0.8rem;
}
.disease-name { font-size: 1.5rem; font-weight: 800; color: #1B5E20; }
.crop-tag {
    font-size: 0.85rem; color: #4E7D50; font-weight: 600;
    text-transform: uppercase; letter-spacing: 0.04em;
}
.confidence-badge {
    background: #2E7D32; color: white; padding: 0.25rem 0.8rem;
    border-radius: 999px; font-size: 0.85rem; font-weight: 700;
    display: inline-block; margin-top: 0.4rem;
}
.severity-mild-badge, .severity-moderate-badge, .severity-severe-badge {
    display: inline-block; padding: 0.25rem 0.8rem; border-radius: 999px;
    font-size: 0.85rem; font-weight: 700; color: white; margin-top: 0.4rem;
}
.severity-mild-badge { background: #2E7D32; }
.severity-moderate-badge { background: #F57F17; }
.severity-severe-badge { background: #C62828; }

.healthy-banner {
    background: linear-gradient(135deg, #43A047, #66BB6A);
    color: white; padding: 1rem 1.4rem; border-radius: 14px;
    font-weight: 700; font-size: 1.05rem;
}

/* ---- AI report bubble ---- */
.ai-report-box {
    background: #F8FAF8 !important;
    border: 1px solid #DCE9DD;
    border-left: 5px solid #2E7D32;
    border-radius: 14px;
    padding: 1.3rem 1.5rem;
    line-height: 1.55;
    font-size: 0.98rem;
    color: #1F2A1F !important;
}
.ai-report-box, .ai-report-box * {
    color: #1F2A1F !important;
}

footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
    <div style="display:flex; align-items:center;">
        <span class="brand-mark">
            <svg width="22" height="22" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                <path d="M20 4C20 4 8 3 4 9C0.5 14.2 4 20 4 20C4 20 10 21 15 16C19 12 20 4 20 4Z" fill="#ffffff" fill-opacity="0.95"/>
                <path d="M4 20C4 20 9 12 20 4" stroke="#1B5E20" stroke-width="1.3" stroke-linecap="round"/>
            </svg>
        </span>
        <div class="hero-title">Plant Leaf Disease Detector</div>
    </div>
    <div class="hero-subtitle">Hybrid CNN + Vision Transformer classification, lesion detection, severity scoring, and AI-generated advisory reports</div>
    <div class="hero-badges">
        <span class="hero-pill">EfficientNetV2 + ViT</span>
        <span class="hero-pill">YOLO lesion detection</span>
        <span class="hero-pill">Grad-CAM++ explainability</span>
        <span class="hero-pill">AI report + PDF export</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ============================================================
# PATHS
# ============================================================
MODELS_DIR = "./models"
HYBRID_CKPT = f"{MODELS_DIR}/hybrid_model.pth"
CLASS_NAMES_PATH = f"{MODELS_DIR}/class_names.json"
YOLO_CKPT = f"{MODELS_DIR}/lesion_detector_best.pt"

IMG_SIZE = 224
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================
# OPTIONAL: fetch large model weights at startup
# ============================================================
# Model checkpoints are usually too large to commit to git. If they're not
# present on disk, this downloads them from URLs supplied via Streamlit
# secrets (e.g. a Hugging Face Hub / Google Drive / S3 direct-download
# link) so the repo itself only needs to hold source code. Safe to leave
# unused if you're providing the weights another way (e.g. Git LFS).
def _maybe_download_weights():
    url_map = {
        HYBRID_CKPT: "HYBRID_CKPT_URL",
        YOLO_CKPT: "YOLO_CKPT_URL",
        CLASS_NAMES_PATH: "CLASS_NAMES_URL",
    }
    for local_path, secret_key in url_map.items():
        if os.path.exists(local_path):
            continue
        url = None
        try:
            url = st.secrets.get(secret_key)
        except Exception:
            url = os.environ.get(secret_key)
        if not url:
            continue
        try:
            import urllib.request
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            with st.spinner(f"Downloading {os.path.basename(local_path)}..."):
                urllib.request.urlretrieve(url, local_path)
        except Exception as e:
            st.warning(f"Could not download {local_path}: {e}")

_maybe_download_weights()

# ============================================================
# MODEL DEFINITIONS (must exactly match training-time architecture)
# ============================================================
class HybridCNNViT(nn.Module):
    def __init__(self, cnn_name, vit_name, num_classes, dropout=0.3):
        super().__init__()
        self.cnn = timm.create_model(cnn_name, pretrained=False, num_classes=0, global_pool="avg")
        self.vit = timm.create_model(vit_name, pretrained=False, num_classes=0, global_pool="avg")
        cnn_feat_dim = self.cnn.num_features
        vit_feat_dim = self.vit.num_features
        self.fusion = nn.Sequential(
            nn.Linear(cnn_feat_dim + vit_feat_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Linear(512, num_classes)

    def forward(self, x):
        cnn_feat = self.cnn(x)
        vit_feat = self.vit(x)
        fused = torch.cat([cnn_feat, vit_feat], dim=1)
        fused = self.fusion(fused)
        return self.classifier(fused)


@st.cache_resource(show_spinner="Loading models (first run only)...")
def load_models():
    errors = []

    class_names = None
    if os.path.exists(CLASS_NAMES_PATH):
        with open(CLASS_NAMES_PATH) as f:
            class_names = json.load(f)
    else:
        errors.append(f"Missing: {CLASS_NAMES_PATH}")

    hybrid_model = None
    if os.path.exists(HYBRID_CKPT) and class_names is not None:
        hybrid_model = HybridCNNViT("tf_efficientnetv2_s", "vit_small_patch16_224", len(class_names))
        hybrid_model.load_state_dict(torch.load(HYBRID_CKPT, map_location=device))
        hybrid_model.to(device)
        hybrid_model.eval()
    else:
        errors.append(f"Missing: {HYBRID_CKPT}")

    yolo_model = None
    if os.path.exists(YOLO_CKPT):
        from ultralytics import YOLO
        yolo_model = YOLO(YOLO_CKPT)
    else:
        errors.append(f"Missing (optional — lesion detection will be skipped): {YOLO_CKPT}")

    return hybrid_model, yolo_model, class_names, errors


hybrid_model, yolo_model, class_names, load_errors = load_models()

if hybrid_model is None or class_names is None:
    st.error("Required model files are missing. See setup instructions in the sidebar.")
    st.stop()

# ============================================================
# PREPROCESSING (must match training exactly)
# ============================================================
def apply_clahe(img_rgb):
    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_enhanced = clahe.apply(l)
    lab_enhanced = cv2.merge((l_enhanced, a, b))
    return cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2RGB)

inference_transform = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def preprocess_for_model(pil_img):
    img_rgb = np.array(pil_img.convert("RGB"))
    enhanced = apply_clahe(img_rgb)
    enhanced_pil = Image.fromarray(enhanced)
    tensor = inference_transform(enhanced_pil).unsqueeze(0).to(device)
    return tensor

# ============================================================
# CLASSICAL CV — leaf segmentation + severity
# ============================================================
cv2.setRNGSeed(42)

def segment_leaf(img_bgr):
    h, w = img_bgr.shape[:2]
    mask = np.zeros((h, w), np.uint8)
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)
    margin_x, margin_y = int(w * 0.05), int(h * 0.05)
    rect = (margin_x, margin_y, w - 2 * margin_x, h - 2 * margin_y)
    cv2.grabCut(img_bgr, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)
    binary_mask = np.where((mask == 2) | (mask == 0), 0, 255).astype("uint8")
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
    if num_labels > 1:
        largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        binary_mask = np.where(labels == largest_label, 255, 0).astype(np.uint8)
    kernel = np.ones((5, 5), np.uint8)
    return cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)


_face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

def is_leaf_present(img_bgr, min_green_fraction=0.05, min_vegetation_fraction=0.12):
    """
    Classical-CV gate to reject clearly non-leaf images BEFORE classification.
    A closed-set classifier (38 disease classes) will always output its best guess even
    for a photo of a wall, a face, or a blank image — it has no built-in "none of these"
    option. This heuristic checks whether the image actually contains plausible plant
    material at all.

    IMPORTANT FIX: an earlier version allowed a broad brown/tan color range (meant to catch
    diseased/dried leaf tissue) to qualify on its own — but that range heavily overlaps with
    human skin tone and beige surfaces (wallpaper, walls), which let selfies and room photos
    pass as "leaf detected". Genuine green presence is now a hard requirement, and an explicit
    face-detection veto catches the skin-tone-overlap case directly.

    Returns: (is_leaf: bool, vegetation_fraction: float, reason: str)
    """
    h, w = img_bgr.shape[:2]
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    hue, sat = hsv[:, :, 0], hsv[:, :, 1]

    # Hard veto: if a face is detected anywhere in the frame, this is not a leaf photo.
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    faces = _face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
    if len(faces) > 0:
        return False, 0.0, "A face was detected in this image — this does not look like a leaf photo."

    # Genuine green tissue/background — required on its own, not satisfied by brown/tan alone
    # (skin tone and beige surfaces sit in a similar hue/saturation range to dried leaf tissue,
    # so brown/tan can no longer qualify an image by itself).
    green_mask = (hue >= 25) & (hue <= 95) & (sat >= 40)
    green_fraction = np.count_nonzero(green_mask) / (h * w)

    if green_fraction < min_green_fraction:
        return False, green_fraction, "Not enough green plant tissue detected in the image."

    # Combined vegetation fraction (green + diseased brown/tan) as a secondary, broader check —
    # only relevant now that the green_fraction requirement above already ruled out skin/walls.
    brown_tan_range = (hue >= 8) & (hue <= 30) & (sat >= 30) & (sat <= 200)
    vegetation_mask = green_mask | brown_tan_range
    vegetation_fraction = np.count_nonzero(vegetation_mask) / (h * w)

    if vegetation_fraction < min_vegetation_fraction:
        return False, vegetation_fraction, "Not enough plant-colored area detected in the image."

    # Segmentation sanity: GrabCut should find a distinct foreground object that's neither
    # ~0% nor ~100% of the frame.
    leaf_mask = segment_leaf(img_bgr)
    leaf_fraction = np.count_nonzero(leaf_mask) / (h * w)
    if leaf_fraction < 0.03 or leaf_fraction > 0.98:
        return False, vegetation_fraction, "Could not isolate a distinct leaf-shaped object in the image."

    return True, vegetation_fraction, ""


def estimate_severity(img_bgr, boxes_xyxy):
    leaf_mask = segment_leaf(img_bgr)
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    hue = hsv[:, :, 0]
    healthy_green = ((hue >= 35) & (hue <= 85)).astype(np.uint8) * 255
    lesion_candidate = cv2.bitwise_and(cv2.bitwise_not(healthy_green), leaf_mask)

    box_mask = np.zeros_like(leaf_mask)
    for (x1, y1, x2, y2) in boxes_xyxy:
        box_mask[int(y1):int(y2), int(x1):int(x2)] = 255
    lesion_mask = cv2.bitwise_and(lesion_candidate, box_mask)

    leaf_area = np.count_nonzero(leaf_mask)
    lesion_area = np.count_nonzero(lesion_mask)
    ratio = (lesion_area / leaf_area) if leaf_area > 0 else 0.0
    tier = "mild" if ratio < 0.10 else ("moderate" if ratio < 0.30 else "severe")
    return ratio, tier, lesion_mask

# ============================================================
# GRAD-CAM (CNN branch of the hybrid model)
# ============================================================
def get_gradcam_overlay(pil_img, tensor, class_idx):
    from pytorch_grad_cam import GradCAMPlusPlus
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    from pytorch_grad_cam.utils.image import show_cam_on_image

    with torch.no_grad():
        vit_feat_fixed = hybrid_model.vit(tensor)

    class Wrapper(nn.Module):
        def __init__(self, hm, vfixed):
            super().__init__()
            self.hm = hm
            self.vfixed = vfixed
        def forward(self, x):
            cnn_feat = self.hm.cnn(x)
            fused = torch.cat([cnn_feat, self.vfixed], dim=1)
            fused = self.hm.fusion(fused)
            return self.hm.classifier(fused)

    wrapper = Wrapper(hybrid_model, vit_feat_fixed).to(device)
    target_layers = [hybrid_model.cnn.conv_head]
    cam = GradCAMPlusPlus(model=wrapper, target_layers=target_layers)
    grayscale_cam = cam(input_tensor=tensor, targets=[ClassifierOutputTarget(class_idx)])[0]

    img_resized = np.array(pil_img.convert("RGB").resize((IMG_SIZE, IMG_SIZE))) / 255.0
    return show_cam_on_image(img_resized, grayscale_cam, use_rgb=True)

# ============================================================
# GROQ REPORT GENERATION
# ============================================================
def _get_groq_api_key():
    try:
        if "GROQ_API_KEY" in st.secrets:
            return st.secrets["GROQ_API_KEY"]
    except Exception:
        pass
    return os.environ.get("GROQ_API_KEY", "")

GROQ_API_KEY = _get_groq_api_key()
GROQ_MODEL = "openai/gpt-oss-120b"


def generate_groq_report(crop, disease, confidence, severity_pct, severity_tier):
    try:
        from groq import Groq

        if not GROQ_API_KEY:
            return None, "No Groq API key configured. Set the GROQ_API_KEY environment variable."

        client = Groq(api_key=GROQ_API_KEY)

        prompt = f"""You are an agricultural assistant helping a farmer understand a plant disease diagnosis.

Crop: {crop}
Detected condition: {disease}
Model confidence: {confidence:.1f}%
Estimated affected leaf area: {severity_pct:.1f}% ({severity_tier})

Write a short, clear, plain-language report (under 200 words) for the farmer covering:
1. What this disease is, in simple terms
2. How serious it looks based on the severity level given
3. 2-3 practical next steps or treatment options
4. A brief note that this is an AI estimate and a local agricultural expert should confirm before major treatment decisions

Write in plain prose paragraphs only. Do not use markdown formatting of any kind — no headers, no asterisks for bold or italics, no numbered or bulleted lists, no emoji."""

        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": "You are a helpful, concise agricultural assistant. You never use markdown syntax in your responses."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
            max_completion_tokens=500,
        )

        return response.choices[0].message.content, None

    except Exception as e:
        return None, str(e)


# ============================================================
# TEXT SANITIZATION FOR PDF
# ============================================================
# ReportLab's built-in Helvetica font only supports the Latin-1 / WinAnsi
# character set. LLM output sometimes contains smart quotes, em-dashes,
# bullet glyphs, or emoji that fall outside that set and render as garbled
# boxes. We normalize those characters and also strip any stray markdown
# syntax (the model is instructed not to use it, but this guards against it)
# so the PDF never shows literal "**", "-", or "#" characters.

_UNICODE_REPLACEMENTS = {
    "\u2018": "'", "\u2019": "'", "\u201a": "'",
    "\u201c": '"', "\u201d": '"', "\u201e": '"',
    "\u2013": "-", "\u2014": "--", "\u2026": "...",
    "\u2022": "-", "\u25cf": "-", "\u2023": "-",
    "\u00a0": " ", "\u200b": "",
}

def _normalize_unicode(text):
    for bad, good in _UNICODE_REPLACEMENTS.items():
        text = text.replace(bad, good)
    # Drop any remaining character Helvetica/WinAnsi can't render
    # (e.g. emoji) instead of letting it print as a black box.
    return text.encode("latin-1", "ignore").decode("latin-1")


def _markdown_inline_to_reportlab(escaped_text):
    """Convert **bold** / *italic* / __bold__ markdown (already XML-escaped)
    into ReportLab paragraph markup, and strip stray '#' headers."""
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", escaped_text)
    text = re.sub(r"__(.+?)__", r"<b>\1</b>", text)
    text = re.sub(r"(?<!\*)\*([^*\n]+?)\*(?!\*)", r"<i>\1</i>", text)
    text = re.sub(r"^\s{0,3}#{1,6}\s*", "", text)
    return text


def build_report_flowables(raw_text, body_style, bullet_style):
    """Turn the AI-generated report text into a clean list of Paragraph
    flowables — regular paragraphs and, if the model still slipped in a
    bullet/numbered list despite instructions, properly indented bullet
    items instead of raw '-'/'1.' characters."""
    flowables = []
    clean_text = _normalize_unicode(raw_text)

    for raw_line in clean_text.split("\n"):
        line = raw_line.strip()
        if not line:
            continue

        bullet_match = re.match(r"^[-*•]\s+(.*)", line)
        numbered_match = re.match(r"^\d+[\.\)]\s+(.*)", line)

        if bullet_match:
            content = _markdown_inline_to_reportlab(escape(bullet_match.group(1)))
            flowables.append(Paragraph(f"&bull;&nbsp;&nbsp;{content}", bullet_style))
        elif numbered_match:
            content = _markdown_inline_to_reportlab(escape(numbered_match.group(1)))
            flowables.append(Paragraph(f"&bull;&nbsp;&nbsp;{content}", bullet_style))
        else:
            content = _markdown_inline_to_reportlab(escape(line))
            flowables.append(Paragraph(content, body_style))

    return flowables


# ============================================================
# PDF REPORT GENERATION
# ============================================================
SEVERITY_COLORS = {
    "mild": colors.HexColor("#2E7D32"),
    "moderate": colors.HexColor("#F57F17"),
    "severe": colors.HexColor("#C62828"),
}

def _pdf_header_footer(canvas, doc):
    canvas.saveState()
    width, height = A4

    # Header bar
    canvas.setFillColor(colors.HexColor("#1B5E20"))
    canvas.rect(0, height - 14 * mm, width, 14 * mm, stroke=0, fill=1)
    canvas.setFillColor(colors.white)
    canvas.setFont("Helvetica-Bold", 11)
    canvas.drawString(20 * mm, height - 9.5 * mm, "PLANT LEAF DISEASE — AI DIAGNOSTIC REPORT")
    canvas.setFont("Helvetica", 8)
    canvas.drawRightString(width - 20 * mm, height - 9.5 * mm,
                            datetime.datetime.now().strftime("%d %b %Y, %H:%M"))

    # Footer
    canvas.setFillColor(colors.HexColor("#666666"))
    canvas.setFont("Helvetica", 8)
    canvas.drawString(20 * mm, 10 * mm, "Generated by Plant Leaf Disease Detector — AI-assisted, not a substitute for expert advice")
    canvas.drawRightString(width - 20 * mm, 10 * mm, f"Page {doc.page}")
    canvas.setStrokeColor(colors.HexColor("#DDDDDD"))
    canvas.line(20 * mm, 13 * mm, width - 20 * mm, 13 * mm)
    canvas.restoreState()


def create_pdf_report(crop, disease, confidence, severity_pct, severity_tier, ai_report):
    """Create a downloadable, well-formatted PDF containing the diagnosis
    and the Groq AI report, with markdown/unicode artifacts cleaned up."""
    buffer = BytesIO()

    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=20 * mm,
        leftMargin=20 * mm,
        topMargin=22 * mm,
        bottomMargin=18 * mm,
        title="Plant Leaf Disease AI Report",
        author="Plant Leaf Disease Detector",
    )

    styles = getSampleStyleSheet()
    subtitle_style = ParagraphStyle(
        "ReportSubtitle", parent=styles["Normal"], fontSize=10,
        textColor=colors.HexColor("#555555"), alignment=TA_CENTER, spaceAfter=14,
    )
    heading_style = ParagraphStyle(
        "ReportHeading", parent=styles["Heading2"], fontSize=13,
        textColor=colors.HexColor("#1B5E20"), leading=16,
        spaceBefore=14, spaceAfter=8,
    )
    body_style = ParagraphStyle(
        "ReportBody", parent=styles["BodyText"], fontSize=10.5,
        leading=15.5, spaceAfter=7, alignment=TA_LEFT,
    )
    bullet_style = ParagraphStyle(
        "ReportBullet", parent=body_style, leftIndent=12, spaceAfter=5,
    )
    disclaimer_style = ParagraphStyle(
        "Disclaimer", parent=styles["BodyText"], fontSize=9, leading=13,
        textColor=colors.HexColor("#7A5B00"),
    )

    story = [
        Spacer(1, 4),
        Paragraph(
            "Hybrid CNN + Vision Transformer classification with lesion and severity analysis",
            subtitle_style,
        ),
    ]

    sev_color = SEVERITY_COLORS.get(severity_tier.lower(), colors.HexColor("#2E7D32"))

    diagnosis_data = [
        ["Parameter", "Result"],
        ["Crop", escape(str(crop))],
        ["Detected condition", escape(str(disease))],
        ["Model confidence", f"{confidence:.1f}%"],
        ["Estimated affected leaf area", f"{severity_pct:.1f}%"],
        ["Severity", escape(str(severity_tier).capitalize())],
    ]

    table = Table(diagnosis_data, colWidths=[2.35 * inch, 3.55 * inch])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2E7D32")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTNAME", (0, 1), (0, -1), "Helvetica-Bold"),
        ("TEXTCOLOR", (1, 5), (1, 5), sev_color),
        ("FONTNAME", (1, 5), (1, 5), "Helvetica-Bold"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#DDDDDD")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F7FAF7")]),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 9),
        ("RIGHTPADDING", (0, 0), (-1, -1), 9),
        ("TOPPADDING", (0, 0), (-1, -1), 7),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
    ]))

    story.extend([
        Paragraph("Diagnosis Summary", heading_style),
        table,
        Spacer(1, 6),
        HRFlowable(width="100%", thickness=0.6, color=colors.HexColor("#E0E0E0")),
        Paragraph("AI-Generated Advisory Report", heading_style),
    ])

    story.extend(build_report_flowables(ai_report, body_style, bullet_style))

    story.extend([
        Spacer(1, 14),
        Table(
            [[Paragraph(
                "<b>Important:</b> This report is an AI-assisted estimate based on the uploaded "
                "leaf image and model outputs. It should not replace diagnosis by a qualified "
                "agricultural expert. Confirm the condition before making major treatment decisions.",
                disclaimer_style,
            )]],
            colWidths=[5.9 * inch],
            style=TableStyle([
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#FFF8E1")),
                ("BOX", (0, 0), (-1, -1), 0.6, colors.HexColor("#F0D98C")),
                ("LEFTPADDING", (0, 0), (-1, -1), 10),
                ("RIGHTPADDING", (0, 0), (-1, -1), 10),
                ("TOPPADDING", (0, 0), (-1, -1), 8),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
            ]),
        ),
    ])

    doc.build(story, onFirstPage=_pdf_header_footer, onLaterPages=_pdf_header_footer)
    buffer.seek(0)
    return buffer.getvalue()


# ============================================================
# SIDEBAR
# ============================================================
def status_row(ok, label_ok, label_bad, warn_only=False):
    if ok:
        dot_class, label = "status-dot-ok", label_ok
    elif warn_only:
        dot_class, label = "status-dot-warn", label_bad
    else:
        dot_class, label = "status-dot-error", label_bad
    return f'<div class="status-row"><span class="status-dot {dot_class}"></span>{label}</div>'


with st.sidebar:
    st.markdown('<div class="section-heading"><i class="bi bi-sliders"></i>Settings &amp; Status</div>', unsafe_allow_html=True)

    with st.container(border=True):
        st.markdown("**Model status**")
        st.markdown(status_row(hybrid_model is not None, "Hybrid CNN+ViT classifier loaded", "Classifier not loaded"), unsafe_allow_html=True)
        st.markdown(status_row(yolo_model is not None, "YOLO lesion detector loaded", "Lesion detector not loaded (color-based severity only)", warn_only=True), unsafe_allow_html=True)
        st.markdown(status_row(bool(GROQ_API_KEY), "Groq API key configured", "Groq API key not set", warn_only=True), unsafe_allow_html=True)
        st.caption(f"Device: `{device.type}`")

    if load_errors:
        with st.expander("Setup warnings"):
            for e in load_errors:
                st.write("- " + e)

    if not GROQ_API_KEY:
        with st.expander("How to enable AI reports"):
            st.caption("Locally, set an environment variable:")
            st.code('export GROQ_API_KEY="your_key_here"', language="bash")
            st.caption("On Streamlit Community Cloud, add it under App settings → Secrets:")
            st.code('GROQ_API_KEY = "your_key_here"', language="toml")
            st.caption("Restart or reboot the app after setting it.")

    st.divider()
    st.caption("Built with a Hybrid EfficientNetV2 + ViT classifier, a YOLO lesion detector, Grad-CAM++ explainability, and Groq-powered advisory reports.")


# ============================================================
# INPUT
# ============================================================
tab_upload, tab_camera = st.tabs(["Upload Image", "Camera Snapshot"])

input_image = None
with tab_upload:
    uploaded_file = st.file_uploader("Upload a leaf image", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        input_image = Image.open(uploaded_file)

with tab_camera:
    st.caption("Captures a single snapshot when you click the button below — not continuous live video.")
    camera_file = st.camera_input("Take a photo of the leaf")
    if camera_file is not None:
        input_image = Image.open(camera_file)

# ============================================================
# RUN PIPELINE
# ============================================================
if input_image is not None:
    left, right = st.columns([1, 1.1], gap="large")

    with left:
        st.image(input_image, caption="Input image", use_container_width=True)

    # --- Leaf-presence gate: check BEFORE running the classifier ---
    _img_bgr_check = cv2.cvtColor(np.array(input_image.convert("RGB")), cv2.COLOR_RGB2BGR)
    _leaf_ok, _veg_fraction, _reject_reason = is_leaf_present(_img_bgr_check)

    if not _leaf_ok:
        with right:
            st.error(
                "**No leaf detected in this image.**\n\n"
                f"{_reject_reason}\n\n"
                "Please upload or capture a clear, well-lit photo of a single plant leaf, "
                "ideally filling most of the frame against a plain background."
            )
        st.stop()

    def pipeline_step(icon, text):
        st.markdown(f'<div class="pipeline-step"><i class="bi bi-{icon}"></i>{text}</div>', unsafe_allow_html=True)

    with st.status("Running AI pipeline...", expanded=True) as status:
        pipeline_step("magic", "Preprocessing image (CLAHE contrast enhancement)...")
        tensor = preprocess_for_model(input_image)

        pipeline_step("cpu", "Running hybrid CNN + ViT classification...")
        with torch.no_grad():
            logits = hybrid_model(tensor)
            probs = torch.softmax(logits, dim=1)[0]
            top_idx = int(torch.argmax(probs).item())
            confidence = float(probs[top_idx].item()) * 100

        predicted_class = class_names[top_idx]
        parts = predicted_class.split("___")
        crop = parts[0].replace("_", " ")
        disease = parts[1].replace("_", " ") if len(parts) > 1 else predicted_class

        img_bgr = cv2.cvtColor(np.array(input_image.convert("RGB")), cv2.COLOR_RGB2BGR)

        boxes = []
        if yolo_model is not None:
            pipeline_step("search", "Detecting lesions with YOLO...")
            results = yolo_model.predict(source=img_bgr, conf=0.25, verbose=False)
            if len(results[0].boxes) > 0:
                boxes = results[0].boxes.xyxy.cpu().numpy()
        else:
            pipeline_step("skip-forward", "Skipping lesion detection (model not loaded)...")

        pipeline_step("rulers", "Segmenting leaf and estimating severity...")
        severity_ratio, severity_tier, lesion_mask = estimate_severity(img_bgr, boxes)

        pipeline_step("bullseye", "Generating Grad-CAM++ explainability map...")
        gradcam_overlay = get_gradcam_overlay(input_image, tensor, top_idx)

        status.update(label="Analysis complete", state="complete", expanded=False)

    is_healthy = "healthy" in disease.lower()

    if confidence < 40:
        st.warning(
            f"The model's confidence for this prediction is low ({confidence:.1f}%). "
            "This can happen with unclear photos, unusual angles, or leaves affected by "
            "a condition outside the 38 classes this model was trained on. Treat this "
            "result as tentative."
        )

    with right:
        st.markdown(f"""
        <div class="result-card">
            <div class="crop-tag">{crop}</div>
            <div class="disease-name">{disease}</div>
            <span class="confidence-badge">{confidence:.1f}% confidence</span>
        </div>
        """, unsafe_allow_html=True)

        m1, m2 = st.columns(2)
        m1.metric("Model confidence", f"{confidence:.1f}%")

        if not is_healthy:
            m2.metric("Affected leaf area", f"{severity_ratio*100:.1f}%")
            st.progress(min(severity_ratio, 1.0))
            st.markdown(
                f'<span class="severity-{severity_tier}-badge">Severity: {severity_tier.capitalize()}</span>',
                unsafe_allow_html=True,
            )
            st.caption(
                f"{len(boxes)} lesion(s) detected by YOLO" if yolo_model is not None
                else "Lesion detector not loaded — severity uses color-based estimation only."
            )
        else:
            m2.metric("Affected leaf area", "0%")
            st.markdown(
                '<div class="healthy-banner"><i class="bi bi-check-circle-fill icon-white"></i>No disease detected — leaf appears healthy.</div>',
                unsafe_allow_html=True,
            )

    st.divider()

    st.markdown('<div class="section-heading"><i class="bi bi-graph-up-arrow"></i>Visual Analysis</div>', unsafe_allow_html=True)
    view_choice = st.segmented_control(
        "Choose view", ["Grad-CAM++ (model attention)", "Detected lesion pixels"],
        default="Grad-CAM++ (model attention)", label_visibility="collapsed",
    ) if hasattr(st, "segmented_control") else "Grad-CAM++ (model attention)"

    vcol1, vcol2 = st.columns(2)
    with vcol1:
        st.image(gradcam_overlay, caption="Grad-CAM++ — where the model is looking", use_container_width=True)
    with vcol2:
        st.image(lesion_mask, caption="Detected lesion pixels", clamp=True, use_container_width=True)

    if not is_healthy:
        st.divider()
        st.markdown('<div class="section-heading"><i class="bi bi-file-earmark-text"></i>AI-Generated Advisory Report</div>', unsafe_allow_html=True)

        if GROQ_API_KEY:
            with st.spinner("Generating AI disease report with Groq..."):
                report_text, error = generate_groq_report(
                    crop, disease, confidence, severity_ratio * 100, severity_tier
                )

            if report_text:
                st.toast("AI report generated")
                with st.container(border=True):
                    st.markdown(
                        '<div style="display:flex; align-items:center; gap:0.5rem; font-weight:700; color:#1B5E20; margin-bottom:0.6rem;">'
                        '<i class="bi bi-robot"></i>Advisory summary</div>',
                        unsafe_allow_html=True,
                    )
                    st.markdown(f'<div class="ai-report-box">{report_text}</div>', unsafe_allow_html=True)

                try:
                    pdf_bytes = create_pdf_report(
                        crop=crop,
                        disease=disease,
                        confidence=confidence,
                        severity_pct=severity_ratio * 100,
                        severity_tier=severity_tier,
                        ai_report=report_text,
                    )

                    safe_crop = "".join(
                        c if c.isalnum() or c in (" ", "_", "-") else "_"
                        for c in crop
                    ).strip().replace(" ", "_")

                    dl_col, _ = st.columns([1, 2])
                    with dl_col:
                        st.download_button(
                            label="Download AI Disease Report (PDF)",
                            data=pdf_bytes,
                            file_name=f"{safe_crop}_disease_report.pdf",
                            mime="application/pdf",
                            use_container_width=True,
                        )
                except Exception as pdf_error:
                    st.error(f"Could not create PDF report: {pdf_error}")

            else:
                st.error(f"Could not generate AI report: {error}")
                st.caption(
                    "Check that your Groq API key is valid, your internet connection is available, "
                    "and the selected Groq model is currently active."
                )
        else:
            st.warning(
                "Set the GROQ_API_KEY environment variable to enable AI-generated disease reports and PDF downloads."
            )

else:
    st.info("Upload an image or take a photo to begin analysis.")