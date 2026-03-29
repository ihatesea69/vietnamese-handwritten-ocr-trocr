"""
OCR Web - Streamlit
Phát hiện chữ: EasyOCR | PaddleOCR
Nhận dạng    : TrOCR fine-tuned (VNOnDB) + ToneSpatialGate + ToneAwareLoss
"""
import sys
import os
import json

sys.path.insert(0, os.path.dirname(__file__))

import io
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
import torch
import torch.nn as nn
from PIL import Image, ImageDraw, ImageFont
from transformers import (
    AutoTokenizer,
    GenerationConfig,
    TrOCRProcessor,
    VisionEncoderDecoderModel,
    ViTImageProcessor,
)

APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parents[1]


def resolve_model_dir() -> Path:
    candidates = []
    env_model_dir = os.environ.get("OCR_MODEL_DIR")
    if env_model_dir:
        candidates.append(Path(env_model_dir).expanduser())

    candidates.append(PROJECT_ROOT / "artifacts" / "models" / "best_model")
    candidates.append(APP_DIR / "models" / "best_model")

    for candidate in candidates:
        if candidate.exists():
            return candidate

    return candidates[0]


# ĐƯỜNG DẪN MODEL
MODEL_DIR = resolve_model_dir()

COLORS = {
    "EasyOCR"  : "#FF4B4B",
    "PaddleOCR": "#1E88E5",
}


# MODULE: ToneSpatialGate
# Kết hợp Channel Attention + Spatial Attention
# Dựa trên SE-Net (channel) và CBAM (spatial),
# điều chỉnh cho Transformer và dấu thanh tiếng Việt.
class ToneSpatialGate(nn.Module):
    def __init__(self, hidden_dim: int = 768):
        super().__init__()
        self.channel_gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 4, hidden_dim),
            nn.Sigmoid()
        )
        self.spatial_gate = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        ch = self.channel_gate(x)
        sp = self.spatial_gate(x)
        gate = ch * sp
        return self.norm(x + x * gate)


# TIỀN XỬ LÝ: Loại bỏ đường kẻ ô ly
def remove_grid_lines(image_np: np.ndarray) -> np.ndarray:
    """
    Loại đường kẻ ngang + dọc trên vở ô ly,
    giữ lại chữ viết tay.

    Quy trình:
    1. Chuyển grayscale
    2. Adaptive threshold -> ảnh nhị phân (chữ = đen, nền = trắng)
    3. Dùng morphological open với kernel ngang dài -> tìm đường ngang
    4. Dùng morphological open với kernel dọc dài -> tìm đường dọc
    5. Gộp đường ngang + dọc -> mask đường kẻ
    6. Dilate mask nhẹ (để phủ hết đường kẻ, kể cả nét đứt)
    7. Inpaint vùng đường kẻ -> "xóa" đường kẻ, giữ chữ
    8. Chuyển lại RGB

    Kết quả: ảnh gần giống nền trắng chữ đen (domain train)
    """
    # 1) Grayscale
    if len(image_np.shape) == 3:
        gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    else:
        gray = image_np.copy()

    h, w = gray.shape

    # 2) Adaptive threshold: chữ = 0 (đen), nền = 255 (trắng)
    #    THRESH_BINARY_INV: đảo → chữ + đường kẻ = trắng (255), nền = đen (0)
    binary = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        blockSize=15, C=10
    )

    # 3) Tìm đường ngang: kernel rộng, cao 1px
    #    Chiều dài kernel = 1/4 chiều rộng ảnh → chỉ bắt đường kẻ dài
    horizontal_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (max(w // 4, 30), 1)
    )
    horizontal_lines = cv2.morphologyEx(
        binary, cv2.MORPH_OPEN, horizontal_kernel, iterations=1
    )

    # 4) Tìm đường dọc: kernel cao, rộng 1px
    vertical_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (1, max(h // 4, 30))
    )
    vertical_lines = cv2.morphologyEx(
        binary, cv2.MORPH_OPEN, vertical_kernel, iterations=1
    )

    # 5) Gộp
    grid_mask = cv2.add(horizontal_lines, vertical_lines)

    # 6) Dilate nhẹ: phủ rộng thêm 2px quanh đường kẻ
    #    Giúp xóa cả đường kẻ nét đứt (các đoạn ngắn cách nhau)
    dilate_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    grid_mask = cv2.dilate(grid_mask, dilate_kernel, iterations=1)

    # 7) Inpaint: "vá" vùng đường kẻ bằng pixel xung quanh
    #    Ưu điểm so với đơn giản fill trắng: giữ lại nét chữ bị đè lên đường kẻ
    cleaned = cv2.inpaint(gray, grid_mask, inpaintRadius=2, flags=cv2.INPAINT_NS)

    # 8) Chuyển RGB
    cleaned_rgb = cv2.cvtColor(cleaned, cv2.COLOR_GRAY2RGB)

    return cleaned_rgb


def enhance_for_ocr(image_np: np.ndarray) -> np.ndarray:
    """
    Tăng cường ảnh cho OCR:
    - Tăng contrast (CLAHE)
    - Giảm nhiễu nhẹ
    Áp dụng sau khi loại đường kẻ (nếu có) hoặc dùng độc lập.
    """
    if len(image_np.shape) == 3:
        gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    else:
        gray = image_np.copy()

    # CLAHE: tăng contrast cục bộ
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    # Giảm nhiễu nhẹ (giữ nét chữ)
    enhanced = cv2.fastNlMeansDenoising(enhanced, h=10)

    return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)


# TIỆN ÍCH
def pil_to_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="PNG")
    return buf.getvalue()


def ensure_model_compat(model_dir: Path) -> None:
    legacy_processor_cfg = model_dir / "processor_config.json"
    preprocessor_cfg = model_dir / "preprocessor_config.json"

    # Older exports may only have processor_config.json, while newer
    # transformers look for preprocessor_config.json.
    if not legacy_processor_cfg.exists():
        return

    with legacy_processor_cfg.open("r", encoding="utf-8") as fh:
        processor_data = json.load(fh)

    image_processor_data = processor_data.get("image_processor", processor_data)

    if (
        not preprocessor_cfg.exists()
        or "image_processor" in processor_data
    ):
        with preprocessor_cfg.open("w", encoding="utf-8") as fh:
            json.dump(image_processor_data, fh, ensure_ascii=False, indent=2)


# LOAD MODEL
@st.cache_resource(show_spinner="Đang tải mô hình TrOCR…")
def load_trocr():
    if not MODEL_DIR.exists():
        st.error(
            f"Không tìm thấy thư mục model: {MODEL_DIR}\n"
            "Hãy đặt best_model vào artifacts/models/best_model/ "
            "hoặc cấu hình biến môi trường OCR_MODEL_DIR."
        )
        st.stop()

    ensure_model_compat(MODEL_DIR)
    image_processor = ViTImageProcessor.from_pretrained(str(MODEL_DIR))
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR))
    processor = TrOCRProcessor(image_processor=image_processor, tokenizer=tokenizer)
    model     = VisionEncoderDecoderModel.from_pretrained(str(MODEL_DIR))
    device    = "cuda" if torch.cuda.is_available() else "cpu"
    model     = model.to(device)

    # Load ToneSpatialGate
    gate_path = MODEL_DIR / "tone_spatial.pt"
    if gate_path.exists():
        tone_spatial = ToneSpatialGate(hidden_dim=768).to(device)
        tone_spatial.load_state_dict(
            torch.load(str(gate_path), map_location=device, weights_only=True)
        )
        tone_spatial.eval()

        def _encoder_hook(module, input, output):
            output.last_hidden_state = tone_spatial(output.last_hidden_state)
            return output

        model.encoder.register_forward_hook(_encoder_hook)
        print("ToneSpatialGate loaded!")
    else:
        # Fallback: thử load ToneAttentionGate cũ
        old_gate_path = MODEL_DIR / "tone_gate.pt"
        if old_gate_path.exists():
            st.warning(
                "Tìm thấy tone_gate.pt (phiên bản cũ). "
                "Hãy train lại với ToneSpatialGate để có kết quả tốt hơn."
            )
        else:
            st.warning("Không tìm thấy tone_spatial.pt - chạy không có module gate")

    model.eval()
    return processor, model, device


# LOAD DETECTOR
@st.cache_resource(show_spinner="Đang tải detector…")
def load_detector(
    name: str,
    det_db_thresh: float = 0.7,
    det_db_box_thresh: float = 0.6,
    det_db_unclip_ratio: float = 2,
):
    if name == "EasyOCR":
        import easyocr
        return easyocr.Reader(["vi", "en"], recognizer=False, verbose=False)
    elif name == "PaddleOCR":
        from paddleocr import PaddleOCR
        return PaddleOCR(
            use_angle_cls=False, lang="en",
            use_gpu=False, show_log=False,
            det_db_thresh=det_db_thresh,
            det_db_box_thresh=det_db_box_thresh,
            det_db_unclip_ratio=det_db_unclip_ratio,
        )
    raise ValueError(f"Detector không hợp lệ: {name}")


# PHÁT HIỆN VÙNG CHỮ
def detect_boxes(
    detector_obj, detector_name: str, image_np: np.ndarray,
    text_threshold: float = 0.7,
    low_text: float = 0.4,
    link_threshold: float = 0.4,
) -> list[tuple[int, int, int, int]]:
    boxes = []

    if detector_name == "EasyOCR":
        bounds = detector_obj.detect(
            image_np,
            text_threshold=text_threshold,
            low_text=low_text,
            link_threshold=link_threshold,
        )
        if bounds and bounds[0] and bounds[0][0]:
            for b in bounds[0][0]:
                x1, x2, y1, y2 = int(b[0]), int(b[1]), int(b[2]), int(b[3])
                boxes.append((min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)))

    elif detector_name == "PaddleOCR":
        result = detector_obj.ocr(image_np, det=True, rec=False, cls=False)
        if result and result[0]:
            for item in result[0]:
                pts = np.array(item, dtype=np.int32)
                x1, y1 = int(pts[:, 0].min()), int(pts[:, 1].min())
                x2, y2 = int(pts[:, 0].max()), int(pts[:, 1].max())
                boxes.append((x1, y1, x2, y2))

    # Sắp xếp: trên→dưới, trái→phải
    if boxes:
        if len(boxes) > 1:
            avg_h = np.mean([b[3] - b[1] for b in boxes])
            boxes = sorted(boxes, key=lambda b: (
                int(b[1] / (avg_h * 0.5)),
                b[0]
            ))
        else:
            boxes = sorted(boxes, key=lambda b: (b[1], b[0]))

    return boxes


# NHẬN DẠNG 1 CROP
def recognize_crop(
    crop     : Image.Image,
    processor: TrOCRProcessor,
    model    : VisionEncoderDecoderModel,
    device   : str,
    gen_cfg  : GenerationConfig,
) -> str:
    pv = processor(crop.convert("RGB"), return_tensors="pt").pixel_values.to(device)
    with torch.no_grad():
        ids = model.generate(pv, generation_config=gen_cfg)
    return processor.batch_decode(ids, skip_special_tokens=True)[0]


# VẼ BOUNDING BOX
def draw_boxes(
    image : Image.Image,
    boxes : list,
    labels: list,
    color : str,
) -> Image.Image:
    img  = image.copy().convert("RGB")
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("arial.ttf", max(12, image.height // 40))
    except Exception:
        font = ImageFont.load_default()

    for (x1, y1, x2, y2), label in zip(boxes, labels):
        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
        bbox_txt = draw.textbbox((x1, y1 - 2), label, font=font)
        draw.rectangle(bbox_txt, fill=color)
        draw.text((x1, y1 - 2), label, fill="white", font=font)

    return img


# TĂNG CƯỜNG CROP (áp dụng SAU detect, TRƯỚC TrOCR)
# Dùng HSV Saturation + Otsu để tách mực bút (saturation cao)
# khỏi đường kẻ nhạt và nền (saturation thấp).
# Kết quả: nền trắng chữ đen → khớp domain train VNOnDB.
#
# ĐỂ VÔ HIỆU HÓA: comment từ dòng này đến hết hàm enhance_crop_for_trocr
def enhance_crop_for_trocr(crop: Image.Image) -> Image.Image:
    arr = np.array(crop.convert("RGB"))
    hsv = cv2.cvtColor(arr, cv2.COLOR_RGB2HSV)
    s   = hsv[:, :, 1]   # saturation: bút (xanh/đen) cao, đường kẻ nhạt thấp
    # Otsu tự tìm ngưỡng tối ưu giữa mực và nền+đường kẻ
    _, mask = cv2.threshold(s, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Dilate nhẹ để điền đầy nét chữ bị thiếu
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    mask   = cv2.dilate(mask, kernel, iterations=1)
    # Đảo: chữ = đen, nền = trắng (khớp domain train)
    result = cv2.bitwise_not(mask)
    return Image.fromarray(cv2.cvtColor(result, cv2.COLOR_GRAY2RGB))
# KẾT THÚC KHỐI enhance_crop_for_trocr


# GIAO DIỆN STREAMLIT
def main():
    st.set_page_config(
        page_title="OCR Chữ Viết Tay Tiếng Việt",
        layout="wide",
    )

    st.title("Nhận Dạng Chữ Viết Tay Tiếng Việt")
    st.markdown(
        "Mô hình **TrOCR + ToneSpatialGate + ToneAwareLoss** "
        "fine-tuned trên **VNOnDB**. "
        "Chọn bộ phát hiện văn bản rồi tải ảnh lên."
    )

    # Sidebar 
    with st.sidebar:
        st.header("Cấu hình")

        detector_name = st.selectbox(
            "Bộ phát hiện văn bản",
            ["EasyOCR", "PaddleOCR"],
            help=(
                "EasyOCR – ổn định, hỗ trợ tiếng Việt tốt\n"
                "PaddleOCR – nhanh, nhẹ"
            ),
        )

        st.divider()

        remove_grid = False
        enhance_img = False

        st.subheader("Padding vùng crop")
        pad_x   = st.slider("Ngang (px)",               0,  40,  20)
        pad_top = st.slider("Dọc phía trên (% height)", 0, 100,  60)
        pad_bot = st.slider("Dọc phía dưới (% height)", 0, 100,  60)

        st.divider()

        # ĐỂ VÔ HIỆU HÓA toggle này: comment 3 dòng dưới (st.subheader + st.toggle)
        # và thêm dòng: enhance_crop = False
        st.subheader("Tăng cường crop")
        enhance_crop = st.toggle(
            "Bật tăng cường (bút xanh / vở ô ly)",
            value=False,
            help=(
                "Chuyển crop sang không gian HSV, lấy kênh Saturation.\n"
                "Otsu tự tìm ngưỡng tách mực bút (saturation cao)\n"
                "khỏi đường kẻ nhạt và nền (saturation thấp).\n"
                "Kết quả: nền trắng chữ đen → khớp domain train VNOnDB.\n"
                "Chỉ áp dụng trên từng crop SAU khi detect."
            ),
        )

        st.divider()

        st.subheader("Ngưỡng phát hiện")
        if detector_name == "EasyOCR":
            text_threshold = st.slider(
                "text_threshold", 0.1, 1.0, 0.7, 0.05,
                help="Ngưỡng tin cậy vùng chữ. Thấp hơn -> phát hiện nhiều hơn nhưng dễ nhầm."
            )
            low_text = st.slider(
                "low_text", 0.1, 1.0, 0.4, 0.05,
                help="Ngưỡng dưới của vùng chữ. Thấp hơn -> bắt được chữ mờ hơn."
            )
            link_threshold = st.slider(
                "link_threshold", 0.1, 1.0, 0.4, 0.05,
                help="Ngưỡng liên kết ký tự thành từ. Thấp hơn -> gộp ký tự rộng hơn."
            )
            det_db_thresh = det_db_box_thresh = det_db_unclip_ratio = None
        else:
            det_db_thresh = st.slider(
                "det_db_thresh", 0.1, 1.0, 0.7, 0.05,
                help="Ngưỡng nhị phân hóa bản đồ phát hiện. Thấp hơn -> bắt được chữ mờ hơn."
            )
            det_db_box_thresh = st.slider(
                "det_db_box_thresh", 0.1, 1.0, 0.6, 0.05,
                help="Ngưỡng lọc bounding box theo điểm tin cậy. Thấp hơn -> giữ nhiều box hơn."
            )
            det_db_unclip_ratio = st.slider(
                "det_db_unclip_ratio", 1.0, 3.0, 2.0, 0.1,
                help="Tỉ lệ mở rộng box phát hiện ra ngoài. Lớn hơn -> box bao rộng hơn."
            )
            text_threshold = low_text = link_threshold = None

        st.divider()

        num_beams = st.slider(
            "Beam search (num_beams)", 1, 8, 4,
            help="Càng lớn càng chính xác nhưng chậm hơn"
        )


    # Upload 
    uploaded = st.file_uploader(
        "Tải ảnh lên (jpg / jpeg / png / bmp)",
        type=["jpg", "jpeg", "png", "bmp"],
    )

    if uploaded is None:
        st.markdown(
            """
            <div style='text-align:center; padding:60px; color:#888;'>
                <h3>Tải ảnh lên để bắt đầu nhận dạng</h3>
                <p>Hỗ trợ: JPG · PNG · BMP</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    # Load models 
    processor, model, device = load_trocr()

    try:
        detector_obj = load_detector(
            detector_name,
            det_db_thresh=det_db_thresh or 0.7,
            det_db_box_thresh=det_db_box_thresh or 0.6,
            det_db_unclip_ratio=det_db_unclip_ratio or 2,
        )
    except Exception as e:
        st.error(
            f"Không thể tải detector **{detector_name}**.\n\n"
            f"Lỗi: `{e}`\n\n"
            "Hãy cài thư viện theo hướng dẫn trong sidebar."
        )
        return

    # Đọc ảnh 
    image_bytes = uploaded.read()
    image       = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image_np    = np.array(image)
    h_img, w_img = image_np.shape[:2]

    # Hiển thị ảnh gốc 
    st.image(pil_to_bytes(image), caption="Ảnh gốc", width="stretch")

    # Tiền xử lý 
    processed_np = image_np.copy()

    processed_image = Image.fromarray(processed_np)

    # Nút xử lý 
    if not st.button("Nhận dạng", type="primary", width="stretch"):
        return

    gen_cfg = GenerationConfig(
        max_new_tokens         = 32,
        num_beams              = num_beams,
        no_repeat_ngram_size   = 3,
        decoder_start_token_id = processor.tokenizer.cls_token_id,
        eos_token_id           = processor.tokenizer.sep_token_id,
        pad_token_id           = processor.tokenizer.pad_token_id,
    )

    # Phát hiện (trên ảnh đã tiền xử lý) 
    detect_np = np.array(processed_image)
    h_proc, w_proc = detect_np.shape[:2]

    with st.spinner(f"Đang phát hiện vùng chữ bằng {detector_name}…"):
        try:
            boxes = detect_boxes(
                detector_obj, detector_name, detect_np,
                text_threshold=text_threshold,
                low_text=low_text,
                link_threshold=link_threshold,
            )
        except Exception as e:
            st.error(f"Lỗi phát hiện: {e}")
            return

    if not boxes:
        st.warning("Không phát hiện được vùng chữ nào trong ảnh.")
        return

    st.success(f"Phát hiện **{len(boxes)}** vùng chữ bằng **{detector_name}**")

    # Nhận dạng từng vùng (crop từ ảnh đã xử lý)
    results  = []
    progress = st.progress(0, text="Đang nhận dạng…")

    for i, (x1, y1, x2, y2) in enumerate(boxes):
        h_box = y2 - y1

        cx1 = max(0,      x1 - pad_x)
        cx2 = min(w_proc, x2 + pad_x)
        cy1 = max(0,      y1 - int(h_box * pad_top / 100))
        cy2 = min(h_proc, y2 + int(h_box * pad_bot / 100))

        crop = processed_image.crop((cx1, cy1, cx2, cy2))
        # ĐỂ VÔ HIỆU HÓA tăng cường: comment 2 dòng dưới
        if enhance_crop:
            crop = enhance_crop_for_trocr(crop)
        text = recognize_crop(crop, processor, model, device, gen_cfg)

        results.append({
            "box"  : (cx1, cy1, cx2, cy2),
            "crop" : crop,
            "text" : text,
        })
        progress.progress(
            (i + 1) / len(boxes),
            text=f"Đã xử lý {i+1}/{len(boxes)}"
        )

    progress.empty()

    # Hiển thị ảnh kết quả 
    st.subheader("Kết quả phát hiện & nhận dạng")
    annotated = draw_boxes(
        processed_image,
        [r["box"]  for r in results],
        [r["text"] for r in results],
        COLORS.get(detector_name, "#FF4B4B"),
    )
    st.image(pil_to_bytes(annotated), caption="Ảnh kết quả", width="stretch")

    # Văn bản đầy đủ 
    full_text = " ".join(r["text"] for r in results)
    st.subheader("Văn bản nhận dạng được")
    st.text_area("Văn bản", full_text, height=120, label_visibility="collapsed")

    # Chi tiết từng vùng 
    st.subheader("Chi tiết từng vùng")
    cols_per_row = 4
    for row_start in range(0, len(results), cols_per_row):
        row_items = results[row_start: row_start + cols_per_row]
        cols      = st.columns(len(row_items))
        for col, r in zip(cols, row_items):
            with col:
                st.image(pil_to_bytes(r["crop"]), width="stretch")
                st.caption(f'**"{r["text"]}"**')

    # Bảng tổng hợp 
    with st.expander("Bảng tổng hợp kết quả"):
        import pandas as pd
        df = pd.DataFrame([
            {
                "STT"              : i + 1,
                "Vùng (x1,y1,x2,y2)": r["box"],
                "Văn bản nhận dạng": r["text"],
            }
            for i, r in enumerate(results)
        ])
        st.dataframe(df, width="stretch", hide_index=True)


if __name__ == "__main__":
    main()
