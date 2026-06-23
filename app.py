import logging

import cv2
import streamlit as st

from utils.app_support import (
    ImageValidationError,
    get_processing_result,
    image_to_bgr,
    load_uploaded_image,
    make_processing_signature,
    resolve_restoration_backend,
    store_processing_result,
)
from utils.ocr import (
    EASYOCR_BACKEND,
    PADDLEOCR_BACKEND,
    calculate_character_error_rate,
    calculate_word_error_rate,
    extract_text_with_details,
    get_backend_label,
)
from utils.restoration import (
    DOCRES_RESTORATION,
    restore_document_image,
)
from utils.text_processing import prepare_image_for_ocr


logger = logging.getLogger(__name__)

AUTO_RESTORATION_LABEL = "Tự động (DocRes, fallback OpenCV)"
OPENCV_ONLY_LABEL = "Chỉ OpenCV"
PREPROCESS_OPTIONS = {
    "Tự động theo engine": "auto",
    "Giữ ảnh màu/gốc": "preserve",
    "OpenCV threshold": "threshold",
}


def resize_for_processing(image_bgr, max_width=1024):
    height, width = image_bgr.shape[:2]
    if width <= max_width:
        return image_bgr

    scale = max_width / width
    resized_height = max(1, int(height * scale))
    return cv2.resize(
        image_bgr,
        (max_width, resized_height),
        interpolation=cv2.INTER_AREA,
    )


def render_ocr_text(title, result, widget_key):
    st.markdown(f"**{title}**")
    if result.ok and result.text:
        st.text_area(
            title,
            result.text,
            height=220,
            key=widget_key,
            label_visibility="collapsed",
        )
    elif not result.ok:
        st.error(f"OCR thất bại: {result.error}")
    else:
        st.info("Không phát hiện văn bản trong ảnh.")


st.set_page_config(
    page_title="Khử mờ và OCR tài liệu",
    page_icon="🪪",
    layout="wide",
)

st.title("🪪 Khử mờ và OCR tài liệu")
st.caption(
    "Tải ảnh lên, chọn mục tiêu xử lý và nhận ảnh cải thiện cùng kết quả OCR."
)

uploaded_file = st.file_uploader(
    "Chọn ảnh cần cải thiện",
    type=["jpg", "jpeg", "png"],
    help="Hỗ trợ JPG/PNG, tối đa 10 MB và 40 triệu pixel.",
)

if uploaded_file is None:
    st.info("Tải một ảnh lên để bắt đầu.")
    st.stop()

try:
    source_image = load_uploaded_image(uploaded_file)
    source_bgr = resize_for_processing(image_to_bgr(source_image))
    uploaded_bytes = uploaded_file.getvalue()
except ImageValidationError as exc:
    st.error(str(exc))
    st.stop()
except Exception:
    logger.exception("Unable to load uploaded image")
    st.error("Không thể đọc ảnh tải lên. Vui lòng thử lại với tệp khác.")
    st.stop()

st.image(
    cv2.cvtColor(source_bgr, cv2.COLOR_BGR2RGB),
    caption="Ảnh đầu vào",
    width="stretch",
)

mode_label = st.radio(
    "Mục tiêu xử lý",
    ["Tự nhiên", "Ưu tiên OCR"],
    horizontal=True,
    help="Tự nhiên giữ màu dễ nhìn; Ưu tiên OCR tăng tương phản chữ mạnh hơn.",
)
enhancement_mode = "natural" if mode_label == "Tự nhiên" else "ocr"

with st.expander("Cấu hình nâng cao", expanded=False):
    restoration_label = st.selectbox(
        "Khôi phục tài liệu",
        [AUTO_RESTORATION_LABEL, OPENCV_ONLY_LABEL],
        help="Mặc định thử DocRes trước và tự dùng OpenCV nếu DocRes lỗi.",
    )
    preprocess_label = st.selectbox(
        "Tiền xử lý OCR",
        list(PREPROCESS_OPTIONS),
    )
    use_deconvolution = st.checkbox(
        "Wiener deconvolution",
        value=False,
        help="Chỉ nên bật cho motion/defocus blur nhẹ và ảnh ít nhiễu.",
    )

restoration_choice = "auto" if restoration_label == AUTO_RESTORATION_LABEL else "opencv"
restoration_backend = resolve_restoration_backend(restoration_choice)
ocr_preprocess_profile = PREPROCESS_OPTIONS[preprocess_label]

processing_signature = make_processing_signature(
    uploaded_bytes,
    (
        enhancement_mode,
        restoration_choice,
        ocr_preprocess_profile,
        use_deconvolution,
    ),
)

if st.button("Cải thiện ảnh", type="primary", width="stretch"):
    processing_stage = "khôi phục ảnh"
    try:
        with st.spinner("Đang khử mờ và nhận dạng văn bản..."):
            restoration_result = restore_document_image(
                source_bgr,
                backend=restoration_backend,
                mode=enhancement_mode,
                use_deconvolution=use_deconvolution,
                docres_task="deblurring",
                fallback_to_opencv=True,
            )
            enhanced_bgr = restoration_result.image

            processing_stage = "trích xuất OCR"
            original_preprocessed = prepare_image_for_ocr(
                source_bgr,
                backend=PADDLEOCR_BACKEND,
                profile=ocr_preprocess_profile,
            )
            enhanced_preprocessed = prepare_image_for_ocr(
                enhanced_bgr,
                backend=PADDLEOCR_BACKEND,
                profile=ocr_preprocess_profile,
            )
            original_ocr_result = extract_text_with_details(
                original_preprocessed,
                backend=PADDLEOCR_BACKEND,
                fallback_backend=EASYOCR_BACKEND,
            )
            enhanced_ocr_result = extract_text_with_details(
                enhanced_preprocessed,
                backend=PADDLEOCR_BACKEND,
                fallback_backend=EASYOCR_BACKEND,
            )

        store_processing_result(
            st.session_state,
            processing_signature,
            {
                "source_bgr": source_bgr.copy(),
                "enhanced_bgr": enhanced_bgr,
                "restoration_result": restoration_result,
                "original_ocr_result": original_ocr_result,
                "enhanced_ocr_result": enhanced_ocr_result,
            },
        )
    except Exception:
        logger.exception("Processing failed during %s", processing_stage)
        st.error(f"Đã xảy ra lỗi khi {processing_stage}. Vui lòng thử lại.")

processing_result = get_processing_result(
    st.session_state,
    processing_signature,
)
if processing_result is None:
    st.stop()

restoration_result = processing_result["restoration_result"]
original_ocr_result = processing_result["original_ocr_result"]
enhanced_ocr_result = processing_result["enhanced_ocr_result"]

if restoration_result.used_fallback:
    st.warning("DocRes không chạy được; ứng dụng đã tự dùng OpenCV thay thế.")
elif restoration_result.backend == DOCRES_RESTORATION:
    st.success("Ảnh được khôi phục bằng DocRes.")
else:
    st.info("Ảnh được xử lý bằng OpenCV theo cấu hình đã chọn.")

st.subheader("Kết quả")
before_col, after_col = st.columns(2)
with before_col:
    st.image(
        cv2.cvtColor(processing_result["source_bgr"], cv2.COLOR_BGR2RGB),
        caption="Trước xử lý",
        width="stretch",
    )
with after_col:
    st.image(
        cv2.cvtColor(processing_result["enhanced_bgr"], cv2.COLOR_BGR2RGB),
        caption="Sau xử lý",
        width="stretch",
    )

actual_ocr_backend = get_backend_label(enhanced_ocr_result.backend)
st.caption(
    f"Khôi phục thực tế: {restoration_result.backend}; "
    f"OCR thực tế: {actual_ocr_backend}; tiền xử lý: {preprocess_label}."
)
if original_ocr_result.used_fallback or enhanced_ocr_result.used_fallback:
    st.warning("PaddleOCR không chạy được; ứng dụng đã fallback sang EasyOCR.")

st.subheader("Văn bản OCR")
original_col, enhanced_col = st.columns(2)
with original_col:
    render_ocr_text(
        "OCR từ ảnh đầu vào",
        original_ocr_result,
        f"original_ocr_{processing_signature}",
    )
with enhanced_col:
    render_ocr_text(
        "OCR từ ảnh sau xử lý",
        enhanced_ocr_result,
        f"enhanced_ocr_{processing_signature}",
    )

with st.expander("Độ chính xác OCR với ground truth", expanded=False):
    ground_truth = st.text_area(
        "Văn bản ground truth",
        height=120,
        key=f"ground_truth_{processing_signature}",
    )
    if ground_truth:
        if original_ocr_result.ok and enhanced_ocr_result.ok:
            original_cer = calculate_character_error_rate(
                ground_truth,
                original_ocr_result.text,
            )
            enhanced_cer = calculate_character_error_rate(
                ground_truth,
                enhanced_ocr_result.text,
            )
            original_wer = calculate_word_error_rate(
                ground_truth,
                original_ocr_result.text,
            )
            enhanced_wer = calculate_word_error_rate(
                ground_truth,
                enhanced_ocr_result.text,
            )

            metric_cols = st.columns(4)
            metric_cols[0].metric("CER đầu vào", f"{original_cer:.1f}%")
            metric_cols[1].metric("CER sau xử lý", f"{enhanced_cer:.1f}%")
            metric_cols[2].metric("WER đầu vào", f"{original_wer:.1f}%")
            metric_cols[3].metric("WER sau xử lý", f"{enhanced_wer:.1f}%")
            st.caption("CER/WER càng thấp càng tốt.")
        else:
            st.info("Cần OCR thành công trước khi tính CER/WER.")
