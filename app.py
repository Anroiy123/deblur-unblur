import logging

import cv2
import streamlit as st

from utils.app_support import (
    ImageValidationError,
    get_processing_result,
    image_to_bgr,
    load_uploaded_image,
    make_processing_signature,
    store_processing_result,
)
from utils.restoration import restore_document_image


logger = logging.getLogger(__name__)

AUTO_RESTORATION_LABEL = "Tự động tốt nhất (DocRes -> Restormer -> NAFNet -> OpenCV)"
AUTO_RESTORATION = "auto"
DOCRES_RESTORATION = "docres"
RESTORMER_RESTORATION = "restormer"
NAFNET_RESTORATION = "nafnet"
OPENCV_RESTORATION = "opencv"
RESTORATION_BACKEND_LABELS = {
    AUTO_RESTORATION: "Tự động tốt nhất",
    DOCRES_RESTORATION: "DocRes",
    RESTORMER_RESTORATION: "Restormer",
    NAFNET_RESTORATION: "NAFNet",
    OPENCV_RESTORATION: "OpenCV",
}
RESTORATION_OPTIONS = {
    AUTO_RESTORATION_LABEL: AUTO_RESTORATION,
    "DocRes": DOCRES_RESTORATION,
    "Restormer": RESTORMER_RESTORATION,
    "NAFNet": NAFNET_RESTORATION,
    "Chỉ OpenCV": OPENCV_RESTORATION,
}
def get_restoration_backend_label(backend):
    return RESTORATION_BACKEND_LABELS.get(str(backend).strip().lower(), str(backend))


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


st.set_page_config(
    page_title="Khử mờ tài liệu",
    page_icon="🪪",
    layout="wide",
)

st.title("🪪 Khử mờ tài liệu")
st.caption("Tải ảnh lên, chọn phương pháp khôi phục và nhận ảnh cải thiện.")

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

with st.expander("Cấu hình nâng cao", expanded=False):
    restoration_label = st.selectbox(
        "Khôi phục tài liệu",
        list(RESTORATION_OPTIONS),
        help=(
            "Mặc định thử DocRes, rồi Restormer, rồi NAFNet trước khi quay về OpenCV. "
            "Bạn cũng có thể ép chạy từng backend nếu đã cấu hình adapter tương ứng."
        ),
    )
    use_deconvolution = st.checkbox(
        "Wiener deconvolution",
        value=False,
        help="Chỉ nên bật cho motion/defocus blur nhẹ và ảnh ít nhiễu.",
    )

restoration_choice = RESTORATION_OPTIONS[restoration_label]
restoration_backend = restoration_choice

processing_signature = make_processing_signature(
    uploaded_bytes,
    (
        restoration_choice,
        use_deconvolution,
    ),
)

if st.button("Cải thiện ảnh", type="primary", width="stretch"):
    processing_stage = "khôi phục ảnh"
    try:
        with st.spinner("Đang khử mờ và khôi phục ảnh..."):
            restoration_result = restore_document_image(
                source_bgr,
                backend=restoration_backend,
                mode="natural",
                use_deconvolution=use_deconvolution,
                docres_task="deblurring",
                fallback_to_opencv=True,
            )
            enhanced_bgr = restoration_result.image

        store_processing_result(
            st.session_state,
            processing_signature,
            {
                "source_bgr": source_bgr.copy(),
                "enhanced_bgr": enhanced_bgr,
                "restoration_result": restoration_result,
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

if restoration_result.used_fallback:
    requested_backend_label = get_restoration_backend_label(
        restoration_result.requested_backend or restoration_backend
    )
    actual_backend_label = get_restoration_backend_label(restoration_result.backend)
    st.warning(
        f"{requested_backend_label} không chạy được trọn vẹn; ứng dụng đã tự dùng "
        f"{actual_backend_label} thay thế."
    )
elif restoration_result.backend == DOCRES_RESTORATION:
    st.success("Ảnh được khôi phục bằng DocRes.")
else:
    st.info(
        f"Ảnh được xử lý bằng {get_restoration_backend_label(restoration_result.backend)} "
        "theo cấu hình đã chọn."
    )

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

st.caption(
    f"Khôi phục thực tế: {get_restoration_backend_label(restoration_result.backend)}."
)
