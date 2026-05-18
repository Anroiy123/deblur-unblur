import streamlit as st
import numpy as np
from PIL import Image
import cv2
from utils.metrics import calculate_sharpness, calculate_psnr, calculate_ssim
from utils.text_processing import prepare_image_for_ocr
from utils.ocr import (
    EASYOCR_BACKEND,
    PADDLEOCR_BACKEND,
    calculate_accuracy,
    calculate_character_error_rate,
    calculate_word_error_rate,
    compare_ocr_results,
    extract_text_with_details,
    get_backend_label,
    is_backend_available,
)
from utils.detection import detect_and_extract_card
from utils.face import enhance_face_in_card
from utils.blur_generator import apply_gaussian_blur, apply_motion_blur, apply_defocus_blur
from utils.restoration import (
    DOCRES_RESTORATION,
    OPENCV_RESTORATION,
    is_docres_configured,
    restore_document_image,
)

st.set_page_config(
    page_title="ID Card Deblur",
    page_icon="🪪",
    layout="wide"
)

# Sidebar for page navigation
page = st.sidebar.selectbox("Chọn trang", ["Khử mờ ảnh", "Tạo ảnh mờ thử nghiệm"])

if page == "Tạo ảnh mờ thử nghiệm":
    st.title("🔧 Trình tạo ảnh mờ thử nghiệm")
    st.markdown("Tải lên ảnh rõ nét để tạo các phiên bản bị làm mờ phục vụ kiểm thử")

    # File uploader for sharp image
    sharp_file = st.file_uploader(
        "Chọn ảnh rõ nét",
        type=["jpg", "jpeg", "png"],
        help="Tải lên ảnh rõ nét để tạo phiên bản bị làm mờ",
        key="blur_gen_upload"
    )

    if sharp_file is not None:
        try:
            # Load image
            sharp_image = Image.open(sharp_file)
            sharp_array = np.array(sharp_image)

            # Convert to BGR
            if len(sharp_array.shape) == 2:
                sharp_bgr = cv2.cvtColor(sharp_array, cv2.COLOR_GRAY2BGR)
            elif sharp_array.shape[2] == 4:
                sharp_bgr = cv2.cvtColor(sharp_array, cv2.COLOR_RGBA2BGR)
            else:
                sharp_bgr = cv2.cvtColor(sharp_array, cv2.COLOR_RGB2BGR)

            # Blur type selection
            st.subheader("Tạo phiên bản ảnh mờ")
            blur_type = st.selectbox("Loại làm mờ", ["Gaussian Blur", "Motion Blur", "Defocus Blur"])
            if blur_type == "Gaussian Blur":
                st.caption("Phù hợp: mô phỏng ảnh nhòe đều do rung nhẹ hoặc resize. Không phù hợp: ảnh có vệt kéo dài theo hướng cụ thể.")
            elif blur_type == "Motion Blur":
                st.caption("Phù hợp: mô phỏng rung tay hoặc chuyển động theo một hướng. Không phù hợp: ảnh chỉ bị mất nét quang học đều.")
            else:
                st.caption("Phù hợp: mô phỏng ảnh out-focus/mất nét do lấy nét sai. Không phù hợp: ảnh bị rung có hướng hoặc nhiễu nặng.")

            blurred_rgb = None
            blurred_caption = None

            if blur_type == "Gaussian Blur":
                kernel_size = st.slider("Kích thước kernel", 3, 31, 15, step=2)
                if st.button("Tạo Gaussian Blur"):
                    blurred = apply_gaussian_blur(sharp_bgr, kernel_size=kernel_size)
                    blurred_rgb = cv2.cvtColor(blurred, cv2.COLOR_BGR2RGB)
                    blurred_caption = f"Ảnh đã tạo (bị mờ): Gaussian Blur (kernel={kernel_size})"

            elif blur_type == "Motion Blur":
                kernel_size = st.slider("Độ dài vệt mờ", 5, 31, 15, step=2)
                angle = st.slider("Góc (độ)", 0, 180, 45)
                if st.button("Tạo Motion Blur"):
                    blurred = apply_motion_blur(sharp_bgr, kernel_size=kernel_size, angle=angle)
                    blurred_rgb = cv2.cvtColor(blurred, cv2.COLOR_BGR2RGB)
                    blurred_caption = f"Ảnh đã tạo (bị mờ): Motion Blur (độ dài={kernel_size}, góc={angle}°)"

            elif blur_type == "Defocus Blur":
                radius = st.slider("Bán kính mất nét", 3, 20, 8)
                if st.button("Tạo Defocus Blur"):
                    blurred = apply_defocus_blur(sharp_bgr, radius=radius)
                    blurred_rgb = cv2.cvtColor(blurred, cv2.COLOR_BGR2RGB)
                    blurred_caption = f"Ảnh đã tạo (bị mờ): Defocus Blur (bán kính={radius})"

            if blurred_rgb is not None:
                st.subheader("Kết quả")
                source_col, generated_col = st.columns(2)

                with source_col:
                    st.markdown("**Ảnh gốc (rõ nét)**")
                    st.image(sharp_image, caption="Ảnh gốc (rõ nét)", use_column_width=True)

                with generated_col:
                    st.markdown("**Ảnh đã tạo (bị mờ)**")
                    st.image(blurred_rgb, caption=blurred_caption, use_column_width=True)

        except Exception as e:
            st.error(f"Lỗi khi xử lý ảnh: {str(e)}")

else:
    # Main deblur page
    st.title("🪪 Ứng dụng khử mờ căn cước công dân")
    st.markdown("Tải lên ảnh căn cước bị mờ để cải thiện chất lượng và trích xuất văn bản")

    def translate_status_message(message):
        translations = {
            "Card detection failed, using original image": "Phát hiện căn cước thất bại, sử dụng ảnh gốc",
            "Card detected and extracted successfully": "Đã phát hiện và tách vùng căn cước thành công",
            "Face region not detected, skipping face enhancement": "Không phát hiện vùng khuôn mặt, bỏ qua bước cải thiện khuôn mặt",
            "Face detected and enhanced successfully": "Đã phát hiện và cải thiện khuôn mặt thành công",
        }
        return translations.get(message, message)

    def render_enhanced_text_diff(original_text, enhanced_text):
        """
        Render enhanced OCR text with inserted/corrected tokens highlighted.
        """
        from difflib import ndiff
        import html

        original_tokens = original_text.split()
        enhanced_tokens = enhanced_text.split()
        diff_tokens = ndiff(original_tokens, enhanced_tokens)

        rendered_tokens = []
        has_highlight = False

        for token in diff_tokens:
            if token.startswith("+ "):
                has_highlight = True
                content = html.escape(token[2:])
                rendered_tokens.append(
                    f"<mark style='background-color:#d4f8d4;padding:0.08em 0.2em;border-radius:0.2em;'>{content}</mark>"
                )
            elif token.startswith("  "):
                rendered_tokens.append(html.escape(token[2:]))

        if has_highlight:
            st.markdown("**Khác biệt văn bản sau cải thiện (từ mới/đã sửa được tô sáng)**")
            st.markdown("<div style='line-height:1.8;'>" + " ".join(rendered_tokens) + "</div>", unsafe_allow_html=True)
            st.caption("Các phần tô sáng màu xanh là những từ mới hoặc đã được sửa trong kết quả OCR sau cải thiện.")


    # File uploader
    uploaded_file = st.file_uploader(
        "Chọn ảnh căn cước",
        type=["jpg", "jpeg", "png"],
        help="Tải lên ảnh căn cước bị mờ (định dạng JPG, PNG hoặc JPEG)"
    )

    if uploaded_file is None:
        st.info("Tải lên ảnh căn cước bị mờ để bắt đầu")
    else:
        # Explicit format validation (before PIL open)
        allowed_extensions = {"jpg", "jpeg", "png"}
        allowed_mime_types = {"image/jpeg", "image/png"}
        file_name = (uploaded_file.name or "").lower()
        file_extension = file_name.rsplit(".", 1)[-1] if "." in file_name else ""
        file_type = (uploaded_file.type or "").lower()

        invalid_extension = file_extension not in allowed_extensions
        invalid_mime = bool(file_type) and file_type not in allowed_mime_types
        if invalid_extension or invalid_mime:
            st.error("Định dạng ảnh không hợp lệ. Vui lòng tải lên tệp JPG, PNG hoặc JPEG.")
            st.stop()

        # Check file size
        file_size_mb = uploaded_file.size / (1024 * 1024)
        if file_size_mb > 10:
            st.warning("Tệp ảnh khá lớn nên có thể mất nhiều thời gian xử lý hơn")

        # Validate and load image
        try:
            image = Image.open(uploaded_file)
            image_array = np.array(image)

            # Convert to BGR for OpenCV
            if len(image_array.shape) == 2:
                # Grayscale
                image_bgr = cv2.cvtColor(image_array, cv2.COLOR_GRAY2BGR)
            elif image_array.shape[2] == 4:
                # RGBA
                image_bgr = cv2.cvtColor(image_array, cv2.COLOR_RGBA2BGR)
            else:
                # RGB
                image_bgr = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)

            # Resize image if too large for performance
            max_width = 1024
            height, width = image_bgr.shape[:2]
            if width > max_width:
                scale = max_width / width
                new_width = max_width
                new_height = int(height * scale)
                image_bgr = cv2.resize(image_bgr, (new_width, new_height), interpolation=cv2.INTER_AREA)
                image = Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
                st.info(f"Ảnh đã được đổi kích thước thành {new_width}x{new_height} để tối ưu hiệu năng")

            st.success("Tải ảnh lên thành công!")

            # Card Detection
            st.subheader("Phát hiện vùng căn cước")
            enable_detection = st.checkbox("Bật phát hiện căn cước và hiệu chỉnh phối cảnh", value=True)
            st.caption("Phù hợp: ảnh còn thấy biên thẻ và nền có tương phản. Không phù hợp: ảnh crop quá sát, thẻ bị che hoặc nền quá rối.")
            detection_success = False
            detection_message = ""
            card_rgb = None

            if enable_detection:
                try:
                    with st.spinner("Đang phát hiện vùng căn cước..."):
                        card_image, detection_success, detection_message = detect_and_extract_card(image_bgr)

                    if detection_success:
                        st.success(translate_status_message(detection_message))
                        card_rgb = cv2.cvtColor(card_image, cv2.COLOR_BGR2RGB)
                    else:
                        st.warning(translate_status_message(detection_message))
                        st.info("Đang xử lý toàn bộ ảnh vì chưa phát hiện được vùng căn cước")
                        card_image = image_bgr
                except Exception:
                    st.warning("Phát hiện căn cước thất bại, sẽ dùng ảnh gốc")
                    st.info("Đang xử lý toàn bộ ảnh vì chưa phát hiện được vùng căn cước")
                    card_image = image_bgr
            else:
                card_image = image_bgr

            # Optional reference image for PSNR/SSIM
            st.markdown("---")
            reference_file = st.file_uploader(
                "Tùy chọn: tải ảnh tham chiếu (ground truth) để tính PSNR/SSIM",
                type=["jpg", "jpeg", "png"],
                key="reference"
            )
            ref_bgr = None
            ref_rgb = None

            if reference_file is not None:
                try:
                    ref_image = Image.open(reference_file)
                    ref_array = np.array(ref_image)

                    # Convert to BGR
                    if len(ref_array.shape) == 2:
                        ref_bgr = cv2.cvtColor(ref_array, cv2.COLOR_GRAY2BGR)
                    elif ref_array.shape[2] == 4:
                        ref_bgr = cv2.cvtColor(ref_array, cv2.COLOR_RGBA2BGR)
                    else:
                        ref_bgr = cv2.cvtColor(ref_array, cv2.COLOR_RGB2BGR)

                    ref_rgb = cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2RGB)
                    st.info("Đã tải ảnh tham chiếu. PSNR/SSIM sẽ được tính sau khi cải thiện ảnh.")
                except Exception:
                    st.error("Không thể đọc ảnh tham chiếu.")

            # Display role-based image previews
            st.subheader("Vai trò của ảnh")
            st.caption("Phù hợp: phân biệt ảnh đầu vào với ảnh chuẩn để so sánh. Không phù hợp: đánh giá PSNR/SSIM nếu chưa có ảnh tham chiếu rõ nét.")
            input_col, reference_col = st.columns(2)

            with input_col:
                st.markdown("**Ảnh đầu vào (bị mờ)**")
                st.image(image, caption="Ảnh đầu vào (bị mờ)", use_column_width=True)

            with reference_col:
                st.markdown("**Ảnh tham chiếu (rõ nét, tùy chọn)**")
                if ref_rgb is not None:
                    st.image(ref_rgb, caption="Ảnh tham chiếu (rõ nét)", use_column_width=True)
                else:
                    st.caption("Chưa tải ảnh tham chiếu.")

            # Calculate and display baseline sharpness
            st.subheader("Chỉ số chất lượng")
            original_sharpness = calculate_sharpness(card_image)
            st.metric("Độ nét ảnh đầu vào", f"{original_sharpness:.2f}")
            st.caption("Phù hợp: theo dõi độ sắc cạnh trước/sau xử lý. Không phù hợp: kết luận trực tiếp OCR đúng hơn hoặc ảnh nhìn tự nhiên hơn.")

            # Enhancement section
            st.markdown("---")
            st.subheader("Cải thiện ảnh")

            enhancement_mode_label = st.selectbox(
                "Chế độ xử lý",
                ["Tự nhiên (giữ màu, dễ xem)", "OCR (tăng chữ, mạnh hơn)"],
                help="Chọn 'Tự nhiên' để preview dễ nhìn hơn, hoặc 'OCR' để ưu tiên tăng độ rõ của chữ."
            )
            if enhancement_mode_label.startswith("Tự nhiên"):
                st.caption("Phù hợp: xem ảnh, giữ màu và chi tiết tổng thể. Không phù hợp: chữ quá mờ cần tăng tương phản mạnh cho OCR.")
            else:
                st.caption("Phù hợp: ưu tiên làm rõ chữ trước khi OCR. Không phù hợp: preview cần màu tự nhiên hoặc ảnh dễ nhìn.")
            enhancement_mode = "natural" if enhancement_mode_label.startswith("Tự nhiên") else "ocr"
            restoration_backend_label = st.selectbox(
                "Lớp khôi phục tài liệu",
                ["OpenCV baseline", "DocRes AI (nếu đã cấu hình)"],
                help="OpenCV chạy ngay. DocRes dùng biến môi trường DOCRES_COMMAND và sẽ fallback về OpenCV nếu chưa cấu hình."
            )
            if restoration_backend_label.startswith("OpenCV"):
                st.caption("Phù hợp: xử lý nhanh, không cần model AI, ảnh lỗi nhẹ. Không phù hợp: tài liệu bị bóng, cong, méo hoặc mờ phức tạp.")
            else:
                st.caption("Phù hợp: tài liệu bị mờ, bóng, méo phối cảnh hoặc lỗi tổng hợp. Không phù hợp: máy yếu/GPU yếu hoặc cần phản hồi rất nhanh.")
            restoration_backend = DOCRES_RESTORATION if restoration_backend_label.startswith("DocRes") else OPENCV_RESTORATION
            docres_task = "deblurring"
            if restoration_backend == DOCRES_RESTORATION:
                docres_task_label = st.selectbox(
                    "Tác vụ DocRes",
                    ["deblurring", "end2end"],
                    index=0,
                    key="docres_task_select_deblurring_default",
                )
                if docres_task_label == "end2end":
                    st.caption("Phù hợp: ảnh tài liệu có nhiều lỗi cùng lúc. Không phù hợp: chỉ cần khử mờ nhẹ và muốn giữ pipeline đơn giản.")
                else:
                    st.caption("Phù hợp: ảnh chủ yếu bị blur/mất nét. Không phù hợp: ảnh bị bóng, cong hoặc biến dạng phối cảnh rõ.")
                docres_task = docres_task_label
                if not is_docres_configured():
                    st.info("DocRes chưa cấu hình DOCRES_COMMAND; khi xử lý app sẽ tự fallback về OpenCV baseline.")

            default_ocr_index = 0 if is_backend_available(PADDLEOCR_BACKEND) else 1
            ocr_backend_label = st.selectbox(
                "Engine OCR",
                ["PaddleOCR (khuyến nghị)", "EasyOCR (baseline hiện tại)"],
                index=default_ocr_index,
                help="PaddleOCR là hướng chính theo recommendation. EasyOCR được giữ làm baseline/fallback."
            )
            if ocr_backend_label.startswith("PaddleOCR"):
                st.caption("Phù hợp: OCR chính cho tài liệu/căn cước, nhất là tiếng Việt. Không phù hợp: môi trường chưa cài PaddleOCR hoặc cần baseline nhẹ.")
            else:
                st.caption("Phù hợp: baseline so sánh nhanh và fallback đơn giản. Không phù hợp: pipeline OCR khuyến nghị cho tiếng Việt/tài liệu.")
            ocr_backend = PADDLEOCR_BACKEND if ocr_backend_label.startswith("PaddleOCR") else EASYOCR_BACKEND
            if ocr_backend == PADDLEOCR_BACKEND and not is_backend_available(PADDLEOCR_BACKEND):
                st.info("PaddleOCR chưa được cài; khi OCR app sẽ fallback sang EasyOCR nếu EasyOCR còn khả dụng.")

            ocr_preprocess_label = st.selectbox(
                "Tiền xử lý trước OCR",
                ["Tự động theo engine", "Giữ ảnh màu/gốc", "OpenCV threshold"],
                help="PaddleOCR thường nên dùng ảnh màu/gốc; EasyOCR baseline vẫn có thể dùng threshold OpenCV."
            )
            if ocr_preprocess_label == "Tự động theo engine":
                st.caption("Phù hợp: dùng mặc định an toàn theo engine OCR đang chọn. Không phù hợp: cần kiểm soát thủ công từng bước tiền xử lý.")
            elif ocr_preprocess_label == "Giữ ảnh màu/gốc":
                st.caption("Phù hợp: PaddleOCR và ảnh có màu/nền phức tạp. Không phù hợp: chữ đen trắng nhiễu mạnh cần threshold rõ.")
            else:
                st.caption("Phù hợp: EasyOCR hoặc ảnh chữ đen trắng cần tách nền. Không phù hợp: phôi thẻ màu, gradient hoặc ảnh cần giữ chi tiết màu.")
            ocr_preprocess_profile = {
                "Tự động theo engine": "auto",
                "Giữ ảnh màu/gốc": "preserve",
                "OpenCV threshold": "threshold",
            }[ocr_preprocess_label]

            use_deconv = st.checkbox("Bật Wiener deconvolution (cho motion/defocus blur)", value=False)
            st.caption("Phù hợp: motion/defocus blur nhẹ và ảnh ít nhiễu. Không phù hợp: ảnh nén JPEG xấu, nhiễu mạnh hoặc blur không đều.")
            enable_face_enhancement = st.checkbox("Bật cải thiện vùng khuôn mặt", value=True)
            st.caption("Phù hợp: cần nhìn vùng mặt rõ hơn sau khử mờ. Không phù hợp: chỉ quan tâm chữ/OCR hoặc ảnh không có vùng mặt.")

            if st.button("Cải thiện ảnh", type="primary"):
                with st.spinner("Đang cải thiện ảnh..."):
                    restoration_result = restore_document_image(
                        card_image,
                        backend=restoration_backend,
                        mode=enhancement_mode,
                        use_deconvolution=use_deconv,
                        docres_task=docres_task,
                    )
                    enhanced_bgr = restoration_result.image
                    if restoration_result.used_fallback:
                        st.warning(f"DocRes chưa chạy được ({restoration_result.error}). Đã dùng OpenCV baseline thay thế.")
                    elif restoration_result.backend == DOCRES_RESTORATION:
                        st.success("Đã khôi phục ảnh bằng DocRes.")

                    # Apply face enhancement if enabled
                    if enable_face_enhancement:
                        enhanced_bgr, face_detected, original_face, enhanced_face, face_message = enhance_face_in_card(enhanced_bgr)

                        if face_detected:
                            st.success(translate_status_message(face_message))
                            # Store face regions for later display
                            st.session_state['original_face'] = original_face
                            st.session_state['enhanced_face'] = enhanced_face
                            st.session_state['face_detected'] = True
                        else:
                            st.info(translate_status_message(face_message))
                            st.session_state['face_detected'] = False
                    else:
                        st.session_state['face_detected'] = False

                    # Convert back to RGB for display
                    enhanced_rgb = cv2.cvtColor(enhanced_bgr, cv2.COLOR_BGR2RGB)

                    # Display before/after comparison
                    st.subheader("So sánh trước/sau")
                    col1, col2 = st.columns(2)

                    with col1:
                        st.markdown("**Ảnh đầu vào (bị mờ)**")
                        st.image(image, caption="Ảnh đầu vào (bị mờ)", use_column_width=True)

                    with col2:
                        st.markdown("**Ảnh kết quả (đã khử mờ)**")
                        st.image(enhanced_rgb, caption="Ảnh kết quả (đã khử mờ)", use_column_width=True)

                    # Calculate enhanced sharpness
                    enhanced_sharpness = calculate_sharpness(enhanced_bgr)
                    improvement = ((enhanced_sharpness - original_sharpness) / original_sharpness) * 100

                    # Display metrics comparison
                    st.subheader("So sánh chỉ số chất lượng")
                    col1, col2, col3 = st.columns(3)

                    with col1:
                        st.metric("Độ nét ban đầu", f"{original_sharpness:.2f}")

                    with col2:
                        st.metric("Độ nét sau cải thiện", f"{enhanced_sharpness:.2f}")

                    with col3:
                        st.metric("Mức cải thiện", f"{improvement:.1f}%", delta=f"{improvement:.1f}%")

                    st.caption("Lưu ý: mức cải thiện này đo độ sắc cạnh, không phải thước đo trực tiếp cho độ đúng của chữ hoặc chất lượng thị giác.")

                    # Calculate PSNR/SSIM if reference provided
                    if ref_bgr is not None:
                        try:
                            # Resize enhanced to match reference if needed
                            if enhanced_bgr.shape != ref_bgr.shape:
                                enhanced_resized = cv2.resize(enhanced_bgr, (ref_bgr.shape[1], ref_bgr.shape[0]))
                            else:
                                enhanced_resized = enhanced_bgr

                            psnr_value = calculate_psnr(ref_bgr, enhanced_resized)
                            ssim_value = calculate_ssim(ref_bgr, enhanced_resized)

                            st.subheader("Chỉ số theo ảnh tham chiếu")
                            col1, col2 = st.columns(2)

                            with col1:
                                st.metric("PSNR", f"{psnr_value:.2f} dB")

                            with col2:
                                st.metric("SSIM", f"{ssim_value:.4f}")

                        except Exception as e:
                            st.warning(f"Không thể tính chỉ số theo ảnh tham chiếu: {str(e)}")

                    # Store enhanced image in session state for later use
                    st.session_state['enhanced_bgr'] = enhanced_bgr

                    details_tab, ocr_tab = st.tabs(["Chi tiết", "OCR"])

                    with details_tab:
                        if detection_success and card_rgb is not None:
                            with st.expander("Xem trước vùng căn cước đã phát hiện", expanded=False):
                                st.image(card_rgb, caption="Vùng căn cước đã phát hiện", use_column_width=True)

                        if st.session_state.get('face_detected', False):
                            with st.expander("So sánh vùng khuôn mặt", expanded=False):
                                col1, col2 = st.columns(2)

                                with col1:
                                    st.markdown("**Vùng khuôn mặt đầu vào**")
                                    original_face_rgb = cv2.cvtColor(st.session_state['original_face'], cv2.COLOR_BGR2RGB)
                                    st.image(original_face_rgb, caption="Vùng khuôn mặt đầu vào", use_column_width=True)

                                with col2:
                                    st.markdown("**Vùng khuôn mặt sau cải thiện**")
                                    enhanced_face_rgb = cv2.cvtColor(st.session_state['enhanced_face'], cv2.COLOR_BGR2RGB)
                                    st.image(enhanced_face_rgb, caption="Vùng khuôn mặt sau cải thiện", use_column_width=True)

                    with ocr_tab:
                        # OCR Extraction and Comparison
                        st.subheader("Trích xuất văn bản OCR")

                        with st.spinner("Đang trích xuất văn bản từ ảnh..."):
                            original_preprocessed = prepare_image_for_ocr(
                                card_image,
                                backend=ocr_backend,
                                profile=ocr_preprocess_profile,
                            )
                            enhanced_preprocessed = prepare_image_for_ocr(
                                enhanced_bgr,
                                backend=ocr_backend,
                                profile=ocr_preprocess_profile,
                            )
                            fallback_backend = EASYOCR_BACKEND if ocr_backend == PADDLEOCR_BACKEND else None
                            original_ocr_result = extract_text_with_details(
                                original_preprocessed,
                                backend=ocr_backend,
                                fallback_backend=fallback_backend,
                            )
                            enhanced_ocr_result = extract_text_with_details(
                                enhanced_preprocessed,
                                backend=ocr_backend,
                                fallback_backend=fallback_backend,
                            )
                            original_text = original_ocr_result.text
                            enhanced_text = enhanced_ocr_result.text

                        actual_backend = get_backend_label(enhanced_ocr_result.backend)
                        st.caption(f"Engine thực tế: {actual_backend}; tiền xử lý: {ocr_preprocess_label}.")
                        if original_ocr_result.used_fallback or enhanced_ocr_result.used_fallback:
                            st.warning("Engine OCR đã fallback sang EasyOCR vì PaddleOCR chưa chạy được trong môi trường hiện tại.")

                        # Display OCR results side-by-side
                        col1, col2 = st.columns(2)

                        with col1:
                            st.markdown("**Văn bản OCR từ ảnh đầu vào (bị mờ)**")
                            if original_ocr_result.ok and original_text:
                                st.text_area("Trích xuất từ ảnh đầu vào", original_text, height=200, key="original_ocr")
                            elif not original_ocr_result.ok:
                                st.info(f"Lỗi khi OCR ảnh đầu vào: {original_ocr_result.error}")
                            else:
                                st.info("Không phát hiện văn bản trong ảnh đầu vào")

                        with col2:
                            st.markdown("**Văn bản OCR từ ảnh kết quả (đã khử mờ)**")
                            if enhanced_ocr_result.ok and enhanced_text:
                                st.text_area("Trích xuất từ ảnh kết quả", enhanced_text, height=200, key="enhanced_ocr")
                            elif not enhanced_ocr_result.ok:
                                st.info(f"Lỗi khi OCR ảnh kết quả: {enhanced_ocr_result.error}")
                            else:
                                st.info("Không phát hiện văn bản trong ảnh kết quả")

                        # Compare OCR results
                        if original_ocr_result.ok and enhanced_ocr_result.ok and original_text and enhanced_text:
                            comparison = compare_ocr_results(original_text, enhanced_text)

                            render_enhanced_text_diff(original_text, enhanced_text)

                            st.subheader("Thống kê so sánh OCR")
                            col1, col2, col3 = st.columns(3)

                            with col1:
                                st.metric("Số ký tự ảnh đầu vào", comparison['original_chars'])

                            with col2:
                                st.metric("Số ký tự ảnh kết quả", comparison['enhanced_chars'])

                            with col3:
                                st.metric("Số ký tự cải thiện",
                                         f"+{comparison['char_improvement']}" if comparison['char_improvement'] >= 0 else str(comparison['char_improvement']),
                                         delta=comparison['char_improvement'])

                        with st.expander("Độ chính xác so với ground truth (tùy chọn)", expanded=False):
                            ground_truth = st.text_area("Nhập văn bản ground truth để tính độ chính xác", height=100)

                            if ground_truth and original_ocr_result.ok and enhanced_ocr_result.ok:
                                original_accuracy = calculate_accuracy(ground_truth, original_text)
                                enhanced_accuracy = calculate_accuracy(ground_truth, enhanced_text)
                                original_cer = calculate_character_error_rate(ground_truth, original_text)
                                enhanced_cer = calculate_character_error_rate(ground_truth, enhanced_text)
                                original_wer = calculate_word_error_rate(ground_truth, original_text)
                                enhanced_wer = calculate_word_error_rate(ground_truth, enhanced_text)

                                st.subheader("Độ chính xác OCR so với ground truth")
                                col1, col2, col3 = st.columns(3)

                                with col1:
                                    st.metric("Độ chính xác ảnh đầu vào", f"{original_accuracy:.1f}%")

                                with col2:
                                    st.metric("Độ chính xác ảnh kết quả", f"{enhanced_accuracy:.1f}%")

                                with col3:
                                    improvement = enhanced_accuracy - original_accuracy
                                    st.metric("Mức cải thiện độ chính xác", f"{improvement:+.1f}%", delta=f"{improvement:.1f}%")

                                st.caption("CER/WER càng thấp càng tốt; đây là thước đo sát mục tiêu đọc chữ hơn độ nét Laplacian.")
                                col1, col2, col3, col4 = st.columns(4)
                                with col1:
                                    st.metric("CER ảnh đầu vào", f"{original_cer:.1f}%")
                                with col2:
                                    st.metric("CER ảnh kết quả", f"{enhanced_cer:.1f}%")
                                with col3:
                                    st.metric("WER ảnh đầu vào", f"{original_wer:.1f}%")
                                with col4:
                                    st.metric("WER ảnh kết quả", f"{enhanced_wer:.1f}%")
                            elif ground_truth:
                                st.info("Cần OCR thành công trước khi tính độ chính xác, CER và WER.")

        except Exception as e:
            st.error("Không thể đọc tệp ảnh. Vui lòng tải lên ảnh hợp lệ.")
            st.stop()
