import streamlit as st
import numpy as np
from PIL import Image
import cv2
from utils.metrics import calculate_sharpness, calculate_psnr, calculate_ssim
from utils.enhancement import enhance_image
from utils.text_processing import preprocess_for_ocr
from utils.ocr import extract_text, compare_ocr_results, calculate_accuracy
from utils.detection import detect_and_extract_card
from utils.face import enhance_face_in_card
from utils.blur_generator import apply_gaussian_blur, apply_motion_blur, apply_defocus_blur

st.set_page_config(
    page_title="ID Card Deblur",
    page_icon="🪪",
    layout="wide"
)

# Sidebar for page navigation
page = st.sidebar.selectbox("Select Page", ["Deblur Image", "Generate Synthetic Blur"])

if page == "Generate Synthetic Blur":
    st.title("🔧 Synthetic Blur Generator")
    st.markdown("Upload a sharp image to generate blurred versions for testing")

    # File uploader for sharp image
    sharp_file = st.file_uploader(
        "Choose a sharp image",
        type=["jpg", "jpeg", "png"],
        help="Upload a sharp image to generate blurred versions",
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

            st.subheader("Original Sharp Image")
            st.image(sharp_image, caption="Sharp Image", use_container_width=True)

            # Blur type selection
            st.subheader("Generate Blurred Versions")

            blur_type = st.selectbox("Select blur type", ["Gaussian Blur", "Motion Blur", "Defocus Blur"])

            if blur_type == "Gaussian Blur":
                kernel_size = st.slider("Kernel Size", 3, 31, 15, step=2)
                if st.button("Generate Gaussian Blur"):
                    blurred = apply_gaussian_blur(sharp_bgr, kernel_size=kernel_size)
                    blurred_rgb = cv2.cvtColor(blurred, cv2.COLOR_BGR2RGB)
                    st.image(blurred_rgb, caption=f"Gaussian Blur (kernel={kernel_size})", use_container_width=True)

            elif blur_type == "Motion Blur":
                kernel_size = st.slider("Blur Length", 5, 31, 15, step=2)
                angle = st.slider("Angle (degrees)", 0, 180, 45)
                if st.button("Generate Motion Blur"):
                    blurred = apply_motion_blur(sharp_bgr, kernel_size=kernel_size, angle=angle)
                    blurred_rgb = cv2.cvtColor(blurred, cv2.COLOR_BGR2RGB)
                    st.image(blurred_rgb, caption=f"Motion Blur (length={kernel_size}, angle={angle}°)", use_container_width=True)

            elif blur_type == "Defocus Blur":
                radius = st.slider("Defocus Radius", 3, 20, 8)
                if st.button("Generate Defocus Blur"):
                    blurred = apply_defocus_blur(sharp_bgr, radius=radius)
                    blurred_rgb = cv2.cvtColor(blurred, cv2.COLOR_BGR2RGB)
                    st.image(blurred_rgb, caption=f"Defocus Blur (radius={radius})", use_container_width=True)

        except Exception as e:
            st.error(f"Error processing image: {str(e)}")

else:
    # Main deblur page
    st.title("🪪 ID Card Deblur Application")
    st.markdown("Upload a blurred ID card image to enhance and extract text")

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
            st.markdown("**Enhanced Text Differences (new/corrected highlighted)**")
            st.markdown("<div style='line-height:1.8;'>" + " ".join(rendered_tokens) + "</div>", unsafe_allow_html=True)
            st.caption("Green highlights indicate words that are new or corrected in the enhanced OCR output.")


    # File uploader
    uploaded_file = st.file_uploader(
        "Choose an ID card image",
        type=["jpg", "jpeg", "png"],
        help="Upload a blurred ID card image (JPG, PNG, or JPEG format)"
    )

    if uploaded_file is None:
        st.info("Upload a blurred ID card image to begin")
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
            st.error("Invalid image format. Please upload JPG, PNG, or JPEG files.")
            st.stop()

        # Check file size
        file_size_mb = uploaded_file.size / (1024 * 1024)
        if file_size_mb > 10:
            st.warning("Image file is large and may take longer to process")

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
                st.info(f"Image resized to {new_width}x{new_height} for optimal performance")

            st.success("Image uploaded successfully!")

            # Card Detection
            st.subheader("Card Detection")
            enable_detection = st.checkbox("Enable card detection and perspective correction", value=True)

            if enable_detection:
                try:
                    with st.spinner("Detecting card region..."):
                        card_image, detection_success, detection_message = detect_and_extract_card(image_bgr)

                    if detection_success:
                        st.success(detection_message)
                        # Convert to RGB for display
                        card_rgb = cv2.cvtColor(card_image, cv2.COLOR_BGR2RGB)
                        st.image(card_rgb, caption="Detected Card Region", use_container_width=True)
                    else:
                        st.warning(detection_message)
                        st.info("Processing full image (card not detected)")
                        card_image = image_bgr
                except Exception as e:
                    st.warning("Card detection failed, using original image")
                    st.info("Processing full image (card not detected)")
                    card_image = image_bgr
            else:
                card_image = image_bgr

            # Display uploaded image
            st.subheader("Uploaded Image")
            st.image(image, caption="Original Image", use_container_width=True)

            # Calculate and display sharpness metrics
            st.subheader("Quality Metrics")
            original_sharpness = calculate_sharpness(card_image)

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Original Sharpness", f"{original_sharpness:.2f}")

            # Optional reference image for PSNR/SSIM
            st.markdown("---")
            reference_file = st.file_uploader(
                "Optional: Upload reference (ground truth) image for PSNR/SSIM",
                type=["jpg", "jpeg", "png"],
                key="reference"
            )

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

                    st.info("Reference image loaded. PSNR/SSIM will be calculated after enhancement.")
                except Exception as e:
                    st.error("Unable to read reference image.")

            # Enhancement section
            st.markdown("---")
            st.subheader("Image Enhancement")

            use_deconv = st.checkbox("Enable Wiener deconvolution (for motion/defocus blur)", value=False)
            enable_face_enhancement = st.checkbox("Enable face region enhancement", value=True)

            if st.button("Enhance Image", type="primary"):
                with st.spinner("Enhancing image..."):
                    # Apply enhancement to card image
                    enhanced_bgr = enhance_image(card_image, use_deconvolution=use_deconv)

                    # Apply face enhancement if enabled
                    if enable_face_enhancement:
                        enhanced_bgr, face_detected, original_face, enhanced_face, face_message = enhance_face_in_card(enhanced_bgr)

                        if face_detected:
                            st.success(face_message)
                            # Store face regions for later display
                            st.session_state['original_face'] = original_face
                            st.session_state['enhanced_face'] = enhanced_face
                            st.session_state['face_detected'] = True
                        else:
                            st.info(face_message)
                            st.session_state['face_detected'] = False
                    else:
                        st.session_state['face_detected'] = False

                    # Convert back to RGB for display
                    enhanced_rgb = cv2.cvtColor(enhanced_bgr, cv2.COLOR_BGR2RGB)

                    # Display before/after comparison
                    st.subheader("Before/After Comparison")
                    col1, col2 = st.columns(2)

                    with col1:
                        st.image(image, caption="Original", use_container_width=True)

                    with col2:
                        st.image(enhanced_rgb, caption="Enhanced", use_container_width=True)

                    # Calculate enhanced sharpness
                    enhanced_sharpness = calculate_sharpness(enhanced_bgr)
                    improvement = ((enhanced_sharpness - original_sharpness) / original_sharpness) * 100

                    # Display metrics comparison
                    st.subheader("Quality Metrics Comparison")
                    col1, col2, col3 = st.columns(3)

                    with col1:
                        st.metric("Original Sharpness", f"{original_sharpness:.2f}")

                    with col2:
                        st.metric("Enhanced Sharpness", f"{enhanced_sharpness:.2f}")

                    with col3:
                        st.metric("Improvement", f"{improvement:.1f}%", delta=f"{improvement:.1f}%")

                    # Calculate PSNR/SSIM if reference provided
                    if reference_file is not None:
                        try:
                            # Resize enhanced to match reference if needed
                            if enhanced_bgr.shape != ref_bgr.shape:
                                enhanced_resized = cv2.resize(enhanced_bgr, (ref_bgr.shape[1], ref_bgr.shape[0]))
                            else:
                                enhanced_resized = enhanced_bgr

                            psnr_value = calculate_psnr(ref_bgr, enhanced_resized)
                            ssim_value = calculate_ssim(ref_bgr, enhanced_resized)

                            st.subheader("Reference-Based Metrics")
                            col1, col2 = st.columns(2)

                            with col1:
                                st.metric("PSNR", f"{psnr_value:.2f} dB")

                            with col2:
                                st.metric("SSIM", f"{ssim_value:.4f}")

                        except Exception as e:
                            st.warning(f"Could not calculate reference metrics: {str(e)}")

                    # Store enhanced image in session state for later use
                    st.session_state['enhanced_bgr'] = enhanced_bgr

                    # Display face comparison if face was detected
                    if st.session_state.get('face_detected', False):
                        st.markdown("---")
                        st.subheader("Face Region Comparison")

                        col1, col2 = st.columns(2)

                        with col1:
                            st.markdown("**Original Face**")
                            original_face_rgb = cv2.cvtColor(st.session_state['original_face'], cv2.COLOR_BGR2RGB)
                            st.image(original_face_rgb, caption="Original Face Region", use_container_width=True)

                        with col2:
                            st.markdown("**Enhanced Face**")
                            enhanced_face_rgb = cv2.cvtColor(st.session_state['enhanced_face'], cv2.COLOR_BGR2RGB)
                            st.image(enhanced_face_rgb, caption="Enhanced Face Region", use_container_width=True)

                    # OCR Extraction and Comparison
                    st.markdown("---")
                    st.subheader("OCR Text Extraction")

                    with st.spinner("Extracting text from images..."):
                        # Preprocess images for OCR
                        original_preprocessed = preprocess_for_ocr(card_image)
                        enhanced_preprocessed = preprocess_for_ocr(enhanced_bgr)

                        # Extract text
                        original_text = extract_text(original_preprocessed)
                        enhanced_text = extract_text(enhanced_preprocessed)

                    # Display OCR results side-by-side
                    col1, col2 = st.columns(2)

                    with col1:
                        st.markdown("**Original Text**")
                        if original_text and not original_text.startswith("Error"):
                            st.text_area("Extracted from original", original_text, height=200, key="original_ocr")
                        else:
                            st.info("No text detected in original image")

                    with col2:
                        st.markdown("**Enhanced Text**")
                        if enhanced_text and not enhanced_text.startswith("Error"):
                            st.text_area("Extracted from enhanced", enhanced_text, height=200, key="enhanced_ocr")
                        else:
                            st.info("No text detected in enhanced image")

                    # Compare OCR results
                    if original_text and enhanced_text and not original_text.startswith("Error") and not enhanced_text.startswith("Error"):
                        comparison = compare_ocr_results(original_text, enhanced_text)

                        render_enhanced_text_diff(original_text, enhanced_text)

                        st.subheader("OCR Comparison Statistics")
                        col1, col2, col3 = st.columns(3)

                        with col1:
                            st.metric("Original Characters", comparison['original_chars'])

                        with col2:
                            st.metric("Enhanced Characters", comparison['enhanced_chars'])

                        with col3:
                            st.metric("Character Improvement",
                                     f"+{comparison['char_improvement']}" if comparison['char_improvement'] >= 0 else str(comparison['char_improvement']),
                                     delta=comparison['char_improvement'])

                    # Optional ground truth for accuracy calculation
                    st.markdown("---")
                    ground_truth = st.text_area("Optional: Enter ground truth text for accuracy calculation", height=100)

                    if ground_truth:
                        original_accuracy = calculate_accuracy(ground_truth, original_text)
                        enhanced_accuracy = calculate_accuracy(ground_truth, enhanced_text)

                        st.subheader("OCR Accuracy (vs Ground Truth)")
                        col1, col2, col3 = st.columns(3)

                        with col1:
                            st.metric("Original Accuracy", f"{original_accuracy:.1f}%")

                        with col2:
                            st.metric("Enhanced Accuracy", f"{enhanced_accuracy:.1f}%")

                        with col3:
                            improvement = enhanced_accuracy - original_accuracy
                            st.metric("Accuracy Improvement", f"{improvement:+.1f}%", delta=f"{improvement:.1f}%")

        except Exception as e:
            st.error("Unable to read image file. Please upload a valid image.")
            st.stop()
