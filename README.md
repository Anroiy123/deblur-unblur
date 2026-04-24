# ID Card Deblur Application

A Computer Vision application for enhancing blurred ID card images using classical image processing techniques. This project demonstrates deblurring, text extraction improvement, and quality metrics calculation.

## Features

- **Image Enhancement**: Bilateral filtering, CLAHE contrast enhancement, unsharp masking, and optional Wiener deconvolution
- **Card Detection**: Automatic ID card region detection with perspective correction
- **Text Enhancement**: OCR-optimized preprocessing with adaptive thresholding and morphological operations
- **Face Enhancement**: Targeted face region detection and enhancement within ID cards
- **OCR Extraction**: Text extraction using EasyOCR with before/after comparison
- **Quality Metrics**: Sharpness calculation (Laplacian variance), optional PSNR/SSIM with reference images
- **Synthetic Blur Generator**: Tool for creating test cases with Gaussian, motion, and defocus blur

## Requirements

- Python 3.8+
- Dependencies listed in `requirements.txt`

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd deblur-unblur
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Running the Application

Start the Streamlit application:
```bash
streamlit run app.py
```

The application will open in your default web browser at `http://localhost:8501`.

### Main Features

#### 1. Deblur Image (Main Page)

1. Upload a blurred ID card image (JPG, PNG, or JPEG)
2. Enable/disable card detection and perspective correction
3. Optionally upload a reference image for PSNR/SSIM calculation
4. Configure enhancement options:
   - Enable Wiener deconvolution for motion/defocus blur
   - Enable face region enhancement
5. Click "Enhance Image" to process
6. View results:
   - Before/after image comparison
   - Quality metrics (sharpness improvement)
   - Face region comparison (if detected)
   - OCR text extraction comparison
7. Optionally enter ground truth text for accuracy calculation

#### 2. Generate Synthetic Blur (Testing Tool)

1. Select "Generate Synthetic Blur" from the sidebar
2. Upload a sharp image
3. Choose blur type:
   - **Gaussian Blur**: Simulates out-of-focus blur
   - **Motion Blur**: Simulates camera shake (adjustable angle)
   - **Defocus Blur**: Simulates lens defocus
4. Adjust parameters and generate blurred versions for testing

## Project Structure

```
deblur-unblur/
├── app.py                      # Main Streamlit application
├── requirements.txt            # Python dependencies
├── utils/
│   ├── enhancement.py          # Classical enhancement algorithms
│   ├── text_processing.py      # Text-specific preprocessing
│   ├── ocr.py                  # OCR extraction and comparison
│   ├── detection.py            # Card detection and perspective correction
│   ├── face.py                 # Face region enhancement
│   ├── metrics.py              # Quality metrics calculation
│   └── blur_generator.py       # Synthetic blur generation
└── README.md                   # This file
```

## Technical Details

### Enhancement Pipeline

1. **Denoising**: Bilateral filter for edge-preserving noise reduction
2. **Contrast Enhancement**: CLAHE (Contrast Limited Adaptive Histogram Equalization)
3. **Sharpening**: Unsharp masking technique
4. **Optional Deconvolution**: Wiener deconvolution for motion/defocus blur

### Text Processing Pipeline

1. Grayscale conversion
2. Text-specific CLAHE tuning
3. Adaptive thresholding (Gaussian/mean)
4. Morphological operations (opening/closing)

### Card Detection

1. Canny edge detection
2. Contour finding and quadrilateral detection
3. Perspective transformation (homography)
4. Fallback to original image if detection fails

### Face Enhancement

1. Haar Cascade face detection
2. Face region extraction
3. Targeted enhancement (CLAHE, sharpening, denoising)
4. Seamless blending back to card image

## Limitations

- Classical CV techniques may not handle severe blur as effectively as deep learning approaches
- OCR accuracy depends on text quality and preprocessing effectiveness
- Face detection may fail on low-quality or non-frontal face photos
- Card detection requires clear card edges and sufficient contrast

## Future Work

- Deep learning-based deblurring (e.g., DeblurGAN, MIMO-UNet)
- Custom OCR model training for ID card text
- Batch processing support
- Real-time video processing
- Mobile application deployment

## License

This project is for educational purposes (Computer Vision course final project).

## Acknowledgments

- OpenCV for computer vision algorithms
- EasyOCR for text extraction
- Streamlit for rapid UI development
- scikit-image for quality metrics
