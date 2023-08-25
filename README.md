Image Text Extraction with YOLOv5 and Tesseract

App Demo

This Flask application allows you to perform text extraction from images using YOLOv5 for object detection and Tesseract for OCR (Optical Character Recognition).
Features

    Object Detection: The application utilizes YOLOv5 to detect and extract regions of interest (ROI) from uploaded images.

    OCR with Tesseract: It then uses Tesseract OCR to extract text from these ROIs.

    CSV Output: Extracted text is organized into subfolders (e.g., 'head', 'main', 'text', 'range') and saved in a CSV file.

Prerequisites

Before running the application, make sure you have the following:

    YOLOv5: Pre-trained weights for YOLOv5 should be available (specified in the --weight argument).

    Tesseract: Ensure that Tesseract OCR is properly installed on your system.

Installation

    Clone this repository to your local machine:

    shell

git clone https://github.com/himacharan128/YOLOTEXTEXTRACTION

Install the required Python packages:

shell

    pip install -r requirements.txt

Usage

    Start the Flask application:

    shell

    python app.py

    Access the application by opening your web browser and navigating to http://localhost:5000.

    Upload an image for text extraction, and the application will perform object detection and text extraction.

    Download the extracted text in a CSV file.

Customization

    YOLOv5 Configuration: You can customize the YOLOv5 configuration (e.g., confidence threshold) by modifying the --conf argument in the run_text_extraction function.

    Tesseract Configuration: Customize the Tesseract OCR configuration in the custom_config variable.

Folder Structure

The project's folder structure is organized as follows:

scss

├── app.py
├── requirements.txt
├── static/
│   ├── css/
│   ├── images/
│   └── js/
├── templates/
│   └── index.html
├── upload/
├── yolov5/
│   └── ... (YOLOv5 files)
└── README.md

License

This project is licensed under the MIT License - see the LICENSE file for details.
Acknowledgments

    YOLOv5: For object detection.

    Tesseract OCR: For text extraction.
