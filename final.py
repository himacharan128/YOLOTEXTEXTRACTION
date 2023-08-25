from flask import Flask, render_template, request, send_file
from flask_ngrok import run_with_ngrok
import os
import cv2
import pytesseract
from PIL import Image
import keras_ocr
import glob
import csv
import io

app = Flask(__name__, template_folder='/drive/templates', static_url_path="", static_folder="static")
run_with_ngrok(app)

UPLOAD_FOLDER = '/content/upload'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


custom_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ.,$:'

pipeline = keras_ocr.pipeline.Pipeline()


main_folder_path = '/content/crop_img/exp3/crops'


subfolder_names = ['head', 'main', 'text', 'range']

def process_image(image):
    prediction_groups = pipeline.recognize([image])
    predicted_image = prediction_groups[0]

    extracted_text = ""
    for text, box in predicted_image:
        x_min, y_min = map(int, box[0])
        x_max, y_max = map(int, box[2])
        text_region = image[y_min:y_max, x_min:x_max]
        pil_image = Image.fromarray(text_region)
        tesseract_text = pytesseract.image_to_string(pil_image, config=custom_config)
        extracted_text += tesseract_text.strip() + " "

    return extracted_text

def run_text_extraction(file_name):
    command = f"python /content/yolov5/detect.py --save-crop --source {file_name} --img 416 --conf 0.4 --weight /content/yolov5/runs/train/exp/weights/best.pt --project /content/crop_img"
    os.system(command)

    extracted_texts_dictionary = {}
    for subfolder_name in subfolder_names:
        subfolder_path = os.path.join(main_folder_path, subfolder_name)
        image_files = glob.glob(os.path.join(subfolder_path, 'uploaded_image*.jpg'))

        if len(image_files) == 0:
            print(f"No images found in {subfolder_name} subfolder.")
        else:
            subfolder_text = []
            for image_file in image_files:
                image = cv2.imread(image_file)
                extracted_text = process_image(image)
                subfolder_text.append(extracted_text)
            extracted_texts_dictionary[subfolder_name] = subfolder_text

    for subfolder_name in subfolder_names:
        print(f"Subfolder: {subfolder_name}")
        for idx, text in enumerate(extracted_texts_dictionary[subfolder_name], start=1):
            print(f"Image {idx}: {text}")
        print()

    print(extracted_texts_dictionary)

    csv_file = 'extracted_texts.csv'
    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(subfolder_names)
        max_len = max(len(texts) for texts in extracted_texts_dictionary.values())
        for i in range(max_len):
            row = [extracted_texts_dictionary[subfolder_name][i] if i < len(extracted_texts_dictionary[subfolder_name]) else '' for subfolder_name in subfolder_names]
            writer.writerow(row)
    return open('extracted_texts.csv', mode='r')

@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":

        uploaded_file = request.files['myfile']
        if uploaded_file.filename != '':
            os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
            image_filename = 'uploaded_image.jpg'
            image_path = os.path.join(app.config['UPLOAD_FOLDER'], image_filename)
            uploaded_file.save(image_path)


            run_text_extraction(image_path)


            csv_content = run_text_extraction(image_path)


            temp_csv_path = '/content/temp.csv'
            with open(temp_csv_path, mode='w', newline='') as file:
                file.write(csv_content.read())

            return send_file(temp_csv_path, as_attachment=True, download_name='extracted_texts.csv')

    return render_template('image_upload.html')

if __name__ == "__main__":
    app.run()