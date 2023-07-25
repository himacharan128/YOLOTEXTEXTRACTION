import torch
import cv2
import time
import re
import numpy as np
import easyocr

EASY_OCR = easyocr.Reader(['en'])
OCR_TH = 0.2

def detectx(frame, model):
    frame = [frame]
    print(f"[INFO] Detecting...")
    results = model(frame)
    labels, cordinates = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1]
    return labels, cordinates

def plot_boxes(results, frame, classes):
    labels, cord = results
    n = len(labels)
    x_shape, y_shape = frame.shape[1], frame.shape[0]
    print(f"[INFO] Total {n} detections...")
    print(f"[INFO] Looping through all detections...")
    plate_num_list = []
    for i in range(n):
        row = cord[i]
        if row[4] >= 0.55:
            print(f"[INFO] Extracting BBox coordinates...")
            x1, y1, x2, y2 = int(row[0] * x_shape), int(row[1] * y_shape), int(row[2] * x_shape), int(row[3] * y_shape)
            text_d = classes[int(labels[i])]
            coords = [x1, y1, x2, y2]
            extracted_text = recognize_plate_easyocr(img=frame, coords=coords, reader=EASY_OCR, region_threshold=OCR_TH)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(frame, (x1, y1 - 20), (x2, y1), (0, 255, 0), -1)
            cv2.putText(frame, f"{extracted_text}", (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            # Append the extracted text as a string, not a list
            plate_num_list.append(str(extracted_text))

    # Join the extracted text list with newline characters to get a single string
    plate_num = "\n".join(plate_num_list)

    return frame, plate_num


def recognize_plate_easyocr(img, coords, reader, region_threshold):
    xmin, ymin, xmax, ymax = coords
    nplate = img[int(ymin):int(ymax), int(xmin):int(xmax)]
    ocr_result = reader.readtext(nplate)
    text = filter_text(region=nplate, ocr_result=ocr_result, region_threshold=region_threshold)
    if len(text) == 1:
        text = text[0].upper()
    return text

def filter_text(region, ocr_result, region_threshold):
    rectangle_size = region.shape[0] * region.shape[1]
    plate = []
    print(ocr_result)
    for result in ocr_result:
        length = np.sum(np.subtract(result[0][1], result[0][0]))
        height = np.sum(np.subtract(result[0][2], result[0][1]))

        if length * height / rectangle_size > region_threshold:
            plate.append(result[1])
    return plate

def main(img_path=None):
    print(f"[INFO] Loading model...")
    model = torch.hub.load('./yolov5', 'custom', source='local', path='best.pt', force_reload=True)
    classes = model.names
    if img_path is not None:
        print(f"[INFO] Working with image: {img_path}")
        img_out_name = f"./output/result_{img_path.split('/')[-1]}"
        frame = cv2.imread(img_path)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = detectx(frame, model=model)
        frame, extracted_text = plot_boxes(results, frame, classes=classes)
        cv2.imwrite(img_out_name, frame)  # Save the processed image to a file
        print(f"[INFO] Image saved to: {img_out_name}")

        # Save extracted text to a TXT file
        text_file_name = f"./output/extracted_text_{img_path.split('/')[-1].split('.')[0]}.txt"
        with open(text_file_name, 'w') as txt_file:
            txt_file.write(extracted_text)
            print(f"[INFO] Extracted text saved to: {text_file_name}")

main(img_path="test_images/test_a.png")
