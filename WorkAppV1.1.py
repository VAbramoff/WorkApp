from tkinter import *
from tkinter import messagebox
from tkinter import filedialog

from ultralytics import YOLO
import cv2
import numpy as np
import pandas as pd
from qreader import QReader
import glob
import os
import time

qreader = QReader(model_size = 's', min_confidence = 0.5, reencode_to = 'shift-jis')

model_path = 'YOLO8n_QR_Code_train.pt'
model = YOLO(model_path)

def DetectingQR(path):
    tic = time.perf_counter()
    df = pd.DataFrame(columns=['Data', 'X', 'Y'])
    image_folder_path = path

    image_files = glob.glob(image_folder_path+ '/*.jpg')    

    results_folder = 'results'
    os.makedirs(results_folder, exist_ok=True)

    for image_file in image_files:
        data = np.fromfile(image_file, dtype=np.uint8)
        image = cv2.imdecode(data, cv2.IMREAD_COLOR)

        results = model.predict(image, conf = 0.4)

        for result in results:
            boxes = result.boxes
            for bbox in boxes.xyxy.tolist():
                x1, y1, x2, y2  = map(int, bbox)  # Преобразование координат в целочисленные значения
                x1 -= 40
                y1 -= 40
                x2 += 40
                y2 += 40    

                # Расширение координат рамки       
                cropped_image = image[y1:y2, x1:x2]
                if cropped_image is not None and cropped_image.size != 0:
                    decoded_text = qreader.detect_and_decode(image=cropped_image)                   
                    if decoded_text:
                        center_x = (x1 + x2) / 2
                        center_y = (y1 + y2) / 2
                        df = df.append({'Data': decoded_text, 'X': center_x, 'Y': center_y}, ignore_index=True)
            
                image_filename = image_file.split('\\')[-1]
                
                result_file_path = os.path.join(results_folder, image_filename.replace('.jpg', '.xlsx'))

        df.to_excel(result_file_path, index=False)
        df.drop(df.index, inplace=True)
    toc = time.perf_counter()    
    messagebox.showinfo(title= 'Уведомление', message= f'Выполнение программы завершено за {toc - tic:0.4f} сек.')


def directory():
    global filepath    
    filepath = filedialog.askdirectory(title="Проводник")

filepath = ''

window = Tk()
window.title('Считыватель QR-кодов с фото')
window.geometry('450x450')

label = Label(text="Считыватель QR-кодов", font=("Arial", 16))
label.pack(side='top', pady= 15)

button_path = Button(text='Выбрать путь', command=directory, width=15, height=2,)
button_path.pack(side='left', anchor='e', expand=True, padx= 15)


button_run = Button(text='Найти QR-коды', command=lambda: DetectingQR(filepath), width=15, height=2,)
button_run.pack(side='right', anchor='w', expand=True, padx= 15)

window.mainloop()