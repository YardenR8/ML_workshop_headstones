# %%
from pytesseract import Output
import imutils
from PIL import Image
import pytesseract
import pandas as pd
import editdistance
import os
import cv2
import image_ops.enhancement as enh
import image_ops.segmentation as seg
import image_ops.thresholding as thr
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from tqdm import tqdm
import re
import numpy as np

# put tesseract path
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# put folder path and xls file path
data_dict = {  # photos dir : xls file path
    r"C:\dev\matzevot\photos\Beshenkovichy": r"C:\dev\matzevot\photos\Beshenkovichy.xls",
    # r"C:\dev\matzevot\photos\Ludza\Ludza" : r"C:\dev\matzevot\photos\Ludza.xls",
    # r"C:\dev\matzevot\photos\Starodub" : r"C:\dev\matzevot\photos\Starodub.xls",
}

# %%

# test Tesseract on an image 

custom_config = r'-c tessedit_char_blacklist=1234567890~!₪@#$%^&*()_-+=[]{}/\\|.<>?;:' \
                    r' --psm 12'
file_path = r"C:\dev\no_angle.jpg"
img = cv2.imread(file_path)
h, w, c = img.shape

ocr_out = pytesseract.image_to_string(img, lang="heb", config=custom_config)
boxes = pytesseract.image_to_boxes(img, lang="heb", config=custom_config)
for b in boxes.splitlines():
    b = b.split(' ')
    img = cv2.rectangle(img, (int(b[1]), h - int(b[2])), (int(b[3]), h - int(b[4])), (0, 255, 0), 1)
img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
plt.imshow(img)
print(ocr_out)




#%%
# Test on a few specific images

custom_config = r'-c tessedit_char_blacklist=1234567890~!@#$%^&*₪()_-+=[]{}/\\|,.<>?;:' \
                    r' --psm 12'

for dir, file_path in data_dict.items():

    df = pd.read_excel(file_path)
    real_text = dict(zip(df.iloc[:, 0], df.iloc[:, 1]))
    file_list = ['BSH0059.JPG']
    for file_name in tqdm(file_list):

        bare_file_name = re.split(', |_|\.|\+', file_name)[0]

        if bare_file_name in real_text:

            real_out = real_text[bare_file_name]
            file_path = os.path.join(dir, file_name)

            img = cv2.imread(file_path)  # This is BGR

            # Segmentation:

            img = seg.crop(img, fraction=3)

            # Enhancement:

            img = enh.fastNlMeansDenoisingColored(img, h=5)
            # img = enh.median(img,size=3)

            img = enh.CLAHE(img)

            # img = enh.bilateral(img)
            # img = enh.median(img,size=3)
            img = enh.fastNlMeansDenoisingColored(img, h=30)


            # Thresholding:
            
            # img = thr.thresholding(img)

            # k means thresholding:

            light, dark, all = thr.kmeans(img, n_clusters=4)

            lengths = []
            for new_img in [light, dark]:
                new_img = cv2.cvtColor(new_img.astype(np.uint8)*255, cv2.COLOR_GRAY2RGB)
                ocr_out = pytesseract.image_to_string(new_img, lang="heb", config=custom_config)
                # ocr_out = pytesseract.image_to_string(new_img, lang="heb")
                lengths.append(len(ocr_out))
            img = light if lengths[0]>=lengths[1] else dark
            img = cv2.cvtColor(img.astype(np.uint8)*255, cv2.COLOR_GRAY2RGB)

            # Run OCR:

            ocr_out = pytesseract.image_to_string(img, lang="heb", config=custom_config)
            # ocr_out = pytesseract.image_to_string(img, lang="heb")

            # Calculate score:

            score = editdistance.eval(ocr_out, real_out) / \
                max([len(real_out), len(ocr_out)])

            print(bare_file_name)

            h, w, c = img.shape  # assumes color image
            boxes = pytesseract.image_to_boxes(img, lang="heb", config=custom_config)
            # boxes = pytesseract.image_to_boxes(img)
            for b in boxes.splitlines():
                b = b.split(' ')
                img = cv2.rectangle(img, (int(b[1]), h - int(b[2])), (int(b[3]), h - int(b[4])), (0, 255, 0), 2)
            
            img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
            
            plt.imshow(img)
            print(real_out)
            print(ocr_out)
            print(score)

#%%

# Run over all dataset

scores = []


custom_config = r'-c tessedit_char_blacklist=1234567890~!₪@#$%^&*()_-+=[]{}/\\|.<>?;:' \
                    r' --psm 12'


for dir, file_path in data_dict.items():

    df = pd.read_excel(file_path)
    real_text = dict(zip(df.iloc[:, 0], df.iloc[:, 1]))
    file_list = os.listdir(dir)

    for file_name in tqdm(file_list):

        bare_file_name = re.split(', |_|\.|\+', file_name)[0]

        if bare_file_name in real_text:

            real_out = real_text[bare_file_name]
            file_path = os.path.join(dir, file_name)


            img = cv2.imread(file_path)  # This is BGR

            # Segmentation:

            img = seg.crop(img, fraction=3)

            # Enhancement:

            img = enh.fastNlMeansDenoisingColored(img, h=5)
            # img = enh.median(img,size=3)

            img = enh.CLAHE(img)

            # img = enh.bilateral(img)
            # img = enh.median(img,size=3)
            img = enh.fastNlMeansDenoisingColored(img, h=30)


            # Thresholding:
            
            # img = thr.thresholding(img)

            # k means thresholding:

            light, dark, all = thr.kmeans(img, n_clusters=4)

            lengths = []
            for new_img in [light, dark]:
                new_img = cv2.cvtColor(new_img.astype(np.uint8)*255, cv2.COLOR_GRAY2RGB)
                ocr_out = pytesseract.image_to_string(new_img, lang="heb", config=custom_config)
                # ocr_out = pytesseract.image_to_string(new_img, lang="heb")
                lengths.append(len(ocr_out))
            img = light if lengths[0]>=lengths[1] else dark
            img = cv2.cvtColor(img.astype(np.uint8)*255, cv2.COLOR_GRAY2RGB)

            # Run OCR:

            ocr_out = pytesseract.image_to_string(img, lang="heb", config=custom_config)
            # ocr_out = pytesseract.image_to_string(img, lang="heb")

            # Calculate score:

            score = editdistance.eval(ocr_out, real_out) / \
                max([len(real_out), len(ocr_out)])
            scores.append(score)

print(f' 5 precentile is {np.percentile(scores,5)}')
print(f' mean is {np.mean(scores)}')
