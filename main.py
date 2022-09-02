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
from tqdm import tqdm
import re
import numpy as np

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
# df = pd.read_excel(r"C:\dev\matzevot\photos\Beshenkovichy.xls")
df = pd.read_excel(r"C:\dev\matzevot\photos\Ludza.xls")
real_text = dict(zip(df.iloc[:, 0], df.iloc[:, 1]))
# folder_path = r"C:\dev\matzevot\photos\white_good"
folder_path = r"C:\dev\matzevot\photos\Ludza\moshe"

data_dict = {  # photos dir : xls file path
    r"C:\dev\matzevot\photos\Beshenkovichy": r"C:\dev\matzevot\photos\Beshenkovichy.xls",
    # r"C:\dev\matzevot\photos\Ludza\Ludza" : r"C:\dev\matzevot\photos\Ludza.xls",
    # r"C:\dev\matzevot\photos\Starodub" : r"C:\dev\matzevot\photos\Starodub.xls",
}
# %%
file_name = "BSH0078"

real_out = real_text[file_name]
file_path = os.path.join(folder_path, file_name + ".jpg")

img = cv2.imread(file_path)  # This is BGR
img = seg.crop(img, fraction=3)
img = enh.fastNlMeansDenoisingColored(img, h=5)
img = thr.kmeans(img, n_clusters=4, invert=True)
ocr_out = pytesseract.image_to_string(img, lang="heb")

print(ocr_out)
score = editdistance.eval(ocr_out, real_out)/max([len(real_out), len(ocr_out)])
print(f"score is {score}")
# %%
file_list = os.listdir(folder_path)[:10]
# file_list = ["BSH0013.jpg"]#"BSH0078.jpg",
# file_list = ["BSH0192.jpg"]  # "BSH0078.jpg",
# The good example was BSH0078

# folder_path = r"C:\dev"
# file_list = ["test_ocr2.png"]
score_list = []
for file_name in tqdm(file_list):

    bare_file_name = re.split(', |_|\.|\+', file_name)[0]

    real_out = real_text[bare_file_name]
    file_path = os.path.join(folder_path, file_name)
    print(file_path)

    img = cv2.imread(file_path)  # This is BGR

    plt.imshow(img)
    img = seg.crop(img, fraction=3)
    # plt.imshow(img)
    img = enh.fastNlMeansDenoisingColored(img, h=5)
    # plt.imshow(img)

    # img = thr.kmeans(img, n_clusters=3, invert=False,chosen_cluster=0)
    img = thr.kmeans(img, n_clusters=5, invert=False)
    # plt.imshow(img)
    img = cv2.cvtColor(img.astype(np.uint8)*255, cv2.COLOR_GRAY2RGB)

    # img2 = imutils.rotate(img,angle = 45)

    ocr_out = pytesseract.image_to_string(img, lang="heb")

    plt.imshow(img)
    print(ocr_out)
    score = editdistance.eval(ocr_out, real_out) / \
        max([len(real_out), len(ocr_out)])
    score_list.append(score)

    print(f"score is {score}")
# %%
# 98 was the best example yet
# BSH0192 was also good
# BSH0323

angle_list = np.linspace(-0, 20, 41)
score_list = []
text_len = []

for angle in angle_list:
    img2 = imutils.rotate(img, angle=angle)
    ocr_out = pytesseract.image_to_string(img2, lang="heb")
    print(ocr_out)
    score_list.append(editdistance.eval(ocr_out, real_out) /
                      max([len(real_out), len(ocr_out)]))
    text_len.append(len(ocr_out))
print(score_list)
# %
text_len_n = np.array(text_len)/max(text_len)
plt.plot(angle_list, score_list)
plt.plot(angle_list, text_len_n)

ind_max = np.argmax(text_len)
center = sum(np.array(text_len) * angle_list) / sum(np.array(text_len))

plt.plot(angle_list[ind_max], score_list[ind_max], 'om')

plt.plot(center, np.interp(center, angle_list, score_list), 'or')
plt.grid()
print(f"center is {center}")
# %%
folder_path = r"C:\dev\matzevot\photos\Beshenkovichy"
file_name = "BSH0016.jpg"
bare_file_name = re.split(', |_|\.|\+', file_name)[0]

real_out = real_text[bare_file_name]
file_path = os.path.join(folder_path, file_name)

img = cv2.imread(file_path)  # This is BGR

pytesseract.image_to_string(img, lang="heb")
# %%
img2 = cv2.cvtColor(~img.astype(np.uint8)*255, cv2.COLOR_GRAY2RGB)
pytesseract.image_to_string(img2, lang="heb")
# %%
d = pytesseract.image_to_data(img, output_type=Output.DICT)
n_boxes = len(d['level'])
for i in range(n_boxes):
    (x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
cv2.imshow("img", img)
cv2.waitKey()
# %%
ocr_out = pytesseract.image_to_string(Image.open(file_path), lang="heb")

editdistance.eval(ocr_out, real_out)/max([len(real_out), len(ocr_out)])

# %%

# run on all folders

scores = []

for dir, file_path in data_dict.items():

    df = pd.read_excel(file_path)
    real_text = dict(zip(df.iloc[:, 0], df.iloc[:, 1]))
    file_list = os.listdir(dir)
    for file_name in tqdm(file_list):

        bare_file_name = re.split(', |_|\.|\+', file_name)[0]

        real_out = real_text[bare_file_name]
        file_path = os.path.join(dir, file_name)
        # print(file_path)

        img = cv2.imread(file_path)  # This is BGR

        # plt.imshow(img)
        # img = seg.crop(img, fraction=3)
        # plt.imshow(img)
        # img = enh.fastNlMeansDenoisingColored(img, h=5)
        # plt.imshow(img)

        # img = thr.kmeans(img, n_clusters=3, invert=False,chosen_cluster=0)
        # img = thr.kmeans(img, n_clusters=5, invert=False)
        # plt.imshow(img)
        # img = cv2.cvtColor(img.astype(np.uint8)*255, cv2.COLOR_GRAY2RGB)

        # img2 = imutils.rotate(img,angle = 45)

        ocr_out = pytesseract.image_to_string(img, lang="heb")

        # plt.imshow(img)
        print(ocr_out)
        score = editdistance.eval(ocr_out, real_out) / \
            max([len(real_out), len(ocr_out)])
        scores.append(score)
