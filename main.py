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

# this is for baseline result regular config

scores = []
identified = []

for dir, file_path in data_dict.items():

    df = pd.read_excel(file_path)
    real_text = dict(zip(df.iloc[:, 0], df.iloc[:, 1]))
    file_list = os.listdir(dir)
    for file_name in tqdm(file_list):

        bare_file_name = re.split(', |_|\.|\+', file_name)[0]

        if bare_file_name in real_text:
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
            # print(ocr_out)
            score = editdistance.eval(ocr_out, real_out) / \
                max([len(real_out), len(ocr_out)])
            scores.append(score)
            if score<0.9:
                identified.append(bare_file_name)
                print(bare_file_name)
                plt.imshow(img)
                print(real_out)
                print(ocr_out)


print(f'identified text in {len(identified)} out of {len(scores)}')
print(f' 5 precentile is {np.percentile(scores,5)}')

#%%
import pandas as pd

# Generate data on commute times.
size, scale = 1000, 10
not1 = [x for x in scores if x<1]
commutes = pd.Series(not1)

commutes.plot.hist(grid=True, bins=20, rwidth=0.9,
                   color='#607c8e')
plt.title('Commute Times for 1,000 Commuters')
plt.xlabel('Counts')
plt.ylabel('Commute Time')
plt.grid(axis='y', alpha=0.75)


# %%

# this is for baseline result costum config

scores = []
identified = []
#-l Hebrew 
custom_config = r'-c tessedit_char_blacklist=1234567890ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz' \
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

            ocr_out = pytesseract.image_to_string(img, config=custom_config)

            # plt.imshow(img)
            # print(ocr_out)
            score = editdistance.eval(ocr_out, real_out) / \
                max([len(real_out), len(ocr_out)])
            scores.append(score)
            if score<0.9:
                identified.append(bare_file_name)
                # print(bare_file_name)
                # plt.imshow(img)
                # print(real_out)
                # print(ocr_out)

print(f'identified text in {len(identified)} out of {len(scores)}')
print(f' 5 precentile is {np.percentile(scores,5)}')

# %%

# run on all folders

# this is for baseline result regular config

scores = []
identified = []

for dir, file_path in data_dict.items():

    df = pd.read_excel(file_path)
    real_text = dict(zip(df.iloc[:, 0], df.iloc[:, 1]))
    file_list = os.listdir(dir)
    for file_name in tqdm(file_list):

        bare_file_name = re.split(', |_|\.|\+', file_name)[0]

        if bare_file_name in real_text:
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
            # print(ocr_out)
            score = editdistance.eval(ocr_out, real_out) / \
                max([len(real_out), len(ocr_out)])
            scores.append(score)
            if score<0.9:
                identified.append(bare_file_name)
                print(bare_file_name)
                print(score)
                plt.imshow(img)
                print(real_out)
                print(ocr_out)

print(f'identified text in {len(identified)} out of {len(scores)}')
print(f' 5 precentile is {np.percentile(scores,5)}')
print(f' mean is {np.mean(scores)}')


#%%

# segmentation

# histogram comparation:

# img = cv2.imread(r'C:\dev\matzevot\photos\BSH0043.JPG')
# img = cv2.imread(r'C:\dev\matzevot\photos\BSH0044.JPG')
# img = (cv2.imread(r'C:\dev\matzevot\photos\BSH0045.JPG'))
img = (cv2.imread(r'C:\dev\matzevot\photos\BSH0046.JPG'))

new_img = seg.hist_comp_segmentation(img)

new_img = cv2.cvtColor(new_img,cv2.COLOR_BGR2RGB)

plt.imshow(new_img)

#%%

# segmentation

# kernel:

img = cv2.imread(r'C:\dev\matzevot\photos\BSH0043.JPG')
# img = (cv2.imread(r'C:\dev\matzevot\photos\BSH0045.JPG'))[:,:,::-1]
# img = (cv2.imread(r'C:\dev\matzevot\photos\BSH0046.JPG'))[:,:,::-1]

ind_top, ind_bot, ind_left, ind_right = seg.edge_detection(img)

img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
plt.imshow(img)
plt.plot([0,img.shape[1]],[ind_top,ind_top],'m')
plt.plot([0,img.shape[1]],[ind_bot,ind_bot],'m')
plt.plot([ind_left,ind_left],[0,img.shape[0]],'m')
plt.plot([ind_right,ind_right],[0,img.shape[0]],'m')


# %%

# run on first 100

# this is for baseline result regular config

scores = []
identified = []

for dir, file_path in data_dict.items():

    df = pd.read_excel(file_path)
    real_text = dict(zip(df.iloc[:, 0], df.iloc[:, 1]))
    file_list = os.listdir(dir) [:100]
    for file_name in tqdm(file_list):

        bare_file_name = re.split(', |_|\.|\+', file_name)[0]

        if bare_file_name in real_text:
            real_out = real_text[bare_file_name]
            file_path = os.path.join(dir, file_name)
            # print(file_path)

            img = cv2.imread(file_path)  # This is BGR

            # plt.imshow(img)
            img = seg.crop(img, fraction=3)
            # plt.imshow(img)
            img = enh.fastNlMeansDenoisingColored(img, h=5)
            # plt.imshow(img)

            # img = thr.kmeans(img, n_clusters=3, invert=False,chosen_cluster=0)
            img = thr.kmeans(img, n_clusters=2, invert=False)
            # plt.imshow(img)
            # img = cv2.cvtColor(img.astype(np.uint8)*255, cv2.COLOR_GRAY2RGB)

            # img2 = imutils.rotate(img,angle = 45)

            ocr_out = pytesseract.image_to_string(img, lang="heb")

            # plt.imshow(img)
            # print(ocr_out)
            score = editdistance.eval(ocr_out, real_out) / \
                max([len(real_out), len(ocr_out)])
            scores.append(score)
            if score<0.9:
                identified.append(bare_file_name)
                print(bare_file_name)
                plt.imshow(img)
                print(real_out)
                print(ocr_out)

print(f'identified text in {len(identified)} out of {len(scores)}')
#%%

# Test custom config on specific file BSH0078, improve between 0.525 to 0.425

scores = []
identified = []
#-l Hebrew 
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
            # print(file_path)

            img = cv2.imread(file_path)  # This is BGR

            # plt.imshow(img)
            # img = seg.crop(img, fraction=3)
            # plt.imshow(img)
            # img = enh.fastNlMeansDenoisingColored(img, h=5)
            # plt.imshow(img)

            # img = thr.kmeans(img, n_clusters=4, invert=False)
            # img = thr.kmeans(img, n_clusters=5, invert=False)
            # plt.imshow(img)
            # img = cv2.cvtColor(img.astype(np.uint8)*255, cv2.COLOR_GRAY2RGB)

            # img2 = imutils.rotate(img,angle = 45)

            ocr_out = pytesseract.image_to_string(img, lang="heb", config=custom_config)

            # plt.imshow(img)
            # print(ocr_out)
            score = editdistance.eval(ocr_out, real_out) / \
                max([len(real_out), len(ocr_out)])
            scores.append(score)
            if score<0.9:
                identified.append(bare_file_name)
                # print(bare_file_name)
                # plt.imshow(img)
                # print(real_out)
                # print(ocr_out)

print(f'identified text in {len(identified)} out of {len(scores)}')
print(f' 5 precentile is {np.percentile(scores,5)}')

#%%
#%%

# Test custom config on specific file BSH0078, improve between 0.525 to 0.425

scores = []
identified1 = []
#-l Hebrew 
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
            # print(file_path)

            img = cv2.imread(file_path)  # This is BGR
            # img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

            # plt.imshow(img)
            img = seg.crop(img, fraction=3)

            # plt.imshow(img)
            img = enh.fastNlMeansDenoisingColored(img, h=5)
            # img = enh.median(img,size=3)
            img = enh.CLAHE(img)
            # img = enh.median(img,size=3)
            # img = enh.bilateral(img)
            img = enh.fastNlMeansDenoisingColored(img, h=30)
            # plt.imshow(img)
            # img = thr.kmeans(img, n_clusters=4, invert=False)
            # img = thr.kmeans(img, n_clusters=5, invert=False)
            # plt.imshow(img)
            # img = cv2.cvtColor(img.astype(np.uint8)*255, cv2.COLOR_GRAY2RGB)


            # img2 = imutils.rotate(img,angle = 45)

            ocr_out = pytesseract.image_to_string(img, lang="heb", config=custom_config)
            # ocr_out = pytesseract.image_to_string(img)
            # plt.imshow(img)
            # print(ocr_out)
            score = editdistance.eval(ocr_out, real_out) / \
                max([len(real_out), len(ocr_out)])
            scores.append(score)
            if score<1.9:
                # identified.append(bare_file_name)
                print(bare_file_name)
                

                h, w, c = img.shape  # assumes color image
                boxes = pytesseract.image_to_boxes(img, lang="heb", config=custom_config)
                # boxes = pytesseract.image_to_boxes(img)
                for b in boxes.splitlines():
                    b = b.split(' ')
                    img = cv2.rectangle(img, (int(b[1]), h - int(b[2])), (int(b[3]), h - int(b[4])), (0, 255, 0), 2)
                img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
                
                plt.imshow(img)
                # cv2.imshow('img',img)
                # cv2.waitKey(0)

                print(real_out)
                print(ocr_out)
                print(score)


#%%

# Test image inhancement techniques

scores = []
identified = []
#-l Hebrew 
custom_config = r'-c tessedit_char_blacklist=1234567890~!₪@#$%^&*()_-+=[]{}/\\|.<>?;:' \
                    r' --psm 12'

for dir, file_path in data_dict.items():

    df = pd.read_excel(file_path)
    real_text = dict(zip(df.iloc[:, 0], df.iloc[:, 1]))
    file_list = os.listdir(dir)[:30]
    # file_list = ['BSH0046.JPG']
    for file_name in tqdm(file_list):

        bare_file_name = re.split(', |_|\.|\+', file_name)[0]

        if bare_file_name in real_text:
            real_out = real_text[bare_file_name]
            file_path = os.path.join(dir, file_name)
            # print(file_path)

            img = cv2.imread(file_path)  # This is BGR

            # plt.imshow(img)
            img = seg.crop(img, fraction=3)
            # plt.imshow(img)
            img = enh.fastNlMeansDenoisingColored(img, h=5)
            # img = enh.median(img,size=3)
            img = enh.CLAHE(img)
            # img = enh.bilateral(img)
            # img = enh.median(img,size=3)
            img = enh.fastNlMeansDenoisingColored(img, h=30)
            # plt.imshow(img)

            # img = thr.kmeans(img, n_clusters=4, invert=False)
            # img = thr.kmeans(img, n_clusters=2, invert=False)
            # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # img = thr.thresholding(img)
            # img = thr.kmeans(img, n_clusters=5, invert=False)
            # plt.imshow(img)
            # img = cv2.cvtColor(img.astype(np.uint8)*255, cv2.COLOR_GRAY2RGB)

            angle = 0
            
            angle_list = np.linspace(-20, 20, 11)
            score_list = []
            text_len = []
            best_angle = 0
            best_len = 0

            # for angle in angle_list:
            
            #     img2 = imutils.rotate(img, angle=angle)
            #     ocr_out = pytesseract.image_to_string(img2, lang="heb", config=custom_config)
            #     # print(ocr_out)
            #     # score_list.append(editdistance.eval(ocr_out, real_out) /
            #                     # max([len(real_out), len(ocr_out)]))
            #     # text_len.append(len(ocr_out))
            #     if len(ocr_out)>best_len:
            #         best_len = len(ocr_out)
            #         best_angle = angle
            # # print(score_list)

            # if best_angle != 0:
            #     print('aaaaaaaaa')
            #     print(best_angle)

            # img = imutils.rotate(img, angle=best_angle)

            ocr_out = pytesseract.image_to_string(img, lang="heb", config=custom_config)

            # plt.imshow(img)
            # print(ocr_out)
            score = editdistance.eval(ocr_out, real_out) / \
                max([len(real_out), len(ocr_out)])
            scores.append(score)
            if score<1.9:
                identified.append(bare_file_name)
                # print(bare_file_name)
                plt.imshow(img)
                # print(real_out)
                print(ocr_out)

print(f'identified text in {len(identified)} out of {len(scores)}')
print(f' 5 precentile is {np.percentile(scores,5)}')

#%%

# Ablation kmeans

scores = []
identified = []
#-l Hebrew 
custom_config = r'-c tessedit_char_blacklist=1234567890~!₪@#$%^&*()_-+=[]{}/\\|.<>?;:' \
                    r' --psm 12'

for dir, file_path in data_dict.items():

    df = pd.read_excel(file_path)
    real_text = dict(zip(df.iloc[:, 0], df.iloc[:, 1]))
    file_list = os.listdir(dir)
    # file_list = ['BSH0046.JPG']
    for file_name in tqdm(file_list):

        bare_file_name = re.split(', |_|\.|\+', file_name)[0]
        # print(bare_file_name)

        if bare_file_name in real_text:
            real_out = real_text[bare_file_name]
            file_path = os.path.join(dir, file_name)
            # print(file_path)

            img = cv2.imread(file_path)  # This is BGR

            # # plt.imshow(img)
            # img = seg.crop(img, fraction=3)
            # # plt.imshow(img)
            # # img = enh.fastNlMeansDenoisingColored(img, h=5)
            # # img = enh.median(img,size=3)
            # img = enh.CLAHE(img)
            # # img = enh.bilateral(img)
            # # img = enh.median(img,size=3)
            # # img = enh.fastNlMeansDenoisingColored(img, h=30)
            # # plt.imshow(img)

            # # img = thr.kmeans(img, n_clusters=4, invert=False)
            # # img = thr.kmeans(img, n_clusters=2, invert=False)
            # # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # light, dark, all = thr.kmeans2(img, n_clusters=4)
            # # img = thr.thresholding(img)
            # # img = thr.OTSU(img)

            # # img = thr.thresholding(img)
            # # img = thr.kmeans(img, n_clusters=5, invert=False)
            # # plt.imshow(img)


            # lengths = []
            # for new_img in [light, dark]:
            #     new_img = cv2.cvtColor(new_img.astype(np.uint8)*255, cv2.COLOR_GRAY2RGB)
            #     ocr_out = pytesseract.image_to_string(new_img, lang="heb", config=custom_config)
            #     lengths.append(len(ocr_out))

            # img = light if lengths[0]>=lengths[1] else dark

            # # kernel = np.ones((3, 3), np.uint8)
            # # img = cv2.morphologyEx(img.astype(np.uint8), cv2.MORPH_ERODE, kernel)

            # # img = dark



            # # img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_GRAY2RGB)
    
            # img = cv2.cvtColor(img.astype(np.uint8)*255, cv2.COLOR_GRAY2RGB)

            angle = 0
            
            angle_list = np.linspace(-20, 20, 11)
            score_list = []
            text_len = []
            best_angle = 0
            best_len = 0

            # for angle in angle_list:
            
            #     img2 = imutils.rotate(img, angle=angle)
            #     ocr_out = pytesseract.image_to_string(img2, lang="heb", config=custom_config)
            #     # print(ocr_out)
            #     # score_list.append(editdistance.eval(ocr_out, real_out) /
            #                     # max([len(real_out), len(ocr_out)]))
            #     # text_len.append(len(ocr_out))
            #     if len(ocr_out)>best_len:
            #         best_len = len(ocr_out)
            #         best_angle = angle
            # # print(score_list)

            # if best_angle != 0:
            #     print('aaaaaaaaa')
            #     print(best_angle)

            # img = imutils.rotate(img, angle=best_angle)

            ocr_out = pytesseract.image_to_string(img, lang="heb", config=custom_config)

            # plt.imshow(img)
            # print(ocr_out)
            score = editdistance.eval(ocr_out, real_out) / \
                max([len(real_out), len(ocr_out)])
            scores.append(score)
            if score<1.9:
                identified.append(bare_file_name)
                # print(bare_file_name)
                # plt.imshow(img)
                # print(real_out)
                # print(ocr_out)

print(f'identified text in {len(identified)} out of {len(scores)}')
print(f' 5 precentile is {np.percentile(scores,5)}')
print(f' mean {np.mean(scores)}')

#%%

# Test thr
scores = []
identified1 = []
#-l Hebrew 
custom_config = r'-c tessedit_char_blacklist=1234567890~!@#$%^&*₪()_-+=[]{}/\\|,.<>?;:' \
                    r' --psm 12'

for dir, file_path in data_dict.items():

    df = pd.read_excel(file_path)
    real_text = dict(zip(df.iloc[:, 0], df.iloc[:, 1]))
    file_list = ['BSH0015.JPG']
    for file_name in tqdm(file_list):

        bare_file_name = re.split(', |_|\.|\+', file_name)[0]

        if bare_file_name in real_text:
            real_out = real_text[bare_file_name]
            file_path = os.path.join(dir, file_name)
            # print(file_path)

            img = cv2.imread(file_path)  # This is BGR
            # img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
            # img = imutils.rotate(img,angle = 20)
            # plt.imshow(img)
            
            img = seg.crop(img, fraction=3)

            # # plt.imshow(img)
            img = enh.fastNlMeansDenoisingColored(img, h=5)
            # img = enh.median(img,size=3)
            img = enh.CLAHE(img)
            # img = enh.median(img,size=3)
            # img = enh.bilateral(img)
            img = enh.fastNlMeansDenoisingColored(img, h=30)
            # # plt.imshow(img)
            # img = thr.kmeans(img, n_clusters=2, invert=False)
            # # img = thr.kmeans(img, n_clusters=5, invert=False)
            # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # thre, img, = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

            # img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
            

            light, dark, all = thr.kmeans2(img, n_clusters=2)

            plt.imshow(all, cmap = 'jet')
            # lengths = {"dark":[],"light":[]}
            # new_img_rot_list = {"dark":[],"light":[]}
            # ocr_out_list = {"dark":[],"light":[]}
            # angle_list = np.linspace(-25,25,11)
            # for angle in angle_list:
            #     print(angle)
            #     for new_img,key in zip([light, dark],["light","dark"]):
            #         new_img = cv2.cvtColor(new_img.astype(np.uint8)*255, cv2.COLOR_GRAY2RGB)
            #         new_img_rot = imutils.rotate(new_img,angle = angle)
            #         # new_img_rot = cv2.cvtColor(new_img_rot.astype(np.uint8)*255, cv2.COLOR_GRAY2RGB)
            #         ocr_out = pytesseract.image_to_string(new_img_rot, lang="heb", config=custom_config)
            #         lengths[key].append(len(ocr_out.replace("\n","").replace("₪","")))
            #         new_img_rot_list[key].append(new_img_rot)
            #         ocr_out_list[key].append(ocr_out)
            
            # key = "light" if np.sum(lengths["light"]) > np.sum(lengths["dark"]) else "dark"
                
            # i = np.argmax(lengths[key])
            
            # # new_angle = np.sum(angle_list * lengths[key]) / np.sum(lengths[key])

            # # img = imutils.rotate(new_img_rot_list[key][len(angle_list)//2] , angle = new_angle)

            # img = new_img_rot_list[key][i]
            # ocr_out = ocr_out_list[key][i]

            # img = cv2.cvtColor(img.astype(np.uint8)*255, cv2.COLOR_GRAY2RGB)


            # ocr_out = pytesseract.image_to_string(img, lang="heb", config=custom_config)
            # # ocr_out = pytesseract.image_to_string(img)
            # # plt.imshow(img)
            # # print(ocr_out)
            # score = editdistance.eval(ocr_out, real_out) / \
            #     max([len(real_out), len(ocr_out)])
            # scores.append(score)
            # if score<1.9:
            #     # identified.append(bare_file_name)
            #     print(bare_file_name)
                

            #     h, w, c = img.shape  # assumes color image
            #     boxes = pytesseract.image_to_boxes(img, lang="heb", config=custom_config)
            #     # boxes = pytesseract.image_to_boxes(img)
            #     for b in boxes.splitlines():
            #         b = b.split(' ')
            #         img = cv2.rectangle(img, (int(b[1]), h - int(b[2])), (int(b[3]), h - int(b[4])), (0, 255, 0), 2)
            #     img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
                
            #     plt.imshow(img)
            #     # cv2.imshow('img',img)
            #     # cv2.waitKey(0)

            #     print(real_out)
            #     print(ocr_out)
            #     print(score)

# %%

#-l Hebrew 
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

# Test angles
scores = []
identified1 = []
#-l Hebrew 
custom_config = r'-c tessedit_char_blacklist=1234567890~!@#$%^&*₪()_-+=[]{}/\\|,.<>?;:' \
                    r' --psm 12'

for dir, file_path in data_dict.items():

    df = pd.read_excel(file_path)
    real_text = dict(zip(df.iloc[:, 0], df.iloc[:, 1]))
    file_list = ['BSH0147.JPG']
    for file_name in tqdm(file_list):

        bare_file_name = re.split(', |_|\.|\+', file_name)[0]

        if bare_file_name in real_text:
            real_out = real_text[bare_file_name]
            file_path = os.path.join(dir, file_name)
            # print(file_path)

            img = cv2.imread(file_path)  # This is BGR
            # img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
            # img = imutils.rotate(img,angle = 20)
            # plt.imshow(img)
            
            img = seg.crop(img, fraction=3)

            # plt.imshow(img)
            # img = enh.fastNlMeansDenoisingColored(img, h=5)
            # # img = enh.median(img,size=3)
            # img = enh.CLAHE(img)
            # # img = enh.median(img,size=3)
            # # img = enh.bilateral(img)
            # img = enh.fastNlMeansDenoisingColored(img, h=30)
            # plt.imshow(img)
            # img = thr.kmeans(img, n_clusters=4, invert=False)
            # img = thr.kmeans(img, n_clusters=5, invert=False)
            # img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
            # plt.imshow(img)


            # light, dark, all = thr.kmeans2(img, n_clusters=4)

            # # plt.imshow(all, cmap = 'jet')

            # lengths = []
            # for new_img in [light, dark]:
            #     new_img = cv2.cvtColor(new_img.astype(np.uint8)*255, cv2.COLOR_GRAY2RGB)
            #     ocr_out = pytesseract.image_to_string(new_img, lang="heb", config=custom_config)
            #     lengths.append(len(ocr_out))

            # img = light if lengths[0]>=lengths[1] else dark

            # img = cv2.cvtColor(img.astype(np.uint8)*255, cv2.COLOR_GRAY2RGB)


            ocr_out = pytesseract.image_to_string(img, lang="heb", config=custom_config)
            # ocr_out = pytesseract.image_to_string(img)
            # plt.imshow(img)
            # print(ocr_out)
            score = editdistance.eval(ocr_out, real_out) / \
                max([len(real_out), len(ocr_out)])
            scores.append(score)
            if score<1.9:
                # identified.append(bare_file_name)
                print(bare_file_name)
                

                h, w, c = img.shape  # assumes color image
                boxes = pytesseract.image_to_boxes(img, lang="heb", config=custom_config)
                # boxes = pytesseract.image_to_boxes(img)
                for b in boxes.splitlines():
                    b = b.split(' ')
                    img = cv2.rectangle(img, (int(b[1]), h - int(b[2])), (int(b[3]), h - int(b[4])), (0, 255, 0), 2)
                img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
                
                plt.imshow(img)
                # cv2.imshow('img',img)
                # cv2.waitKey(0)

                print(real_out)
                print(ocr_out)
                print(score)

#%%
# Test thr
scores = []
identified1 = []
#-l Hebrew 
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
            # print(file_path)

            img = cv2.imread(file_path)  # This is BGR
            # img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
            # img = imutils.rotate(img,angle = 20)
            # plt.imshow(img)
            
            img = seg.crop(img, fraction=3)

            # # plt.imshow(img)
            img = enh.fastNlMeansDenoisingColored(img, h=5)
            # # img = enh.median(img,size=3)
            img = enh.CLAHE(img)
            # # img = enh.median(img,size=3)
            # # img = enh.bilateral(img)
            img = enh.fastNlMeansDenoisingColored(img, h=30)
            # # plt.imshow(img)
            # # img = thr.kmeans(img, n_clusters=4, invert=False)
            # # img = thr.kmeans(img, n_clusters=5, invert=False)
            # # plt.imshow(img)

            light, dark, all = thr.kmeans2(img, n_clusters=4)


            # lengths = {"dark":[],"light":[]}
            # new_img_rot_list = {"dark":[],"light":[]}
            # ocr_out_list = {"dark":[],"light":[]}
            # angle_list = np.linspace(-25,25,11)
            # for angle in angle_list:
            #     print(angle)
            #     for new_img,key in zip([light, dark],["light","dark"]):
            #         new_img = cv2.cvtColor(new_img.astype(np.uint8)*255, cv2.COLOR_GRAY2RGB)
            #         new_img_rot = imutils.rotate(new_img,angle = angle)
            #         # new_img_rot = cv2.cvtColor(new_img_rot.astype(np.uint8)*255, cv2.COLOR_GRAY2RGB)
            #         ocr_out = pytesseract.image_to_string(new_img_rot, lang="heb", config=custom_config)
            #         lengths[key].append(len(ocr_out.replace("\n","").replace("₪","")))
            #         new_img_rot_list[key].append(new_img_rot)
            #         ocr_out_list[key].append(ocr_out)
            
            # key = "light" if np.sum(lengths["light"]) > np.sum(lengths["dark"]) else "dark"
                
            # i = np.argmax(lengths[key])
            
            # # new_angle = np.sum(angle_list * lengths[key]) / np.sum(lengths[key])

            # # img = imutils.rotate(new_img_rot_list[key][len(angle_list)//2] , angle = new_angle)

            # img = new_img_rot_list[key][i]
            # ocr_out = ocr_out_list[key][i]

            # img = cv2.cvtColor(img.astype(np.uint8)*255, cv2.COLOR_GRAY2RGB)


            ocr_out = pytesseract.image_to_string(img, lang="heb", config=custom_config)
            # ocr_out = pytesseract.image_to_string(img)
            # plt.imshow(img)
            # print(ocr_out)
            score = editdistance.eval(ocr_out, real_out) / \
                max([len(real_out), len(ocr_out)])
            scores.append(score)
            if score<1.9:
                # identified.append(bare_file_name)
                print(bare_file_name)
                

                h, w, c = img.shape  # assumes color image
                boxes = pytesseract.image_to_boxes(img, lang="heb", config=custom_config)
                # boxes = pytesseract.image_to_boxes(img)
                for b in boxes.splitlines():
                    b = b.split(' ')
                    img = cv2.rectangle(img, (int(b[1]), h - int(b[2])), (int(b[3]), h - int(b[4])), (0, 255, 0), 2)
                img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
                
                plt.imshow(img)
                # cv2.imshow('img',img)
                # cv2.waitKey(0)

                print(real_out)
                print(ocr_out)
                print(score)

#%%
# Ablation costum config

scores = []
identified = []
#-l Hebrew 
# custom_config = r'-c tessedit_char_blacklist=1234567890~!₪@#$%^&*()_-+=[]{}/\\|.<>?;:' \
#                     r' --psm 12'
custom_config = r' --psm 3 tessedit_char_blacklist=1' 

for dir, file_path in data_dict.items():

    df = pd.read_excel(file_path)
    real_text = dict(zip(df.iloc[:, 0], df.iloc[:, 1]))
    file_list = os.listdir(dir)[:30]
    # file_list = ['BSH0046.JPG']
    for file_name in tqdm(file_list):

        bare_file_name = re.split(', |_|\.|\+', file_name)[0]
        # print(bare_file_name)

        if bare_file_name in real_text:
            real_out = real_text[bare_file_name]
            file_path = os.path.join(dir, file_name)
            # print(file_path)

            img = cv2.imread(file_path)  # This is BGR

            # plt.imshow(img)
            img = seg.crop(img, fraction=3)
            # plt.imshow(img)
            img = enh.fastNlMeansDenoisingColored(img, h=5)
            # img = enh.median(img,size=3)
            img = enh.CLAHE(img)
            # img = enh.bilateral(img)
            # img = enh.median(img,size=3)
            img = enh.fastNlMeansDenoisingColored(img, h=30)
            # plt.imshow(img)

            # img = thr.kmeans(img, n_clusters=4, invert=False)
            # img = thr.kmeans(img, n_clusters=2, invert=False)
            # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            light, dark, all = thr.kmeans2(img, n_clusters=4)
            # img = thr.thresholding(img)
            # img = thr.OTSU(img)

            # img = thr.thresholding(img)
            # img = thr.kmeans(img, n_clusters=5, invert=False)
            # plt.imshow(img)


            lengths = []
            for new_img in [light, dark]:
                new_img = cv2.cvtColor(new_img.astype(np.uint8)*255, cv2.COLOR_GRAY2RGB)
                ocr_out = pytesseract.image_to_string(new_img, lang="heb", config=custom_config)
                # ocr_out = pytesseract.image_to_string(new_img, lang="heb")
                lengths.append(len(ocr_out))

            img = light if lengths[0]>=lengths[1] else dark

            # kernel = np.ones((3, 3), np.uint8)
            # img = cv2.morphologyEx(img.astype(np.uint8), cv2.MORPH_ERODE, kernel)

            # img = dark



            # img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_GRAY2RGB)
    
            img = cv2.cvtColor(img.astype(np.uint8)*255, cv2.COLOR_GRAY2RGB)

            angle = 0
            
            angle_list = np.linspace(-20, 20, 11)
            score_list = []
            text_len = []
            best_angle = 0
            best_len = 0

            # for angle in angle_list:
            
            #     img2 = imutils.rotate(img, angle=angle)
            #     ocr_out = pytesseract.image_to_string(img2, lang="heb", config=custom_config)
            #     # print(ocr_out)
            #     # score_list.append(editdistance.eval(ocr_out, real_out) /
            #                     # max([len(real_out), len(ocr_out)]))
            #     # text_len.append(len(ocr_out))
            #     if len(ocr_out)>best_len:
            #         best_len = len(ocr_out)
            #         best_angle = angle
            # # print(score_list)

            # if best_angle != 0:
            #     print('aaaaaaaaa')
            #     print(best_angle)


            # img = imutils.rotate(img, angle=best_angle)

            ocr_out = pytesseract.image_to_string(img, lang="heb", config=custom_config)
            # ocr_out = pytesseract.image_to_string(img, lang="heb")

            # plt.imshow(img)
            # print(ocr_out)
            score = editdistance.eval(ocr_out, real_out) / \
                max([len(real_out), len(ocr_out)])
            scores.append(score)
            if score<1.9:
                identified.append(bare_file_name)
                # print(bare_file_name)
                # plt.imshow(img)
                # print(real_out)
                # print(ocr_out)

print(f'identified text in {len(identified)} out of {len(scores)}')
print(f' 5 precentile is {np.percentile(scores,5)}')
print(f' mean {np.mean(scores)}')
