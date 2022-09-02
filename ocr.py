import pytesseract
import cv2
import os
import zipfile
import numpy as np

def unzip(file, to_dir):
    try:
        os.mkdir(to_dir)
        with zipfile.ZipFile(file, 'r') as zip_ref:
            zip_ref.extractall(to_dir)
    except OSError as e:
        print(e)


# get grayscale image
def get_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


# noise removal
def remove_noise(image):
    return cv2.medianBlur(image, 5)


# thresholding
def thresholding(image):
    return cv2.threshold(image, 127, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    # return cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)


# dilation
def dilate(image):
    kernel = np.ones((5, 5), np.uint8)
    return cv2.dilate(image, kernel, iterations=1)


# erosion
def erode(image):
    kernel = np.ones((5, 5), np.uint8)
    return cv2.erode(image, kernel, iterations=1)


# opening - erosion followed by dilation
def opening(image):
    kernel = np.ones((5, 5), np.uint8)
    return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)


# canny edge detection
def canny(image):
    return cv2.Canny(image, 100, 200)


# skew correction
def deskew(image):
    coords = np.column_stack(np.where(image > 0))
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        return rotated


# template matching
def match_template(image, template):
    return cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)


if __name__ == '__main__':
    neg = cv2.imread('5.jpg')
    # orig = cv2.imread('2.jpeg')
    gray = get_grayscale(neg)
    thresh = thresholding(gray)
    opening = opening(gray)
    canny = canny(gray)
    # custom_config = r'-l Hebrew -c tessedit_char_blacklist= --oem 3 --psm 11'
    custom_config = r'-l Hebrew -c tessedit_char_blacklist=1234567890ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz' \
                    r' --psm 12'

    h, w = gray.shape  # assumes color image
    boxes = pytesseract.image_to_boxes(gray)

    for b in boxes.splitlines():
        b = b.split(' ')
        img = cv2.rectangle(gray, (int(b[1]), h - int(b[2])), (int(b[3]), h - int(b[4])), (0, 255, 0), 2)

    # # show annotated image and wait for keypress
    cv2.imshow('img', neg)
    cv2.waitKey(0)

    s = pytesseract.image_to_string(image=neg, config=custom_config)
    print(s)

