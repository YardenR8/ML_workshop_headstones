import cv2
def fastNlMeansDenoisingColored(img, h = 5):
    return cv2.fastNlMeansDenoisingColored(img,h = h)

def bilateral(img , diam=10, sigma_r=100, sigma_s=100):
    return cv2.bilateralFilter(img, d=diam, sigmaColor=sigma_r, sigmaSpace=sigma_s)

def gaussian(img, ker_size = (7,7)):
    return cv2.GaussianBlur(img,ker_size)