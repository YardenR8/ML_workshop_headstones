import cv2
def fastNlMeansDenoisingColored(img, h = 5):
    return cv2.fastNlMeansDenoisingColored(img,h = h)

def bilateral(img , diam=10, sigma_r=100, sigma_s=100):
    return cv2.bilateralFilter(img, d=diam, sigmaColor=sigma_r, sigmaSpace=sigma_s)

def gaussian(img, ker_size = (7,7)):
    return cv2.GaussianBlur(img,ker_size)

def median(img, size=7):
    return cv2.medianBlur(img,size)

def CLAHE(img):
    lab= cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l_channel = lab[:,:,0]
    a = lab[:,:,1]
    b = lab[:,:,2]
    clahe = cv2.createCLAHE(clipLimit=100, tileGridSize=(5,5))
    cl = clahe.apply(l_channel)
    limg = cv2.merge((cl,a,b))
    enhanced_img = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

    return enhanced_img

