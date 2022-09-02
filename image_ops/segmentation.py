import cv2
import numpy as np

def crop(img, fraction = 3):
    
    ker_x = np.array([[1]*100+[0]+[-1]*100]*201)

    grad_x = np.abs(cv2.filter2D(img[:,:,0],cv2.CV_32F,ker_x))
    grad_y = np.abs(cv2.filter2D(img[:,:,0],cv2.CV_32F,ker_x.transpose()))

    grad_x += np.abs(cv2.filter2D(img[:,:,1],cv2.CV_32F,ker_x))
    grad_y += np.abs(cv2.filter2D(img[:,:,1],cv2.CV_32F,ker_x.transpose()))

    grad_x += np.abs(cv2.filter2D(img[:,:,2],cv2.CV_32F,ker_x))
    grad_y += np.abs(cv2.filter2D(img[:,:,2],cv2.CV_32F,ker_x.transpose()))

    tot_grad = grad_x+grad_y
    val = []
    for i in range(tot_grad.shape[0]//fraction):
        val.append(sum(grad_y [i,:]))
    ind_top = np.argmax(val)
    val = []
    for i in range((fraction-1)*tot_grad.shape[0]//fraction,tot_grad.shape[0]):
        val.append(sum(grad_y [i,:]))
    ind_bot = np.argmax(val) + (fraction-1)*tot_grad.shape[0]//fraction

    val = []
    for i in range(tot_grad.shape[1]//fraction):
        val.append(sum(grad_x [:,i]))
    ind_left = np.argmax(val)
    val = []
    for i in range((fraction-1)*tot_grad.shape[1]//fraction,tot_grad.shape[1]):
        val.append(sum(grad_x [:,i]))
    ind_right = np.argmax(val) + (fraction-1)*tot_grad.shape[1]//fraction

    cropped = img[ind_top:ind_bot, ind_left:ind_right]

    return cropped