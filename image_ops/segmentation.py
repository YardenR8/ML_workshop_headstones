import cv2
import numpy as np
import matplotlib.pyplot as plt
# from skimage.color import rgb2hsv

def edge_detection(img, ker_size=100, fraction = 3):
    
    ker_x = np.array([[1]*ker_size+[0]+[-1]*ker_size]*(2*ker_size+1))

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

    return ind_top, ind_bot, ind_left, ind_right


def crop(img, ker_size=100, fraction = 3 ):
    ind_top, ind_bot, ind_left, ind_right = edge_detection(img, ker_size=ker_size, fraction = fraction)
    cropped = img[ind_top:ind_bot, ind_left:ind_right]
    return cropped

def hist_comp_segmentation(img, n=10, thresh = 0.4):

    n =10

    shape = img.shape

    # find center histogram

    center_x = shape[0]//2
    center_y = shape[1]//2
    ref_size_fraction = n

    ref_img = img[center_x - shape[0]//ref_size_fraction//2:center_x + shape[0]//ref_size_fraction//2,center_y - shape[1]//ref_size_fraction//2:center_y + shape[1]//ref_size_fraction//2,:]

    x_space = int(shape[0]/n)
    y_space = int(shape[1]/n)

    xy = np.meshgrid(range(n),range(n))
    x_ind = xy[0].flatten()
    y_ind = xy[1].flatten()


    # create histogram for each channel
    ref_x = []
    ref_y = []
    ref_x_bins = []
    for i in range(3):
        y,x = np.histogram(ref_img[:,:,i].flatten(),bins=10)
        ref_x.append((x[:1]+x[:-1])/2)
        ref_x_bins.append(x)
        ref_y.append(y)

    dists =[] 
    for i,x,y in zip(range(len(x_ind)), x_ind * x_space,y_ind*y_space):
        dist = []
        
        for color in range(3):
            yy,xx = np.histogram(img[x:x+x_space,y:y+y_space,color].flatten(),bins = ref_x_bins[color])
            dist.append( sum((np.cumsum(yy) - np.cumsum(ref_y[color])) **2))
            
        dists.append(sum(dist))

    prob = np.array(dists) / sum(dists) * len(dists) # the probability that this rectangle is part of the headstone

    for i,x,y in zip(range(len(x_ind)), x_ind * x_space,y_ind*y_space):

        if prob[i] > thresh:
            img[x:x+x_space,y:y+y_space,:] = 0

    return img