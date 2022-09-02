#%%
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
import numpy as np
import os
from tqdm import tqdm
# img = mpimg.imread(r'C:\dev\matzevot\photos\BSH0043.JPG')
folder = r'C:\dev\matzevot\photos\white_good'
outputfolder = r'C:\dev\matzevot\photos\output8'
# name = r"BSH0078.JPG"
file_list = os.listdir(folder)
for name in tqdm(file_list):
    # name = r"BSH0008.JPG"

    # img = mpimg.imread(r'C:\dev\matzevot\photos\Beshenkovichy\BSH0078.JPG')
    # imgplot = plt.imshow(img)
    #%

    img = cv2.imread(os.path.join(folder,name))
    ker_x = np.array([[1]*100+[0]+[-1]*100]*201)

    grad_x = np.abs(cv2.filter2D(img[:,:,0],cv2.CV_32F,ker_x))
    grad_y = np.abs(cv2.filter2D(img[:,:,0],cv2.CV_32F,ker_x.transpose()))

    grad_x += np.abs(cv2.filter2D(img[:,:,1],cv2.CV_32F,ker_x))
    grad_y += np.abs(cv2.filter2D(img[:,:,1],cv2.CV_32F,ker_x.transpose()))

    grad_x += np.abs(cv2.filter2D(img[:,:,2],cv2.CV_32F,ker_x))
    grad_y += np.abs(cv2.filter2D(img[:,:,2],cv2.CV_32F,ker_x.transpose()))

    # sobel_x = abs(cv2.Sobel(img[:,:,0],cv2.CV_32F,1,0,ksize=31))
    # sobel_y = abs(cv2.Sobel(img[:,:,0],cv2.CV_32F,0,1,ksize=31))

    # sobel_x += abs(cv2.Sobel(img[:,:,1],cv2.CV_32F,1,0,ksize=31))
    # sobel_y += abs(cv2.Sobel(img[:,:,1],cv2.CV_32F,0,1,ksize=31))

    # sobel_x += abs(cv2.Sobel(img[:,:,2],cv2.CV_32F,1,0,ksize=31))
    # sobel_y += abs(cv2.Sobel(img[:,:,2],cv2.CV_32F,0,1,ksize=31))

    tot_grad = grad_x+grad_y
    val = []
    for i in range(tot_grad.shape[0]//3):
        val.append(sum(grad_y [i,:]))
    ind_top = np.argmax(val)
    val = []
    for i in range(2*tot_grad.shape[0]//3,tot_grad.shape[0]):
        val.append(sum(grad_y [i,:]))
    ind_bot = np.argmax(val) + 2*tot_grad.shape[0]//3

    val = []
    for i in range(tot_grad.shape[1]//3):
        val.append(sum(grad_x [:,i]))
    ind_left = np.argmax(val)
    val = []
    for i in range(2*tot_grad.shape[1]//3,tot_grad.shape[1]):
        val.append(sum(grad_x [:,i]))
    ind_right = np.argmax(val) + 2*tot_grad.shape[1]//3



    # plt.imshow(img)
    # plt.imshow(tot_grad)

    # plt.plot([0,tot_grad.shape[1]],[ind_top,ind_top],'m')
    # plt.plot([0,tot_grad.shape[1]],[ind_bot,ind_bot],'m')
    # plt.plot([ind_left,ind_left],[0,tot_grad.shape[0]],'m')
    # plt.plot([ind_right,ind_right],[0,tot_grad.shape[0]],'m')
    # plt.pause(0.01)

    cropped = img[ind_top:ind_bot, ind_left:ind_right]
    lab= cv2.cvtColor(cropped, cv2.COLOR_BGR2LAB)
    l_channel = lab[:,:,0]
    a = lab[:,:,1]
    b = lab[:,:,2]
    # plt.imshow(l_channel)
    # plt.hist(np.array(l_channel).flatten())
    #%
    # plt.imshow(np.array(l_channel > 170))
    #%
    from sklearn.cluster import KMeans

    # l_channel_clean = cv2.fastNlMeansDenoising(l_channel,h = 5)
    # kmeans = KMeans(n_clusters = 4).fit(np.array(l_channel_clean).flatten().reshape(-1,1))
    # out = kmeans.predict(l_channel_clean.flatten().reshape(-1,1)).reshape(np.shape(l_channel_clean))
    # new_image = (out != np.argmax(kmeans.cluster_centers_))
    # new_image_clean = cv2.fastNlMeansDenoising(new_image.astype(np.uint8)*255,h = 20)
    # plt.imshow(new_image_clean,cmap = "gray")
    # plt.imsave(os.path.join(outputfolder,name), new_image_clean,cmap = "gray")

    img_clean = cv2.fastNlMeansDenoisingColored(cropped,h = 5)
    image_for_kmeans = np.array([img_clean[:,:,i].flatten() for i in range(3)]).T
    kmeans = KMeans(n_clusters=4).fit(image_for_kmeans)
    out = kmeans.predict(image_for_kmeans)
    new_image = (out != np.argmax(np.sum(kmeans.cluster_centers_,axis=1))).reshape(np.shape(img_clean)[0:2])
    new_image_clean = cv2.fastNlMeansDenoising(new_image.astype(np.uint8)*255,h = 50)
    plt.imshow(new_image_clean,cmap = "gray")
    plt.imsave(os.path.join(outputfolder,name), new_image_clean,cmap = "gray")
#%%
#%%
plt.imshow(out==0)
#%%
plt.imshow(out==1)
#%%
plt.imshow(out==2)
#%%

cropped = img[ind_top+250:ind_bot-260, ind_left+250:ind_right-220]
lab= cv2.cvtColor(cropped, cv2.COLOR_BGR2LAB)
l_channel = lab[:,:,0]
l_channel2 = cv2.fastNlMeansDenoising(lab[:,:,0],h = 30)
l_channel2 = l_channel2.astype(np.int32)
l_channel = l_channel.astype(np.int32)

integrated = np.cumsum(l_channel2,axis=0)

slope = np.zeros_like(integrated)
length = np.shape(slope)[0]
for i in range(length):
    slope[i,:] = integrated[-1,:] * i / (length-1)

alpha = 0.2
plt.imshow(alpha*(integrated - slope) + (1-alpha)*l_channel2)
#%%
plt.imshow(l_channel)
#%%
aaa = cv2.fastNlMeansDenoising(lab[:,:,0],h = 30)
plt.imshow(aaa)
#%%
plt.imshow(lab[:,:,0])
#%%
