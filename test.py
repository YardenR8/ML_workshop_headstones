#%%
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
img = mpimg.imread(r'C:\dev\matzevot\photos\BSH0043.JPG')
imgplot = plt.imshow(img)
plt.show()
#%%
import numpy as np
from skimage.color import rgb2hsv
np.shape(img)
plt.figure(2)
plt.clf()
cropped = img[300:600,400:700,:]
cropped_hsv = rgb2hsv(cropped)
imgplot = plt.imshow(cropped_hsv)
#%%
plt.clf()
imgplot = plt.imshow(cropped_hsv[:,:,2])
#%%
plt.figure(3)
plt.clf()
for i in range(3):
    y,x = np.histogram(cropped[:,:,i].flatten(),bins=251)
    plt.plot((x[1:] + x[:-1])/2 , y)
#%%
mean = [np.mean(cropped[:,:,i].flatten()) for i in range(3)]
std = [np.std(cropped[:,:,i].flatten()) for i in range(3)]
rgb = []
for i in range(3):
    c = (img[:,:,i] - mean[i])/std[i]*8*125 + 125
    c[c>255] = 255
    c[c<0] = 0
    rgb.append(c)
new = np.transpose(np.array(rgb),[1,2,0])
plt.figure(5)
plt.clf()
imgplot = plt.imshow(new)
#%%
import cv2
import matplotlib.pyplot as plt
# %matplotlib qt
# Read the image.
img = cv2.imread(r'C:\dev\matzevot\photos\BSH0043.JPG')
 
# Apply bilateral filter with d = 15,
# sigmaColor = sigmaSpace = 75.
bilateral = cv2.bilateralFilter(img, 15, 75, 75)
 
plt.clf()
imgplot = plt.imshow(bilateral)
# Save the output.
# cv2.imwrite('taj_bilateral.jpg', bilateral)
#%%
import numpy as np
img = cv2.imread(r'C:\dev\matzevot\photos\BSH0044.JPG')
lab= cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
l_channel = lab[:,:,0]
a = lab[:,:,1]
b = lab[:,:,2]

# Applying CLAHE to L-channel
# feel free to try different values for the limit and grid size:
clahe = cv2.createCLAHE(clipLimit=100.0, tileGridSize=(16,16))
cl = clahe.apply(l_channel)

# merge the CLAHE enhanced L-channel with the a and b channel
limg = cv2.merge((cl,a,b))

# Converting image from LAB Color model to BGR color spcae
enhanced_img = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

# Stacking the original image with the enhanced image
plt.imsave(r'C:\dev\matzevot\photos\BSH0044_new.JPG',enhanced_img)
bilateral = cv2.bilateralFilter(enhanced_img, 20, 50, 50)


plt.imsave(r'C:\dev\matzevot\photos\BSH0044_new2.JPG',bilateral)
# plt.imshow(result)
# cv2.imshow('Result', result)
#%%
from sklearn.cluster import KMeans
import numpy as np
X = np.array([[1, 2], [1, 4], [1, 0],
              [10, 2], [10, 4], [10, 0]])
kmeans = KMeans(n_clusters=2, random_state=0).fit(X)
kmeans.labels_
np.array([1, 1, 1, 0, 0, 0], dtype=np.int32)
kmeans.predict([[0, 0], [12, 3]])
np.array([1, 0], dtype=np.int32)
kmeans.cluster_centers_
np.array([[10.,  2.],
       [ 1.,  2.]])

#%%
import cv2
# Read the original image
img = cv2.imread(r'C:\dev\matzevot\photos\BSH0043.JPG')
# Display original image
cv2.imshow('Original', img)
cv2.waitKey(0)
# Convert to graycsale
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# Blur the image for better edge detection
img_blur = cv2.GaussianBlur(img_gray, (3,3), 0)
# Sobel Edge Detection
sobelx = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=1, dy=0, ksize=5) # Sobel Edge Detection on the X axis
sobely = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=0, dy=1, ksize=5) # Sobel Edge Detection on the Y axis
sobelxy = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=1, dy=1, ksize=5) # Combined X and Y Sobel Edge Detection
# Display Sobel Edge Detection Images
cv2.imshow('Sobel X', sobelx)
cv2.waitKey(0)
cv2.imshow('Sobel Y', sobely)
cv2.waitKey(0)
cv2.imshow('Sobel X Y using Sobel() function', sobelxy)
cv2.waitKey(0)
# Canny Edge Detection
edges = cv2.Canny(image=img_blur, threshold1=100, threshold2=200) # Canny Edge Detection
# Display Canny Edge Detection Image
cv2.imshow('Canny Edge Detection', edges)
cv2.waitKey(0)
cv2.destroyAllWindows()

#%%
import cv2
import matplotlib.pyplot as plt
from skimage.color import rgb2hsv
import numpy as np
# img = cv2.imread(r'C:\dev\matzevot\photos\BSH0043.JPG')
img = (cv2.imread(r'C:\dev\matzevot\photos\BSH0045.JPG'))[:,:,::-1]
# img = (cv2.imread(r'C:\dev\matzevot\photos\BSH0046.JPG'))[:,:,::-1]

# rgb2hsv
n =10

shape = img.shape

center_x = shape[0]//2
center_y = shape[1]//2
ref_size_fraction = n

ref_img = img[center_x - shape[0]//ref_size_fraction//2:center_x + shape[0]//ref_size_fraction//2,center_y - shape[1]//ref_size_fraction//2:center_y + shape[1]//ref_size_fraction//2,:]

x_space = int(shape[0]/n)
y_space = int(shape[1]/n)

xy = np.meshgrid(range(n),range(n))
x_ind = xy[0].flatten()
y_ind = xy[1].flatten()

plt.figure(1)
plt.clf()
# for x,y in zip(x_ind * x_space,y_ind*y_space):
#     # plt.figure()
    
#     plt.imshow( img[x:x+x_space,y:y+y_space,:] ,extent=[x,x+x_space,y,y+y_space])
#     plt.pause(0.001)
#     print(f"{x},{y}")

#%
ref_x = []
ref_y = []
ref_x_bins = []
for i in range(3):
    y,x = np.histogram(ref_img[:,:,i].flatten(),bins=10)
    ref_x.append((x[:1]+x[:-1])/2)
    ref_x_bins.append(x)
    ref_y.append(y)

prob =[] 
for i,x,y in zip(range(len(x_ind)), x_ind * x_space,y_ind*y_space):
    # plt.figure()
    dist = []
    
    for color in range(3):
        yy,xx = np.histogram(img[x:x+x_space,y:y+y_space,color].flatten(),bins = ref_x_bins[color])
        dist.append( sum((np.cumsum(yy) - np.cumsum(ref_y[color])) **2))
        # dist.append( np.interp(img[x:x+x_space,y:y+y_space,color].flatten(), ref_x[color],ref_y[color]) )
    prob.append(sum(dist))
    # prob.append(np.sum(1/(dist[0]+0.01) + 1/(dist[1]+0.01) + 1/(dist[2]+0.01)))
    # prob.append(  np.sum(((1/(dist[0]+1e-9))**2)+((1/(dist[1]+1e-9))**2)+((1/(dist[2]+1e-9))**2  )))
    
prob = np.array(prob) / sum(prob) * len(prob)

for i,x,y in zip(range(len(x_ind)), x_ind * x_space,y_ind*y_space):
    # print(f"{prob[i]}")
    # for k in range(3):
    #     yy,xx = np.histogram(img[x:x+x_space,y:y+y_space,k].flatten(),bins=30)
    #     plt.plot((xx[:1]+xx[:-1])/2,yy)
    if prob[i] > 0.4:
        img[x:x+x_space,y:y+y_space,:] = 0
    # plt.imshow( img[x:x+x_space,y:y+y_space,:] ,extent=[x,x+x_space,y,y+y_space])
    plt.pause(0.001)
    # print(f"{x},{y}")
plt.imshow(img)
#%%
img = (cv2.imread(r'C:\dev\matzevot\photos\BSH0046.JPG'))
plt.imshow(img[:,:,::-1])
#%%
img = (cv2.imread(r'C:\dev\matzevot\photos\BSH0043.JPG'))
# img = (cv2.imread(r'C:\dev\matzevot\photos\BSH0046.JPG'))

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

plt.imshow(grad_x+grad_y)
#%%
img = (cv2.imread(r'C:\dev\matzevot\photos\BSH0046.JPG'))
# img = (cv2.imread(r'C:\dev\matzevot\photos\BSH0045.JPG'))
img = (cv2.imread(r'C:\dev\matzevot\photos\BSH0043.JPG'))

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
for i in range(tot_grad.shape[0]//2):
    val.append(sum(grad_y [i,:]))
ind_top = np.argmax(val)
val = []
for i in range(tot_grad.shape[0]//2,tot_grad.shape[0]):
    val.append(sum(grad_y [i,:]))
ind_bot = np.argmax(val) + tot_grad.shape[0]//2

val = []
for i in range(tot_grad.shape[1]//2):
    val.append(sum(grad_x [:,i]))
ind_left = np.argmax(val)
val = []
for i in range(tot_grad.shape[1]//2,tot_grad.shape[1]):
    val.append(sum(grad_x [:,i]))
ind_right = np.argmax(val) + tot_grad.shape[1]//2



plt.imshow(img[:,:,::-1])
plt.plot([0,tot_grad.shape[1]],[ind_top,ind_top],'m')
plt.plot([0,tot_grad.shape[1]],[ind_bot,ind_bot],'m')
plt.plot([ind_left,ind_left],[0,tot_grad.shape[0]],'m')
plt.plot([ind_right,ind_right],[0,tot_grad.shape[0]],'m')
#%%
import os
folder = r"C:\dev\matzevot\photos\Beshenkovichy"
output_folder = r"C:\dev\matzevot\photos\output4"
file_list = os.listdir(folder)
for name in file_list[:50]:
    
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

    # Applying CLAHE to L-channel
    # feel free to try different values for the limit and grid size:
    clahe = cv2.createCLAHE(clipLimit=100.0, tileGridSize=(16,16))
    cl = clahe.apply(l_channel)

    medl = cv2.medianBlur(cl,7)

    # merge the CLAHE enhanced L-channel with the a and b channel
    limg = cv2.merge((medl,a,b))

    # Converting image from LAB Color model to BGR color spcae
    enhanced_img = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

    # Stacking the original image with the enhanced image
    # plt.imsave(r'C:\dev\matzevot\photos\BSH0044_new.JPG',enhanced_img)
    bilateral = cv2.bilateralFilter(enhanced_img, 100, 100, 100)
    # gauss = cv2.GaussianBlur(enhanced_img,(7,7))
    # gauss = cv2.GaussianBlur(enhanced_img,(5,5),cv2.BORDER_DEFAULT)

    plt.imsave(os.path.join(output_folder,name), bilateral)

#%%

img = (cv2.imread(r'C:\dev\matzevot\photos\BSH0046.JPG'))
# img = (cv2.imread(r'C:\dev\matzevot\photos\BSH0045.JPG'))
img = (cv2.imread(r'C:\dev\matzevot\photos\BSH0043.JPG'))


# img = cv2.imread(os.path.join(folder,name))
ker_x = np.array([[1]*100+[0]+[-1]*100]*201)

grad_x = np.abs(cv2.filter2D(img[:,:,0],cv2.CV_32F,ker_x))
grad_y = np.abs(cv2.filter2D(img[:,:,0],cv2.CV_32F,ker_x.transpose()))

grad_x += np.abs(cv2.filter2D(img[:,:,1],cv2.CV_32F,ker_x))
grad_y += np.abs(cv2.filter2D(img[:,:,1],cv2.CV_32F,ker_x.transpose()))

grad_x += np.abs(cv2.filter2D(img[:,:,2],cv2.CV_32F,ker_x))
grad_y += np.abs(cv2.filter2D(img[:,:,2],cv2.CV_32F,ker_x.transpose()))


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

# Applying CLAHE to L-channel
# feel free to try different values for the limit and grid size:
clahe = cv2.createCLAHE(clipLimit=100.0, tileGridSize=(16,16))
cl = clahe.apply(l_channel)

######
medl = cv2.medianBlur(cl,7)



# merge the CLAHE enhanced L-channel with the a and b channel
limg = cv2.merge((medl,a,b))

# Converting image from LAB Color model to BGR color spcae
enhanced_img = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)



# Stacking the original image with the enhanced image
# plt.imsave(r'C:\dev\matzevot\photos\BSH0044_new.JPG',enhanced_img)
bilateral = cv2.bilateralFilter(enhanced_img, 100, 100, 100)
# gauss = cv2.GaussianBlur(enhanced_img,(7,7))
# gauss = cv2.GaussianBlur(enhanced_img,(5,5),cv2.BORDER_DEFAULT)

plt.imshow(enhanced_img[:,:,::-1])
#%% Get single letter
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
folder = r"C:\dev\matzevot\photos\Beshenkovichy"
output_folder = r"C:\dev\matzevot\photos\letters"
file_list = os.listdir(folder)

cors = {file_list[46] : [452,160,89,82],
        file_list[47] : [57,379,51,79],
        file_list[48] : [405,657,70,67],
        file_list[49] : [259,669,64,87]}

def save_letter(name, img, ext_name):
    if name in cors:
        cor = cors[name]
        new = img[cor[1]:(cor[1]+cor[3]),cor[0]:(cor[0]+cor[2]),:]
        new2 = cv2.cvtColor(new, cv2.COLOR_BGR2RGB)
        plt.imsave(os.path.join(output_folder,name.split(".")[0] + "_" + ext_name + ".png"), new2)

for name in file_list[46:50]:
    
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

    plt.imshow(cropped)
    
    save_letter(name, cropped, "original")
    lab= cv2.cvtColor(cropped, cv2.COLOR_BGR2LAB)
    l_channel = lab[:,:,0]
    a = lab[:,:,1]
    b = lab[:,:,2]

    # Applying CLAHE to L-channel
    # feel free to try different values for the limit and grid size:
    clahe = cv2.createCLAHE(clipLimit=100.0, tileGridSize=(16,16))
    cl = clahe.apply(l_channel)
    save_letter(name, cv2.cvtColor(cv2.merge((cl,a,b)), cv2.COLOR_LAB2BGR), "contrast")
    medl = cv2.medianBlur(cl,7)

    # merge the CLAHE enhanced L-channel with the a and b channel
    limg = cv2.merge((medl,a,b))
    # Converting image from LAB Color model to BGR color spcae
    enhanced_img = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    save_letter(name, enhanced_img, "enhanced_img")
    # Stacking the original image with the enhanced image
    # plt.imsave(r'C:\dev\matzevot\photos\BSH0044_new.JPG',enhanced_img)
    bilateral = cv2.bilateralFilter(enhanced_img, 100, 100, 100)
    # gauss = cv2.GaussianBlur(enhanced_img,(7,7))
    # gauss = cv2.GaussianBlur(enhanced_img,(5,5),cv2.BORDER_DEFAULT)

    
    plt.imsave(os.path.join(output_folder,name), cv2.cvtColor(bilateral, cv2.COLOR_BGR2RGB) )
    save_letter(name, bilateral, "bilateral")
