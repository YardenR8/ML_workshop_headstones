from sklearn.cluster import KMeans
import cv2
import numpy as np
def kmeans(img, n_clusters = 4, invert = False, chosen_cluster = None):
    if chosen_cluster is None:
        chosen_cluster = n_clusters - 1
        
    image_for_kmeans = np.array([img[:,:,i].flatten() for i in range(3)]).T
    kmeans = KMeans(n_clusters).fit(image_for_kmeans)
    out = kmeans.predict(image_for_kmeans)
    target_cluster = np.argsort(np.sum(kmeans.cluster_centers_,axis=1))[chosen_cluster]
    # new_image = (out != np.argmax(np.sum(kmeans.cluster_centers_,axis=1))).reshape(np.shape(img)[0:2])
    new_image = (out != target_cluster).reshape(np.shape(img)[0:2])

    new_image = ~new_image if invert else new_image

    return new_image

def kmeans2(img, n_clusters = 4):

    image_for_kmeans = np.array([img[:,:,i].flatten() for i in range(3)]).T
    kmeans = KMeans(n_clusters).fit(image_for_kmeans)
    out = kmeans.predict(image_for_kmeans)
    target_cluster_light = np.argsort(np.sum(kmeans.cluster_centers_,axis=1))[n_clusters-1]
    target_cluster_dark = np.argsort(np.sum(kmeans.cluster_centers_,axis=1))[0]
    # new_image = (out != np.argmax(np.sum(kmeans.cluster_centers_,axis=1))).reshape(np.shape(img)[0:2])
    new_image_light = (out != target_cluster_light).reshape(np.shape(img)[0:2])
    new_image_dark = (out != target_cluster_dark).reshape(np.shape(img)[0:2])

    all = out.reshape(np.shape(img)[0:2])

    # new_image = ~new_image if invert else new_image

    return new_image_light, new_image_dark, all


def thresholding(image):
    print(image.shape)
    image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    # return cv2.threshold(image, 127, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    return cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)

def OTSU(image):
    print(image.shape)
    image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    return cv2.threshold(image, 127, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    # return cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
