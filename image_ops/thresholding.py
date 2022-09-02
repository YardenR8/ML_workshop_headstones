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
