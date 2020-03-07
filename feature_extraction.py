#from pylab import *
import numpy as np
from os import listdir
from sklearn.svm import LinearSVC
import cv2
from PIL import Image
import imagehash
from scipy.cluster.vq import *
import scipy.spatial.distance as space
from sklearn.preprocessing import StandardScaler


def preprocess_image(path, display=False):
    raw_image = cv2.imread(path)
    bw_image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2GRAY)
    bw_image = 255 - bw_image

    if display:
        cv2.imshow("RGB to Gray", bw_image)
        cv2.waitKey()

    _, threshold_image = cv2.threshold(bw_image, 30, 255, 0)

    if display:
        cv2.imshow("Threshold", threshold_image)
        cv2.waitKey()

    return threshold_image


def get_contour_features(im, display=False):
    '''
    :param im: input preprocessed image
    :param display: flag - if true display images
    :return:aspect ratio of bounding rectangle, area of : bounding rectangle, contours and convex hull
    '''

    rect = cv2.minAreaRect(cv2.findNonZero(im))
    box = cv2.boxPoints(rect)
    box = np.int0(box)

    w = np.linalg.norm(box[0] - box[1])
    h = np.linalg.norm(box[1] - box[2])

    aspect_ratio = max(w, h) / min(w, h)
    bounding_rect_area = w * h

    if display:
        image1 = cv2.drawContours(im.copy(), [box], 0, (120, 120, 120), 2)
        cv2.imshow("a", cv2.resize(image1, (0, 0), fx=2.5, fy=2.5))
        cv2.waitKey()

    hull = cv2.convexHull(cv2.findNonZero(im))

    if display:
        convex_hull_image = cv2.drawContours(im.copy(), [hull], 0, (120, 120, 120), 2)
        cv2.imshow("a", cv2.resize(convex_hull_image, (0, 0), fx=2.5, fy=2.5))
        cv2.waitKey()

    im2, contours, hierarchy = cv2.findContours(im.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    if display:
        contour_image = cv2.drawContours(im.copy(), contours, -1, (120, 120, 120), 3)
        cv2.imshow("a", cv2.resize(contour_image, (0, 0), fx=2.5, fy=2.5))
        cv2.waitKey()

    contour_area = 0
    for cnt in contours:
        contour_area += cv2.contourArea(cnt)
    hull_area = cv2.contourArea(hull)

    return aspect_ratio, bounding_rect_area, hull_area, contour_area


def sift(im, path, display=False):
    raw_image = cv2.imread(path)
    sift = cv2.xfeatures2d.SIFT_create()
    kp, des = sift.detectAndCompute(im, None)

    if display:
        cv2.drawKeypoints(im, kp, raw_image)
        cv2.imshow('sift_keypoints.jpg', cv2.resize(raw_image, (0, 0), fx=3, fy=3))
        cv2.waitKey()

    return (path, des)



def feature_extraction(image_path, im_contour_features, des_list):

    preprocessed_image = preprocess_image(image_path)
    hash = imagehash.phash(Image.open(image_path))
    hash = int(str(hash), 16)

    aspect_ratio, bounding_rect_area, convex_hull_area, contours_area = \
        get_contour_features(preprocessed_image.copy(), display=False)

    im_contour_features.append(
        [hash, aspect_ratio, convex_hull_area / bounding_rect_area, contours_area / bounding_rect_area])

    des_list.append(sift(preprocessed_image, image_path))
    return im_contour_features, des_list



def extract_im_feature(des_list, m, n, im_contour_features):
    descriptors = des_list[0][1];
    #print(des_list)
    #print(descriptors); print("------")
    for image_path, descriptor in des_list:
        descriptors = np.vstack((descriptors, descriptor)); #print(image_path)
    #print(descriptors); print("------")
    k = 500
    centroids, variance = kmeans(descriptors, k, 1)
    #print(centroids); print("|||||||||||||||"); print(variance)

    # Calculate the histogram of features
    im_features = np.zeros((m, k+n), "float32")
    for i in range(m):
        words, distance = vq(whiten(des_list[i][1]), centroids)
        #print(words); print("----"); print(distance)
        for w in words:
            im_features[i][w] = im_features[i][w]+1
        #print(im_features)
        for j in range(n):
            im_features[i][k + j] = im_contour_features[i][j]
        #print(im_features)

    # nbr_occurences = np.sum((im_features > 0) * 1, axis=0)
    # idf = np.array(np.log((1.0 * len(image_paths) + 1) / (1.0 * nbr_occurences + 1)), 'float32')

    # Scaling the words
    # stdSlr = StandardScaler().fit(im_features)
    # im_features = stdSlr.transform(im_features)

    return im_features


genuine_image_paths = "data/genuine"
forged_image_paths = "data/forged"

genuine_image_features = [[] for x in range(12)]
forged_image_features = [[] for x in range(12)]

genuine_image_filenames = listdir("data/genuine")
forged_image_filenames = listdir("data/forged")

for name in genuine_image_filenames:
    signature_id = int(name.split('_')[0][-3:])
    genuine_image_features[signature_id - 1].append({"name": name})

for name in forged_image_filenames:
    signature_id = int(name.split('_')[0][-3:])
    forged_image_features[signature_id - 1].append({"name": name})


def run_all():
    cor = 0
    wrong = 0
    im_contour_features = []

    for i in range(12):
        des_list = []
        for im in genuine_image_features[i]:
            paths = genuine_image_paths
            image_path = paths + "/" + im['name']
            im_contour_features, des_list = feature_extraction(image_path, im_contour_features, des_list)

        for im in forged_image_features[i]:
            paths = forged_image_paths
            image_path = paths + "/" + im['name']
            im_contour_features, des_list = feature_extraction(image_path, im_contour_features, des_list)

        m = len(genuine_image_features[i]) + len(forged_image_features[i])
        n = 4
        im_features = extract_im_feature(des_list, m, n, im_contour_features)

        train_genuine_features, test_genuine_features = im_features[0:3], im_features[3:5]
        train_forged_features, test_forged_features = im_features[5:8], im_features[8:10]

        clf = LinearSVC()
        clf.fit(np.concatenate((train_forged_features,train_genuine_features)), np.array([1 for x in range(len(train_forged_features))] + [2 for x in range(len(train_genuine_features))]))
        #print("2" + str(clf.predict(test_genuine_features)))
        genuine_res = clf.predict(test_genuine_features)

        for res in genuine_res:
            if int(res) == 2:
                cor += 1
            else:
                wrong += 1

        #print("1" + str(clf.predict(test_forged_features)))
        forged_res = clf.predict(test_forged_features)

        for res in forged_res:
            if int(res) == 1:
                cor += 1
            else:
                wrong += 1

    print(float(cor)/(cor+wrong))


def get_features(image_path):
    icf, dl = feature_extraction(image_path, [], [])
    imf = extract_im_feature(dl, 1, 4, icf)
    return imf

# run_all()
image_path1 = "data/genuine/001001_000.png"
image_path2 = "data/forged/021001_000.png"
imf1 = get_features(image_path1)
imf2 = get_features(image_path2)

#print(icf); print("----------------------"); print(dl); print("----------------------"); print(imf)
d = space.cdist(imf1, imf2)[0]; print(d)