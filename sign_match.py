import cv2
from difflib import SequenceMatcher as sim
import os
import math
from scipy.stats.stats import pearsonr

size = (100, 200)
scale_threshold = 20
bin_threshold = 2
alpha_s = 0.1
lambda_v=0.3


def roundup(x,n):
    return int(math.ceil(x / n)) * n

def contain_image(bw_image):
    st = 0; ed = len(bw_image)

    for r in range(0, ed):
        row = bw_image[r]
        blen = len(row[row <= scale_threshold])
        if blen / len(row) < 0.95:
            st = r
            break

    for r in range(-ed, 0):
        row = bw_image[r]
        blen = len(row[row <= scale_threshold])
        if blen / len(row) < 0.95:
            ed = abs(r)
            break

    bw_image = bw_image[st:ed]; #print(st, ed)
    return bw_image


def preprocess_image(path, label):
    raw_image = cv2.imread(path)
    bw_image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2GRAY)
    bw_image = 255 - bw_image
    cv2.imwrite("input/"+str(label) + "_1.png", bw_image)
    #cv2.imshow("BlacknWhite", cv2.resize(bw_image, (0, 0), fx=3, fy=3)); cv2.waitKey()
    _, threshold_image = cv2.threshold(bw_image, scale_threshold, 255, 0)
    cv2.imwrite("input/"+str(label) + "_2.png", threshold_image)
    #cv2.imshow("Threshold", cv2.resize(threshold_image, (0, 0), fx=3, fy=3)); cv2.waitKey()
    return threshold_image


def surf(im, path, label):
    raw_image = cv2.imread(path)
    surf = cv2.xfeatures2d.SURF_create()
    kp, des = surf.detectAndCompute(im, None)
    cv2.drawKeypoints(im, kp, raw_image)
    cv2.imwrite("input/"+str(label)+"_3.png",raw_image)
    #cv2.imshow('Keypoints', cv2.resize(raw_image, (0, 0), fx=3, fy=3)); cv2.waitKey()
    return des


def nomalize_image(im):
    bw_image = contain_image(im)
    bw_image = contain_image(cv2.transpose(bw_image))
    #bw_image = cv2.transpose(bw_image); #do not transpose again if features are vertically strong
    trans_image = cv2.resize(bw_image, size)
    #cv2.imshow("Transposed", cv2.resize(trans_image, (0, 0), fx=3, fy=3)); cv2.waitKey()
    return trans_image


def flat_features(im):
    flat = []
    im_len = len(im);
    sample_v = round(im_len * alpha_s)
    r = 0
    while r < im_len:
        sample_space = im[r:r + sample_v]
        r = r + sample_v + 1
        sample_space = sample_space.transpose()
        sample_space = sample_space.flatten()
        vector_len = len(sample_space)
        iter_v = round(vector_len * alpha_s)
        c = 0
        while c < vector_len:
            window = sample_space[c:c + iter_v]
            c = c + iter_v
            if sum(window) > (iter_v * lambda_v):
                flat.append(1)
            else:
                flat.append(0)
    return flat


def hist_features(im):
    hist = []
    for row in im:
        sted=[0,len(row)]

        for f in range(0,len(row)):
            if row[f]>0 and row[f+1]>0:
                sted[0]=f; break

        for b in range(-len(row),0):
            if row[b]>0 and row[b+1]>0:
                sted[1]=abs(b); break

        hist.append(abs(sted[0]-sted[1]))
        c=1; i=0; s=0; j=0
        histc=[]
        while i < len(hist):
            s = s + hist[i]; i = i + 1; j = j + 1
            if j>=c:
                histc.append(s); s = 0; j = 0
    return histc


def get_features(image_path, label):
    im = preprocess_image(image_path, label)
    des = surf(im, image_path, label)
    im = nomalize_image(im)

    im[im <= bin_threshold] = 0
    im[im > bin_threshold] = 1

    flat = flat_features(im)
    hist = hist_features(im)
    return flat, hist


def compare(a, ar, b, br):
    match = round(sim(None, a, b).ratio(), 2)
    corr = round(pearsonr(ar, br)[0],2)
    w1=1.0; w2=0
    if corr>=0.5: w2=0.5
    if corr < 0.5 and corr >= 0.4: w2 = 0.2
    if corr < 0.4 and corr >= 0.2: w2 = 0.1
    if corr < 0.2 and corr >= 0.1: w2 = 0.01
    if corr < 0.1 : match=0
    if match < 0.75 and corr < 0.5 : match=0
    if match < 1:
        y = round(((match * w1) + (corr * w2)), 2)
        if y < 1:
            score = y
        else:
            score = match
    else:
        score = match
    score = str(round(score*100,2))+"%"
    return score, match, corr


def compare_api(image_path1, image_path2):
    a, ar = get_features(image_path1, 1)
    b, br = get_features(image_path2, 2)
    score, match, corr = compare(a, ar, b, br)
    return score


def test():
    fol = "data/genuine/"
    image_path1 = fol+"001001_000.png"
    a, ar = get_features(image_path1, 1); #print(len(ar), ar)
    i=0
    for im in os.listdir(fol):
        i=i+1; #im="001001_001.png"
        image_path2 = fol+im
        b, br = get_features(image_path2, 2); #print(len(br), br)
        score, match, corr = compare(a, ar, b, br)
        print(i, " ", im, " >> ", match, " >> ", corr, " >> ", score)
        #break

#test()



