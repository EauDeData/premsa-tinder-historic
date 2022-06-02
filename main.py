from dataloader import *
import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
import cv2 as cv # so annoying in tutorials ngl
import random
from skimage import data, feature, exposure
import pytesseract
from docx import Document
import pickle as pkl
import multiprocessing as mp
from mmocr.utils.ocr import MMOCR
#mp.set_start_method("spawn")




def get_boxes_lines(sample, prop_vertical = 0.7, iter_dilate = 5, dilate_size = 4):
    
    
    ret, thresh = cv2.threshold(sample, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)
    cols = thresh.shape[1]
    horizontal_size = cols // 30
    horizontalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (horizontal_size, 1))
    horizontal = cv2.erode(thresh, horizontalStructure)
    horizontal = cv2.dilate(horizontal, horizontalStructure)

    # Specify size on vertical axis
    rows = thresh.shape[0]
    verticalsize = rows // 30
    # Create structure element for extracting vertical lines through morphology operations
    verticalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (1, verticalsize))
    # Apply morphology operations
    vertical = cv2.erode(thresh, verticalStructure)
    vertical = cv2.dilate(vertical, verticalStructure)
    
    anuncis = vertical*prop_vertical + horizontal*(1 - prop_vertical) # What will we do when the blocks arent delimited by lines
    anuncis = ((anuncis - anuncis.min()) / (anuncis.max() - anuncis.min())) * 255
    ret, anuncis = cv2.threshold(anuncis.astype(np.uint8), 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)
    anuncis = ~ anuncis
    
    
    dilation = cv2.dilate(anuncis,np.ones((dilate_size, dilate_size)),iterations = iter_dilate)
    
    
    contours, hierarchy = cv2.findContours(dilation, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    sample2 = np.zeros_like(sample.copy())
    bbxs = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if 1: #w/h < 5 and w/h > 1:
            bbxs.append((x, y, w, h))
            # Drawing a rectangle on copied image
            # We need a discard criteria
            sample2 = cv2.rectangle(sample2, (x, y), (x + w, y + h), (255, 255, 255), 2)


    img = sample2*0.8 + sample*0.2
    img = ((img - img.min())/(img.max() - img.min()))*255

    return img, bbxs


def find_blblurred(sample, blur_size = 11, conv_kernel = np.array([[0, 1, 0], [0, 1, 0], [0, 1, 0]]), erosion_size = 5, erosion_iter = 1, ret_boxes = False):
    
    ret, thresh = cv2.threshold(sample, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)
    blurred = cv2.GaussianBlur(thresh, (blur_size, blur_size), 10)
    #blurred = cv2.Canny(thresh, 0, 100)
    kernel = conv_kernel
    kernel2 = kernel.T

    convolved = cv2.filter2D(blurred, -1, kernel = kernel) + cv2.filter2D(blurred, -1, kernel = kernel2)
    erosion = cv2.erode(convolved,np.ones(erosion_size),iterations = erosion_iter)
    ret, thresh = cv2.threshold(erosion, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)
    
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(~thresh)


    # Map component labels to hue val, 0-179 is the hue range in OpenCV
    label_hue = np.uint8(179*labels/np.max(labels))
    blank_ch = 255*np.ones_like(label_hue)
    labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])

    # Converting cvt to BGR
    labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)

    # set bg label to black
    labeled_img[label_hue==0] = 0
    labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_BGR2RGB)
    output = np.zeros_like(sample).copy()

    boxes = []
    for i in range(0, num_labels):

        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        area = stats[i, cv2.CC_STAT_AREA]
        (cX, cY) = centroids[i]
        if w*h < 1e4 or w/h < 1e-2 or h/w < 1e-2: continue # Hardcoded, do it from histogram
        
        
        output = cv2.rectangle(output, (x, y), (x + w, y + h), (255, 255, 255), 3)
        boxes.append((x, y, w, h))
    
    if ret_boxes: return output, boxes
    return output


def corner_detector(sample):
    ret, thresh = cv2.threshold(sample, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)
    dst = cv.cornerHarris(thresh,10,3,0.04)
    #ret, thresh = cv2.threshold(dst, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)
    dst = (dst - dst.min()) / (dst.max() - dst.min()) * 255
    ret, thresh = cv2.threshold(dst.astype(np.uint8), 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)
    
    thresh = ~thresh
    conv_kernel = np.array([[0, 1, 0], [0, 1, 0], [0, 1, 0]])
    kernel = conv_kernel
    kernel2 = kernel.T
    convolved = cv2.filter2D(thresh, -1, kernel = kernel) + cv2.filter2D(thresh, -1, kernel = kernel2)
    
    
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(convolved)


    # Map component labels to hue val, 0-179 is the hue range in OpenCV
    label_hue = np.uint8(179*labels/np.max(labels))
    blank_ch = 255*np.ones_like(label_hue)
    labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])

    # Converting cvt to BGR
    labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)

    # set bg label to black
    labeled_img[label_hue==0] = 0
    labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_BGR2RGB)

    output = np.zeros_like(sample).copy()
    outputFilled = np.zeros_like(sample).copy()

    boxes = []
    for i in range(0, num_labels):

        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        area = stats[i, cv2.CC_STAT_AREA]
        (cX, cY) = centroids[i]
        if h*w < 1000 or h*w > 0.9*sample.shape[-1] * sample.shape[-2]: continue
        
        
        outputFilled = cv2.rectangle(outputFilled, (x, y), (x + w, y + h), ((255, 255, 255)), -1)
        output = cv2.rectangle(output, (x, y), (x + w, y + h), (255, 255, 255), 10)
        
        boxes.append((x, y, w, h))
    
    ### CONVEX HULL ###
    ret = cv.findContours((outputFilled).astype(np.uint8), cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    drawing = np.zeros_like(sample).copy()

    color = (random.randint(50,255), random.randint(50,255), random.randint(50,255))
    
    return output, labeled_img, boxes, convolved, outputFilled

def extract_boxes(img, filled_image):
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(filled_image)
    boxes = []
    output = np.zeros_like(img).copy()
    for i in range(0, num_labels):

        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        area = stats[i, cv2.CC_STAT_AREA]
        (cX, cY) = centroids[i]
        if h*w > 0.8*img.shape[-1] * img.shape[-2]: continue
        
        output = cv2.rectangle(output, (x, y), (x + w, y + h), (255, 255, 255), 10)
        boxes.append(img[y:y+h, x:x+w])
        
    # Map component labels to hue val, 0-179 is the hue range in OpenCV
    label_hue = np.uint8(179*labels/np.max(labels))
    blank_ch = 255*np.ones_like(label_hue)
    labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])

    # Converting cvt to BGR
    labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)

    # set bg label to black
    labeled_img[label_hue==0] = 0
    labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_BGR2RGB)
        
    return boxes, output, labeled_img


def get_string(bbx):
    return pytesseract.image_to_string(bbx, lang = 'spa', nice = 10)
    



#### MP INNER FUNCTION ####
def mp_get_strings(strings, out, n_threads, thread):
    for i in range(thread, len(strings), n_threads):
        out.append(get_string(strings[i]))
###########################

def ad_in_page(files, file_number):
    
    
    threads = 8
    
    i = file_number
    bbxs, _, _ = extract_boxes(files[i][0], corner_detector(files[i][0])[-1])
    bbxs = [cv.threshold(x,127,255,cv.THRESH_BINARY_INV)[1] for x in bbxs if x.shape[0] * x.shape[1] > 2000] # TODO: Use classifier

    batch = [np.stack([bbx, bbx, bbx]).transpose(1, 2, 0) for bbx in bbxs]
    print(f"Doing document {file_number}", end = '\n')
    
    strings = mp.Manager().list()
    proc = [mp.Process(target = mp_get_strings, args=(batch, strings, threads, i ,)) for i in range(threads)]
    [p.start() for p in proc]
    [p.join() for p in proc]

    return list(strings)



def all_strings(files):
    out=[]
    for i in range(len(files)):
        out.append(ad_in_page(files, i))
    return out


#all_strings = [pytesseract.image_to_string(i, lang = 'spa', nice = 10) for i in [extract_boxes(files[j], corner_detector(files[j])[-1])[0] for j in range(len(files)) ]]
files = DataAnuncis('/home/adri/Desktop/cvc/data/tinder-historic/filenames.txt')

alL_strings = all_strings(files)

pkl.dump(list(all_strings), open('all_strings.pkl', 'wb'))
