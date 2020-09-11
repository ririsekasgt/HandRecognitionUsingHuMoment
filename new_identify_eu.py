
import os
import cv2
import sys
import numpy as np
import math
from sklearn import preprocessing
import matplotlib.pyplot as plt

def get_class_names(path="Preproc/"):  
    class_names = os.listdir(path)
    return class_names

def get_total_files(path="Preproc/",train_percentage=0.8): 
    sum_total = 0
    sum_train = 0
    sum_test = 0
    subdirs = os.listdir(path)
    for subdir in subdirs:
        files = os.listdir(path+subdir)
        n_files = len(files)
        sum_total += n_files
    return sum_total

def calculate_mean(path='new_datatrain/'):
    class_names = get_class_names(path=path)
    print("class_names = ",class_names)

    total_files = get_total_files(path=path)
    print("total files = ",total_files)

    nb_classes = len(class_names)
    global sr

    for idx, classname in enumerate(class_names):
        class_files = os.listdir(path+classname)
        n_files = len(class_files)
        n_load =  n_files
        total = np.zeros(int(7),dtype=np.float32)
        print(total.shape)
        for idx2, infilename in enumerate(class_files[0:n_load]):
            image_path = path + classname + '/' + infilename
            img = cv2.imread(image_path)
            
            #preprocessing img
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (35,35), 0)
            _, edges = cv2.threshold(blurred, 155, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
            
            contours, hierarchy = cv2.findContours(edges.copy(),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
            y=len(contours)
            area= np.zeros(y)
            for i in range(0, y):
                area[i] = cv2.contourArea(contours[i])
            index = area.argmax()
            hand = contours[index]

            cv2.drawContours(img, [hand], -1, (255, 255, 255), -1)
            
            #extracting hu moments
            moments = cv2.HuMoments(cv2.moments(hand))
            #make it log scale
            for i in range(0,7):
                moments[i] = -1* math.copysign(1.0, moments[i]) * math.log10(abs(moments[i]))
                #print (str(abs(moments[0])) + ' ' + str(abs(moments[1])) + ' ' + str(abs(moments[2])) + ' ' + str(abs(moments[3])) + ' ' + str(abs(moments[4])) + ' ' + str(abs(moments[5])) + ' ' + str(abs(moments[6])))
            total += abs(moments [:,0])
        hasil = total/n_files
        print(classname)
        print(hasil)

        #save data train
        outpath = 'new_euclidean/' + classname
        outfile = outpath + '/' + 'huMom' + '.npy'
        if not os.path.exists(outpath):
                os.mkdir( outpath, 0o755 )
        np.save(outfile,hasil)

calculate_mean()


#distance = math.sqrt(sum([(a - b) ** 2 for a, b in zip(x, y)]))
#print("Euclidean distance from x to y: ",distance)
