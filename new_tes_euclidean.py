
import os
import sys
import numpy as np
import math
import cv2
import time

def get_class_names(path="Preproc/"):  
    class_names = os.listdir(path)
    return class_names

def myEcli(data, classes):
    hasil = np.zeros(len(classes), dtype=np.float32)
    for idx, i in enumerate(classes):
        hasil[idx] = math.sqrt(np.sum((data-i)**2))
        #print(hasil[idx])
        #print(i.shape)
    val = min(hasil)
    #print(hasil)
    index = np.where(hasil == val)
    index = np.asarray(index)
    return val, index[0,0]

def readEcli_Database(path='new_euclidean/'):
    class_names = get_class_names(path=path)
    print("class_names = ",class_names)

    nb_classes = len(class_names)
    data = np.zeros((nb_classes,7),dtype=np.float32)

    for idx, classname in enumerate(class_names):
        data[idx,:] = np.load(path + classname)

    return data

def ecliData(path='new_euclidean/'):
    class_names = get_class_names(path=path)
    print("class_names = ",class_names)

    nb_classes = len(class_names)
    data = np.zeros((nb_classes,7),dtype=np.float32)

    for idx, classname in enumerate(class_names):
        
        class_files = os.listdir(path+classname)
        n_files = len(class_files)
        n_load =  n_files

        for idx2, infilename in enumerate(class_files[0:n_load]):
            data_path = path + classname + '/' + infilename
            data[idx,:] = np.load(data_path)
            
    return data, class_names

count = 0
h = ''
s = ''
v = ''
w = ''
y = ''
lastDecision = ''
data, class_names = ecliData()
cap = cv2.VideoCapture(0)

#open camera
while (cap.isOpened()):
    ret, img= cap.read()
    img= cv2.flip(img, 1)
    cv2.rectangle(img,(500,300),(200,0),(0,255,0),0)
    roi= img[0:300, 200:500]
    #preprocessing img
    gray= cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (35,35), 0)
    _, edges= cv2.threshold(blurred, 155, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    contours, hierarchy = cv2.findContours(edges.copy(),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    y=len(contours) 
    area= np.zeros(y)   
    for i in range(0, y):
            area[i] = cv2.contourArea(contours[i])
    
    indx = area.argmax()
    hand = contours[indx]
    x,y,w,h = cv2.boundingRect(hand)
    cv2.rectangle(roi,(x,y),(x+w,y+h),(0,0,255),0)
    temp = np.zeros(roi.shape, np.uint8)

    cv2.drawContours(temp, [hand], -1, (255, 255, 255), -1)
    #cv2.drawContours(img, [hand], -1, (255, 255, 255), -1)

    img = cv2.putText(img, "Place your hand in rectangle", (310,30) , cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

    #extracting hu moments
    moments = cv2.HuMoments(cv2.moments(hand))
    #print(moments.shape)
    #make it logscale
    for i in range (0,7):
        moments[i] = -1 * math.copysign(1.0, moments[i]) * math.log10(abs(moments[i]))
           
    nilai, index = myEcli(abs(moments),data)
    #print(class_names[index])
    decision = class_names[index]
    #print(decision)
    time.sleep(.1)
    print(count)

    
    if(decision == lastDecision):
        count += 1
    else:
        count = 0
        
    if(count == 30):
        finalDecision = decision
        print(finalDecision)

        if(finalDecision == 'H'):
            print('Opening File Explorer....')
            #V -> Spotify
            os.startfile(r"C:\Windows\explorer.exe")
        elif(finalDecision == 'S'):
            print('Opening Calculator....')
            #V -> Spotify
            os.startfile(r"C:\Windows\System32\calc.exe")
        elif(finalDecision == 'V'):
            print('Opening Spotify....')
            #V -> Spotify
            os.startfile(r"C:\Users\Riris Eka Sgt\AppData\Local\Microsoft\WindowsApps\Spotify.exe")
        elif(finalDecision == 'W'):
            print('Opening Microsoft Word...')
            #W -> Word
            os.startfile(r"C:\Program Files\Microsoft Office\root\Office16\WINWORD.exe")
        elif(finalDecision == 'Y'):
            print('Opening Browser...')
            #Y -> Browser
            os.startfile(r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe")
          
    elif(count > 30):
            cap.release()
            cv2.destroyAllWindows()

    lastDecision = decision

    img = cv2.putText(img, decision, (450,100) , cv2.FONT_HERSHEY_SIMPLEX, 2, (0,255,255), 2)
    cv2.imshow('Gesture Recognition', img)
    #guide = cv2.imread('guide.png')
    #cv2.imshow('Guide', guide)
    #cv2.moveWindow('Guide',350,0)
    cv2.moveWindow('Gesture Recognition',350,150)
    cv2.imshow('Contour', temp)
    cv2.moveWindow('Contour',0,150)

    #keluar
    if cv2.waitKey(1) == ord('q'):
        break

