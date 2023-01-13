from PyQt5.QtGui import * 
from PyQt5.QtWidgets import * 
from PyQt5.QtCore import *
from email.mime import image

import numpy as np
import cv2
import sys  
import math
import glob
import os

class Window(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("2022 Opencvdl Hw1")
        self.setGeometry(50, 50, 800, 550)

        self.path = ''
        self.image = []
        self.path2 = ''
        self.image2 = []
        self.pathf=[]
        self.folder_path=[]

        self.rvecs=[]
        self.tvecs=[]
        self.dist=[]
        self.mtx=[]

        self.UiComponents()
        self.show()  

    def UiComponents(self):
        self.textbox = QLineEdit(self)
        self.textbox.setGeometry(550, 125, 160, 50)

        self.mycombobox = QComboBox(self)
        self.mycombobox.addItems(['1', '2', '3', '4', '5','6', '7', '8', '9', '10','11', '12', '13', '14', '15'])
        self.mycombobox.setGeometry(120, 350, 150, 50)

        button_load_folder = QPushButton("Load folder", self)
        button_load_folder.setGeometry(120, 125, 150, 50)
        button_load_folder.clicked.connect(self.load_folder)

        button_load_image_1 = QPushButton("Load Image1", self)
        button_load_image_1.setGeometry(120, 200, 150, 50)
        button_load_image_1.clicked.connect(lambda: self.load_img(0))

        button_load_image_2 = QPushButton("Load Image2", self)
        button_load_image_2.setGeometry(120, 275, 150, 50)
        button_load_image_2.clicked.connect(lambda: self.load_img2(1))

        button_one =  QPushButton("1.1 Find Contour", self)
        button_one.setGeometry(300, 50, 200, 50)
        button_one.clicked.connect(self.q11)

        button_two =  QPushButton("1.2 Find Contour ", self)
        button_two.setGeometry(550, 50, 160, 50)
        button_two.clicked.connect(self.q12)
        
        button_third =  QPushButton("2.1 Corner Detection", self)
        button_third.setGeometry(300, 125, 200, 50)
        button_third.clicked.connect(self.q21)

        button_fourth =  QPushButton("2.2 Find the Intrinsic Matrix", self)
        button_fourth.setGeometry(300, 200, 200, 50)
        button_fourth.clicked.connect(self.q22)

        button_five =  QPushButton("2.3 Find the Extrinsic Matrix", self)
        button_five.setGeometry(300, 275, 200, 50)
        button_five.clicked.connect(self.q23)

        button_six =  QPushButton("2.4 Find the Distortion Matrix ", self)
        button_six.setGeometry(300, 350, 200, 50)
        button_six.clicked.connect(self.q24)

        button_seven =  QPushButton("2.5 Show the undistorted result ", self)
        button_seven.setGeometry(300, 425, 200, 50)
        button_seven.clicked.connect(self.q25)

        button_eight =  QPushButton("4.1 Stereo Disparity Map", self)
        button_eight.setGeometry(550, 350, 160, 50)
        button_eight.clicked.connect(self.q41)

        button_nine =  QPushButton("4.1 Stereo Disparity Map", self)
        button_nine.setGeometry(550, 425, 160, 50)

        button_ten =  QPushButton("3.1 Show words on board", self)
        button_ten.setGeometry(550, 200, 160, 50)
        button_ten.clicked.connect(self.q31)

        button_eleven =  QPushButton("3.2 Show words vertically", self)
        button_eleven.setGeometry(550, 275, 160, 50)
        button_eleven.clicked.connect(self.q32)

    def load_img(self,n):

        self.path,_ = QFileDialog.getOpenFileName(self, 'Open a file', '', 'All Files (*.*)')
        self.image = cv2.imread(self.path, flags=1)

    def load_img2(self,n):

        self.path2,_ = QFileDialog.getOpenFileName(self, 'Open a file', '', 'All Files (*.*)')
        self.image2 = cv2.imread(self.path2, flags=1)

    def load_folder(self):
        self.folder_path = QFileDialog.getExistingDirectory(None,"Open folder","./")
        print('load folder : ', self.folder_path)
        paths = glob.glob(self.folder_path + "/*.bmp")
        self.pathf = []
        for path in paths:
            print('load image : ', path)
            self.pathf.append(path)

    def q11(self):

        im = self.image
        imgray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(imgray, 127, 255, 0)

        contours, hierarchy = cv2.findContours(image=thresh, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)
                                            
        image_copy = im.copy()
        cv2.drawContours(image=image_copy, contours=contours, contourIdx=-1, color=(255,0,0), thickness=2, lineType=cv2.LINE_AA)
                        
        cv2.imshow('draw', image_copy)

    def q12(self):

        im = self.image
        imgray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(imgray, 127, 255, 0)

        contours, hierarchy = cv2.findContours(image=thresh, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)
                                            
        image_copy = im.copy()
        cv2.drawContours(image=image_copy, contours=contours, contourIdx=-1, color=(255,0,0), thickness=2, lineType=cv2.LINE_AA)
                        
        circles = cv2.HoughCircles(imgray, cv2.HOUGH_GRADIENT, 1.2, 100)
        print("number of circle:",len(circles[0]))

    def q21(self):

        for i in range(1,16):

            img=cv2.imread(self.folder_path+'/'+str(i)+'.bmp')
            size=(11,8)

            ret, corners = cv2.findChessboardCorners(img, size, None)
            cv2.drawChessboardCorners(img, size, corners, ret)

            image = cv2.resize(img, (1024,1024), interpolation=cv2.INTER_AREA)
            cv2.imshow('Result', image) 
            cv2.waitKey(500)

    def q22(self):

        CHECKERBOARD = (11,8)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        
        objpoints = []
        imgpoints = [] 
        gray=[]
        img=[]

        objp = np.zeros((1, CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
        objp[0,:,:2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
        prev_img_shape = None
        

        images = glob.glob(self.folder_path+'/*.bmp')
        h=0
        for fname in images:
            img = cv2.imread(fname)
            gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)
            
            if ret == True:
                objpoints.append(objp)
                corners2 = cv2.cornerSubPix(gray, corners, (11,11),(-1,-1), criteria)
                
                imgpoints.append(corners2)
        
                img = cv2.drawChessboardCorners(img, CHECKERBOARD, corners2, ret)
            
            cv2.waitKey(0)
            h+=1
        print(h)
        ret, self.mtx, self.dist, self.rvecs, self.tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
        print("Intrinsic Matrix : ")
        print(self.mtx)

    def q23(self):
        num=int(self.mycombobox.currentText())
        R = cv2.Rodrigues(self.rvecs[num-1])
        ext = np.hstack((R[0], self.tvecs[num-1]))
        print("Extrinsic Matrix of picture "+str(num)+ " : ")
        print(ext)

    def q24(self):
        print("Distortion Matrix : ")
        print(self.dist)

    def q25(self):
        for i in range(1,16):
            img = cv2.imread(self.folder_path+'/'+str(i)+'.bmp')
            image2=img
            h,  w = img.shape[:2]
            newcameramtx, roi = cv2.getOptimalNewCameraMatrix(self.mtx, self.dist, (w,h), 1, (w,h))
            dst = cv2.undistort(img, self.mtx, self.dist, None, newcameramtx)
            s=256*3
            x, y, w, h = roi
            dst = dst[y:y+h, x:x+w]
            image1 = cv2.resize(dst, (s,s), interpolation=cv2.INTER_AREA)
            image2 = cv2.resize(image2, (s,s), interpolation=cv2.INTER_AREA)
            image=np.concatenate((image1,image2),axis=1)
            cv2.imshow('Result', image) 
            cv2.waitKey(500)
            # break
    def q31(self):
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        objp = np.zeros((8 * 11, 3), np.float32)
        objp[:, :2] = np.mgrid[0:11, 0:8].T.reshape(-1, 2)
        # word="CAMARA"
        word=self.textbox.text()
        objpoints = [] 
        imgpoints = []  
        chess_images = glob.glob(self.folder_path+'/*.bmp')

        for i in range(len(chess_images)):
            image = cv2.imread(chess_images[i])

            
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            ret, corners = cv2.findChessboardCorners(gray, (11,8), None)

            if ret == True:
                objpoints.append(objp)
                corners2 = cv2.cornerSubPix(gray, corners, (7, 7), (-1, -1), criteria)
                imgpoints.append(corners2)
                ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, (2048, 2048), None, None)
                
                def draw(image, imgpts):
                    vector = np.vectorize(np.int_)
                    for i in range(0,int(c)):
                        image = cv2.line(image, tuple(vector(imgpts[i*2].ravel())), tuple(vector(imgpts[i*2+1].ravel())), (0, 0, 255), 5)

                    return image
                for j in range(0,len(word)):    
                    fs = cv2.FileStorage(self.folder_path+'/Q2_lib/alphabet_lib_onboard.txt', cv2.FILE_STORAGE_READ)
                    chs = np.float32(fs.getNode(word[j]).mat()).reshape(-1, 3)
                    c=chs.size/6
                    tx=chs[:,1]
                    ty=chs[:,0]
                    tx=tx+5
                    ty=ty+7
                    if(j==1):
                        ty-=3

                    elif(j==2):
                        ty-=6

                    elif(j==3):
                        tx-=3

                    elif(j==4):
                        tx-=3
                        ty-=3

                    elif(j==5):
                        tx-=3
                        ty-=6

                    chs[:,1]=tx
                    chs[:,0]=ty
                    print(chs)
                    imgpts, jac = cv2.projectPoints(chs, rvecs[i], tvecs[i], mtx, dist)
                    img = draw(image, imgpts)
                img = cv2.resize(img, (1024, 1024), interpolation=cv2.INTER_AREA)
                cv2.imshow('img', img)
                cv2.waitKey(500)
                
    def q32(self):
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        objp = np.zeros((8 * 11, 3), np.float32)
        objp[:, :2] = np.mgrid[0:11, 0:8].T.reshape(-1, 2)
        # word="CAMARA"
        word=self.textbox.text()
        objpoints = [] 
        imgpoints = []  
        chess_images = glob.glob(self.folder_path+'/*.bmp')

        for i in range(len(chess_images)):
            image = cv2.imread(chess_images[i])

            
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            ret, corners = cv2.findChessboardCorners(gray, (11,8), None)

            if ret == True:
                objpoints.append(objp)
                corners2 = cv2.cornerSubPix(gray, corners, (7, 7), (-1, -1), criteria)
                imgpoints.append(corners2)
                ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, (2048, 2048), None, None)
                
                def draw(image, imgpts):
                    vector = np.vectorize(np.int_)
                    for i in range(0,int(c)):
                        image = cv2.line(image, tuple(vector(imgpts[i*2].ravel())), tuple(vector(imgpts[i*2+1].ravel())), (0, 0, 255), 5)

                    return image
                for j in range(0,len(word)):    
                    fs = cv2.FileStorage(self.folder_path+'/Q2_lib/alphabet_lib_vertical.txt', cv2.FILE_STORAGE_READ)
                    chs = np.float32(fs.getNode(word[j]).mat()).reshape(-1, 3)
                    c=chs.size/6
                    tx=chs[:,1]
                    ty=chs[:,0]
                    tx=tx+5
                    ty=ty+7
                    if(j==1):
                        ty-=3

                    elif(j==2):
                        ty-=6

                    elif(j==3):
                        tx-=3

                    elif(j==4):
                        tx-=3
                        ty-=3

                    elif(j==5):
                        tx-=3
                        ty-=6

                    chs[:,1]=tx
                    chs[:,0]=ty
                    print(chs)
                    imgpts, jac = cv2.projectPoints(chs, rvecs[i], tvecs[i], mtx, dist)
                    img = draw(image, imgpts)
                img = cv2.resize(img, (1024, 1024), interpolation=cv2.INTER_AREA)
                cv2.imshow('img', img)
                cv2.waitKey(500)

    def q41(self):

        imgL=self.image
        imgR=self.image2

        imgL=cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
        imgR=cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)

        stereo = cv2.StereoBM_create(numDisparities=256, blockSize=25)
        disparity = stereo.compute(imgL, imgR).astype(np.float32) / 16.0
        disparity = cv2.resize(disparity, (1024, 256*3), interpolation=cv2.INTER_AREA)
        cv2.imshow('disparity', disparity)

cv2.waitKey(0)
cv2.destroyAllWindows()

App = QApplication(sys.argv)
window = Window()
sys.exit(App.exec())