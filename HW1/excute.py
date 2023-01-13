import sys
from PyQt5.QtWidgets import (QApplication, QWidget, QPushButton)
from PyQt5.QtWidgets import QFileDialog
import sys
import cv2
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QGridLayout, QWidget
from PyQt5.QtGui import QPixmap
import numpy as np
import matplotlib.pyplot as plt
import glob
class MyWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
        path=' '
        self.origin_img=[]
        path2=' '
        self.origin_img2=[]

    def initUI(self):

        self.setWindowTitle('1.Image Processing')
        self.setGeometry(50, 50, 500, 500)

        self.mybutton = QPushButton('1.1 Color Seperate', self)
        self.mybutton.move(130, 205)
        self.mybutton.clicked.connect(self.one)

        self.mybutton2 = QPushButton('1.2 Color Transform', self)
        self.mybutton2.move(130, 235)
        self.mybutton2.clicked.connect(self.two)

        self.mybutton3 = QPushButton('1.3 Color Detection', self)
        self.mybutton3.move(130, 265)
        self.mybutton3.clicked.connect(self.three)

        self.mybutton4 = QPushButton('1.4 Blending', self)
        self.mybutton4.move(130, 295)
        self.mybutton4.clicked.connect(self.four)

        self.mybutton5 = QPushButton('2.1 Gaussian Blur', self)
        self.mybutton5.move(280, 205)
        self.mybutton5.clicked.connect(self.five)

        self.mybutton6 = QPushButton('2.2 Bilateral Filter', self)
        self.mybutton6.move(280, 235)
        self.mybutton6.clicked.connect(self.six)

        self.mybutton7 = QPushButton('2.3 Median Filter', self)
        self.mybutton7.move(280, 265)
        self.mybutton7.clicked.connect(self.seven)

        self.mybutton = QPushButton('Load image', self)
        self.mybutton.move(20, 235)
        self.mybutton.clicked.connect(self.all)

        self.mybutton = QPushButton('Load image2', self)
        self.mybutton.move(20, 285)
        self.mybutton.clicked.connect(self.all2)
    
    def all(self):
        self.path = QFileDialog.getOpenFileName(self, 'Open a file', '','')
        self.im = QPixmap(self.path[0])
        self.label = QLabel()
        self.label.setPixmap(self.im)

        # self.grid = QGridLayout()
        # self.grid.addWidget(self.label,1,1)
        # self.setLayout(self.grid)

        # self.setGeometry(50,50,320,200)
        # self.setWindowTitle("PyQT show image")
        # self.show()
        
        self.origin_img = cv2.imread(self.path[0],flags=1)

    def all2(self):
        self.path2 = QFileDialog.getOpenFileName(self, 'Open a file', '','')
        self.im2 = QPixmap(self.path2[0])
        self.label2 = QLabel()
        self.label2.setPixmap(self.im2)

        # self.grid = QGridLayout()
        # self.grid.addWidget(self.label,1,1)
        # self.setLayout(self.grid)

        # self.setGeometry(50,50,320,200)
        # self.setWindowTitle("PyQT show image")
        # self.show()
        
        self.origin_img2 = cv2.imread(self.path2[0],flags=1)   
        

    def one(self):

        # self.path = QFileDialog.getOpenFileName(self, 'Open a file', '','')
        # self.im = QPixmap(self.path[0])
        # self.label = QLabel()
        # self.label.setPixmap(self.im)

        # self.grid = QGridLayout()
        # self.grid.addWidget(self.label,1,1)
        # self.setLayout(self.grid)

        # self.setGeometry(50,50,320,200)
        # self.setWindowTitle("PyQT show image")
        # self.show()
        
        # self.origin_img = cv2.imread(self.path[0],flags=1)
        
        self.split_RGBThreeChannel(self.origin_img)

    def two(self):

        # self.path = QFileDialog.getOpenFileName(self, 'Open a file', '','')
        # self.im = QPixmap(self.path[0])
        # self.label = QLabel()
        # self.label.setPixmap(self.im)

        # self.grid = QGridLayout()
        # self.grid.addWidget(self.label,1,1)
        # self.setLayout(self.grid)

        # self.setGeometry(50,50,320,200)
        # self.setWindowTitle("PyQT show image")
        # self.show()
        
        # self.origin_img = cv2.imread(self.path[0],flags=1)
        
        # self.split_RGBThreeChannel(self.origin_img)

        self.gray(self.origin_img)

    def three(self):

        # self.path = QFileDialog.getOpenFileName(self, 'Open a file', '','')
        # self.im = QPixmap(self.path[0])
        # self.label = QLabel()
        # self.label.setPixmap(self.im)

        # self.grid = QGridLayout()
        # self.grid.addWidget(self.label,1,1)
        # self.setLayout(self.grid)

        # self.setGeometry(50,50,320,200)
        # self.setWindowTitle("PyQT show image")
        # self.show()
        
        # self.origin_img = cv2.imread(self.path[0],flags=1)
        
        self.detection(self.origin_img)

    def split_RGBThreeChannel(self,img):
        
        B, G, R = cv2.split(img) 
        zeros = np.zeros(img.shape[:2], dtype = np.uint8)

        self.a=self.merge_RGBThreeChannel(R=R, G=zeros, B=zeros)
        self.show_img(self.a)

        self.b=self.merge_RGBThreeChannel(R=zeros, G=G, B=zeros)
        self.show_img(self.b)

        self.c=self.merge_RGBThreeChannel(R=zeros, G=zeros, B=B)
        self.show_img(self.c)

        self.out=np.concatenate((self.a,self.b,self.c),axis=1)
        self.show_img(self.out)

        return R, G, B

    def merge_RGBThreeChannel(self,R, G, B):

        img = cv2.merge([B, G, R])

        return img

    def show_img(self,img):

        image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.imshow(image_rgb)
        plt.show()

    def show_histogram(self,img):

        color = ('b','g','r')
        plt.style.use('dark_background')
        plt.figure(figsize=(10,5))
        for idx, color in enumerate(color):
            histogram = cv2.calcHist([img],[idx],None,[256],[0, 256])
            plt.plot(histogram, color = color)
            plt.xlim([0, 256])

        plt.show()

    def gray(self,img):

        w = np.array([[[0.07, 0.72,  0.21]]])
        self.img_gray1 = cv2.convertScaleAbs(np.sum(img*w, axis=2))

        w = np.array([[[ 1/3, 1/3,  1/3]]])
        self.img_gray2 = cv2.convertScaleAbs(np.sum(img*w, axis=2))

        self.out2=np.concatenate((self.img_gray1,self.img_gray2),axis=1)
        self.show_img(self.out2)
    
    def detection(self,img):
        
        GREEN_MIN = np.array([(40, 50,20) ],np.uint8)
        GREEN_MAX = np.array([(80, 255,255)],np.uint8)
        hsv_img = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
        self.frame_threshed = cv2.inRange(hsv_img, GREEN_MIN, GREEN_MAX)
        self.masked_data = cv2.bitwise_and(img, img, mask=self.frame_threshed)

        WHITE_MIN = np.array([(0, 0,200) ],np.uint8)
        WHITE_MAX = np.array([(180, 20,255)],np.uint8)
        hsv_img2 = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
        self.frame_threshed2 = cv2.inRange(hsv_img2, WHITE_MIN, WHITE_MAX)
        self.masked_data2 = cv2.bitwise_and(img, img, mask=self.frame_threshed2)
        
        self.out3=np.concatenate((self.masked_data,self.masked_data2),axis=1)
        self.show_img(self.out3)
    
    def nothing(self,x):
        pass

    def four(self):
        self.origin_img
        # self.img1=cv2.imread('./Dog_Strong.jpg')
        # self.img2=cv2.imread('./Dog_Weak.jpg')
        
        self.img1=self.origin_img
        self.img2=self.origin_img2

        self.dst=cv2.addWeighted(self.img1,0.7,self.img2,0.3,0)
        
        cv2.namedWindow('image')
        cv2.createTrackbar('a','image',0,100,self.nothing)
        cv2.setTrackbarPos('a','image', 5)

        while(1):
            cv2.imshow('image',self.dst)
            k = cv2.waitKey(100) 
            if k == 27:
                cv2.destroyAllWindows()
                break        
            if cv2.getWindowProperty('image',cv2.WND_PROP_VISIBLE) < 1:        
                break   

            r = cv2.getTrackbarPos('a','image')
            r=float(r)/100.0
            self.dst=cv2.addWeighted(self.img1,r,self.img2,1.0-r,0)

        cv2.destroyAllWindows()


    def five(self):
        cv2.namedWindow('image')
        cv2.createTrackbar('a','image',0,10,self.nothing)
        cv2.setTrackbarPos('a','image', 1)
        img = cv2.imread('image1.jpg')
        output1 = cv2.blur(img, (5, 5)) 

        while(1):
            cv2.imshow('image', output1)
            k = cv2.waitKey(100) 
            if k == 27:
                cv2.destroyAllWindows()
                break        
            if cv2.getWindowProperty('image',cv2.WND_PROP_VISIBLE) < 1:        
                break  

            r = cv2.getTrackbarPos('a','image')
            output1 = cv2.blur(img, (2*r+1, 2*r+1)) 
              
        cv2.destroyAllWindows()

    def six(self):

        cv2.namedWindow('image')
        cv2.createTrackbar('a','image',0,10,self.nothing)
        cv2.setTrackbarPos('a','image', 1)
        img = cv2.imread('image1.jpg')
        output1 = cv2.bilateralFilter(img, 9, 5, 5)

        while(1):
            cv2.imshow('image', output1)
            k = cv2.waitKey(100) 
            if k == 27:
                cv2.destroyAllWindows()
                break        
            if cv2.getWindowProperty('image',cv2.WND_PROP_VISIBLE) < 1:        
                break  

            r = cv2.getTrackbarPos('a','image')
            output1 = cv2.bilateralFilter(img, 2*r+1,90, 90)            
              
        cv2.destroyAllWindows()

    def seven(self):

        cv2.namedWindow('image')
        cv2.createTrackbar('a','image',0,10,self.nothing)
        cv2.setTrackbarPos('a','image', 1)
        img = cv2.imread('image2.jpg')
        output1 = cv2.medianBlur(img, 3)

        while(1):
            cv2.imshow('image', output1)
            k = cv2.waitKey(100) 
            if k == 27:
                cv2.destroyAllWindows()
                break        
            if cv2.getWindowProperty('image',cv2.WND_PROP_VISIBLE) < 1:        
                break  

            r = cv2.getTrackbarPos('a','image')
            output1 = cv2.medianBlur(img, 2*r+1)           
              
        cv2.destroyAllWindows()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    w = MyWidget()
    w.show()
    sys.exit(app.exec_())
 


    