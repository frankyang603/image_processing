from PyQt5.QtGui import * 
from PyQt5.QtWidgets import * 
from PyQt5.QtCore import *
from email.mime import image

import numpy as np
import cv2
import sys  
import math

class Window(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("2022 Opencvdl Hw1")
        self.setGeometry(50, 50, 800, 550)

        self.path = ''
        self.image = [[],[]]

        self.ans = []
        self.ans_x = []
        self.ans_y = []
        self.ans_m = []
        
        self.UiComponents()
        self.show()
  
    def UiComponents(self):
  
        button_load_image_1 = QPushButton("Load Image 1", self)
        button_load_image_1.setGeometry(80, 220, 150, 80)
        button_load_image_1.clicked.connect(lambda: self.load_img(0))

        button_one =  QPushButton("Gaussian Blur", self)
        button_one.setGeometry(300, 100, 160, 60)
        button_one.clicked.connect(self.show_gaussian_blur)

        button_two =  QPushButton("Sobel X", self)
        button_two.setGeometry(300, 200, 160, 60)
        button_two.clicked.connect(self.show_sobel_x)

        button_third =  QPushButton("Sobel Y", self)
        button_third.setGeometry(300, 300, 160, 60)
        button_third.clicked.connect(self.show_sobel_y)

        button_fourth =  QPushButton("Magnitude", self)
        button_fourth.setGeometry(300, 400, 160, 60)
        button_fourth.clicked.connect(self.magnitude)

        button_five =  QPushButton("Resize", self)
        button_five.setGeometry(550, 100, 160, 60)
        button_five.clicked.connect(self.resize)

        button_six =  QPushButton("Translation", self)
        button_six.setGeometry(550, 200, 160, 60)
        button_six.clicked.connect(self.translation)

        button_seven =  QPushButton("Rotation, Scaling", self)
        button_seven.setGeometry(550, 300, 160, 60)
        button_seven.clicked.connect(self.rotation_scaling)

        button_eight =  QPushButton("Shearing", self)
        button_eight.setGeometry(550, 400, 160, 60)
        button_eight.clicked.connect(self.shearing)

    def load_img(self,n):

        self.path,_ = QFileDialog.getOpenFileName(self, 'Open a file', '', 'All Files (*.*)')
        self.image[n] = cv2.imread(self.path, flags=1)

    def gaussian_blur(self):
        
        gray = cv2.cvtColor(self.image[0], cv2.COLOR_BGR2GRAY)

        filter=[0.045,0.122,0.045,0.122,0.332,0.122,0.045,0.122,0.045] 

        self.ans=np.zeros_like(gray)

        for i in range(1,gray.shape[0]-1):
            for j in range(1,gray.shape[1]-1):
                cal = 0
                cal += gray[i-1][j-1]*filter[0]
                cal += gray[i-1][j]*filter[1]
                cal += gray[i-1][j+1]*filter[2]
                cal += gray[i][j-1]*filter[3]
                cal += gray[i][j]*filter[4]
                cal += gray[i][j+1]*filter[5]
                cal += gray[i+1][j-1]*filter[6]
                cal += gray[i+1][j]*filter[7]
                cal += gray[i+1][j+1]*filter[8]

                self.ans[i][j] = cal
                
        self.ans = self.ans.astype(np.uint8)
    
    def show_gaussian_blur(self):

        self.gaussian_blur()
        cv2.imshow('', self.ans)

    def show_sobel_x(self):

        self.gaussian_blur()

        filter=[-1,0,1,-2,0,2,-1,0,1] 
        self.ans_x=np.zeros_like(self.ans)

        for i in range(1,self.ans.shape[0]-1):
            for j in range(1,self.ans.shape[1]-1):
                cal = 0
                cal += self.ans[i-1][j-1]*filter[0]
                cal += self.ans[i-1][j]*filter[1]
                cal += self.ans[i-1][j+1]*filter[2]
                cal += self.ans[i][j-1]*filter[3]
                cal += self.ans[i][j]*filter[4]
                cal += self.ans[i][j+1]*filter[5]
                cal += self.ans[i+1][j-1]*filter[6]
                cal += self.ans[i+1][j]*filter[7]
                cal += self.ans[i+1][j+1]*filter[8]

                self.ans_x[i][j] = cal

        self.ans_x = self.ans_x.astype(np.uint8)

        cv2.imshow('', self.ans_x)


    def show_sobel_y(self):

        self.gaussian_blur()

        filter=[1,2,1,0,0,0,-1,-2,-1] 
        self.ans_y=np.zeros_like(self.ans)

        for i in range(1,self.ans.shape[0]-1):
            for j in range(1,self.ans.shape[1]-1):
                cal = 0
                cal += self.ans[i-1][j-1]*filter[0]
                cal += self.ans[i-1][j]*filter[1]
                cal += self.ans[i-1][j+1]*filter[2]
                cal += self.ans[i][j-1]*filter[3]
                cal += self.ans[i][j]*filter[4]
                cal += self.ans[i][j+1]*filter[5]
                cal += self.ans[i+1][j-1]*filter[6]
                cal += self.ans[i+1][j]*filter[7]
                cal += self.ans[i+1][j+1]*filter[8]

                self.ans_y[i][j] = cal

        self.ans_y = self.ans_y.astype(np.uint8)

        cv2.imshow('', self.ans_y)

    def magnitude(self):

        # self.sobel_x()
        self.gaussian_blur()

        filter=[-1,0,1,-2,0,2,-1,0,1] 
        self.ans_x=np.zeros_like(self.ans)

        for i in range(1,self.ans.shape[0]-1):
            for j in range(1,self.ans.shape[1]-1):
                cal = 0
                cal += self.ans[i-1][j-1]*filter[0]
                cal += self.ans[i-1][j]*filter[1]
                cal += self.ans[i-1][j+1]*filter[2]
                cal += self.ans[i][j-1]*filter[3]
                cal += self.ans[i][j]*filter[4]
                cal += self.ans[i][j+1]*filter[5]
                cal += self.ans[i+1][j-1]*filter[6]
                cal += self.ans[i+1][j]*filter[7]
                cal += self.ans[i+1][j+1]*filter[8]

                self.ans_x[i][j] = cal

        self.ans_x = self.ans_x.astype(np.uint8)

        self.gaussian_blur()

        filter=[1,2,1,0,0,0,-1,-2,-1] 
        self.ans_y=np.zeros_like(self.ans)

        for i in range(1,self.ans.shape[0]-1):
            for j in range(1,self.ans.shape[1]-1):
                cal = 0
                cal += self.ans[i-1][j-1]*filter[0]
                cal += self.ans[i-1][j]*filter[1]
                cal += self.ans[i-1][j+1]*filter[2]
                cal += self.ans[i][j-1]*filter[3]
                cal += self.ans[i][j]*filter[4]
                cal += self.ans[i][j+1]*filter[5]
                cal += self.ans[i+1][j-1]*filter[6]
                cal += self.ans[i+1][j]*filter[7]
                cal += self.ans[i+1][j+1]*filter[8]

                self.ans_y[i][j] = cal

        self.ans_y = self.ans_y.astype(np.uint8)


        # self.sobel_y()
        self.ans_m=np.zeros_like(self.ans_x)

        for i in range(self.ans_x.shape[0]):
            for j in range(self.ans_x.shape[1]):
                self.ans_m[i][j] = math.sqrt(self.ans_x[i][j]**2+self.ans_y[i][j]**2)

        self.ans_m = self.ans_m.astype(np.uint8)

        cv2.imshow('', self.ans_m)

    def resize(self):

        origin = np.float32([[0, 0],[0, self.image[0].shape[0]],[self.image[0].shape[1], 0]]) 

        r = np.float32([[0, 0],[0, 215],[215, 0]]) 

        trans = cv2.getAffineTransform(origin, r) 

        ans_r = cv2.warpAffine(self.image[0], trans, (self.image[0].shape[1],self.image[0].shape[0]))

        cv2.imshow('', ans_r)

    def translation(self):

        origin = np.float32([[0, 0],[0, self.image[0].shape[0]],[self.image[0].shape[1], 0]])  

        resize_marix1 = np.float32([[0, 0],[0, 215],[215, 0]])
        resize_marix2 = np.float32([[215, 215],[215, 430],[430, 215]])

        trans_1 = cv2.getAffineTransform(origin, resize_marix1)
        trans_2 = cv2.getAffineTransform(origin, resize_marix2)

        ans_r = cv2.warpAffine(self.image[0], trans_1, (self.image[0].shape[1],self.image[0].shape[0]))
        ans_r += cv2.warpAffine(self.image[0], trans_2, (self.image[0].shape[1],self.image[0].shape[0]))

        cv2.imshow('', ans_r)

    def rotation_scaling(self):

        origin = np.float32([[0, 0],[0, self.image[0].shape[0]],[self.image[0].shape[1], 0]])  

        resize_marix1 = np.float32([[0, 0],[0, 215],[215, 0]])
        resize_marix2 = np.float32([[215, 215],[215, 430],[430, 215]])
 
        trans_1 = cv2.getAffineTransform(origin, resize_marix1)
        trans_2 = cv2.getAffineTransform(origin, resize_marix2)

        ans_r = cv2.warpAffine(self.image[0], trans_1, (self.image[0].shape[1],self.image[0].shape[0]))
        ans_r += cv2.warpAffine(self.image[0], trans_2, (self.image[0].shape[1],self.image[0].shape[0]))

        mat_rotate = cv2.getRotationMatrix2D((ans_r.shape[1]/2,ans_r.shape[0]/2),45,0.5)
        out_rotate = cv2.warpAffine(ans_r,mat_rotate,(ans_r.shape[1],ans_r.shape[0]))

        cv2.imshow('', out_rotate)

    def shearing(self):

        origin = np.float32([[0, 0],[0, self.image[0].shape[0]],[self.image[0].shape[1], 0]])  

        resize_marix1 = np.float32([[0, 0],[0, 215],[215, 0]])
        resize_marix2 = np.float32([[215, 215],[215, 430],[430, 215]])
 
        trans_1 = cv2.getAffineTransform(origin, resize_marix1)
        trans_2 = cv2.getAffineTransform(origin, resize_marix2)

        ans_r = cv2.warpAffine(self.image[0], trans_1, (self.image[0].shape[1],self.image[0].shape[0]))
        ans_r += cv2.warpAffine(self.image[0], trans_2, (self.image[0].shape[1],self.image[0].shape[0]))

        mat_rotate = cv2.getRotationMatrix2D((ans_r.shape[1]/2,ans_r.shape[0]/2),45,0.5)
        out_rotate = cv2.warpAffine(ans_r,mat_rotate,(ans_r.shape[1],ans_r.shape[0]))

        old = np.float32([[[50,50],[200,50],[50,200]]])  
        new = np.float32([[10,100],[100,50],[100,250]])
 
        shearing_matrix = cv2.getAffineTransform(old, new)
        ans_s = cv2.warpAffine(out_rotate, shearing_matrix, (self.image[0].shape[1],self.image[0].shape[0]))

        cv2.imshow('', ans_s)
    
App = QApplication(sys.argv)
window = Window()
sys.exit(App.exec())