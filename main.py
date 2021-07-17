################################################################################
##
## BY: WANDERSON M.PIMENTA
## PROJECT MADE WITH: Qt Designer and PySide2
## V: 1.0.0
##
## This project can be used freely for all uses, as long as they maintain the
## respective credits only in the Python scripts, any information in the visual
## interface (GUI) can be modified without any implication.
##
## There are limitations on Qt licenses if you want to use your products
## commercially, I recommend reading them on the official website:
## https://doc.qt.io/qtforpython/licenses.html
##
################################################################################

import sys
import os
import math
from skimage.io import imsave
import matplotlib.pyplot as plt
import platform
from tensorflow.keras import layers, models
import glob
import cv2 as cv
from builtins import staticmethod
from datetime import datetime

import tensorflow as tf
import numpy as np
import random
from PySide2 import QtCore, QtGui, QtWidgets
from PIL import Image
from PySide2.QtCore import (QCoreApplication, QPropertyAnimation, QDate, QDateTime, QMetaObject, QObject, QPoint, QRect, QSize, QTime, QUrl, Qt, QEvent)
from PySide2.QtGui import (QBrush, QColor, QConicalGradient, QCursor, QFont, QFontDatabase, QIcon, QKeySequence, QLinearGradient, QPalette, QPainter, QPixmap, QRadialGradient)
from PySide2.QtWidgets import *

# GUI FILE
from app_modules import *
from ui_main import Ui_MainWindow


class MainWindow(QMainWindow):
    def __init__(self):
        QMainWindow.__init__(self)
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.folder_path = None
        self.original_img = None
        self.noised_img = None
        self.denoised_img = None
        self.type_img = "rgb"
        self.type_noise = "gauss"
        self.noise_intensity = 0.0

        UIFunctions.removeTitleBar(True)
        self.setWindowTitle('AI Denoise - Graduation application')
        UIFunctions.labelTitle(self, 'AI Denoise - Graduation application')
        UIFunctions.labelDescription(self, 'AI - Denoise')
        startSize = QSize(1000, 750)
        self.resize(startSize)
        self.setMinimumSize(startSize)

        self.ui.tableWidget.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
        self.ui.tableWidget.setSelectionBehavior(QTableView.SelectRows)
        self.ui.tableWidget.viewport().installEventFilter(self)

        self.ui.btn_toggle_menu.clicked.connect(lambda: UIFunctions.toggleMenu(self, 220, True))
        self.ui.stackedWidget.setMinimumWidth(20)
        UIFunctions.addNewMenu(self, "Home", "btn_home", "url(:/16x16/icons/16x16/cil-home.png)", True)
        UIFunctions.addNewMenu(self, "History", "btn_history", "url(:/16x16/icons/16x16/cil-check-circle.png)", True)
        UIFunctions.addNewMenu(self, "Settings", "btn_settings", "url(:/16x16/icons/16x16/cil-equalizer.png)", False)
        self.ui.btn_open_image.clicked.connect(self.Button)
        self.ui.btn_generate_noise.clicked.connect(self.Button)
        self.ui.btn_classification.clicked.connect(self.Button)
        self.ui.btn_denoising.clicked.connect(self.Button)
        self.ui.slider_intensity.valueChanged.connect(self.slider_listener)


        UIFunctions.selectStandardMenu(self, "btn_home")
        self.ui.stackedWidget.setCurrentWidget(self.ui.page_home)
        UIFunctions.userIcon(self, "AI", "", True)

        # move window method
        def moveWindow(event):
            # IF MAXIMIZED CHANGE TO NORMAL
            if UIFunctions.returStatus() == 1:
                UIFunctions.maximize_restore(self)

            # MOVE WINDOW
            if event.buttons() == Qt.LeftButton:
                self.move(self.pos() + event.globalPos() - self.dragPos)
                self.dragPos = event.globalPos()
                event.accept()

        self.ui.frame_label_top_btns.mouseMoveEvent = moveWindow
        UIFunctions.uiDefinitions(self)
        self.show()

    # Add Salt-and-Pepper noise to an image.
    @staticmethod
    def sp_noise(image, probability):
        output = np.copy(image)
        x = 1 - probability
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                rand = random.random()
                if rand < probability:
                    if random.randint(0, 1) == 0:
                        output[i][j] = 0
                    else:
                        output[i][j] = 255
                elif rand > x:
                    output[i][j] = image[i][j]
        return output

    # Add gauss noise to an image.
    @staticmethod
    def gauss_noise(image, probability):
        # generate noise
        noise = np.random.normal(loc=0, scale=1, size=image.shape)

        # noise overlaid over image
        result = np.clip((image + noise * probability), 0, 1)

        return result

    # Extract patch from image matrix form AMF filter
    @staticmethod
    def _extract_patch(matrix, x, y, patch_size=3):
        height, width = matrix.shape
        size = patch_size // 2

        # initialize x index
        if x - size >= 0:
            x_begin = x - size
        else:
            x_begin = 0

        if x + size < height:
            x_end = x + size
        else:
            x_end = height - 1

        # initialize y index
        if y - size >= 0:
            y_begin = y - size
        else:
            y_begin = 0

        if y + size < width:
            y_end = y + size
        else:
            y_end = width - 1

        # loop inside patch
        output = []
        for i in range(x_begin, x_end + 1):
            for j in range(y_begin, y_end + 1):
                output.append(matrix[i][j])
        return output

    # Adaptive median filter function
    @staticmethod
    def amf(matrix, max_patch_size=15):
        # prepare output
        output = np.copy(matrix)
        height, width = matrix.shape

        for x in range(height):
            for y in range(width):
                patch_size = 3
                patch = MainWindow._extract_patch(matrix, x, y, patch_size)

                # extract min, max and median value of patch
                patch_min = np.min(patch)
                patch_max = np.max(patch)
                patch.sort()
                patch_median = patch[len(patch) // 2]

                # check if pixel is corrupted
                if patch_min < matrix[x][y] < patch_max:
                    output[x][y] = matrix[x][y]
                else:
                    # check if median value is also corrupted
                    finish = False
                    while not finish:
                        if 0 < patch_median < 255:
                            output[x][y] = patch_median
                            finish = True
                        else:
                            # calculate new patch
                            patch_size = patch_size + 2
                            if patch_size <= max_patch_size:
                                patch = MainWindow._extract_patch(matrix, x, y, patch_size)
                                patch.sort()
                                patch_median = patch[len(patch) // 2]
                            else:
                                finish = True

        return output

    # Extract patches from grayscale image with resize
    @staticmethod
    def extract_patches_gray(image, patch_size):
        # resize image so that it's dimensions are dividable by patch_height and patch_width
        h, w = image.shape
        height = math.ceil(h / patch_size) * patch_size
        width = math.ceil(w / patch_size) * patch_size
        image = cv.resize(image, (height, width))

        patches = tf.image.extract_patches(images=tf.expand_dims(image[:, :, np.newaxis], 0),
                                           sizes=[1, patch_size, patch_size, 1],
                                           strides=[1, patch_size, patch_size, 1],
                                           rates=[1, 1, 1, 1],
                                           padding='SAME')
        _, x, y, _ = patches.shape
        list = []
        for imgs in patches:
            for r in range(x):
                for c in range(y):
                    list.append(np.array(tf.reshape(imgs[r, c], shape=(patch_size, patch_size, 1))))
        return list

    # Extract patches from grayscale image without resize
    @staticmethod
    def image_to_patches_gray(image, patch_size=64):
        W = image.shape[0]
        H = image.shape[1]
        if W < patch_size or H < patch_size:
            return []

        ret = []
        for ws in range(0, W // patch_size):
            for hs in range(0, H // patch_size):
                patch = image[ws * patch_size: (ws + 1) * patch_size, hs * patch_size: (hs + 1) * patch_size]
                ret.append(patch)
        return ret

    # Extract patches from rgb image without resize
    @staticmethod
    def image_to_patches(image, patch_size=64):
        W = image.shape[0]
        H = image.shape[1]
        if W < patch_size or H < patch_size:
            return []

        ret = []
        for ws in range(0, W // patch_size):
            for hs in range(0, H // patch_size):
                patch = image[ws * patch_size: (ws + 1) * patch_size, hs * patch_size: (hs + 1) * patch_size]
                ret.append(patch)
        return ret

    # Reconstruct image from patches
    @staticmethod
    def reconstruct_patches(patches):
        hstack = []
        count = 0
        for patch in patches:
            if count == 0:
                stack = patch
                count += 1
            elif count < math.sqrt(len(patches)):
                stack = np.hstack((stack, patch))
                count += 1
            else:
                hstack.append(stack)
                count = 1
                stack = patch
        hstack.append(stack)
        if math.sqrt(len(patches)).is_integer():
            count = 0
            for stack in hstack:
                if count == 0:
                    vstack = stack
                else:
                    vstack = np.vstack((vstack, stack))
                count += 1
        else:
            count = 0
            for stack in hstack:
                if count == 0:
                    vstack = stack
                else:
                    vstack = np.hstack((vstack, stack))
                count += 1
        return vstack

    # image prediction from auto-encoder de-noising
    @staticmethod
    def predict(noised_patches, model):
        patches = []
        for patch in noised_patches:
            patch = tf.expand_dims(patch, axis=0)
            patches.append(model.predict(patch)[0])
        return MainWindow.reconstruct_patches(patches)

    # calculate PSNR
    def calculate_psnr(self, image1, image2):
        if self.type_noise == "gauss":
            if self.type_img == "RGB":
                image1_patches = MainWindow.image_to_patches(image1, 64)
                image2_patches = MainWindow.image_to_patches(image2, 64)
            else:
                image1_patches = MainWindow.image_to_patches_gray(image1, 64)
                image2_patches = MainWindow.image_to_patches_gray(image2, 64)
        else:
            image1_patches = MainWindow.image_to_patches_gray(image1, 40)
            image2_patches = MainWindow.image_to_patches_gray(image2, 40)

        psnr = 0
        for i in range(len(image1_patches)):
            psnr += cv.PSNR(image1_patches[i], image2_patches[i])

        return psnr / len(image1_patches)

    # Save image
    def save_image(self, image, directory, file_name):
        # save grayscale image
        if self.type_img == "grayscale":
            img = Image.fromarray(image)
            if not os.path.isdir(directory):
                os.makedirs(directory)
            img = img.convert("L")
            img.save(f"{directory}{file_name}.png")
        # save color image
        else:
            if not os.path.isdir(directory):
                os.makedirs(directory)
            imsave(f"{directory}{file_name}.png", image)

    # Grayscale classification
    def gray_predict(self):
        # load model
        class_names = ['gauss', 'none', 'sp']
        model = models.load_model('./models/noise_classification_gray.model')

        # loading & normalization of test image
        image_path = f"{self.folder_path}noised.png"
        target_size = (64, 64)
        image = tf.keras.preprocessing.image.load_img(image_path, color_mode="grayscale", target_size=target_size)
        image = tf.keras.preprocessing.image.img_to_array(image)
        image = image / 255.0
        image = tf.expand_dims(image, axis=0)

        # predict type of noise
        prediction = model.predict(image)
        noise = class_names[np.argmax(prediction)]
        self.ui.checkBox_noise_rand.setChecked(False)
        self.ui.checkBox_noise_sp.setChecked(False)
        self.ui.checkBox_noise_gauss.setChecked(False)
        self.type_noise = None

        if noise == "gauss":
            self.ui.checkBox_noise_gauss.setChecked(True)
            self.type_noise = "gauss"
        elif noise == "sp":
            self.ui.checkBox_noise_sp.setChecked(True)
            self.type_noise = "sp"
        else:
            pass

    # Color classification
    def color_predict(self):
        # load model
        class_names = ['gauss', 'none']
        model = models.load_model('./models/noise_classification_color.model')

        # loading & normalization of test image
        image_path = f"{self.folder_path}noised.png"
        target_size = (64, 64)
        image = tf.keras.preprocessing.image.load_img(image_path, color_mode="rgb", target_size=target_size)
        image = tf.keras.preprocessing.image.img_to_array(image)
        image = image / 255.0
        image = tf.expand_dims(image, axis=0)

        # predict type of noise
        prediction = model.predict(image)
        noise = class_names[np.argmax(prediction)]
        self.ui.checkBox_noise_rand.setChecked(False)
        self.ui.checkBox_noise_sp.setChecked(False)
        self.ui.checkBox_noise_gauss.setChecked(False)
        self.type_noise = None

        if noise == "gauss":
            self.ui.checkBox_noise_gauss.setChecked(True)
            self.type_noise = "gauss"
        else:
            pass

    # Display Alert box
    def alert(self, title, context):
        dlg = QMessageBox(self)
        dlg.setWindowTitle(title)
        dlg.setText(context)
        button = dlg.exec()

        if button == QMessageBox.Ok:
            print("OK!")

    # Classification loading model
    def classify_image(self):
        # check if noised image exists
        path = glob.glob(self.folder_path+"*")
        print(len(path))
        if len(path) <= 1:
            # save noised image as original image
            self.save_image(self.original_img, self.folder_path, "noised")
            self.ui.image_noised.setPixmap(self.folder_path + "noised.png")

        # check image type
        img = Image.open(f"{self.folder_path}noised.png")
        color_mode = img.mode
        self.ui.checkBox_type_rand.setChecked(False)
        self.ui.checkBox_type_gray.setChecked(False)
        self.ui.checkBox_type_rgb.setChecked(False)
        # image grayscale
        if color_mode == "L":
            self.type_img = "grayscale"
            self.ui.checkBox_type_gray.setChecked(True)
        # color image
        else:
            self.type_img = "rgb"
            self.ui.checkBox_type_rgb.setChecked(True)

        # classification
        if self.type_img == "grayscale":
            self.gray_predict()
        else:
            self.color_predict()

    # Image de-noising with auto-encoders
    def denoise_image(self):
        # color images
        if self.type_img == "rgb":
            if self.type_noise == "gauss":
                # denoise image with color gauss auto-encoder model
                self.load_model(model_key=1)
            else:
                # display error: image doesn't contain any noise
                self.alert("Image Denoising", "Image doesn't contain any noise!")
        # gray images
        else:
            if self.type_noise == "gauss":
                # denoise image with grayscale gauss auto-encoder model
                self.load_model(model_key=2)

            elif self.type_noise == "sp":
                # denoise image with grayscale sp auto-encoder model
                self.load_model(model_key=3)

            else:
                # display error: image doesn't contain any noise
                self.alert("Image Denoising", "Image doesn't contain any noise!")

    # loading auto-encoder model
    def load_model(self, model_key=1):
        # load original & noised images
        original_path = f"{self.folder_path}original.png"
        original_image = tf.keras.preprocessing.image.load_img(original_path, color_mode=self.type_img)
        original_image = tf.keras.preprocessing.image.img_to_array(original_image)

        noised_path = f"{self.folder_path}noised.png"
        noised_image = tf.keras.preprocessing.image.load_img(noised_path, color_mode=self.type_img)
        noised_image = tf.keras.preprocessing.image.img_to_array(noised_image)

        # load color gaussian de-noising auto-encoder
        if model_key == 1:
            model = models.load_model('./models/G-COLOR.model')
            original_image = original_image / 255.0
            noised_image = noised_image / 255.0
            patch_size = 64

            intensity = self.noise_intensity
            if self.noise_intensity >= 0.35:
                intensity -= 0.1
            else:
                intensity -= 0.05

            if self.noise_intensity == 0.0:
                noisy_image = noised_image
            else:
                noisy_image = self.gauss_noise(original_image, intensity)

            prediction = MainWindow.predict(MainWindow.image_to_patches(np.array(noisy_image), patch_size), model)
            prediction_image = (prediction * 255).clip(0, 255).astype(int)
            original_image = (MainWindow.reconstruct_patches(MainWindow.image_to_patches(original_image, patch_size)) * 255).astype(int)
            noised_image = (MainWindow.reconstruct_patches(MainWindow.image_to_patches(noised_image, patch_size)) * 255).astype(int)
            median5_image = cv.medianBlur(np.uint8(noised_image), 5)
            average_image = cv.blur(np.uint8(noised_image), (5, 5))
            gaussian_image = cv.GaussianBlur(np.uint8(noised_image), (5, 5), 0)

            self.save_image((original_image).astype(int), self.folder_path, "original")
            self.save_image((prediction_image).astype(int), self.folder_path, "auto-encoder")
            self.save_image((noised_image).astype(int), self.folder_path, "noised")
            self.save_image(median5_image, self.folder_path, "median5")
            self.save_image(average_image, self.folder_path, "average")
            self.save_image(gaussian_image, self.folder_path, "gaussian")

            # change name of execution folder
            now = datetime.now()
            dt_string = now.strftime("%d-%m-%Y %H-%M-%S")
            image_size = f"{np.array(prediction_image).shape[0]}x{np.array(prediction_image).shape[0]}"
            noise_intensity = str(int(self.noise_intensity * 100))
            new_path = f"./history/{dt_string}+{image_size}+RGB+Gaussian+{noise_intensity}/"
            os.rename(self.folder_path, new_path)
            self.folder_path = new_path

        # load gray gaussian de-noising auto-encoder
        elif model_key == 2:
            model = models.load_model('./models/G-GRAY.model')
            original_image = original_image / 255.0
            noised_image = noised_image / 255.0
            patch_size = 64

            intensity = self.noise_intensity
            if self.noise_intensity >= 0.35:
                intensity -= 0.2
            else:
                intensity -= 0.1

            if self.noise_intensity == 0.0:
                noisy_image = noised_image
            else:
                noisy_image = self.gauss_noise(original_image, intensity)

            prediction = MainWindow.predict(MainWindow.image_to_patches(np.array(noisy_image), patch_size), model)
            prediction_image = (prediction * 255).astype(int)
            original_image = (MainWindow.reconstruct_patches(MainWindow.image_to_patches(original_image[..., 0], patch_size)) * 255).astype(int)
            noised_image = (MainWindow.reconstruct_patches(MainWindow.image_to_patches(noised_image[..., 0], patch_size)) * 255).astype(int)
            median5_image = cv.medianBlur(np.float32(noised_image), 5)
            average_image = cv.blur(np.float32(noised_image), (5, 5))
            gaussian_image = cv.GaussianBlur(np.float32(noised_image), (5, 5), 0)

            self.save_image(np.array(original_image).astype(int), self.folder_path, "original")
            self.save_image(np.array(prediction_image[..., 0]).astype(int), self.folder_path, "auto-encoder")
            self.save_image(np.array(noised_image).astype(int), self.folder_path, "noised")
            self.save_image(median5_image, self.folder_path, "median5")
            self.save_image(average_image, self.folder_path, "average")
            self.save_image(gaussian_image, self.folder_path, "gaussian")

            # change name of execution folder
            now = datetime.now()
            dt_string = now.strftime("%d-%m-%Y %H-%M-%S")
            image_size = f"{np.array(prediction_image).shape[0]}x{np.array(prediction_image).shape[0]}"
            noise_intensity = str(int(self.noise_intensity * 100))
            new_path = f"./history/{dt_string}+{image_size}+GRAY+Gaussian+{noise_intensity}/"
            os.rename(self.folder_path, new_path)
            self.folder_path = new_path

        # load gray salt & pepper de-noising auto-encoder & AMF filter
        elif model_key == 3:
            model = models.load_model('./models/S-GRAY.model')
            patch_size = 40
            filtred_image = tf.expand_dims(MainWindow.amf(noised_image[...,0]), -1)
            prediction_image = MainWindow.predict(np.float32(MainWindow.extract_patches_gray(np.array(filtred_image[..., 0]) / 255, patch_size)), model)
            original_image = MainWindow.reconstruct_patches(MainWindow.extract_patches_gray(np.array(original_image[..., 0]), patch_size))
            amf_image = MainWindow.reconstruct_patches(MainWindow.extract_patches_gray(np.array(filtred_image[..., 0]), patch_size))
            noised_image = MainWindow.reconstruct_patches(MainWindow.extract_patches_gray(np.array(noised_image[..., 0]), patch_size))
            median3_image = cv.medianBlur(noised_image, 3)
            median5_image = cv.medianBlur(noised_image, 5)

            self.save_image(np.array(original_image)[..., 0].astype(int), self.folder_path, "original")
            self.save_image((np.array(prediction_image)[..., 0] * 255).astype(int), self.folder_path, "auto-encoder")
            self.save_image(np.array(noised_image)[..., 0].astype(int), self.folder_path, "noised")
            self.save_image(median5_image, self.folder_path, "median5")
            self.save_image(median3_image, self.folder_path, "median3")
            self.save_image(np.array(amf_image)[..., 0].astype(int), self.folder_path, "median_Filter")

            # change name of execution folder
            now = datetime.now()
            dt_string = now.strftime("%d-%m-%Y %H-%M-%S")
            image_size = f"{np.array(prediction_image).shape[0]}x{np.array(prediction_image).shape[0]}"
            noise_intensity = str(int(self.noise_intensity * 100))
            new_path = f"./history/{dt_string}+{image_size}+GRAY+Salt & pepper+{noise_intensity}/"
            os.rename(self.folder_path, new_path)
            self.folder_path = new_path

    # Display de-noising result images for gaussian noise
    def display_gauss_images(self):
        # load images
        # original image
        image_path = f"{self.folder_path}original.png"
        original_image = tf.keras.preprocessing.image.load_img(image_path, color_mode=self.type_img)
        original_image = tf.keras.preprocessing.image.img_to_array(original_image).astype(int)
        if self.type_img == "grayscale":
            original_image = original_image[..., 0]
        original_image = MainWindow.reconstruct_patches(MainWindow.image_to_patches(original_image, 64))

        # noised image
        image_path = f"{self.folder_path}noised.png"
        noised_image = tf.keras.preprocessing.image.load_img(image_path, color_mode=self.type_img)
        noised_image = tf.keras.preprocessing.image.img_to_array(noised_image).astype(int)
        if self.type_img == "grayscale":
            noised_image = noised_image[..., 0]
        noised_image = MainWindow.reconstruct_patches(MainWindow.image_to_patches(noised_image, 64))

        # de-noised image
        image_path = f"{self.folder_path}auto-encoder.png"
        denoised_image = tf.keras.preprocessing.image.load_img(image_path, color_mode=self.type_img)
        denoised_image = tf.keras.preprocessing.image.img_to_array(denoised_image).astype(int)
        if self.type_img == "grayscale":
            denoised_image = denoised_image[..., 0]
        denoised_image = MainWindow.reconstruct_patches(MainWindow.image_to_patches(denoised_image, 64))

        # gaussian image
        image_path = f"{self.folder_path}gaussian.png"
        gaussian_image = tf.keras.preprocessing.image.load_img(image_path, color_mode=self.type_img)
        gaussian_image = tf.keras.preprocessing.image.img_to_array(gaussian_image).astype(int)
        if self.type_img == "grayscale":
            gaussian_image = gaussian_image[..., 0]
        gaussian_image = MainWindow.reconstruct_patches(MainWindow.image_to_patches(gaussian_image, 64))

        # average image
        image_path = f"{self.folder_path}average.png"
        average_image = tf.keras.preprocessing.image.load_img(image_path, color_mode=self.type_img)
        average_image = tf.keras.preprocessing.image.img_to_array(average_image).astype(int)
        if self.type_img == "grayscale":
            average_image = average_image[..., 0]
        average_image = MainWindow.reconstruct_patches(MainWindow.image_to_patches(average_image, 64))

        # median 5x5 image
        image_path = f"{self.folder_path}median5.png"
        median5_image = tf.keras.preprocessing.image.load_img(image_path, color_mode=self.type_img)
        median5_image = tf.keras.preprocessing.image.img_to_array(median5_image).astype(int)
        if self.type_img == "grayscale":
            median5_image = median5_image[..., 0]
        median5_image = MainWindow.reconstruct_patches(MainWindow.image_to_patches(median5_image, 64))

        # display images with matplotlib
        fig = plt.figure(figsize=(20, 20))
        if self.type_img == "grayscale":
            plt.gray()
        fig.add_subplot(2, 3, 1)
        plt.imshow(original_image)
        plt.axis('off')
        plt.rcParams['font.size'] = 10
        plt.title('ORIGINAL')

        fig.add_subplot(2, 3, 2)
        plt.imshow(noised_image)
        plt.axis('off')
        psnr = self.calculate_psnr(noised_image, original_image)
        psnr = "{:.2f}".format(psnr)
        plt.rcParams['font.size'] = 10
        plt.title(f"NOISED \n PSNR = {psnr}")

        fig.add_subplot(2, 3, 3)
        plt.imshow(denoised_image)
        plt.axis('off')
        psnr = self.calculate_psnr(denoised_image, original_image)
        psnr = "{:.2f}".format(psnr)
        plt.rcParams['font.size'] = 10
        plt.title(f"DENOISED \n PSNR = {psnr}")

        fig.add_subplot(2, 3, 4)
        plt.imshow(gaussian_image)
        plt.axis('off')
        psnr = self.calculate_psnr(gaussian_image, original_image)
        psnr = "{:.2f}".format(psnr)
        plt.rcParams['font.size'] = 10
        plt.title(f"GAUSSIAN \n PSNR = {psnr}")

        fig.add_subplot(2, 3, 5)
        plt.imshow(average_image)
        plt.axis('off')
        psnr = self.calculate_psnr(average_image, original_image)
        psnr = "{:.2f}".format(psnr)
        plt.rcParams['font.size'] = 10
        plt.title(f"AVERAGE \n PSNR = {psnr}")

        fig.add_subplot(2, 3, 6)
        plt.imshow(median5_image)
        plt.axis('off')
        psnr = self.calculate_psnr(median5_image, original_image)
        psnr = "{:.2f}".format(psnr)
        plt.rcParams['font.size'] = 10
        plt.title(f"MEDIAN 5x5 \n PSNR = {psnr}")

        plt.show()

    # Display de-noising result images for salt & pepper noise
    def display_sp_images(self):
        # load images
        # original image
        image_path = f"{self.folder_path}original.png"
        original_image = tf.keras.preprocessing.image.load_img(image_path, color_mode=self.type_img)
        original_image = tf.keras.preprocessing.image.img_to_array(original_image).astype(int)
        if self.type_img == "grayscale":
            original_image = original_image[..., 0]
        original_image = MainWindow.reconstruct_patches(MainWindow.image_to_patches(original_image, 64))

        # noised image
        image_path = f"{self.folder_path}noised.png"
        noised_image = tf.keras.preprocessing.image.load_img(image_path, color_mode=self.type_img)
        noised_image = tf.keras.preprocessing.image.img_to_array(noised_image).astype(int)
        if self.type_img == "grayscale":
            noised_image = noised_image[..., 0]
        noised_image = MainWindow.reconstruct_patches(MainWindow.image_to_patches(noised_image, 64))

        # de-noised image
        image_path = f"{self.folder_path}auto-encoder.png"
        denoised_image = tf.keras.preprocessing.image.load_img(image_path, color_mode=self.type_img)
        denoised_image = tf.keras.preprocessing.image.img_to_array(denoised_image).astype(int)
        if self.type_img == "grayscale":
            denoised_image = denoised_image[..., 0]
        denoised_image = MainWindow.reconstruct_patches(MainWindow.image_to_patches(denoised_image, 64))

        # median 3x3 image
        image_path = f"{self.folder_path}median3.png"
        median3_image = tf.keras.preprocessing.image.load_img(image_path, color_mode=self.type_img)
        median3_image = tf.keras.preprocessing.image.img_to_array(median3_image).astype(int)
        if self.type_img == "grayscale":
            median3_image = median3_image[..., 0]
        median3_image = MainWindow.reconstruct_patches(MainWindow.image_to_patches(median3_image, 64))

        # amf filter
        image_path = f"{self.folder_path}median_Filter.png"
        amf_image = tf.keras.preprocessing.image.load_img(image_path, color_mode=self.type_img)
        amf_image = tf.keras.preprocessing.image.img_to_array(amf_image).astype(int)
        if self.type_img == "grayscale":
            amf_image = amf_image[..., 0]
        amf_image = MainWindow.reconstruct_patches(MainWindow.image_to_patches(amf_image, 64))

        # median 5x5 image
        image_path = f"{self.folder_path}median5.png"
        median5_image = tf.keras.preprocessing.image.load_img(image_path, color_mode=self.type_img)
        median5_image = tf.keras.preprocessing.image.img_to_array(median5_image).astype(int)
        if self.type_img == "grayscale":
            median5_image = median5_image[..., 0]
        median5_image = MainWindow.reconstruct_patches(MainWindow.image_to_patches(median5_image, 64))

        # display images with matplotlib
        fig = plt.figure(figsize=(20, 20))
        plt.gray()
        fig.add_subplot(2, 3, 1)
        plt.imshow(original_image)
        plt.axis('off')
        plt.rcParams['font.size'] = 10
        plt.title('ORIGINAL')

        fig.add_subplot(2, 3, 2)
        plt.imshow(noised_image)
        plt.axis('off')
        psnr = self.calculate_psnr(noised_image, original_image)
        psnr = "{:.2f}".format(psnr)
        plt.rcParams['font.size'] = 10
        plt.title(f"NOISED \n PSNR = {psnr}")

        fig.add_subplot(2, 3, 3)
        plt.imshow(denoised_image)
        plt.axis('off')
        psnr = self.calculate_psnr(denoised_image, original_image)
        psnr = "{:.2f}".format(psnr)
        plt.rcParams['font.size'] = 10
        plt.title(f"DENOISED \n PSNR = {psnr}")

        fig.add_subplot(2, 3, 4)
        plt.imshow(median3_image)
        plt.axis('off')
        psnr = self.calculate_psnr(median3_image, original_image)
        psnr = "{:.2f}".format(psnr)
        plt.rcParams['font.size'] = 10
        plt.title(f"MEDIAN 3x3 \n PSNR = {psnr}")

        fig.add_subplot(2, 3, 5)
        plt.imshow(amf_image)
        plt.axis('off')
        psnr = self.calculate_psnr(amf_image, original_image)
        psnr = "{:.2f}".format(psnr)
        plt.rcParams['font.size'] = 10
        plt.title(f"ADAPTATIVE MEDIAN FILTER \n PSNR = {psnr}")

        fig.add_subplot(2, 3, 6)
        plt.imshow(median5_image)
        plt.axis('off')
        psnr = self.calculate_psnr(median5_image, original_image)
        psnr = "{:.2f}".format(psnr)
        plt.rcParams['font.size'] = 10
        plt.title(f"MEDIAN 5x5 \n PSNR = {psnr}")

        plt.show()

    # Add buttons listener
    def Button(self):
        # GET BT CLICKED
        btnWidget = self.sender()

        # PAGE HOME
        if btnWidget.objectName() == "btn_home":
            self.ui.stackedWidget.setCurrentWidget(self.ui.page_home)
            UIFunctions.resetStyle(self, "btn_home")
            UIFunctions.labelPage(self, "Home")
            btnWidget.setStyleSheet(UIFunctions.selectMenu(btnWidget.styleSheet()))

        # PAGE HISTORY
        if btnWidget.objectName() == "btn_history":
            self.ui.stackedWidget.setCurrentWidget(self.ui.page_history)
            UIFunctions.resetStyle(self, "btn_history")
            UIFunctions.labelPage(self, "History")
            btnWidget.setStyleSheet(UIFunctions.selectMenu(btnWidget.styleSheet()))
            self.ui.tableWidget.clear()
            self.filling_table()

        # PAGE SETTINGS
        if btnWidget.objectName() == "btn_settings":
            # self.ui.stackedWidget.setCurrentWidget(self.ui.page_settings)
            UIFunctions.resetStyle(self, "btn_settings")
            UIFunctions.labelPage(self, "Settings")
            btnWidget.setStyleSheet(UIFunctions.selectMenu(btnWidget.styleSheet()))

        # OPEN IMAGE
        if btnWidget.objectName() == "btn_open_image":
            file_name, _ = QtWidgets.QFileDialog.getOpenFileName(self, 'Single File', QtCore.QDir.rootPath(), '*.png')
            if file_name != '':
                self.ui.edit_image_path.setText(file_name)
                self.ui.image_original.setPixmap(file_name)

                # open image
                if self.ui.checkBox_type_rgb.isChecked():
                    # open color RGB image
                    self.type_img = "rgb"
                elif self.ui.checkBox_type_gray.isChecked():
                    # open grayscale image
                    self.type_img = "grayscale"

                self.original_img = tf.keras.preprocessing.image.load_img(file_name, color_mode=self.type_img)
                self.original_img = tf.keras.preprocessing.image.img_to_array(self.original_img)

                if self.type_img == "grayscale":
                    self.original_img = self.original_img[..., 0]

                # save original image in temporary folder
                now = datetime.now()
                dt_string = now.strftime("%d-%m-%Y %H-%M-%S")
                self.folder_path = f"./history/{dt_string}/"
                self.save_image(self.original_img, self.folder_path, "original")

        # GENERATE NOISE
        if btnWidget.objectName() == "btn_generate_noise":
            # get noise intensity
            if self.ui.checkBox_intensity_rand.isChecked():
                self.noise_intensity = random.randint(10, 50)/100
                self.ui.slider_intensity.setValue(self.noise_intensity*100)
                self.ui.label_intensity.setText(f"{str(int(self.noise_intensity*100))}%")
            else:
                self.noise_intensity = int(self.ui.slider_intensity.value())/100

            # get noise type
            if self.ui.checkBox_noise_rand.isChecked():
                if self.type_img == "rgb":
                    self.type_noise = "gauss"
                else:
                    rand = random.randint(1, 2)
                    if rand == 1:
                        self.type_noise = "gauss"
                    else:
                        self.type_noise = "sp"
            elif self.ui.checkBox_noise_gauss.isChecked():
                self.type_noise = "gauss"
            elif self.ui.checkBox_noise_sp.isChecked():
                self.type_noise = "sp"

            # normalize image /.255 before applying gaussian noise
            if self.type_noise == "gauss":
                self.original_img = np.array(self.original_img / 255)

            # apply noise to original image
            if self.type_noise == "gauss":
                self.noised_img = MainWindow.gauss_noise(self.original_img, self.noise_intensity)
            else:
                self.noised_img = MainWindow.sp_noise(self.original_img, self.noise_intensity)
                self.noised_img = np.array(self.noised_img / 255)

            # save image
            self.noised_img = (self.noised_img * 255).astype(int)
            self.save_image(self.noised_img, self.folder_path, "noised")

            # display noised image
            self.ui.image_noised.setPixmap(self.folder_path+"noised.png")

        # IMAGE CLASSIFICATION
        if btnWidget.objectName() == "btn_classification":
            self.classify_image()
            if self.type_noise == "gauss":
                self.alert("Image classification",
                           "The execution of the classification is finished, the introduced image "
                           "is a noisy image with Gaussian noise")
            elif self.type_noise == "sp":
                self.alert("Image classification",
                           "The execution of the classification is finished, the introduced image "
                           "is a noisy image with the salt & pepper noise")
            else:
                self.alert("Image classification",
                           "The execution of the classification is finished, No noise detected on "
                           "the input image")

        # IMAGE DENOISING
        if btnWidget.objectName() == "btn_denoising":
            self.denoise_image()
            self.alert("Image Denoising", "Input image was successfully denoised!")
            if self.type_noise == "gauss":
                self.display_gauss_images()
            else:
                self.display_sp_images()

    # Slider listener
    def slider_listener(self):
        self.ui.label_intensity.setText(f"{str(self.ui.slider_intensity.value())}%")

    # Click event filter
    def eventFilter(self, watched, event):
        if event.type() == QtCore.QEvent.MouseButtonPress:
            index = self.ui.tableWidget.indexAt(event.pos())
            row = index.row()
            path = glob.glob("./history/*")
            path = path[::-1]
            counter = 0
            for folder in path:
                if counter == row:
                    # load image params
                    params = folder.split("+")
                    self.folder_path = f"{folder}/"
                    image_type = params[2]
                    if image_type == "RGB":
                        self.type_img = "rgb"
                    else:
                        self.type_img = "grayscale"
                    noise_type = params[3]
                    if noise_type == "Gaussian":
                        self.type_noise = "gauss"
                    else:
                        self.type_noise = "sp"

                    # load image results
                    if self.type_noise == "gauss":
                        self.display_gauss_images()
                    else:
                        self.display_sp_images()
                counter += 1

    # Mouse Press event
    def mousePressEvent(self, event):
        self.dragPos = event.globalPos()
        if event.buttons() == Qt.LeftButton:
            pass
        if event.buttons() == Qt.RightButton:
            pass
        if event.buttons() == Qt.MidButton:
            pass

    # Key press event
    def keyPressEvent(self, event):
        pass

    # Resize window event
    def resizeEvent(self, event):
        self.resizeFunction()
        return super(MainWindow, self).resizeEvent(event)

    # Resize window function
    def resizeFunction(self):
        pass

    # Filling history execution table
    def filling_table(self):
        path = glob.glob("./history/*")
        for folder in path:
            params = folder.split("+")
            image_size = params[1]
            image_type = params[2]
            noise_type = params[3]
            noise_intensity = params[4]+" %"
            self.filling_item(f"{folder}/", image_size, image_type, noise_type, noise_intensity)

    # Filling execution item widget inside history table
    def filling_item(self, folder_path, image_size, image_type, noise_type, noise_intensity):
        image = QLabel()
        pic = QtGui.QPixmap(folder_path+"original.png")
        image.setPixmap(pic)
        image.setScaledContents(True)
        image.setMinimumSize(128, 128)
        self.ui.tableWidget.insertRow(0)
        self.ui.tableWidget.setCellWidget(0, 0, image)
        self.ui.tableWidget.cellWidget(0, 0).setFixedWidth(128)
        self.ui.tableWidget.setCellWidget(0, 1, QLabel(image_size))
        self.ui.tableWidget.setCellWidget(0, 2, QLabel(image_type))
        self.ui.tableWidget.setCellWidget(0, 3, QLabel(noise_type))
        self.ui.tableWidget.setCellWidget(0, 4, QLabel(noise_intensity))


# main method
if __name__ == "__main__":
    app = QApplication(sys.argv)
    QtGui.QFontDatabase.addApplicationFont('fonts/segoeui.ttf')
    QtGui.QFontDatabase.addApplicationFont('fonts/segoeuib.ttf')
    window = MainWindow()
    window.setWindowIcon(QIcon("./icons/logo.png"))
    window.show()
    sys.exit(app.exec_())
