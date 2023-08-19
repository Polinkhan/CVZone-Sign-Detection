import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
classifier = Classifier("Model/keras_model.h5", "Model/labels.txt")

margin = 20
imgSize = 300

labels = ["A", "B", "C"]

while True:
    try:
        seccess, img = cap.read()
        copyImg = img.copy()
        hands, img = detector.findHands(img)

        if hands:
            x, y, w, h = hands[0]['bbox']
            croppedImg = img[y-margin:y+h+margin, x-margin:x+w+margin]

            whiteImg = np.ones((imgSize, imgSize, 3), np.uint8)*255

            # if croppedImg.shape[0] < imgSize and croppedImg.shape[1] < imgSize:
            #     whiteImg[0:croppedImg.shape[0], 0:croppedImg.shape[1]] = croppedImg

            aspectRation = h/w

            if aspectRation > 1:
                k = imgSize/h
                calW = math.ceil(k*w)
                gapW = math.ceil((imgSize - calW)/2)
                resizeImg = cv2.resize(croppedImg, (calW, imgSize))
                whiteImg[:, gapW:calW+gapW] = resizeImg

            else:
                k = imgSize/w
                calH = math.ceil(k*h)
                gapH = math.ceil((imgSize - calH)/2)
                resizeImg = cv2.resize(croppedImg, (imgSize, calH))
                whiteImg[gapH:calH+gapH, :] = resizeImg

            predection, index = classifier.getPrediction(img)
            # print(labels[index])
            cv2.imshow("cropped Img", croppedImg)
            cv2.imshow("White Img", whiteImg)

        cv2.imshow("image", copyImg)
        cv2.waitKey(1)

    except KeyboardInterrupt:
        print("Exiting Program")
        exit()
    except Exception as e:
        print(e)
