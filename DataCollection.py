import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import time

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)

margin = 20
imgSize = 300

saveDir = "Data/C"

counter = 0
while True:
    try:
        seccess, img = cap.read()
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

            cv2.imshow("cropped Img", croppedImg)
            cv2.imshow("White Img", whiteImg)

        cv2.imshow("image", img)
        key = cv2.waitKey(1)

        if (key == ord('s')):
            counter += 1
            cv2.imwrite(f'{saveDir}/Image_{time.time()}.jpg', whiteImg)
            print(f'{counter} Image Saved')

    except KeyboardInterrupt:
        print("Exiting Program")
        exit()
    except:
        print("Error")
