import cv2
import matplotlib.pyplot as plt
import numpy as np
import imutils
import easyocr

## Read in Image, Greyscale/Blur

img = cv2.imread("C:/Users/asus/Desktop/Python Project/License Plate Recog/IMG_9.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
# plt.imshow(cv2.cvtColor(gray, cv2.COLOR_BGR2RGB)) # to test this step

## Apply filter & find edges

bfilter = cv2.bilateralFilter(gray, 11, 17, 17) #noise reduction/smoothing
edged = cv2.Canny(bfilter, 150, 200, True) #edge detection
img2=cv2.cvtColor(edged, cv2.COLOR_BGR2RGB)
plt.figure(1)
plt.imshow(img2) # to test step

## Find Contours & apply mask

keypoints = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
contours = imutils.grab_contours(keypoints)
contours = sorted(contours, key=cv2.contourArea, reverse=True) #[:10]

# x,y,w,h = cv2.boundingRect(edged.copy())
# img3 = cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
# plt.imshow(img3)

# rect = cv2.minAreaRect(img)
# box = cv2.boxPoints(rect)
# box = np.int0(box)
# img3 = cv2.drawContours(img,[box],0,(0,0,255),2)
# plt.imshow(img3)

location = None
for contour in contours:
    approx = cv2.approxPolyDP(contour, 10, True)
    if len(approx) == 4:
        location = approx
        break
#print(location)




mask = np.zeros(gray.shape, np.uint8)
new_image = cv2.drawContours(mask, [location], 0,255, -1)
new_image = cv2.bitwise_and(img, img, mask = mask)
plt.figure(2)
plt.imshow(cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB)) #for test

(x, y) = np.where(mask==255)
(x1, y1) = (np.min(x), np.min(y))
(x2, y2) = (np.max(x), np.max(y))
cropped_image = gray[x1-1:x2+1, y1-1:y2+1]
plt.figure(3)
plt.imshow(cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB))

# Use EasyOCR to read text

reader = easyocr.Reader(['en'], gpu=True)
result = reader.readtext(cropped_image)
res=result[0]

# Render result

text = res[1]
font = cv2.FONT_HERSHEY_PLAIN
rez = cv2.putText(img, text=text, org=(approx[0][0][0], approx[1][0][1]+60), fontFace=font, fontScale=2, color=(0,255,0), thickness=2)   #, linetype = cv2.LINE_AA)
rez = cv2.rectangle(img, tuple(approx[0][0]), tuple(approx[2][0]), (0,255,0), 3)
plt.figure(4)
plt.imshow(cv2.cvtColor(rez, cv2.COLOR_BGR2RGB))