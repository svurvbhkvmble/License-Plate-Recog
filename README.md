# License Plate Recognition


This Python project uses OpenCV and EasyOCR to identify and read car license plates.
This combines edge detection and optical character recognition powered by deep learning libaries to extract the exact license plate numbers and overlay them on the original image.

Due to the complexity and additional stickers on American license plates, this script works best on European license plates at the moment. I am tweaking the edge detection method to better recognize American plates despite the additional details.  

I am also exploring the .ginput method to pick and choose which contours the OCR decides to read rather than having it set to the edge detection keypoints.

Libraries Used:  
-MatPlotLib  
-OpenCV  
-Numpy  
-Imutils  
-EasyOCR  
