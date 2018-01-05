import cv2
import numpy as np

#image = cv2.imread("images/barCode.jpg")
camera = cv2.VideoCapture(0)
while True:
    ret,frame = camera.read()
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    # Sobel and Scharr (set ksize=-1, 3x3) first do gaussian blur then perform vertical or horizontal differential op
    # shows vertical lines
    gradX = cv2.Sobel(gray, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
    # shows horizontal lines
    gradY = cv2.Sobel(gray, ddepth=cv2.CV_32F, dx=0, dy=1, ksize=-1)

    # since the bar code is vertical lines, we can subtract the y-gradient from the x-gradient
    gradient = cv2.subtract(gradX, gradY)
    gradient = cv2.convertScaleAbs(gradient)

    blurred = cv2.blur(gradient, (5, 5))
    (_, thresh) = cv2.threshold(blurred, 225, 255, cv2.THRESH_BINARY)


    # construct a closing kernel and apply it to the thresholded image
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 7))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel)

    #cv2.imshow("closed",opened)
    #cv2.waitKey(0)

    # perform a series of erosions and dilations
    closed = cv2.erode(opened, kernel, iterations = 2)
    closed = cv2.dilate(closed, kernel, iterations = 2)


    #cv2.imshow("dilated",closed)
    #cv2.waitKey(0)

    # find the contours in the thresholded image, then sort the contours
    # by their area, keeping only the largest one
    (img,cnts,hier) = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    # if no contours were found, return None
    if len(cnts) == 0:
        continue
    c = sorted(cnts, key=cv2.contourArea, reverse=True)[0]

    # compute the rotated bounding box of the largest contour
    rect = cv2.minAreaRect(c)
    box = np.int0(cv2.boxPoints(rect))

    # draw a bounding box arounded the detected barcode and display the image
    cv2.drawContours(frame, [box], -1, (0, 255, 0), 3)
    cv2.imshow("images", frame)

    # if the 'q' key is pressed, stop the loop
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# release the video capture
camera.release()
cv2.destroyAllWindows()