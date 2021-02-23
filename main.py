import cv2
import operator
import numpy as np

from ANN import *
from solver import *


# -- initalising ML  --------------------------
shape = [784,128,10]          
net = ANN( shape ) 
#net.load('Params/params_900.pickle')
#train_net()

# -- Camera object  --------------------------
cam = cv2.VideoCapture(0)

# --  Misc varibles  --------------------------
margin = 10
case = 28 + 2*margin
perspective_size = 9*case

flag = 0
ans = 0

if __name__ == '__main__':
  while True:

    # -- collect images  --------------------------
    check, frame = cam.read()
    #frame = imutils.resize(frame, width = 1000)
    #(height, width) = frame.shape[:2]

    # -- Noise reduction  --------------------------
    grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 
    blur = cv2.GaussianBlur(grey,(5,5),0)        

    # -- Segmentation --------------------------
    # (Edge detection) Adaptive threshold for to obtain a Binary Image from the frame
    adTH  = cv2.adaptiveThreshold(grey, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 9, 2)

    # -- feature extraction --------------------------
    # (Finding Contours) Use a copy of the image e.g. edged.copy() since findContours alters the image 
    contours_, hierarchy = cv2.findContours(adTH, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    maxArea, contour = 0, None # temp varibles to hold new contour

    #Find the largest contour(Sudoku Grid)
    for c in contours_:
      area = cv2.contourArea(c)
      if area > 25000:
        peri = cv2.arcLength(c, True)
        polygon = cv2.approxPolyDP(c, 0.01*peri, True)
        
        if area>maxArea and len(polygon)==4:
          contour = polygon
          maxArea = area
          

    # -- Drawing contour for feedback --------------------------
    #Draw the contour and extract Sudoku Grid
    if contour is not None:
      cv2.drawContours(frame, [contour], 0, (0, 255, 0), 2) # Draw contours
      points = np.vstack(contour).squeeze()
      points = sorted(points, key=operator.itemgetter(1))
        
    # -- extracting image inside largest contour  --------------------------
      if points[0][0]<points[1][0]:
        pts1 = np.float32([points[0], points[1], points[2], points[3]])
        if points[3][0]<points[2][0]:
          pts1 = np.float32([points[0], points[1], points[3], points[2]])
      else:
        pts1 = np.float32([points[1], points[0], points[2], points[3]])
        if points[3][0]<points[2][0]:
          pts1 = np.float32([points[1], points[0], points[3], points[2]])

      pts2 = np.float32([[0, 0], [perspective_size, 0], [0, perspective_size], [perspective_size, perspective_size]])
      matrix = cv2.getPerspectiveTransform(pts1, pts2)
      perspective_window =cv2.warpPerspective(frame, matrix, (perspective_size, perspective_size))
      result = perspective_window.copy()
      
      # -- pre processing extracted window --------------------------
      p_window = cv2.cvtColor(perspective_window, cv2.COLOR_BGR2GRAY)
      p_window = cv2.GaussianBlur(p_window, (5, 5), 0)
      p_window = cv2.adaptiveThreshold(p_window, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 9, 2)
      vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))
      p_window = cv2.morphologyEx(p_window, cv2.MORPH_CLOSE, vertical_kernel)

      # -- highlighting edges of window --------------------------
      lines = cv2.HoughLinesP(p_window, 1, np.pi/180, 120, minLineLength=40, maxLineGap=10)
      for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(perspective_window, (x1, y1), (x2, y2), (0, 255, 0), 2)

      #Invert the grid for digit recognition
      invert = 255 - p_window
      invert_window = invert.copy()
      
      invert_window = invert_window /255
      i = 0
      #If not predict the answer
      #Else only get the cell regions
      if flag != 1:
        predicted_digits = []
        pixels_sum = []
      
      #To get individual cells
      for y in range(9):
        predicted_line = []
        for x in range(9):
          y2min = y*case+margin
          y2max = (y+1)*case-margin
          x2min = x*case+margin
          x2max = (x+1)*case-margin
          
          #Obtained Cell
          image = invert_window[y2min:y2max, x2min:x2max]
          
          #Process the cell to feed it into model
          img = cv2.resize(image,(28,28))

          ###########################################################
          #img = img.reshape((784,1))
          img = img.reshape((1,784))
          pixel_sum = np.sum(img)
          pixels_sum.append(pixel_sum)

          #  model training
          pred = net.forward(img)
          pred = net.soft_loss.forward(pred)               

          predicted_digit = np.argmax(pred)
          #print(predicted_digit, end='\r')

          ###########################################################
        
          #For blank cells set predicted digit to 0
          if pixel_sum > 775.0: predicted_digit = 0
          predicted_line.append(predicted_digit)                        
         
          #If we already have predicted result, display it on window
          if flag == 1:
            ans = 1
            x_pos = int((x2min + x2max)/ 2)+10
            y_pos = int((y2min + y2max)/ 2)-5
            image = cv2.putText(result, str(pred_digits[i]), (y_pos, x_pos), cv2.FONT_HERSHEY_SIMPLEX,  
            1, (255, 0, 0), 2, cv2.LINE_AA)
          i = i + 1

        #Get predicted digit list
        if flag != 1: predicted_digits.append(predicted_line)
                      
      #Get solved Sudoku
      ans = solveSudoku(predicted_digits)
      if ans==True:
          flag = 1
          pred_digits = display_predList(predicted_digits)
          
          #Display the final result
          if ans == 1:
              cv2.imshow("Result", result)
              results = cv2.warpPerspective(result, matrix, (perspective_size, perspective_size), flags=cv2.WARP_INVERSE_MAP)
      
      #cv2.imshow("results", result)
      cv2.imshow('P-Window', p_window)
    cv2.imshow("frame", frame)


    if cv2.waitKey(1) == 27: break  # esc to quit
  cam.release()
  cv2.destroyAllWindows()


