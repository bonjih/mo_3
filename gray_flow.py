import cv2
import numpy as np
from collections import deque

def process_frame(previous, n_of_frames, sum_of_frames, sumsq_of_frames, gray, roi_coords):
    if len(previous) == n_of_frames:
        stdev_gray = np.sqrt(sumsq_of_frames / n_of_frames - np.square(sum_of_frames / n_of_frames))
        cv2.imshow('stdev_gray', stdev_gray * (1/255))
        sum_of_frames -= previous[0]
        sumsq_of_frames -= np.square(previous[0])
        previous.popleft()
    previous.append(gray)
    sum_of_frames += gray
    sumsq_of_frames += np.square(gray)
    
    # Extract the ROI from the grayscale frame
    x1, y1, x2, y2 = roi_coords
    roi_gray = gray[y1:y2, x1:x2]

    # Calculate the standard deviation within the ROI
    stdev_roi = np.std(roi_gray)

    # Display the standard deviation within the ROI
    cv2.imshow('stdev_roi', stdev_roi * (1/255))

    return previous, sum_of_frames, sumsq_of_frames

video = cv2.VideoCapture(0)
previous = deque(maxlen=200)  # Define maxlen for the deque
n_of_frames = 200
sum_of_frames = 0
sumsq_of_frames = 0  # Initialize sum of squares to zero

# Define ROI coordinates (x1, y1, x2, y2)
roi_coords = (300, 70, 1000, 650)  # Example coordinates

while True:
    ret, frame = video.read()
    if ret:
        cropped_img = frame[0:150, 0:500]
        gray = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2GRAY)
        gray = gray.astype('f4')

        previous, sum_of_frames, sumsq_of_frames = process_frame(previous, n_of_frames, sum_of_frames, sumsq_of_frames, gray, roi_coords)

        key = cv2.waitKey(1)
        if key == ord('q'):
            break

video.release()
cv2.destroyAllWindows()
