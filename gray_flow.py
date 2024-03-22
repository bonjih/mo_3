import cv2
import numpy as np
from collections import deque


def isolate(img, roi_coords):
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, [roi_coords], (255, 255, 255))
    masked = cv2.bitwise_and(img, mask)
    return masked


def op_isolate(roi_coord, frame_shape):
    # Create a binary mask covering the entire frame
    full_frame_mask = np.zeros(frame_shape[:2], dtype=np.uint8)
    cv2.fillPoly(full_frame_mask, [roi_coord], (255, 255, 255))

    op_roi_mask = cv2.bitwise_not(full_frame_mask)

    return op_roi_mask



def process_frame(previous, n_of_frames, sum_of_frames, sumsq_of_frames, gray, roi_coords):
    if len(previous) == n_of_frames:
        stdev_gray = np.sqrt(sumsq_of_frames / n_of_frames - np.square(sum_of_frames / n_of_frames))
        stdev_gray_scaled = stdev_gray * (1 / 255)
        sum_of_frames -= previous[0]
        sumsq_of_frames -= np.square(previous[0])
        previous.popleft()
    else:
        stdev_gray_scaled = None

    previous.append(gray)
    sum_of_frames += gray
    sumsq_of_frames += np.square(gray)

    # Calculate the standard deviation within the ROI
    roi_mask = isolate(gray, roi_coords)
    stdev_roi = np.std(roi_mask)

    return previous, sum_of_frames, sumsq_of_frames, stdev_gray_scaled, stdev_roi


video = cv2.VideoCapture('./data/crusher_bin_bridge2.mkv')
previous = deque(maxlen=200)  # Define maxlen for the deque
n_of_frames = 200
sum_of_frames = 0
sumsq_of_frames = 0  # Initialize sum of squares to zero

# Define ROI polygon vertices
roi_coords = np.array([[350, 650], [300, 300], [1000, 300], [1000, 650]], np.int32)

frame_masked = None  # Initialize frame_masked

while True:
    ret, frame = video.read()
    if ret:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = gray.astype('f4')

        previous, sum_of_frames, sumsq_of_frames, stdev_gray_scaled, stdev_roi = process_frame(previous, n_of_frames,
                                                                                               sum_of_frames,
                                                                                               sumsq_of_frames, gray,
                                                                                               roi_coords)

        cv2.polylines(frame, [roi_coords], isClosed=True, color=(0, 255, 0), thickness=2)

        # Convert stdev_roi to string and display it within the ROI
        stdev_roi_str = f"Stdev ROI: {stdev_roi:.2f}"
        cv2.putText(frame, stdev_roi_str, (roi_coords[0][0], roi_coords[0][1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (255, 255, 255), 1)

        # Overlay stdev_gray on the original frame
        if stdev_gray_scaled is not None:
            stdev_gray_bgr = cv2.cvtColor(stdev_gray_scaled, cv2.COLOR_GRAY2BGR)

            frame_masked = isolate(frame, roi_coords)
            frame_masked += (stdev_gray_bgr * 255).astype(frame.dtype)

        # Display the frame
        cv2.imshow('Frame with Stdev ROI', frame_masked if frame_masked is not None else frame)

        key = cv2.waitKey(1)
        if key == ord('q'):
            break

video.release()
cv2.destroyAllWindows()
