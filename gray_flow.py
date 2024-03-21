mport cv2
import numpy as np

def process_frame(previous, n_of_frames, sum_of_frames, sumsq_of_frames, gray):
    if len(previous) == n_of_frames:
        stdev_gray = np.sqrt(sumsq_of_frames / n_of_frames - np.square(sum_of_frames / n_of_frames))
        cv2.imshow('stdev_gray', stdev_gray * (1/255))
        sum_of_frames -= previous[0]
        sumsq_of_frames -= np.square(previous[0])
        previous.pop(0)
    previous.append(gray)
    sum_of_frames = sum_of_frames + gray
    sumsq_of_frames = sumsq_of_frames + np.square(gray)
    return previous, sum_of_frames, sumsq_of_frames

video = cv2.VideoCapture(0)
previous = []
n_of_frames = 200
sum_of_frames = 0
sumsq_of_frames = 0

while True:
    ret, frame = video.read()
    if ret:
        cropped_img = frame[0:150, 0:500]
        gray = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2GRAY)
        gray = gray.astype('f4')

        previous, sum_of_frames, sumsq_of_frames = process_frame(previous, n_of_frames, sum_of_frames, sumsq_of_frames, gray)

        key = cv2.waitKey(1)
        if key == ord('q'):
            break

video.release()
cv2.destroyAllWindows()
