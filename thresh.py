import time
import cv2
import pandas as pd

roi_result_fieldnames = ["ts", "pos_msec", "movement_instant", "auto1sec", "bridged"]


def frame_diff(first_frame, prev_frame):
    diff_im = cv2.absdiff(prev_frame, first_frame)

    conv_gray = cv2.cvtColor(diff_im, cv2.COLOR_RGB2GRAY)
    _, mask = cv2.threshold(conv_gray, 200, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

    diff_im_colored = cv2.cvtColor(diff_im, cv2.COLOR_RGB2BGR)
    diff_im_colored[mask != 255] = [0, 0, 255]

    return diff_im_colored


def frame_preprocess(frame_ref):
    return cv2.cvtColor(frame_ref, cv2.COLOR_BGR2RGB)


def autocorrelation_window(results_queue, thresh, delta_frame):  # autocorr n-13
    """
    Using autocorrelation window with the delta_frame-th frame
    add {'auto1sec': diff value, 'bridged': diff value is less than thresh}
    """
    clamper = lambda v: clamped(-60.0e6, 60.0e6, v)

    results = dict()

    if len(results_queue) < delta_frame:
        print(f"rx{len(results)}, {delta_frame}")
        return results

    else:
        delta = clamper(results_queue[delta_frame - 1]['movement_instant']) - clamper(
            results_queue[0]['movement_instant'])
        results['auto1sec'] = abs(delta)
        results['bridged'] = [f"<{ti}:{int(abs(delta) < ti)}" for i, ti in enumerate(thresh)]
        # print(results['bridged'])
    return results


def clamped(min_v, max_v, val):
    return min(max(float(val), float(min_v)), float(max_v))


def curried_autocorrelaton_window(thresh, delta_frame):
    """
    Allows you to configure the autocorrelation_window function ahead of time
    with known values, thresh and delta_frame
    """

    def curried(rq):
        return autocorrelation_window(rq, thresh, delta_frame)

    return curried


def process_roi(roi_comp, prev_frame, frame):
    if frame is not None:
        for roi in roi_comp:
            points = roi.points

            x1, y1 = points['x2'], points['y2']
            x4, y4 = points['x3'], points['y4']
            prev_roi_region = prev_frame[y1:y4, x1:x4]
            frame_roi_region = frame[y1:y4, x1:x4]

            diff_im = frame_diff(frame_preprocess(prev_roi_region), frame_preprocess(frame_roi_region))

            frame_roi_region = diff_im[:, :, 2]
            frame[y1:y4, x1:x4][frame_roi_region == 255] = [0, 0, 255]

            results = {"ts": time.time(), "movement_instant": round(diff_im.sum(), 4)}
            results['ts'] = pd.to_datetime(results['ts'], unit='s')

            return frame, results
    else:
        frame = prev_frame
        return frame, None