import time
import cv2
import pandas as pd

from video_grabber import VideoGrabber

roi_result_fieldnames = ["ts", "pos_msec", "movement_instant", "auto1sec", "thresh", "bridge status"]


def frame_diff(first_frame, prev_frame):
    diff_im = cv2.absdiff(prev_frame, first_frame)

    conv_gray = cv2.cvtColor(diff_im, cv2.COLOR_RGB2GRAY)
    _, mask = cv2.threshold(conv_gray, 200, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

    diff_im_colored = cv2.cvtColor(diff_im, cv2.COLOR_RGB2BGR)
    diff_im_colored[mask != 255] = [0, 0, 255]

    return diff_im_colored


def frame_preprocess(frame_ref):
    return cv2.cvtColor(frame_ref, cv2.COLOR_BGR2RGB)


processed_items = []  # List to keep track of processed items


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
        results['thresh'] = [f"<{ti}:{int(abs(delta) < ti)}" for i, ti in enumerate(thresh)]

        processed_items.append(results['auto1sec'])  # Append the processed item

        if len(processed_items) >= 13:  # Check if 13 items are processed
            print(len(processed_items))

            # No need to clear processed_items here

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


def process_roi(roi_comp, prev_frame, frame, millis_diff):
    millis_diff_list = []
    bridge_status = ""
    results = {}

    if frame is not None:
        for roi in roi_comp:
            points = roi.points

            x1, y1 = points['x2'], points['y2']
            x4, y4 = points['x3'], points['y4']
            prev_roi_region = prev_frame[y1:y4, x1:x4]
            frame_roi_region = frame[y1:y4, x1:x4]

            diff_im = frame_diff(frame_preprocess(prev_roi_region), frame_preprocess(frame_roi_region))
            results.update({"ts": time.time(), "movement_instant": round(diff_im.sum(), 4)})
            results['ts'] = pd.to_datetime(results['ts'], unit='s')

            frame_roi_region = diff_im[:, :, 2]
            frame[y1:y4, x1:x4][frame_roi_region == 255] = [0, 0, 255]

            if millis_diff is not None:
                if millis_diff > 100000 or results['movement_instant'] > 46000000:
                    millis_diff_list.append(millis_diff)
                    if bridge_status != 'BRIDGED':
                        bridge_status = 'BRIDGED'
                        results.update({"bridge status": bridge_status})
                elif bridge_status != 'BRIDGED':  # Only update status if not already BRIDGED
                    bridge_status = 'No Bridge'
                    results.update({"bridge status": [bridge_status, millis_diff, results['movement_instant']]})
            print(results)
            return frame, results
    else:
        frame = prev_frame
        return frame, None
