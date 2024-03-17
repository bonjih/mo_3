from RoiClass import ComposeROI
from video_streamer import VideoPlayer
from video_grabber import CV2Grabber, CV2VideoFileWriter
from video_datalogger import CSVDataLogger
from thresh import curried_autocorrelaton_window, process_roi, roi_result_fieldnames
from utils import json_file

other_params = json_file("params.json")

comp_roi = ComposeROI("params.json")
thresh_comp = comp_roi.thresholds

logger = CSVDataLogger("./out/datalog.csv", roi_result_fieldnames)
grabber = CV2Grabber('./data/crusher_bin_bridge2.mkv')  # TODO move all params to json
writer = CV2VideoFileWriter("./out/debug.mkv", overlay_labels=other_params['addlabels']
                            )

# allow you to experiment with different window sizes and thresholds
# TODO windowing should manage its own queue and be a pluggable class
window_fn = curried_autocorrelaton_window(
    other_params["autocorrelation"]["thresholds"],
    other_params["autocorrelation"]["delta"]
)

other_params = json_file("params.json")
video_player = VideoPlayer(
    roi_comp=comp_roi,
    cap_iface=grabber,
    out_video_writer=writer,
    datalogger_fn=logger.trace,
    process_roi_fn=process_roi,
    windowing_roi_fn=window_fn,
    window_size_frames=other_params["autocorrelation"]["delta"],
    start_time_seconds=other_params["offset"]
)

video_player.run()
