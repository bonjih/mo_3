import time
from collections import deque
import cv2
from video_grabber import VideoGrabber, VideoWriter
import logging


class VideoPlayer:
    def __init__(self, roi_comp, cap_iface: VideoGrabber = None,
                 process_roi_fn=None, datalogger_fn=None,
                 out_video_writer: VideoWriter = None, windowing_roi_fn=None,
                 window_size_frames=2, start_time_seconds=None):
        """
        Class to compose the components you would like to use for the
        pipeline

        :process_roi_fn:
        will send -> roi_comp, prev_frame and frame to your process_roi_fn
            recommend supply a class function to process_roi_fn to maintain state (if required)
            between calls

            for example:

            def my_process_roi_fn(self, prev_frame, frame):
                stat = some_func
                self.stat_list.append(stat)

                some_derived_stat = derive_stat(self.stat_list)

                return (frame, {"ts": time.time(), "derived_stat": some_derived_stat, "stat": stat})

        :datalogger_fn:
        will send the results of the process_roi_fn to this function, the use a class to keep track of
        logfile etc (see video_datalogger.py, CSVDataLogger) for an example
        """

        # information about the region of interest in the frame, this must be
        # tailored for your process_roi_fn function ( they are dependent)
        self.roi_comp = roi_comp
        # VideoGrabber object to begin abstracting away from directly using
        # opencv2
        self.cap_iface = cap_iface
        # for cumulatve summing of time delta
        self.timestamps = []
        # used to get underlying access to the video object, this is to manage
        # breaking changes when non-cv2 VideoGrabbers are used
        # self.cap_obj = None

        # used to stop the loop
        self.stopped = False
        # perform framen, framen + 1 statistic calculations
        self.process_roi = process_roi_fn
        # when not None, this function will generate additional statistics based
        # on the last windows_size_frames of results
        self.windowing_roi = windowing_roi_fn
        # a function that supports a key value dict, the expectation is that it
        # will log the values on a row by row basis somewhere
        self.datalog = datalogger_fn
        # the class of type VideoWriter to output frames from the process,
        # see CV2VideoWriter and CV2ImshowWriter in video_grabber.py for
        # examples
        self.out = out_video_writer
        # size of the results queue to send to windowing_roi function if it
        # exists
        self.window_size_frames = window_size_frames
        # when not none will be used to skip `start_time_seconds` seconds of the
        # video file
        self.start_time_seconds = start_time_seconds

    def run(self):
        self.cap_iface.open_video()
        # self.cap_obj = self.cap_iface.get_video_obj()
        if self.start_time_seconds is not None:
            self.cap_iface.set_pos_seconds(819.0)

        prev_frame = None
        # used for debug type stuff
        frame_counter = 0

        # local state fifo queue, this queue drops items > maxlen
        # the 0th index is the oldest and th (len -1)th is the newes
        results_dequeue = deque(maxlen=self.window_size_frames)
        try:
            while not self.stopped:
                # Performance measurement
                prev_timestamp = time.time()

                ret, frame = self.cap_iface.get_frame()
                if not ret:
                    self.cap_iface.release_video()
                    self.stopped = True

                if prev_frame is None:
                    prev_frame = frame
                    continue

                # Frame delta statistic calculation
                millis = self.cap_iface.get_current_timestamp()
                fps = self.cap_iface.get_fps()
                processed_frame, results = self.process_roi(
                    self.roi_comp, prev_frame, frame)

                # Operations on a queue of frame statistics
                if self.windowing_roi is not None and results is not None:
                    results_dequeue.append(results)
                    results.update(self.windowing_roi(results_dequeue))

                if results is not None:
                    results.update(
                        {"ts": frame_counter/fps, "pos_msec": millis})

                # Output data to datalogger
                self.datalog(results)

                # Perf / cum-sum measurement
                # post_frame_proc_ts = time.time()
                # frame_time_diff = post_frame_proc_ts - prev_timestamp
                # self.timestamps.append(frame_time_diff)

                # Video outputter when it exists
                if self.out is not None:
                    if not self.out.is_ready():
                        params = {
                            "fourcc": self.cap_iface.get_fourcc(),
                            "fps": self.cap_iface.get_fps(),
                            "dimension": self.cap_iface.get_dimension()
                        }
                        self.out.open(params)
                    self.out.add_frame_withlabel(processed_frame, results)

                cv2.imshow('Video', processed_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

                # Frame counter for diagnostics
                frame_counter = frame_counter + 1

        except Exception as e:
            logging.exception("Exception in frameloop ", e)
        finally:
            self.cap_iface.release_video()
            if self.out is not None:
                self.out.release_video()
