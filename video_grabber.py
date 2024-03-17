import cv2
import numpy as np


class VideoGrabber:
    """
     subclass and bind to your video library i.e. CV2Grabber, could be pointless because CV2 seems to bind to everything,
     however there can be speedups by accessing FFMPEG directly and presenting frames for CV2 as shown here

        - https://stackoverflow.com/questions/67352282/pipe-video-frames-from-ffmpeg-to-numpy-array-without-loading-whole-movie-into-me
    """

    def __init__(self, file):
        self.fps = None
        self.fourcc = None
        self.dimension = None
        self.video = None
        self.file = file
        self.current_ts = None

    def open_video(self):
        raise (NotImplementedError(
            "Invalid usage: interface definition called directly"))

    def release_video(self):
        raise (NotImplementedError(
            "Invalid usage: interface definition called directly"))

    def get_frame(self) -> tuple[bool, np.ndarray]:
        raise (NotImplementedError(
            "Invalid usage: interface definition called directly"))

    def get_video_obj(self) -> object:
        return self.video

    def get_current_timestamp(self) -> object:
        """
        calls the underlying timestamp function supplied
        """
        return self.current_ts

    def set_pos_seconds(self, seconds: float):
        """
        skip the video to seconds position
        """
        raise (NotImplementedError(
            "Invalid usage: interface definition called directly"))

    def get_fps(self):
        """
        Returns the frames per second (fps) of the video.
        """
        return self.fps

    def get_fourcc(self):
        return self.fourcc

    def get_dimension(self):
        return self.dimension


class CV2Grabber(VideoGrabber):

    def __init__(self, file):
        super().__init__(file)

    def open_video(self):
        self.video = cv2.VideoCapture(self.file)
        self.fourcc = self.video.get(cv2.CAP_PROP_FOURCC)
        self.fps = self.video.get(cv2.CAP_PROP_FPS)

    def release_video(self):
        self.video.release()

    def get_frame(self) -> tuple[bool, np.ndarray]:
        tup = self.video.read()
        self.current_ts = self.video.get(cv2.CAP_PROP_POS_MSEC)

        if self.dimension is None and tup[0]:
            self.dimension = tup[1].shape
            print(
                f"Opened Video or stream with: FPS: {self.fps}, dimension {self.dimension}, fourcc: {encoding(self.fourcc)}")

        return tup

    def set_pos_seconds(self, seconds: float):
        self.video.set(cv2.CAP_PROP_POS_MSEC, seconds * 1.0e3)
        self.current_ts = self.video.get(cv2.CAP_PROP_POS_MSEC)


class VideoWriter:

    def __init__(self, destination_params, overlay_labels: list = []):
        self.video = None
        self.destination_params = destination_params
        self.overlay_labels = set(overlay_labels)

    def open(self, params) -> None:
        raise (NotImplementedError(
            "Invalid usage: interface definition called directly"))

    def release_video(self) -> None:
        raise (NotImplementedError(
            "Invalid usage: interface definition called directly"))

    def add_frame(self, frame: np.ndarray) -> None:
        raise (NotImplementedError(
            "Invalid usage: interface definition called directly"))

    def add_frame_withlabel(self, frame: np.ndarray, labels: dict) -> None:
        raise (NotImplementedError(
            "Invalid usage: interface definition called directly"))

    def is_ready(self):
        raise (NotImplementedError(
            "Invalid usage: interface definition called directly"))


class CV2VideoFileWriter(VideoWriter):

    def __init__(self, destination_params, overlay_labels: list = []):
        super().__init__(destination_params, overlay_labels)

    def open(self, params):
        self.open_cv2(params['fourcc'], params['fps'], params['dimension'])

    def open_cv2(self, fourcc, fps, dim):

        self.video = cv2.VideoWriter(
            self.destination_params,
            cv2.VideoWriter_fourcc('m', 'p', '4', 'v'),
            fps,
            (dim[1], dim[0])
        )

    def add_frame(self, frame):
        if self.video is not None:
            self.video.write(frame)

    def add_frame_withlabel(self, frame: np.ndarray, labels: dict) -> None:
        if self.video is not None:
            if labels is not None:  # Add a check for None
                filt = filtered_dict(self.overlay_labels, labels)
                self.add_frame(frame_label(frame, filt))

    def release_video(self):
        if self.video is not None:
            self.video.release()

    def is_ready(self):
        return self.video is not None


class CV2ImshowWriter(VideoWriter):

    def __init__(self, destination_params):
        super().__init__(destination_params)

    def open(self, params):
        self.video = "no video file required"

    def add_frame(self, frame):
        if self.video is not None:
            cv2.imshow(self.destination_params, frame)

    def add_frame_withlabel(self, frame: np.ndarray, labels: dict) -> None:
        if self.video is not None:
            if labels is not None:
                filt = filtered_dict(self.overlay_labels, labels)
                self.add_frame(frame_label(frame, filt))

            if cv2.waitkey(1) & 0xFF == ord('q'):
                raise KeyboardInterrupt(
                    "q pressed, bubbling up keyboard interrupt")

    def release_video(self):
        if self.video is not None:
            cv2.destroyAllWindows()

    def is_ready(self):
        return self.video is not None


def encoding(fourcc):
    h = int(fourcc)
    codec = chr(h & 0xff) + chr((h >> 8) & 0xff) + \
        chr((h >> 16) & 0xff) + chr((h >> 24) & 0xff)
    return codec


def frame_label(frame: np.ndarray, labels: dict, font_scalef: float = 0.5) -> np.ndarray:
    font = cv2.FONT_HERSHEY_SIMPLEX
    color = (255, 255, 0)
    start = 50.50
    font_scale = 1 * font_scalef
    thickness = 1
    temp_frame = frame
    line_height = 30 * font_scalef
    y = 50 * font_scale
    for k in labels:
        temp_frame = cv2.putText(temp_frame, str(f"{k}:{labels[k]}"), (int(
            50), int(y)), font, font_scale, color, thickness, cv2.LINE_AA)
        y = y + line_height

    return temp_frame


def filtered_dict(fkeys: set, unfiltered: dict):
    return {k: v for k, v in unfiltered.items() if k in fkeys}
