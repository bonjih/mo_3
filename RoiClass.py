import json

class ROI:
    def __init__(self, **kwargs):
        self.points = kwargs


class ComposeROI:
    """
    calls a json file, iterates over all ROI's.
    ROI key must start with roi
    """

    def __init__(self, json_file):
        self.rois = []
        self.thresholds = []

        with open(json_file, 'r') as f:
            data = json.load(f)

            for key, value in data.items():
                if key.startswith("roi"):
                    roi = ROI(**value)
                    self.rois.append(roi)

                if key == "thresholds":
                    for thresh_key, thresh_value in value.items():

                        thresh = ROI(**thresh_value)
                        self.thresholds.append(thresh)

    def add_roi(self, roi):
        self.rois.append(roi)

    def add_threshold(self, thresh):
        self.thresholds.append(thresh)
        print(self.thresholds)

    def __iter__(self):
        return iter(self.rois + self.thresholds)




