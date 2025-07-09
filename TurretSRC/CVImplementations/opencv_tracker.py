import cv2
import numpy as np
from typing import Sequence


class PersonTracker:

    def __init__(self) -> None:
        """
        tracker_list: list[cv2.tracker] is a list of tracker objects for the things
                                        we are tracking.
        num_frames missing:list[int] is how a targetID has been missing for
                                    where the targetID is the index.
        last_seen_location:list[tuple] is an array of quadruples which specify
                                    the last bbox where a target was seen.
                                    In particular, the tuples are of the form (x,y,width,height)
        """
        self.tracker_list: list[cv2.Tracker] = []
        self.num_frames_missing: list[int] = []
        self.last_seen_location: list[np.ndarray] = []

    def add_trackers(self, frame, bboxes: list[np.ndarray]) -> None:
        """
        This function shall be called every time we switch to detection mode.
        If this function is not called, the tracking mode will not know what to track
        and will default back to idle.
        Note that this function does not clear existing trackers.
        If you wanted to purge trackers and use a new set, you call flush first.
        """
        print("trackers initialized")
        params = cv2.TrackerVit.Params()
        params.net = "CVModel/Tracking/vittrack.onnx"
        params.backend = 0
        params.target = 0
        for bbox in bboxes:
            tracker = cv2.TrackerVit.create(parameters=params)
            tracker.init(frame, bbox)
            self.tracker_list.append(tracker)
            self.num_frames_missing.append(0)
            self.last_seen_location.append(bbox)

    def tracking_update(self, new_frame) -> None:
        """
        This function will update all trackers and their corresponding bounding
        boxes.
        """
        for i in range(len(self.tracker_list)):
            ok, bbox = self.tracker_list[i].update(new_frame)
            if ok:
                height, width, _ = new_frame.shape
                bbox = PersonTracker._normalize_bbox(bbox, img_height=height, img_width=width)
                self.last_seen_location[i] = bbox
                self.num_frames_missing[i] = 0
            else:
                print("Unable to update tracker ", i)
                self.num_frames_missing[i] += 1
                continue

    def get_last_seen_bbox(self, targetID) -> np.ndarray:
        """
        This function will get the last seen location of specified target.
        """
        return self.last_seen_location[targetID]

    def get_num_frames_missing(self, targetID) -> int:
        """
        This function returns the number of frames that a target has been missing.
        """
        return self.num_frames_missing[targetID]

    def get_smallest_num_frame_missing(self) -> int:
        """
        This function returns the targetID of the tracker with the smallest number of
        frames missing. (First instance in cases of draw.)
        """
        return self.num_frames_missing.index(min(self.num_frames_missing))

    def flush(self) -> None:
        """
        Flush the current tracked list.
        Flushing just clears the entire tracked list.
        We would want to do this every couple of frames to realign
        our detection and tracked thing.
        """
        self.tracker_list.clear()
        self.num_frames_missing.clear()
        self.last_seen_location.clear()

    def is_empty(self) -> bool:
        """
        This function returns true if the list is empty.
        """
        return len(self.tracker_list) == 0

    @staticmethod
    def _normalize_bbox(bbox: Sequence[int], img_height: int, img_width: int) -> np.ndarray:
        """
        This function will normalize the bbox so that it does not try to interpolate off the screen.
        bbox: Sequence[int] a rectangle of the form (x,y,width,height)
        img_height: int the height of the image in pixels
        img_width: int the width of the image in pixels
        """
        x = max(0, bbox[0])
        y = max(0, bbox[1])
        if bbox[0] + bbox[2] > img_width:
            w = img_width - bbox[0]
        else:
            w = bbox[2]

        if bbox[1] + bbox[3] > img_height:
            h = img_height - bbox[1]
        else:
            h = bbox[3]
        return np.array((x, y, w, h))