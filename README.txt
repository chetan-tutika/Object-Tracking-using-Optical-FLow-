Requirements: Python 3.5 and OpenCV 3.4.1

Directory Structure:
videos_src/ - contains all source experiment videos.[input videos picked up from here]
videos_track/ - contains the final tracking videos. [output videos dumped here]
bbox_preload/- contains pre-saved bounding box points for Easy and Medium video
Results/- Precomputed final results for Easy,Medium,Hard and other videos after parameter turning.


How to Run:
python objectTracking.py <video_name> <bbox_name>

Example: 

Easy video:
If you want to use preloaded bounding box:
    python objectTracking.py Easy.mp4 Easy_bbox

If you want to select bounding box:
    python objectTracking.py Easy.mp4 

(check parameters.py for number of features to track)

Medium video:
If you want to use preloaded bounding box:
    python objectTracking.py Medium.mp4 Medium_bbox

If you want to select bounding box:
    python objectTracking.py Medium.mp4


Parameter Selection:
Check parameters.py and comment/uncomment to select the parameters for the videos