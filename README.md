# Object-Tracking-using-Optical-FLow
Built a multi-feature tracker that takes in object bounding boxes for the first frame and tracks them over the
remaining frames<br />

## Results
![2ncyec](https://user-images.githubusercontent.com/41950483/48973639-8e75df80-f011-11e8-9508-632f70d91f4d.gif)
![2ncygs](https://user-images.githubusercontent.com/41950483/48973640-8e75df80-f011-11e8-97d1-34dccd9094a4.gif)

## Parameter tuning:
1. Number of features:<br />
a. As the features tracked are increased per object, we get better approximation of how the object
is moving over successive frames. But selecting too many features will affect the efficiency of
the algorithm as the neighboring features across the window may be pulled in. This might
adversely affect the tracking.<br />
b. Selecting less features might result faster feature drop. To counteract the effect, one might have
to decrease the refresh rate for calculating the features which might be computationally
expensive<br />
2. Transform:<br />
a. Similarity transform worked better than Affine and Holography for estimating the displacement
of features<br />
3. Threshold:<br />
a. Threshold for detecting outliers depends on the displacement between successive frames. As
the displacement increases the threshold increases.<br />
4. Update Frequency:<br />
a. Depends on how frequently the features are dropped in the video. As the angle of view change
the features are dropped much more frequently and hence the refresh rate is much higher<br />
