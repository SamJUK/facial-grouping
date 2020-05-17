# Python Face Extraction

Extracts faces from videos and tries to group them by person


```sh
$ python face_detector.py --video=input/face-demographics-walking.mp4 --tolerance=.55 --blur=100
```

- If classifying same person as different people, raise tolerance
- If classifying different people as the same, lower tolerance


## Arguments

- **video:** Path to video file
- **frames:** What frame multiplier to process (e.g 0.5 = Half the frames, 0.1 = tenth of the frames)
	- `0.01` = Default
	- `-1` = Every frame
	- `High Number` = Fast, Less Accurate
	- `Low number` = Slower, more accurate
- **matchaccuracy:** What percentage of faces to match in a person group to be classified the same
	- `0.15` = Default
- **blur:** Integer of blur intensity (only affects preview)
- **framesize:** float from `0 - 1` that scales the image size (smaller is faster but less accurate)
- **tolerance:** Face detection match tolerance