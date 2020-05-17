# Python Face Extraction

Extracts faces from videos and tries to group them by person


```sh
$ python face_detector.py --video="input/face-demographics-walking.mp4"
```

- **video:** Path to video file
- **frames:** What frame multiplier to process (e.g 0.5 = Half the frames, 0.1 = tenth of the frames)
	- `0.01` = Default
	- `-1` = Every frame
	- `High Number` = Fast, Less Accurate
	- `Low number` = Slower, more accurate
- matchaccuracy
	- `0.15` = Default

