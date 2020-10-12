# Python Face Extraction

This project is made up of two components, a video face extractor and an age & gender detector. The idea of the project is to be able to run community made videos through it and classify by the age of the people within them. So those of certain age groups can be flagged for manual review.


```sh
$ python face_detector.py --video=videos/face-demographics-walking.mp4 --tolerance=.55 --blur=100
$ python age_gender_detect.py --images=faces/videos_face_demographics_walking_mp4/E1
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

## Networks
The Age & Gender detection models are based from the [AgeGenderDeepLearning](https://github.com/GilLevi/AgeGenderDeepLearning) project.