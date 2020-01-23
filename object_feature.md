# NAFAE object feature

## File Structure

Unzip the file and obtain the file structure

```shell
feature
├── train_box.json
├── val_box.json
├── test_box.json
├── training
		├── 101 
				├── 0O4bxhpFX9o
						├── 0000000000.npy
						└── ...
				└── ...
    └── ...
├── testing
		└── ...
└── testing
		└── ...
```

- The feature is stored as `.npy` for each frame. For `0003000194.npy` , the first four digits `0003` is the video segment index, starting from zero; and the last six digits `000194` is the frame index. The frame is indexed at 16 fps, starting with zero at the beginnning of each video segment. 

  For training data, we evenly sample 5 frames per video segment, including its boundary frames. 

  For val and test data, we sample 1 frame per second where bounding box is annotated.

- The box is stored in `${PHASE}_box.json`. The structure of the dictionary is 

  - key `testing/101/YSes0R7EksY/0000000008`: the key is the name of the frame. Its value is a list of box with shape (20, 4), representing 20 boxes each frame with (x1, y1, x2, y2) coordinate. The box coordinate is located in frame size (224x224). The order of boxes is the same with the feature order.





