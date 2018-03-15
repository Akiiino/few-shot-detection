# Few-shot traffic sign detection and classification

Roadmap:

- [x] Implement baseline non-few-shot traffic sign classification.
    - [x] Parse `classification` part of Russian Traffic Sign Dataset
    - [x] Implement basic training and testing routine
- [x] Extend classifier for heatmap generation
    - [x] Baseline implementation
    - [x] Improving with own ideas
    - [x] Improving with ideas from [paper](http://cnnlocalization.csail.mit.edu/Zhou_Learning_Deep_Features_CVPR_2016_paper.pdf)

##### TBD by 22.03.2018
-----
- [] Sample own classification dataset from full RTSD dataset
    - [] Include resampling for most/least frequent classes
    - [] Include data augmentation by resizing signs and adding background
- [] Create visualisations
    - [] Heatmap visualizations
    - [] Best/worst performing classes, confusion matrix
    - [] Heatmaps for full scenes
    
##### TBD by 1.04.2018
-----
- [] Apply classifier base for detection
    - [] Baseline implementation using fixed bounding boxes around heatmap maxima
    - [] Baseline implementation using already made detection networks
 
