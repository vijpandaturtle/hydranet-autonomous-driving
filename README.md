# Multi-task Learning "Hydranets" for Autonomous Driving
This project demonstrates the use of a multi-task deep learning algorithm to learn and perform the tasks of semantic segmentation and depth estimation simultaneously.

## Project Structure 
- ```cmaps``` storing cmaps for both the datasets
- ```models``` storing the pre-trained models for the hydranets
- ```notebooks``` contains the inference and training notebooks
- ```output``` contains the output videos and point clouds
- ```lib``` contains code for loading the dataset
- ```lib```/```network``` contains the scripts for network architecture
- ```lib```/```utils``` contains the scripts for utility functions


## Output (Work in progress)
The current segmentation output is noisy. 




## References
This project is based on the paper "Real-Time Joint Semantic Segmentation and Depth Estimation Using Asymmetric Annotations". Some of the code has been adapted from the official [repository](https://github.com/DrSleep/multi-task-refinenet).
