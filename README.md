# Try out analysis on human movement

This is a project trying to help researchers to analyse human movement without knowing all the details of pose estimation models. We implement this as a Python3 package called pose_playground which is a folder at the root of this directory.

## Basic interfaces

Basic interfaces are introduced in `example.py`. Essentially, one manages pose model by creating an instance of `pose_playground.pose_models.model_managed.ModelManaged`.

## Analysing

`compare_patients_spine_angle.py` is an simple analysis comparing neck-spine-pelvis angles between healthy subjects and patients.

## Further development (for developers)

Currently only a single pose estimation [VNect](http://gvv.mpi-inf.mpg.de/projects/VNect/) model (Tensorflow 1 implementation) is supported. We are working to add more models.

New analysis helpers function should live directly under `pose_playground` directory like `pose_playground.joint_angle.py` does. Other non-sharable analysis code may live outside of `pose_playground` package like `compare_patients_spine_angle.py` does.

## Required packages

We want to have the easiest setup for all users. However, it is difficult when it comes to deep learning models that we collected from elsewhere. Anyway, here is a listed of required packages to run `example.py` and tested versions (most package versions should be fine):

| pip package name | tested version |
|------------------|----------------|
| opencv-python    | 4.2.0.34       |
| numpy            | 1.18.4         |
| matplotlib       | 3.2.1          |

For `VNect` this implementation requires `tensorflow 1.x` version.

## Acknowledgements

A large portion of `VNect` implementation code comes from [Wu Xin's repo](https://github.com/XinArkh/VNect) which also contains code from [Tim Ho's repo](https://github.com/timctho/VNect-tensorflow) and [EJ Shim](https://github.com/EJShim/).