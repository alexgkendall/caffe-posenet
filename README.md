# PoseNet
**This is a modified version of [Caffe](https://github.com/BVLC/caffe) which supports the [PoseNet architecture](http://mi.eng.cam.ac.uk/projects/relocalisation/)**

As described in the ICCV 2015 paper **PoseNet: A Convolutional Network for Real-Time 6-DOF Camera Relocalization** Alex Kendall, Matthew Grimes and Roberto Cipolla [http://mi.eng.cam.ac.uk/projects/relocalisation/]

## Getting Started

 * Download the Cambridge Landmarks dataset [from here.](http://mi.eng.cam.ac.uk/projects/relocalisation/)
 * Download models and trained weights [from here.](http://mi.eng.cam.ac.uk/~agk34/resources/PoseNet.zip)

Create an LMDB localisation dataset with ```caffe-posenet/posenet/scripts/create_posenet_lmdb_dataset.py``` Change lines 1, 11 & 12 to the appropriate directories.

Test PoseNet with ```caffe-posenet/posenet/scripts/test_posenet.py``` using the command ```python test_posenet.py --model your_model.prototxt --weights your_weights.caffemodel --iter size_of_dataset```

## Publications

If you use this software in your research, please cite our publications:

http://arxiv.org/abs/1505.07427
Alex Kendall, Matthew Grimes and Roberto Cipolla "PoseNet: A Convolutional Network for Real-Time 6-DOF Camera Relocalization." Proceedings of the International Conference on Computer Vision (ICCV), 2015. 

http://arxiv.org/abs/1509.05909
Alex Kendall and Roberto Cipolla "Modelling Uncertainty in Deep Learning for Camera Relocalization." The International Conference on Robotics and Automation, 2015. 


## License

This extension to the Caffe library and the PoseNet models are released under a creative commons license which allows for personal and research use only. For a commercial license please contact the authors. You can view a license summary here:
http://creativecommons.org/licenses/by-nc/4.0/
