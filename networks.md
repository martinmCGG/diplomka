# Neural Networks

This page lists all neural networks contained in our framework. For each network there is a link to original paper describing the network and its original implementation. The input format of the network is specified by the path of the directory containing scripts for data conversion to this particular format. Also some further explanation of specific parameters is given.

### VRNENS

Paper:  [Generative and Discriminative Voxel Modeling with Convolutional Neural Networks](https://arxiv.org/pdf/1608.04236.pdf)  
Original Code: [https://github.com/ajbrock/Generative-and-Discriminative-Voxel-Modeling](https://github.com/ajbrock/Generative-and-Discriminative-Voxel-Modeling)  
Docker: dockers/vrnens  
Data: dockers/data_conversion/vrnens_data  
Framework: Theano with Lasagne

Details: This network takes voxel grid as an input compressed in .npz format. Size of the voxel grid is set in dim parameter. Network uses more than one rotation of the 3D model, you can change the number of rotations by setting num_rotations parameter. This network is very big and slow and even compilation of the model takes up fifteen minutes.

### OCTREE

Paper:  [O-CNN: Octree-based Convolutional Neural Networks](https://wang-ps.github.io/O-CNN_files/CNN3D.pdf)  
Original Code: [https://github.com/Microsoft/O-CNN](https://github.com/Microsoft/O-CNN)  
Docker: dockers/octree  
Data: dockers/data_conversion/octree_data  
Framework: Caffe

Details: This network is implemented in Caffe framework, which works differently then other frameworks. The network is defined in a .prototxt file and the training procedure is defined in similiar file and is called solver. In order to have the same parameters as other networks we have to do some tricks. The above mentioned files are located in examples directory and the parameters in config.ini are automatically copied into these files if they are named the same.  
The solver and net parameters have to contain the paths to solver and net definition file respectively and has to be enclosed in quotation marks. Similarly the snapshot_prefix has to contain full path and has to be enclosed in quotation marks as well.  
Some of the parameters are in special section [ITER_PARAMETERS] containing parameters that are measured in epochs. Parameters in this section will be automatically converted to iterations which caffe uses. This will not affect you, only weights of trained networks will be saved with number of iterations instead of number of epochs.

### OCTREE ADAPTIVE  

Paper: [Adaptive O-CNN: A Patch-based Deep Representation of 3D Shapes](https://wang-ps.github.io/AO-CNN_files/AOCNN.pdf)  
Original Code: [https://github.com/Microsoft/O-CNN](https://github.com/Microsoft/O-CNN)  
Docker: dockers/octree_adaptive  
Data: dockers/data_conversion/octree_data (with adaptive=True)  
Framework: Caffe

Details: Functions the same as original octree network described above.

### VGG

Paper: [Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/pdf/1409.1556.pdf)  
Original Code: [https://github.com/machrisaa/tensorflow-vgg](https://github.com/machrisaa/tensorflow-vgg)  
Docker: dockers/vgg  
Data: dockers/data_conversion/mvcnn_data, dockers/data_conversion/blender_data  
Framework: TensorFlow

Details: This is an implementation of classic VGG network which we modified to vote across multiple views to classify 3D models. Before running the network download pretrained [here](https://mega.nz/#%21xZ8glS6J%21MAnE91ND_WyfZ_8mvkuSa2YcA7q-1ehfSm-Q1fxOvvs) and copy it to dockers/vgg. As with all following networks using multiple images views you can set the number of used views in parameter num_views.

VGG is also used to extract features for SEQ2SEQ network. If you want to do this set weights parameter to number of version of trained VGG you want to use and extract to True. The features will be saved to the root directory of dataset and ready to use by SEQ2SEQ network.

### MVCNN

Paper:  [Multi-view Convolutional Neural Networks for 3D Shape Recognition](https://arxiv.org/pdf/1706.02413.pdf)  
Original Code: [https://github.com/WeiTang114/MVCNN-TensorFlow](https://github.com/WeiTang114/MVCNN-TensorFlow)  
Docker: dockers/mvcnn  
Data: dockers/data_conversion/mvcnn_data, dockers/data_conversion/blender_data  
Framework: TensorFlow

Details: Uses pretrained AlexNet which is prepared automatically. If you want to use different weight, you have to copy them to docker container and change parameter pretrained_network_file.

### MVCNN2

Paper:  [A Deeper Look at 3D Shape Classifiers](http://people.cs.umass.edu/%7Ejcsu/papers/shape_recog/shape_recog.pdf)  
Original Code: [https://github.com/jongchyisu/mvcnn_pytorch](https://github.com/jongchyisu/mvcnn_pytorch)  
Docker: dockers/mvcnn2  
Data: dockers/data_conversion/mvcnn_data, dockers/data_conversion/blender_data  
Framework: PyTorch

Details: This network trains in two phases and requires two different batch sizes to be used, so it has two separate parameters for this. You can choose one of several pretrained networks to use by setting cnn_name parameter. Or you can try training from scratch by setting no_pretraining=True.

### ROTATIONNET

Paper: [Joint Object Categorization and Pose Estimation Using Multiviews from Unsupervised Viewpoints](https://arxiv.org/abs/1603.06208)  
Original Code: [https://github.com/kanezaki/rotationnet](https://github.com/kanezaki/rotationnet)  
Docker: dockers/rotnet  
Data: dockers/data_conversion/mvcnn_data, dockers/data_conversion/blender_data  
Framework: Caffe

Details: Rotation Net is implemented in caffe therefore all issues mentioned in octree section apply. Only there are two separate network file definitions one for training and one for testing.

### SEQ2SEQ

Paper: [SeqViews2SeqLabels: Learning 3D Global Features via Aggregating Sequential Views by RNN with Attention](https://www.cs.umd.edu/%7Ezwicker/publications/SeqView2SeqLabels-TIP18.pdf)  
Original Code: [https://github.com/mingyangShang/SeqViews2SeqLabels](https://github.com/mingyangShang/SeqViews2SeqLabels)  
Docker: dockers/seq2seq  
Data: vgg_features (explained below)  
Framework: TensorFlow

Details: This network gets feature vectors extracted by pretrained image classification network as its input. You can get these features by training and using above described VGG network.  The dimensionality of the feature space is controlled by n_input_fc parameter, VGG vector has 4096 dimensions by default. Paths to .npy files containing extracted features inside the docker container are controlled by train_feature_file, train_label_file and analogous for test dataset.

### POINTNET

Paper: [PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation](https://arxiv.org/abs/1612.00593)  
Original Code: [https://github.com/charlesq34/pointnet](https://github.com/charlesq34/pointnet)  
Docker: dockers/pointnet  
Data: dockers/data_conversion/pnet_data  
Framework: TensorFlow

Details: This network gets point cloud as its input. Number of points used is specified by num_points parameter. During testing you can use voting across several views to get better results, number of rotations is controlled by num_votes parameter.

### POINTNET++

Paper:  [PointNet++: Deep Hierarchical Feature Learning on Point Sets in a Metric Space](https://arxiv.org/pdf/1706.02413.pdf)  
Original Code: [https://github.com/charlesq34/pointnet2](https://github.com/charlesq34/pointnet2)  
Docker: dockers/pointnet2  
Data: dockers/data_conversion/pnet_data  
Framework: TensorFlow

Details: Same as the original pointnet described above.

### SONET

Paper: [SO-Net: Self-Organizing Network for Point Cloud Analysis](http://openaccess.thecvf.com/content_cvpr_2018/papers/Li_SO-Net_Self-Organizing_Network_CVPR_2018_paper.pdf)  
Original Code: [https://github.com/lijx10/SO-Net](https://github.com/lijx10/SO-Net)  
Docker: dockers/sonet  
Data: dockers/data_conversion/sonet_data  
Framework: PyTorch

Details: Similarly to pointnet this network gets specific number of points as its inputvspecified by num_points parameter. In addition to points it uses a self organizing network to get better representation of these points. This is computed by sonet_data docker. The config.ini file contains great number of parameters, for their explanation check original paper and original code.

### KDNET

Paper: [Escape from Cells: Deep Kd-Networks for the Recognition of 3D Point Cloud Models](http://openaccess.thecvf.com/content_ICCV_2017/papers/Klokov_Escape_From_Cells_ICCV_2017_paper.pdf)  
Original Code: [https://github.com/fxia22/kdnet.pytorch](https://github.com/fxia22/kdnet.pytorch)  
Docker: dockers/kdnet  
Data: dockers/data_conversion/kdnet_data  
Framework: PyTorch

Details: This network constructs its input which is kd-trees on the fly. But you still need to prepare mesh data to some more convenient format. You can find several parameters to set in config.ini such as depth of the network which also determines number of used points (2 to the power of depth of the network). You can also explore different data augmentation options under the corresponding section..

## [<<< BACK](README.md)