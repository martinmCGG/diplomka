
    <h1>Manual to 3D Classification Survey</h1>
    
    <h2>Abstract</h2>
    <p>We compiled set of publicly available neural networks for classification
      of 3D models. The code works with ModelNet40 and ShapeNetCore datasets
      which are also available online. This is a manual explaining how to
      convert datasets, train and test these networks. <br>
    </p>
    <h2>Requirements</h2>
    <p>To run the code you will need a computer with Linux operating system and
      NVIDIA GPU.</p>
    <p>You will need to install:</p>
    <ul>
      <li>NVIDIA drivers (<a href="https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#package-manager-installation"

          target="_blank">Installation guide here</a>)</li>
      <li>Docker version 1.12 or higher (<a href="https://docs.docker.com/install/"

          target="_blank">Installation guide here</a>)</li>
      <li>NVIDIA Container Runtime for Docker (<a href="https://github.com/NVIDIA/nvidia-docker"

          target="_blank">Installation guide here</a>)</li>
    </ul>
    <p>And that is all! Each neural network is an independent Docker image and
      all its dependencies are installed when building the image. All code is
      written in python.<br>
    </p>
    <h2>Datasets Setup</h2>
    <p dir="ltr" style="line-height:1.38;margin-top:0pt;margin-bottom:0pt;">The
      code is made to work with ModelNet40 and ShapeNetCore datasets. The
      easiest way to run it with custom dataset is to restructure your data so
      it copies the structure of one of these datasets. </p>
    <ul>
      <li>
        <h3 dir="ltr" style=" line-height:1.38;margin-top:0pt;margin-bottom:0pt;">ModelNet40<br>
        </h3>
        <ul>
          <li>Get the dataset <a href="http://modelnet.cs.princeton.edu/" target="_blank">here</a>.
            For experiments we used manually aligned version which can be
            downloaded <a href="https://github.com/lmb-freiburg/orion" target="_blank">here</a>.</li>
        </ul>
      </li>
      <ul>
        <li>Unpack the downloaded archive and you are ready to go!</li>
      </ul>
    </ul>
    <ul>
      <li>ShapeNetCore</li>
      <ul>
        <li>Get the dataset <a href="https://www.shapenet.org/download/shapenetcore"

            target="_blank">here</a>. You need to register and wait for
          confirmation email.</li>
        <li>Unpack the downloaded archive.</li>
        <li>Download official dataset split <a href="http://shapenet.cs.stanford.edu/shapenet/obj-zip/SHREC16/all.csv"

            target="_blank">here</a> and copy it to the root directory of the
          dataset.</li>
      </ul>
    </ul>
    <h2>General Setup</h2>
    <p>You can download all the code from Github <a href="https://github.com/Alobal123/diplomka"

        target="_blank">here</a>.</p>
    <p>Each network is implemented as a separate Docker image. To learn more
      about Docker, images and containers visit <a href="https://docs.docker.com/get-started/"

        target="_blank">this page</a>.</p>
    <p>Each neural network is contained in one directory in /dockers. None of
      the networks accepts mesh files as their input directly, so some data
      conversion is required. All data conversion is implemented in docker with
      same structure as neural networks themselves. Code for data conversion is
      located in /dockers/data_conversion.</p>
    <p>Each directory contains two important files - config.ini and run.sh,
      which you will need to open and edit. Another important file is Dockerfile
      which contains the definition of the docker image. Remaining files contain
      the code which differ from the original network implementation. Original
      network code is downloaded automatically when building the image and you
      can find the download link below.</p>
    <p>run.sh is a runnable script which builds the docker image, runs the
      docker container and executes the neural network or data conversion. You
      will need to setup a couple of variables here:</p>
    <ul>
      <li>name - will be used as a name of the docker image and docker
        container. You can leave this at default value unless it is in conflict
        with some already existing image or you want to run more instances of
        this image at once. With data conversion scripts the name is the name of
        the converted dataset and directory of the same name will be created.
        The name of the image can be changed by changing variable image_name in
        this case.</li>
      <li>dataset_path -&nbsp; contains path to the root directory of the
        dataset on your filesystem.</li>
      <li>out_path - contains path to the directory where training logs and
        network weights will be saved. This directory will be created
        automatically.</li>
      <li>GPU - index of GPU which will be visible to docker container. Have to
        be a single integer. We currently do not support multiple GPUs.</li>
      <li>docker_hidden - Must be one of t or d. With t the container will be
        run in interactive mode, meaning it will run in your console. With d it
        will in detached mode i.e. in the background. For more information check
        docker documentation <a href="https://docs.docker.com/engine/reference/run/"

          target="_blank">here</a>.</li>
    </ul>
    <p>config.ini contains most of the relevant parameters of the network or
      data conversion. The file is split to sections where each section is
      started by [SECTION] statement. Then on each line a parameter in format
      key = value. You can find explanation of network parameters in later
      sections. </p>
    <ul>
    </ul>
    <h2> Data conversion</h2>
    <p>To convert your dataset you need to set the parameters described above
      and then simply run script run.sh in your console. This will convert the
      dataset to various formats directly readable by the neural networks.</p>
    <p>Parameters for data conversion in config.ini file: </p>
    <ul>
      <li>data - path to the dataset inside the container. Does not have to be
        changed.</li>
      <li>output - path to the directory inside the container where converted
        dataset will be saved. Does not have to be changed.</li>
      <li>log_file - path and name of the file where progress of the data
        conversion will be written.</li>
      <li>num_threads - maximum number of threads to use.</li>
      <li>dataset_type -&nbsp; which dataset is converting. Must be one of
        modelnet or shapenet currently.</li>
    </ul>
    <p>For more detail about individual data conversion scripts, continue <a href="conversions.html"

        target="_top">here</a>.</p>
    <h2>Neural Networks</h2>
    <p>Each of the neural networks is implemented in python but in different
      framework. That is why we used the docker infrastructure. We try to
      present some unified framework to easily test and train the networks
      without changing the code. This section will briefly introduce used
      networks and some of their most important parameters. </p>
    <p>Parameters common to all neural networks: </p>
    <ul>
      <li>name - will be used as the name of the experiment used in log files.</li>
      <li>data - path to the dataset inside the container. Does not have to be
        changed.</li>
      <li>log_dir - path to the directory inside the container where logs and
        weights will be saved. Does not have to be changed.</li>
      <li>num_classes - number of classes in the dataset. </li>
      <li>batch_size - size of the batch for training and testing neural
        networks.</li>
      <li>weights - if you want to test or finetune already trained network,
        this should be the number of this model. If you want to train from
        scratch, this should be -1. </li>
      <li>snapshot_prefix - name of the file where weights will be saved. Number
        of training epoch when these weights are saved will be added to this.</li>
    </ul>
    <ul>
      <li>max_epoch - number of epochs to train for. One epoch means one pass
        through the training part of the dataset. </li>
      <li>save_period - the trained network will be saved every epoch divisible
        by save_period. </li>
      <li>train_log_frq - frequency of logging during training. It is roughly
        number of examples seen by network.</li>
    </ul>
    <ul>
      <li> test - if you want to only test already trained network, set this to
        True. weights parameter has to have a valid value bigger than -1. Should
        be False for training. </li>
    </ul>
    <p>For more details about individual networks, continue <a href="networks.html"

        target="_top">here</a>.</p>
    <h2>Logging and Evaluation</h2>
    <p>Our framework offers some basic logging options. It saves several .csv to
      the logging directory. The logger keeps track of time of the training,
      training epochs and some other value. By default four values are tracked:
      training loss, training accuracy, test loss and test accuracy. Evaluation
      on the test set is performed after each epoch of training. Also some basic
      graphs using matplotlib library are created and saved during training.
      When testing your already trained network (using test = True) text file is
      saved where network category prediction is saved along with a simple html
      confusion matrix. </p>
    <p><br>
    </p>
    <p> </p>

