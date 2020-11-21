# Image Classifier

This project is implemented in scope of [Intro to Machine Learning with Pytorch Nanodegree Program](https://www.udacity.com/course/intro-to-machine-learning-nanodegree--nd229).
It will take set of images, train it's model and predict class for a given image.

### Install

This project requires **Python 3.x** and the following Python libraries installed:

- [pytorch](https://pytorch.org/get-started/locally/)
- [NumPy](http://www.numpy.org/)
- [matplotlib](http://matplotlib.org/)

You will also need to have software installed to run and execute an [Jupyter Notebook](https://jupyter.org/)

You can download sample flower dataset from [here](https://s3.amazonaws.com/content.udacity-data.com/nd089/flower_data.tar.gz).

### How to run

- Execute `jupyter notebook` in parent folder to open [Image Classifier Project.ipynb](Image Classifier Project.ipynb)
- Execute `python train.py data_dir` to train data.
    
    -  `python train.py -h` lists script options.
    - Default model is vgg16. You can also use AlexNet. See [Pytorch models](https://pytorch.org/docs/stable/torchvision/models.html).
    - Script outputs checkpoint.pth file which is the trained model and used for prediction.
     
- Execute `python predict.py path/to/image path/to/checkpoint` to predict image.     


