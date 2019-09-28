Expert Gate: Lifelong Learning with a Network of Experts
========================================

Code for the Paper

**[Expert Fate: Lifelong Learning with a Network of Experts][7]**
<br />
Rahaf Aljundi, Punarjay Chakravarty, Tinne Tuytelaars
[CVPR 2017]

If you find this code useful, please consider citing the original work by authors:

```
@InProceedings{Aljundi_2017_CVPR,
author = {Aljundi, Rahaf and Chakravarty, Punarjay and Tuytelaars, Tinne},
title = {Expert Gate: Lifelong Learning With a Network of Experts},
booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
month = {July},
year = {2017}
}
```

Introduction
---------------------------

Lifelong Machine Learning, or LML, considers systems that can learn many tasks over a lifetime from one or more domains. They retain the knowledge they have learned and use that knowledge to more efficiently and effectively learn new tasks more effectively and efficiently (This is a case of positive inductive bias where the past knoweledge helps the model to perform better on the newer task). 

The problem of Catastrophic Inference or Catstrophic Forgetting is one of the major hurdles facing this domain where the performance of the model inexplicably declines on the older tasks once the newer tasks are introduced into the learning pipeline. 

![Expert Gate Architecture](https://i.imgur.com/0F9gR7P.png)

This paper advocates the use of seperate "experts" for each task such that each expert is called into action when
it faces a training sample that is pertinent to the task on which it is the "expert". They theorize that a shared model would be unable to account for the nuances of each task and hence lead to a performance degradation on
the older tasks as the number of tasks grows

In order to help distinguish between these tasks, the paper proposes to train a single layer autoencoder on each task, wherein the autoencoders; given the input are taught to reconstruct it. In theory, when a task arrives, the autoencoder having the lowest reconstruction error on the task activates it's corresponding "expert". Therby, for each task, the authors train a single layer autoencoder model and an "expert" which derives it's architecture from AlexNet 


Requisites
-----------------------------

* PyTorch: Use the instructions that are outlined on [PyTorch Homepage][1] for installing PyTorch 



Datasets and Designing the experiments
-----------------------------

The original paper uses [Caltech-UCSD Birds][2], [MIT Scenes][3] and [Oxford Flowers][4] in addition to the [ImageNet][5] (for training the autoencoder). Compuatational and hardware limitations necessitated the design of experiments such that the smaller versions of these standard datasets were used. However this was complicated by the two major reasons:

* The smaller versions of most of the standard datsets were not available publically
* The ones that could be found (Oxford 17 categories dataset, Birds 200 categories) were getting corrupted by the system such that the dataloaders in PyTorch were reading in files that were prepended with a _ sign.

The [Tiny-Imagenet][6] dataset was used and the 200 odd classses were split into 4 tasks with 50 classes being assigned to each task randomly. This division can also be arbitrary and no speciaal consideration has been given to the decision to split the dataset evenly. Each of these tasks has a "train" and a "test" folder to validate the performance on these wide ranging tasks.


Training
------------------------------

Training a model on a given task takes place using the **`main.py`** file. Simply execute the following lines to begin the training process

```sh
python3 main.py
```

The file takes the following arguments

* ***data_file***: Path to the data folder to train the model on. **Default**: Empty
* ***init_lr***: Initial learning rate for the model. The learning rate is decayed every 5 epochs.**Default**: 0.1 
* ***num_epochs***: Number of epochs you want to train the model for. **Default**: 40
* ***batch_size***: Batch Size. **Default**: 8

Once you invoke the **`main.py`** file with the appropriate arguments, the following things shall happen

1) The Autoencoder model is trained on the features of the last convolutional layer of an Alexnet model (the preprocessing steps are as detailed in section 3.1 of the paper) and the model is stored in **`./models/autoencoders`** with the appropriate task number. The facility to restart training from a given checkpoint is provided so as to protect agianst abrupt failures whilst training

2) After an autoencoder model is trained, a search is carried out over the already existing autoencoders to determine the "expert" which is most closely related to the new task using the task-relatedness metric described in section 3.3 of the paper

3) After the appropriate model is decided upon, there are two ways to train this model depending on the task-relatedness value as described in section 3.3 of the paper:

	a) LwF approach: Use the method outlined in the [Learning Without Forgetting][8] paper when the value for the task-relatedness is greater than 0.85
	b) Finetuning: If the value if less than 0.85 proceed with the finetuning approach as described in [Learning Without Forgetting][8] 

   The implementation comes with a slight caveat: PyTorch does not allow setting the *train* attribute of some weights in a layer to `True` and the could be set to `False` and some weights could be set to `True`. In order to implement this idea, the gradients for these weights are manually set to zero so as to ensure that these weights don't train

4) The final model is stored in /models/trained_models with a text file `classes.txt` which describe the number of classes that the model was exposed to in this particular task

Refer to the docstrings and the inline comments that are made in `encoder_train.py` and `model_train.py` for a more detailed view


Evaluating the model
-------------------------------

To recreate the experiments performed, first execute the following lines of code

```sh
cd Data
python3 download_datasets.py

```

This will download the tiny-imagenet dataset to the Data folder and split it into 4 tasks with each task consisting of 50 classes each. The directory structure of the downloaded datasets would be: 

```
Data
├── test
│   ├── n01443537
│   └── n01629819
└── train
    ├── n01443537
    └── n01629819

```


Execute the following lines of code to generate to generate the expert models for the 4 tasks

```sh 
cd ../
python3 generate_models.py

``` 

Next to assess how well the model adapts to the different tasks at hand, execute the following lines to generate the final scores

```sh
python3 test_2.py

```

References
----------
1. **Rahaf Aljundi, Punarjay Chakravarty, Tinne Tuytelaars** _Expert Gate: Lifelong Learning with a Network of Experts_ CVPR 2017. [[arxiv][7]]
2. **Zhizhong Li, Derek Hoiem** _Learning without Forgetting_ ECCV 2016 [[arxiv][8]]
3. **Geoffrey Hinton, Oriol Vinyals, Jeff Dean** _Distilling the Knowledge in a Neural Network_ NeurIPs 2014 (formerly known as NIPS). [[arxiv][9]]
4. PyTorch Docs. [[http://pytorch.org/docs/master](http://pytorch.org/docs/master)]

This repository owes a huge credit to the authors of the original [implementation][11] (code in Matlab). This code repository could only be built due to the help offered by countless people on Stack Overflow and PyTorch Discus blogs

License
-------

BSD

[1]: https://pytorch.org 
[2]: http://www.vision.caltech.edu/visipedia/CUB-200.html
[3]: http://places2.csail.mit.edu/
[4]: http://www.robots.ox.ac.uk/~vgg/data/flowers/17/
[5]: http://www.image-net.org/
[6]: https://tiny-imagenet.herokuapp.com/
[7]: https://arxiv.org/abs/1611.06194v2
[8]: https://arxiv.org/abs/1606.09282
[9]: https://arxiv.org/abs/1503.02531
[10]: https://github.com/rahafaljundi/Expert-Gate
