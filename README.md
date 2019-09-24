Expert Gate: Lifelong Learning with a Network of Experts
========================================

Code for the Paper

**[Expert Fate: Lifelong Learning with a Network of Experts][1]**
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

This paper advocates the use of seperate "experts" for each task such that each expert is called into action when
it faces a training sample that is pertinent to the task on which it is the "expert"

In order to help distinguish these tasks, the paper proposes the use of single layer autoencoders to   

