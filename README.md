# object-detection

[TOC]

This is a list of awesome articles about object detection. If you want to read the paper according to time, you can refer to [Date](Date.md).

- R-CNN
- Fast R-CNN
- Faster R-CNN
- Mask R-CNN
- Light-Head R-CNN
- Cascade R-CNN
- SPP-Net
- YOLO
- YOLOv2
- YOLOv3
- YOLT
- SSD
- DSSD
- FSSD
- ESSD
- MDSSD
- Pelee
- Fire SSD
- R-FCN
- FPN
- DSOD
- RetinaNet
- MegDet
- RefineNet
- DetNet
- SSOD
- CornerNet
- M2Det
- 3D Object Detection
- ZSD（Zero-Shot Object Detection）
- OSD（One-Shot object Detection）
- Weakly Supervised Object Detection
- Softer-NMS
- 2018
- 2019
- Other

Based on handong1587's github: https://handong1587.github.io/deep_learning/2015/10/09/object-detection.html

# Survey

**Recent Advances in Deep Learning for Object Detection**

- intro: From 2013 (OverFeat) to 2019 (DetNAS)
- arXiv: <https://arxiv.org/abs/1908.03673>

**A Survey of Deep Learning-based Object Detection**

- intro：From Fast R-CNN to NAS-FPN

- arXiv：<https://arxiv.org/abs/1907.09408>

**Object Detection in 20 Years: A Survey**

- intro：This work has been submitted to the IEEE TPAMI for possible publication
- arXiv：<https://arxiv.org/abs/1905.05055>

**《Recent Advances in Object Detection in the Age of Deep Convolutional Neural Networks》**

- intro: awesome


- arXiv: https://arxiv.org/abs/1809.03193

**《Deep Learning for Generic Object Detection: A Survey》**

- intro: Submitted to IJCV 2018
- arXiv: https://arxiv.org/abs/1809.02165

# Papers&Codes

## R-CNN

**Rich feature hierarchies for accurate object detection and semantic segmentation**

- intro: R-CNN
- arxiv: <http://arxiv.org/abs/1311.2524>
- supp: <http://people.eecs.berkeley.edu/~rbg/papers/r-cnn-cvpr-supp.pdf>
- slides: <http://www.image-net.org/challenges/LSVRC/2013/slides/r-cnn-ilsvrc2013-workshop.pdf>
- slides: <http://www.cs.berkeley.edu/~rbg/slides/rcnn-cvpr14-slides.pdf>
- github: <https://github.com/rbgirshick/rcnn>
- notes: <http://zhangliliang.com/2014/07/23/paper-note-rcnn/>
- caffe-pr("Make R-CNN the Caffe detection example"): <https://github.com/BVLC/caffe/pull/482>

## Fast R-CNN

**Fast R-CNN**

- arxiv: <http://arxiv.org/abs/1504.08083>
- slides: <http://tutorial.caffe.berkeleyvision.org/caffe-cvpr15-detection.pdf>
- github: <https://github.com/rbgirshick/fast-rcnn>
- github(COCO-branch): <https://github.com/rbgirshick/fast-rcnn/tree/coco>
- webcam demo: <https://github.com/rbgirshick/fast-rcnn/pull/29>
- notes: <http://zhangliliang.com/2015/05/17/paper-note-fast-rcnn/>
- notes: <http://blog.csdn.net/linj_m/article/details/48930179>
- github("Fast R-CNN in MXNet"): <https://github.com/precedenceguo/mx-rcnn>
- github: <https://github.com/mahyarnajibi/fast-rcnn-torch>
- github: <https://github.com/apple2373/chainer-simple-fast-rnn>
- github: <https://github.com/zplizzi/tensorflow-fast-rcnn>

**A-Fast-RCNN: Hard Positive Generation via Adversary for Object Detection**

- intro: CVPR 2017
- arxiv: <https://arxiv.org/abs/1704.03414>
- paper: <http://abhinavsh.info/papers/pdfs/adversarial_object_detection.pdf>
- github(Caffe): <https://github.com/xiaolonw/adversarial-frcnn>

## Faster R-CNN

**Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks**

- intro: NIPS 2015
- arxiv: <http://arxiv.org/abs/1506.01497>
- gitxiv: <http://www.gitxiv.com/posts/8pfpcvefDYn2gSgXk/faster-r-cnn-towards-real-time-object-detection-with-region>
- slides: <http://web.cs.hacettepe.edu.tr/~aykut/classes/spring2016/bil722/slides/w05-FasterR-CNN.pdf>
- github(official, Matlab): <https://github.com/ShaoqingRen/faster_rcnn>
- github(Caffe): <https://github.com/rbgirshick/py-faster-rcnn>
- github(MXNet): <https://github.com/msracver/Deformable-ConvNets/tree/master/faster_rcnn>
- github(PyTorch--recommend): <https://github.com//jwyang/faster-rcnn.pytorch>
- github: <https://github.com/mitmul/chainer-faster-rcnn>
- github(Torch):: <https://github.com/andreaskoepf/faster-rcnn.torch>
- github(Torch):: <https://github.com/ruotianluo/Faster-RCNN-Densecap-torch>
- github(TensorFlow): <https://github.com/smallcorgi/Faster-RCNN_TF>
- github(TensorFlow): <https://github.com/CharlesShang/TFFRCNN>
- github(C++ demo): <https://github.com/YihangLou/FasterRCNN-Encapsulation-Cplusplus>
- github(Keras): <https://github.com/yhenon/keras-frcnn>
- github: <https://github.com/Eniac-Xie/faster-rcnn-resnet>
- github(C++): <https://github.com/D-X-Y/caffe-faster-rcnn/tree/dev>

**R-CNN minus R**

- intro: BMVC 2015
- arxiv: <http://arxiv.org/abs/1506.06981>

**Faster R-CNN in MXNet with distributed implementation and data parallelization**

- github: <https://github.com/dmlc/mxnet/tree/master/example/rcnn>

**Contextual Priming and Feedback for Faster R-CNN**

- intro: ECCV 2016. Carnegie Mellon University
- paper: <http://abhinavsh.info/context_priming_feedback.pdf>
- poster: <http://www.eccv2016.org/files/posters/P-1A-20.pdf>

**An Implementation of Faster RCNN with Study for Region Sampling**

- intro: Technical Report, 3 pages. CMU
- arxiv: <https://arxiv.org/abs/1702.02138>
- github: <https://github.com/endernewton/tf-faster-rcnn>
- github: https://github.com/ruotianluo/pytorch-faster-rcnn

**Interpretable R-CNN**

- intro: North Carolina State University & Alibaba
- keywords: AND-OR Graph (AOG)
- arxiv: <https://arxiv.org/abs/1711.05226>

**Domain Adaptive Faster R-CNN for Object Detection in the Wild**

- intro: CVPR 2018. ETH Zurich & ESAT/PSI
- arxiv: <https://arxiv.org/abs/1803.03243>

## Mask R-CNN

- arxiv: <http://arxiv.org/abs/1703.06870>
- github(Keras): https://github.com/matterport/Mask_RCNN
- github(Caffe2): https://github.com/facebookresearch/Detectron
- github(Pytorch): <https://github.com/wannabeOG/Mask-RCNN>
- github(MXNet): https://github.com/TuSimple/mx-maskrcnn
- github(Chainer): https://github.com/DeNA/Chainer_Mask_R-CNN

## Light-Head R-CNN

**Light-Head R-CNN: In Defense of Two-Stage Object Detector**

- intro: Tsinghua University & Megvii Inc
- arxiv: <https://arxiv.org/abs/1711.07264>
- github(offical): https://github.com/zengarden/light_head_rcnn
- github: <https://github.com/terrychenism/Deformable-ConvNets/blob/master/rfcn/symbols/resnet_v1_101_rfcn_light.py#L784>

## Cascade R-CNN

**Cascade R-CNN: Delving into High Quality Object Detection**

- arxiv: <https://arxiv.org/abs/1712.00726>
- github: <https://github.com/zhaoweicai/cascade-rcnn>

## SPP-Net

**Spatial Pyramid Pooling in Deep Convolutional Networks for Visual Recognition**

- intro: ECCV 2014 / TPAMI 2015
- arxiv: <http://arxiv.org/abs/1406.4729>
- github: <https://github.com/ShaoqingRen/SPP_net>
- notes: <http://zhangliliang.com/2014/09/13/paper-note-sppnet/>

**DeepID-Net: Deformable Deep Convolutional Neural Networks for Object Detection**

- intro: PAMI 2016
- intro: an extension of R-CNN. box pre-training, cascade on region proposals, deformation layers and context representations
- project page: <http://www.ee.cuhk.edu.hk/%CB%9Cwlouyang/projects/imagenetDeepId/index.html>
- arxiv: <http://arxiv.org/abs/1412.5661>

**Object Detectors Emerge in Deep Scene CNNs**

- intro: ICLR 2015
- arxiv: <http://arxiv.org/abs/1412.6856>
- paper: <https://www.robots.ox.ac.uk/~vgg/rg/papers/zhou_iclr15.pdf>
- paper: <https://people.csail.mit.edu/khosla/papers/iclr2015_zhou.pdf>
- slides: <http://places.csail.mit.edu/slide_iclr2015.pdf>

**segDeepM: Exploiting Segmentation and Context in Deep Neural Networks for Object Detection**

- intro: CVPR 2015
- project(code+data): <https://www.cs.toronto.edu/~yukun/segdeepm.html>
- arxiv: <https://arxiv.org/abs/1502.04275>
- github: <https://github.com/YknZhu/segDeepM>

**Object Detection Networks on Convolutional Feature Maps**

- intro: TPAMI 2015
- keywords: NoC
- arxiv: <http://arxiv.org/abs/1504.06066>

**Improving Object Detection with Deep Convolutional Networks via Bayesian Optimization and Structured Prediction**

- arxiv: <http://arxiv.org/abs/1504.03293>
- slides: <http://www.ytzhang.net/files/publications/2015-cvpr-det-slides.pdf>
- github: <https://github.com/YutingZhang/fgs-obj>

**DeepBox: Learning Objectness with Convolutional Networks**

- keywords: DeepBox
- arxiv: <http://arxiv.org/abs/1505.02146>
- github: <https://github.com/weichengkuo/DeepBox>

## YOLO

**You Only Look Once: Unified, Real-Time Object Detection**

[![img](https://camo.githubusercontent.com/e69d4118b20a42de4e23b9549f9a6ec6dbbb0814/687474703a2f2f706a7265646469652e636f6d2f6d656469612f66696c65732f6461726b6e65742d626c61636b2d736d616c6c2e706e67)](https://camo.githubusercontent.com/e69d4118b20a42de4e23b9549f9a6ec6dbbb0814/687474703a2f2f706a7265646469652e636f6d2f6d656469612f66696c65732f6461726b6e65742d626c61636b2d736d616c6c2e706e67)

- arxiv: <http://arxiv.org/abs/1506.02640>
- code: <https://pjreddie.com/darknet/yolov1/>
- github: <https://github.com/pjreddie/darknet>
- blog: <https://pjreddie.com/darknet/yolov1/>
- slides: <https://docs.google.com/presentation/d/1aeRvtKG21KHdD5lg6Hgyhx5rPq_ZOsGjG5rJ1HP7BbA/pub?start=false&loop=false&delayms=3000&slide=id.p>
- reddit: <https://www.reddit.com/r/MachineLearning/comments/3a3m0o/realtime_object_detection_with_yolo/>
- github: <https://github.com/gliese581gg/YOLO_tensorflow>
- github: <https://github.com/xingwangsfu/caffe-yolo>
- github: <https://github.com/frankzhangrui/Darknet-Yolo>
- github: <https://github.com/BriSkyHekun/py-darknet-yolo>
- github: <https://github.com/tommy-qichang/yolo.torch>
- github: <https://github.com/frischzenger/yolo-windows>
- github: <https://github.com/AlexeyAB/yolo-windows>
- github: <https://github.com/nilboy/tensorflow-yolo>

**darkflow - translate darknet to tensorflow. Load trained weights, retrain/fine-tune them using tensorflow, export constant graph def to C++**

- blog: <https://thtrieu.github.io/notes/yolo-tensorflow-graph-buffer-cpp>
- github: <https://github.com/thtrieu/darkflow>

**Start Training YOLO with Our Own Data**

[![img](https://camo.githubusercontent.com/2f99b692dd7ce47d7832385f3e8a6654e680d92a/687474703a2f2f6775616e6768616e2e696e666f2f626c6f672f656e2f77702d636f6e74656e742f75706c6f6164732f323031352f31322f696d616765732d34302e6a7067)](https://camo.githubusercontent.com/2f99b692dd7ce47d7832385f3e8a6654e680d92a/687474703a2f2f6775616e6768616e2e696e666f2f626c6f672f656e2f77702d636f6e74656e742f75706c6f6164732f323031352f31322f696d616765732d34302e6a7067)

- intro: train with customized data and class numbers/labels. Linux / Windows version for darknet.
- blog: <http://guanghan.info/blog/en/my-works/train-yolo/>
- github: <https://github.com/Guanghan/darknet>

**YOLO: Core ML versus MPSNNGraph**

- intro: Tiny YOLO for iOS implemented using CoreML but also using the new MPS graph API.
- blog: <http://machinethink.net/blog/yolo-coreml-versus-mps-graph/>
- github: <https://github.com/hollance/YOLO-CoreML-MPSNNGraph>

**TensorFlow YOLO object detection on Android**

- intro: Real-time object detection on Android using the YOLO network with TensorFlow
- github: <https://github.com/natanielruiz/android-yolo>

**Computer Vision in iOS – Object Detection**

- blog: <https://sriraghu.com/2017/07/12/computer-vision-in-ios-object-detection/>
- github:<https://github.com/r4ghu/iOS-CoreML-Yolo>

## YOLOv2

**YOLO9000: Better, Faster, Stronger**

- arxiv: <https://arxiv.org/abs/1612.08242>
- code: <http://pjreddie.com/yolo9000/>    https://pjreddie.com/darknet/yolov2/
- github(Chainer): <https://github.com/leetenki/YOLOv2>
- github(Keras): <https://github.com/allanzelener/YAD2K>
- github(PyTorch): <https://github.com/longcw/yolo2-pytorch>
- github(Tensorflow): <https://github.com/hizhangp/yolo_tensorflow>
- github(Windows): <https://github.com/AlexeyAB/darknet>
- github: <https://github.com/choasUp/caffe-yolo9000>
- github: <https://github.com/philipperemy/yolo-9000>
- github(TensorFlow): <https://github.com/KOD-Chen/YOLOv2-Tensorflow>
- github(Keras): <https://github.com/yhcc/yolo2>
- github(Keras): <https://github.com/experiencor/keras-yolo2>
- github(TensorFlow): <https://github.com/WojciechMormul/yolo2>

**darknet_scripts**

- intro: Auxilary scripts to work with (YOLO) darknet deep learning famework. AKA -> How to generate YOLO anchors?
- github: <https://github.com/Jumabek/darknet_scripts>

**Yolo_mark: GUI for marking bounded boxes of objects in images for training Yolo v2**

- github: <https://github.com/AlexeyAB/Yolo_mark>

**LightNet: Bringing pjreddie's DarkNet out of the shadows**

<https://github.com//explosion/lightnet>

**YOLO v2 Bounding Box Tool**

- intro: Bounding box labeler tool to generate the training data in the format YOLO v2 requires.
- github: <https://github.com/Cartucho/yolo-boundingbox-labeler-GUI>

**Loss Rank Mining: A General Hard Example Mining Method for Real-time Detectors**

- intro: **LRM** is the first hard example mining strategy which could fit YOLOv2 perfectly and make it better applied in series of real scenarios where both real-time rates and accurate detection are strongly demanded.
- arxiv: https://arxiv.org/abs/1804.04606

**Object detection at 200 Frames Per Second**

- intro: faster than Tiny-Yolo-v2
- arxiv: https://arxiv.org/abs/1805.06361

**Event-based Convolutional Networks for Object Detection in Neuromorphic Cameras**

- intro: YOLE--Object Detection in Neuromorphic Cameras
- arxiv:https://arxiv.org/abs/1805.07931

**OmniDetector: With Neural Networks to Bounding Boxes**

- intro: a person detector on n fish-eye images of indoor scenes（NIPS 2018）
- arxiv:https://arxiv.org/abs/1805.08503
- datasets:https://gitlab.com/omnidetector/omnidetector

## YOLOv3

**YOLOv3: An Incremental Improvement**

- arxiv:https://arxiv.org/abs/1804.02767
- paper:https://pjreddie.com/media/files/papers/YOLOv3.pdf
- code: <https://pjreddie.com/darknet/yolo/>
- github(Official):https://github.com/pjreddie/darknet
- github:https://github.com/mystic123/tensorflow-yolo-v3
- github:https://github.com/experiencor/keras-yolo3
- github:https://github.com/qqwweee/keras-yolo3
- github:https://github.com/marvis/pytorch-yolo3
- github:https://github.com/ayooshkathuria/pytorch-yolo-v3
- github:https://github.com/ayooshkathuria/YOLO_v3_tutorial_from_scratch
- github:https://github.com/eriklindernoren/PyTorch-YOLOv3
- github:https://github.com/ultralytics/yolov3
- github:https://github.com/BobLiu20/YOLOv3_PyTorch
- github:https://github.com/andy-yun/pytorch-0.4-yolov3
- github:https://github.com/DeNA/PyTorch_YOLOv3

## YOLT

**You Only Look Twice: Rapid Multi-Scale Object Detection In Satellite Imagery**

- intro: Small Object Detection


- arxiv:https://arxiv.org/abs/1805.09512
- github:https://github.com/avanetten/yolt

## SSD

**SSD: Single Shot MultiBox Detector**

[![img](https://camo.githubusercontent.com/ad9b147ed3a5f48ffb7c3540711c15aa04ce49c6/687474703a2f2f7777772e63732e756e632e6564752f7e776c69752f7061706572732f7373642e706e67)](https://camo.githubusercontent.com/ad9b147ed3a5f48ffb7c3540711c15aa04ce49c6/687474703a2f2f7777772e63732e756e632e6564752f7e776c69752f7061706572732f7373642e706e67)

- intro: ECCV 2016 Oral
- arxiv: <http://arxiv.org/abs/1512.02325>
- paper: <http://www.cs.unc.edu/~wliu/papers/ssd.pdf>
- slides: [http://www.cs.unc.edu/%7Ewliu/papers/ssd_eccv2016_slide.pdf](http://www.cs.unc.edu/~wliu/papers/ssd_eccv2016_slide.pdf)
- github(Official): <https://github.com/weiliu89/caffe/tree/ssd>
- video: <http://weibo.com/p/2304447a2326da963254c963c97fb05dd3a973>
- github: <https://github.com/zhreshold/mxnet-ssd>
- github: <https://github.com/zhreshold/mxnet-ssd.cpp>
- github: <https://github.com/rykov8/ssd_keras>
- github: <https://github.com/balancap/SSD-Tensorflow>
- github: <https://github.com/amdegroot/ssd.pytorch>
- github(Caffe): <https://github.com/chuanqi305/MobileNet-SSD>

**What's the diffience in performance between this new code you pushed and the previous code? #327**

<https://github.com/weiliu89/caffe/issues/327>

## DSSD

**DSSD : Deconvolutional Single Shot Detector**

- intro: UNC Chapel Hill & Amazon Inc
- arxiv: <https://arxiv.org/abs/1701.06659>
- github: <https://github.com/chengyangfu/caffe/tree/dssd>
- github: <https://github.com/MTCloudVision/mxnet-dssd>
- demo: <http://120.52.72.53/www.cs.unc.edu/c3pr90ntc0td/~cyfu/dssd_lalaland.mp4>

**Enhancement of SSD by concatenating feature maps for object detection**

- intro: rainbow SSD (R-SSD)
- arxiv: <https://arxiv.org/abs/1705.09587>

**Context-aware Single-Shot Detector**

- keywords: CSSD, DiCSSD, DeCSSD, effective receptive fields (ERFs), theoretical receptive fields (TRFs)
- arxiv: <https://arxiv.org/abs/1707.08682>

**Feature-Fused SSD: Fast Detection for Small Objects**

<https://arxiv.org/abs/1709.05054>

## FSSD

**FSSD: Feature Fusion Single Shot Multibox Detector**

<https://arxiv.org/abs/1712.00960>

**Weaving Multi-scale Context for Single Shot Detector**

- intro: WeaveNet
- keywords: fuse multi-scale information
- arxiv: <https://arxiv.org/abs/1712.03149>

## ESSD

**Extend the shallow part of Single Shot MultiBox Detector via Convolutional Neural Network**

<https://arxiv.org/abs/1801.05918>

**Tiny SSD: A Tiny Single-shot Detection Deep Convolutional Neural Network for Real-time Embedded Object Detection**

<https://arxiv.org/abs/1802.06488>

## MDSSD

**MDSSD: Multi-scale Deconvolutional Single Shot Detector for small objects**

- arxiv: https://arxiv.org/abs/1805.07009

## Pelee

**Pelee: A Real-Time Object Detection System on Mobile Devices**

https://github.com/Robert-JunWang/Pelee

- intro: (ICLR 2018 workshop track)


- arxiv: https://arxiv.org/abs/1804.06882
- github: https://github.com/Robert-JunWang/Pelee

## Fire SSD

**Fire SSD: Wide Fire Modules based Single Shot Detector on Edge Device**

- intro:low cost, fast speed and high mAP on  factor edge computing devices


- arxiv:https://arxiv.org/abs/1806.05363

## R-FCN

**R-FCN: Object Detection via Region-based Fully Convolutional Networks**

- arxiv: <http://arxiv.org/abs/1605.06409>
- github: <https://github.com/daijifeng001/R-FCN>
- github(MXNet): <https://github.com/msracver/Deformable-ConvNets/tree/master/rfcn>
- github: <https://github.com/Orpine/py-R-FCN>
- github: <https://github.com/PureDiors/pytorch_RFCN>
- github: <https://github.com/bharatsingh430/py-R-FCN-multiGPU>
- github: <https://github.com/xdever/RFCN-tensorflow>

**R-FCN-3000 at 30fps: Decoupling Detection and Classification**

<https://arxiv.org/abs/1712.01802>

**Recycle deep features for better object detection**

- arxiv: <http://arxiv.org/abs/1607.05066>

## FPN

**Feature Pyramid Networks for Object Detection**

- intro: Facebook AI Research
- arxiv: <https://arxiv.org/abs/1612.03144>

**Action-Driven Object Detection with Top-Down Visual Attentions**

- arxiv: <https://arxiv.org/abs/1612.06704>

**Beyond Skip Connections: Top-Down Modulation for Object Detection**

- intro: CMU & UC Berkeley & Google Research
- arxiv: <https://arxiv.org/abs/1612.06851>

**Wide-Residual-Inception Networks for Real-time Object Detection**

- intro: Inha University
- arxiv: <https://arxiv.org/abs/1702.01243>

**Attentional Network for Visual Object Detection**

- intro: University of Maryland & Mitsubishi Electric Research Laboratories
- arxiv: <https://arxiv.org/abs/1702.01478>

**Learning Chained Deep Features and Classifiers for Cascade in Object Detection**

- keykwords: CC-Net
- intro: chained cascade network (CC-Net). 81.1% mAP on PASCAL VOC 2007
- arxiv: <https://arxiv.org/abs/1702.07054>

**DeNet: Scalable Real-time Object Detection with Directed Sparse Sampling**

- intro: ICCV 2017 (poster)
- arxiv: <https://arxiv.org/abs/1703.10295>

**Discriminative Bimodal Networks for Visual Localization and Detection with Natural Language Queries**

- intro: CVPR 2017
- arxiv: <https://arxiv.org/abs/1704.03944>

**Spatial Memory for Context Reasoning in Object Detection**

- arxiv: <https://arxiv.org/abs/1704.04224>

**Accurate Single Stage Detector Using Recurrent Rolling Convolution**

- intro: CVPR 2017. SenseTime
- keywords: Recurrent Rolling Convolution (RRC)
- arxiv: <https://arxiv.org/abs/1704.05776>
- github: <https://github.com/xiaohaoChen/rrc_detection>

**Deep Occlusion Reasoning for Multi-Camera Multi-Target Detection**

<https://arxiv.org/abs/1704.05775>

**LCDet: Low-Complexity Fully-Convolutional Neural Networks for Object Detection in Embedded Systems**

- intro: Embedded Vision Workshop in CVPR. UC San Diego & Qualcomm Inc
- arxiv: <https://arxiv.org/abs/1705.05922>

**Point Linking Network for Object Detection**

- intro: Point Linking Network (PLN)
- arxiv: <https://arxiv.org/abs/1706.03646>

**Perceptual Generative Adversarial Networks for Small Object Detection**

<https://arxiv.org/abs/1706.05274>

**Few-shot Object Detection**

<https://arxiv.org/abs/1706.08249>

**Yes-Net: An effective Detector Based on Global Information**

<https://arxiv.org/abs/1706.09180>

**SMC Faster R-CNN: Toward a scene-specialized multi-object detector**

<https://arxiv.org/abs/1706.10217>

**Towards lightweight convolutional neural networks for object detection**

<https://arxiv.org/abs/1707.01395>

**RON: Reverse Connection with Objectness Prior Networks for Object Detection**

- intro: CVPR 2017
- arxiv: <https://arxiv.org/abs/1707.01691>
- github: <https://github.com/taokong/RON>

**Mimicking Very Efficient Network for Object Detection**

- intro: CVPR 2017. SenseTime & Beihang University
- paper: <http://openaccess.thecvf.com/content_cvpr_2017/papers/Li_Mimicking_Very_Efficient_CVPR_2017_paper.pdf>

**Residual Features and Unified Prediction Network for Single Stage Detection**

<https://arxiv.org/abs/1707.05031>

**Deformable Part-based Fully Convolutional Network for Object Detection**

- intro: BMVC 2017 (oral). Sorbonne Universités & CEDRIC
- arxiv: <https://arxiv.org/abs/1707.06175>

**Adaptive Feeding: Achieving Fast and Accurate Detections by Adaptively Combining Object Detectors**

- intro: ICCV 2017
- arxiv: <https://arxiv.org/abs/1707.06399>

**Recurrent Scale Approximation for Object Detection in CNN**

- intro: ICCV 2017
- keywords: Recurrent Scale Approximation (RSA)
- arxiv: <https://arxiv.org/abs/1707.09531>
- github: <https://github.com/sciencefans/RSA-for-object-detection>

## DSOD

**DSOD: Learning Deeply Supervised Object Detectors from Scratch**

![img](https://user-images.githubusercontent.com/3794909/28934967-718c9302-78b5-11e7-89ee-8b514e53e23c.png)

- intro: ICCV 2017. Fudan University & Tsinghua University & Intel Labs China
- arxiv: <https://arxiv.org/abs/1708.01241>
- github: <https://github.com/szq0214/DSOD>
- github:https://github.com/Windaway/DSOD-Tensorflow
- github:https://github.com/chenyuntc/dsod.pytorch

**Learning Object Detectors from Scratch with Gated Recurrent Feature Pyramids**

- arxiv:https://arxiv.org/abs/1712.00886
- github:https://github.com/szq0214/GRP-DSOD

**Tiny-DSOD: Lightweight Object Detection for Resource-Restricted Usages**

- intro: BMVC 2018
- arXiv: https://arxiv.org/abs/1807.11013

**Object Detection from Scratch with Deep Supervision**

- intro: This is an extended version of DSOD
- arXiv: https://arxiv.org/abs/1809.09294

## RetinaNet

**Focal Loss for Dense Object Detection**

- intro: ICCV 2017 Best student paper award. Facebook AI Research
- keywords: RetinaNet
- arxiv: <https://arxiv.org/abs/1708.02002>

**CoupleNet: Coupling Global Structure with Local Parts for Object Detection**

- intro: ICCV 2017
- arxiv: <https://arxiv.org/abs/1708.02863>

**Incremental Learning of Object Detectors without Catastrophic Forgetting**

- intro: ICCV 2017. Inria
- arxiv: <https://arxiv.org/abs/1708.06977>

**Zoom Out-and-In Network with Map Attention Decision for Region Proposal and Object Detection**

<https://arxiv.org/abs/1709.04347>

**StairNet: Top-Down Semantic Aggregation for Accurate One Shot Detection**

<https://arxiv.org/abs/1709.05788>

**Dynamic Zoom-in Network for Fast Object Detection in Large Images**

<https://arxiv.org/abs/1711.05187>

**Zero-Annotation Object Detection with Web Knowledge Transfer**

- intro: NTU, Singapore & Amazon
- keywords: multi-instance multi-label domain adaption learning framework
- arxiv: <https://arxiv.org/abs/1711.05954>

## MegDet

**MegDet: A Large Mini-Batch Object Detector**

- intro: Peking University & Tsinghua University & Megvii Inc
- arxiv: <https://arxiv.org/abs/1711.07240>

**Receptive Field Block Net for Accurate and Fast Object Detection**

- intro: RFBNet
- arxiv: <https://arxiv.org/abs/1711.07767>
- github: <https://github.com//ruinmessi/RFBNet>

**An Analysis of Scale Invariance in Object Detection - SNIP**

- arxiv: <https://arxiv.org/abs/1711.08189>
- github: <https://github.com/bharatsingh430/snip>

**Feature Selective Networks for Object Detection**

<https://arxiv.org/abs/1711.08879>

**Learning a Rotation Invariant Detector with Rotatable Bounding Box**

- arxiv: <https://arxiv.org/abs/1711.09405>
- github: <https://github.com/liulei01/DRBox>

**Scalable Object Detection for Stylized Objects**

- intro: Microsoft AI & Research Munich
- arxiv: <https://arxiv.org/abs/1711.09822>

**Learning Object Detectors from Scratch with Gated Recurrent Feature Pyramids**

- arxiv: <https://arxiv.org/abs/1712.00886>
- github: <https://github.com/szq0214/GRP-DSOD>

**Deep Regionlets for Object Detection**

- keywords: region selection network, gating network
- arxiv: <https://arxiv.org/abs/1712.02408>

**Training and Testing Object Detectors with Virtual Images**

- intro: IEEE/CAA Journal of Automatica Sinica
- arxiv: <https://arxiv.org/abs/1712.08470>

**Large-Scale Object Discovery and Detector Adaptation from Unlabeled Video**

- keywords: object mining, object tracking, unsupervised object discovery by appearance-based clustering, self-supervised detector adaptation
- arxiv: <https://arxiv.org/abs/1712.08832>

**Spot the Difference by Object Detection**

- intro: Tsinghua University & JD Group
- arxiv: <https://arxiv.org/abs/1801.01051>

**Localization-Aware Active Learning for Object Detection**

- arxiv: <https://arxiv.org/abs/1801.05124>

**Object Detection with Mask-based Feature Encoding**

- arxiv: <https://arxiv.org/abs/1802.03934>

**LSTD: A Low-Shot Transfer Detector for Object Detection**

- intro: AAAI 2018
- arxiv: <https://arxiv.org/abs/1803.01529>

**Pseudo Mask Augmented Object Detection**

<https://arxiv.org/abs/1803.05858>

**Revisiting RCNN: On Awakening the Classification Power of Faster RCNN**

<https://arxiv.org/abs/1803.06799>

**Learning Region Features for Object Detection**

- intro: Peking University & MSRA
- arxiv: <https://arxiv.org/abs/1803.07066>

**Single-Shot Bidirectional Pyramid Networks for High-Quality Object Detection**

- intro: Singapore Management University & Zhejiang University
- arxiv: <https://arxiv.org/abs/1803.08208>

**Object Detection for Comics using Manga109 Annotations**

- intro: University of Tokyo & National Institute of Informatics, Japan
- arxiv: <https://arxiv.org/abs/1803.08670>

**Task-Driven Super Resolution: Object Detection in Low-resolution Images**

- arxiv: <https://arxiv.org/abs/1803.11316>

**Transferring Common-Sense Knowledge for Object Detection**

- arxiv: <https://arxiv.org/abs/1804.01077>

**Multi-scale Location-aware Kernel Representation for Object Detection**

- intro: CVPR 2018
- arxiv: <https://arxiv.org/abs/1804.00428>
- github: <https://github.com/Hwang64/MLKP>


**Loss Rank Mining: A General Hard Example Mining Method for Real-time Detectors**

- intro: National University of Defense Technology
- arxiv: https://arxiv.org/abs/1804.04606

**Robust Physical Adversarial Attack on Faster R-CNN Object Detector**

- arxiv: https://arxiv.org/abs/1804.05810

## RefineNet

**Single-Shot Refinement Neural Network for Object Detection**

- intro: CVPR 2018

- arxiv: <https://arxiv.org/abs/1711.06897>
- github: <https://github.com/sfzhang15/RefineDet>
- github: https://github.com/lzx1413/PytorchSSD
- github: https://github.com/ddlee96/RefineDet_mxnet
- github: https://github.com/MTCloudVision/RefineDet-Mxnet

## DetNet

**DetNet: A Backbone network for Object Detection**

- intro: Tsinghua University & Face++
- arxiv: https://arxiv.org/abs/1804.06215


## SSOD

**Self-supervisory Signals for Object Discovery and Detection**

- Google Brain
- arxiv:https://arxiv.org/abs/1806.03370

## CornerNet

**CornerNet: Detecting Objects as Paired Keypoints**

- intro: ECCV 2018
- arXiv: https://arxiv.org/abs/1808.01244
- github: <https://github.com/umich-vl/CornerNet>

## M2Det

**M2Det: A Single-Shot Object Detector based on Multi-Level Feature Pyramid Network**

- intro: AAAI 2019
- arXiv: https://arxiv.org/abs/1811.04533
- github: https://github.com/qijiezhao/M2Det

## 3D Object Detection

**3D Backbone Network for 3D Object Detection**

- arXiv: https://arxiv.org/abs/1901.08373

**LMNet: Real-time Multiclass Object Detection on CPU using 3D LiDARs**

- arxiv: https://arxiv.org/abs/1805.04902
- github: https://github.com/CPFL/Autoware/tree/feature/cnn_lidar_detection


## ZSD（Zero-Shot Object Detection）

**Zero-Shot Detection**

- intro: Australian National University
- keywords: YOLO
- arxiv: <https://arxiv.org/abs/1803.07113>

**Zero-Shot Object Detection**

- arxiv: https://arxiv.org/abs/1804.04340

**Zero-Shot Object Detection: Learning to Simultaneously Recognize and Localize Novel Concepts**

- arxiv: https://arxiv.org/abs/1803.06049

**Zero-Shot Object Detection by Hybrid Region Embedding**

- arxiv: https://arxiv.org/abs/1805.06157

## OSD（One-Shot Object Detection）

**Comparison Network for One-Shot Conditional Object Detection**

- arXiv: https://arxiv.org/abs/1904.02317

**One-Shot Object Detection**

RepMet: Representative-based metric learning for classification and one-shot object detection

- intro: IBM Research AI
- arxiv:https://arxiv.org/abs/1806.04728
- github: TODO

## Weakly Supervised Object Detection

**Weakly Supervised Object Detection in Artworks**

- intro: ECCV 2018 Workshop Computer Vision for Art Analysis
- arXiv: https://arxiv.org/abs/1810.02569
- Datasets: https://wsoda.telecom-paristech.fr/downloads/dataset/IconArt_v1.zip

**Cross-Domain Weakly-Supervised Object Detection through Progressive Domain Adaptation**

- intro: CVPR 2018
- arXiv: https://arxiv.org/abs/1803.11365
- homepage: https://naoto0804.github.io/cross_domain_detection/
- paper: http://openaccess.thecvf.com/content_cvpr_2018/html/Inoue_Cross-Domain_Weakly-Supervised_Object_CVPR_2018_paper.html
- github: https://github.com/naoto0804/cross-domain-detection

## Softer-NMS

**《Softer-NMS: Rethinking Bounding Box Regression for Accurate Object Detection》**

- intro: CMU & Face++
- arXiv: https://arxiv.org/abs/1809.08545
- github: https://github.com/yihui-he/softer-NMS

## 2019

**Feature Selective Anchor-Free Module for Single-Shot Object Detection**

- intro: CVPR 2019

- arXiv: https://arxiv.org/abs/1903.00621

**Object Detection based on Region Decomposition and Assembly**

- intro: AAAI 2019

- arXiv: https://arxiv.org/abs/1901.08225

**Bottom-up Object Detection by Grouping Extreme and Center Points**

- intro: one stage 43.2% on COCO test-dev
- arXiv: https://arxiv.org/abs/1901.08043
- github: https://github.com/xingyizhou/ExtremeNet

**ORSIm Detector: A Novel Object Detection Framework in Optical Remote Sensing Imagery Using Spatial-Frequency Channel Features**

- intro: IEEE TRANSACTIONS ON GEOSCIENCE AND REMOTE SENSING

- arXiv: https://arxiv.org/abs/1901.07925

**Consistent Optimization for Single-Shot Object Detection**

- intro: improves RetinaNet from 39.1 AP to 40.1 AP on COCO datase

- arXiv: https://arxiv.org/abs/1901.06563

**Learning Pairwise Relationship for Multi-object Detection in Crowded Scenes**

- arXiv: https://arxiv.org/abs/1901.03796

**RetinaMask: Learning to predict masks improves state-of-the-art single-shot detection for free**

- arXiv: https://arxiv.org/abs/1901.03353
- github: https://github.com/chengyangfu/retinamask

**Region Proposal by Guided Anchoring**

- intro: CUHK - SenseTime Joint Lab
- arXiv: https://arxiv.org/abs/1901.03278

**Scale-Aware Trident Networks for Object Detection**

- intro: mAP of **48.4** on the COCO dataset
- arXiv: https://arxiv.org/abs/1901.01892

## 2018

**Large-Scale Object Detection of Images from Network Cameras in Variable Ambient Lighting Conditions**

- arXiv: https://arxiv.org/abs/1812.11901

**Strong-Weak Distribution Alignment for Adaptive Object Detection**

- arXiv: https://arxiv.org/abs/1812.04798

**AutoFocus: Efficient Multi-Scale Inference**

- intro: AutoFocus obtains an **mAP of 47.9%** (68.3% at 50% overlap) on the **COCO test-dev** set while processing **6.4 images per second on a Titan X (Pascal) GPU** 
- arXiv: https://arxiv.org/abs/1812.01600

**NOTE-RCNN: NOise Tolerant Ensemble RCNN for Semi-Supervised Object Detection**

- intro: Google Could
- arXiv: https://arxiv.org/abs/1812.00124

**SPLAT: Semantic Pixel-Level Adaptation Transforms for Detection**

- intro: UC Berkeley
- arXiv: https://arxiv.org/abs/1812.00929

**Grid R-CNN**

- intro: SenseTime
- arXiv: https://arxiv.org/abs/1811.12030

**Deformable ConvNets v2: More Deformable, Better Results**

- intro: Microsoft Research Asia

- arXiv: https://arxiv.org/abs/1811.11168

**Anchor Box Optimization for Object Detection**

- intro: Microsoft Research
- arXiv: https://arxiv.org/abs/1812.00469

**Efficient Coarse-to-Fine Non-Local Module for the Detection of Small Objects**

- intro: https://arxiv.org/abs/1811.12152

**NOTE-RCNN: NOise Tolerant Ensemble RCNN for Semi-Supervised Object Detection**

- arXiv: https://arxiv.org/abs/1812.00124

**Learning RoI Transformer for Detecting Oriented Objects in Aerial Images**

- arXiv: https://arxiv.org/abs/1812.00155

**Integrated Object Detection and Tracking with Tracklet-Conditioned Detection**

- intro: Microsoft Research Asia
- arXiv: https://arxiv.org/abs/1811.11167

**Deep Regionlets: Blended Representation and Deep Learning for Generic Object Detection**

- arXiv: https://arxiv.org/abs/1811.11318

 **Gradient Harmonized Single-stage Detector**

- intro: AAAI 2019
- arXiv: https://arxiv.org/abs/1811.05181

**CFENet: Object Detection with Comprehensive Feature Enhancement Module**

- intro: ACCV 2018
- github: https://github.com/qijiezhao/CFENet

**DeRPN: Taking a further step toward more general object detection**

- intro: AAAI 2019
- arXiv: https://arxiv.org/abs/1811.06700
- github: https://github.com/HCIILAB/DeRPN

**Hybrid Knowledge Routed Modules for Large-scale Object Detection**

- intro: Sun Yat-Sen University & Huawei Noah’s Ark Lab
- arXiv: https://arxiv.org/abs/1810.12681
- github: https://github.com/chanyn/HKRM

**《Receptive Field Block Net for Accurate and Fast Object Detection》**

- intro: ECCV 2018
- arXiv: [https://arxiv.org/abs/1711.07767](https://arxiv.org/abs/1711.07767)
- github: [https://github.com/ruinmessi/RFBNet](https://github.com/ruinmessi/RFBNet)

**Deep Feature Pyramid Reconfiguration for Object Detection**

- intro: ECCV 2018
- arXiv: https://arxiv.org/abs/1808.07993

**Unsupervised Hard Example Mining from Videos for Improved Object Detection**

- intro: ECCV 2018
- arXiv: https://arxiv.org/abs/1808.04285

**Acquisition of Localization Confidence for Accurate Object Detection**

- intro: ECCV 2018
- arXiv: https://arxiv.org/abs/1807.11590
- github: https://github.com/vacancy/PreciseRoIPooling

**Toward Scale-Invariance and Position-Sensitive Region Proposal Networks**

- intro: ECCV 2018
- arXiv: https://arxiv.org/abs/1807.09528

**MetaAnchor: Learning to Detect Objects with Customized Anchors**

- arxiv: https://arxiv.org/abs/1807.00980

**Relation Network for Object Detection**

- intro: CVPR 2018
- arxiv: https://arxiv.org/abs/1711.11575
- github:https://github.com/msracver/Relation-Networks-for-Object-Detection

**Quantization Mimic: Towards Very Tiny CNN for Object Detection**

- Tsinghua University1 & The Chinese University of Hong Kong2 &SenseTime3
- arxiv: https://arxiv.org/abs/1805.02152

**Learning Rich Features for Image Manipulation Detection**

- intro: CVPR 2018 Camera Ready
- arxiv: https://arxiv.org/abs/1805.04953

**SNIPER: Efficient Multi-Scale Training**

- arxiv:https://arxiv.org/abs/1805.09300
- github:https://github.com/mahyarnajibi/SNIPER

**Soft Sampling for Robust Object Detection**

- intro: the robustness of object detection under the presence of missing annotations
- arxiv:https://arxiv.org/abs/1806.06986

**Cost-effective Object Detection: Active Sample Mining with Switchable Selection Criteria**

- intro: TNNLS 2018
- arxiv:https://arxiv.org/abs/1807.00147
- code: http://kezewang.com/codes/ASM_ver1.zip

## Other

**R3-Net: A Deep Network for Multi-oriented Vehicle Detection in Aerial Images and Videos**

- arxiv: https://arxiv.org/abs/1808.05560
- youtube: https://youtu.be/xCYD-tYudN0

# Detection Toolbox

- [Detectron(FAIR)](https://github.com/facebookresearch/Detectron): Detectron is Facebook AI Research's software system that implements state-of-the-art object detection algorithms, including [Mask R-CNN](https://arxiv.org/abs/1703.06870). It is written in Python and powered by the [Caffe2](https://github.com/caffe2/caffe2) deep learning framework.

- [maskrcnn-benchmark(FAIR)](https://github.com/facebookresearch/maskrcnn-benchmark): Fast, modular reference implementation of Instance Segmentation and Object Detection algorithms in PyTorch.

- [mmdetection(SenseTime&CUHK)](https://github.com/open-mmlab/mmdetection): mmdetection is an open source object detection toolbox based on PyTorch. It is a part of the open-mmlab project developed by [Multimedia Laboratory, CUHK](http://mmlab.ie.cuhk.edu.hk/).
