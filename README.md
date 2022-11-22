# Awesome-3D-Human-Pose
A summary on 3D human pose estimation

## Survey
* [Recovering 3D Human Mesh from Monocular Images: A Survey](https://arxiv.org/abs/2203.01923)</br>
Yating Tian, Hongwen Zhang, Yebin Liu, Limin Wang</br>

## Body Models
* **\[SCAPE\]** [SCAPE: Shape Completion and Animation of People.](https://ai.stanford.edu/~drago/Papers/shapecomp.pdf) D. Anguelov, P. Srinivasan, D. Koller, S. Thrun, J. Rodgers, and J. Davis. *ACM Trans. Graphics, 2005* </br>
**One Sentence Summary**: First body model disentangling human body into rigid transformation of pose, id-related shape, and pose-related shape.

* **\[SMPL\]** [SMPL: A Skinned Multi-Person Linear Model.](https://smpl.is.tue.mpg.de) Loper, Matthew and Mahmood, Naureen and Romero, Javier and Pons-Moll, Gerard and Black, Michael J. *ACM Trans. Graphics, 2015* </br>
**One Sentence Summary**: The most widely-used body model which can be easily used in rendering engines for animation (with bones).

* **\[SMPL-X\]** [Expressive Body Capture: 3D Hands, Face, and Body from a Single Image](https://smpl-x.is.tue.mpg.de/) Pavlakos, Georgios and Choutas, Vasileios and Ghorbani, Nima and Bolkart, Timo and Osman, Ahmed A. A. and Tzionas, Dimitrios and Black, Michael J. *CVPR 2019* </br>
**One Sentence Summary**: SMPL + MANO (hand model) + FLAME (head model)

* **\[STAR\]** [STAR: A Sparse Trained Articulated Human Body Regressor](https://star.is.tue.mpg.de) Osman, Ahmed A A and Bolkart, Timo and Black, Michael J. *ECCV 2020* </br>
**One Sentence Summary**: Disentangling the pose-related blend shapes in SMPL to per-joint pose-related blend shapes

* **\[DeepDaz\]** [UltraPose: Synthesizing Dense Pose with 1 Billion Points by Human-body Decoupling 3D Model](https://github.com/MomoAILab/ultrapose) Haonan Yan, Jiaqi Chen, Xujie Zhang, Shengkai Zhang, Nianhong Jiao, Xiaodan Liang, Tianxiang Zheng. *ICCV 2021* </br>
**One Sentence Summary**: Human body model with parameters having a specific physical meaning and decoupled with each other (based on [Daz](https://www.daz3d.com/) model)

* **\[GHUM\]** [GHUM & GHUML: Generative 3D Human Shape and Articulated Pose Models](https://openaccess.thecvf.com/content_CVPR_2020/papers/Xu_GHUM__GHUML_Generative_3D_Human_Shape_and_Articulated_Pose_CVPR_2020_paper.pdf) Hongyi Xu, Eduard Gabriel Bazavan, Andrei Zanfir, William T. Freeman, Rahul Sukthankar, Cristian Sminchisescu. *CVPR 2020* </br>
**One Sentence Summary**: Human body model with non-linear (VAEs) id-related shape and face expression embedding spaces.

## Human Mesh Recovery

### From Single Images

* **\[SMPLify\]** [Keep it SMPL: Automatic Estimation of 3D Human Pose and Shape from a Single Image.](https://smplify.is.tue.mpg.de) Bogo, Federica and Kanazawa, Angjoo and Lassner, Christoph and Gehler, Peter and Romero, Javier and Black, Michael J. *ECCV 2016* </br>
**One Sentence Summary**: One optimizion-based method using the reprojection loss of keypoints as well as several regularization terms.

* **\[HMR\]** [End-to-end Recovery of Human Shape and Pose.](https://openaccess.thecvf.com/content_cvpr_2018/papers/Kanazawa_End-to-End_Recovery_of_CVPR_2018_paper.pdf) Angjoo Kanazawa, Michael J. Black, David W. Jacobs, Jitendra Malik. *CVPR 2018* </br>
**One Sentence Summary**: Human mesh recovery using reprojection loss of keypoints and adversary training to avoid unreasonable pose.

* **\[\]** [Learning to Estimate 3D Human Pose and Shape from a Single Color Image.](https://openaccess.thecvf.com/content_cvpr_2018/papers/Pavlakos_Learning_to_Estimate_CVPR_2018_paper.pdf) Georgios Pavlakos, Luyang Zhu, Xiaowei Zhou, Kostas Daniilidis *CVPR 2018* </br>
**One Sentence Summary**: Training the network with keypoint heatmaps and masks as supervision for SMPL parameters regression.

* **\[SPIN\]** [SPIN：Learning to Reconstruct 3D Human Pose and Shape via Model-fitting in the Loop.](https://openaccess.thecvf.com/content_ICCV_2019/papers/Kolotouros_Learning_to_Reconstruct_3D_Human_Pose_and_Shape_via_Model-Fitting_ICCV_2019_paper.pdf) Kolotouros, Nikos and Pavlakos, Georgios and Black, Michael J and Daniilidis, Kostas. *ICCV 2019* </br>
**One Sentence Summary**: HMR + SMPLify (HMR is used to inilize the body model parameters & SMPLify is used to refine these parameters. The refined parameters are further used as the surpervision for the network.)

* **\[\]** [On the Continuity of Rotation Representations in Neural Networks.](https://openaccess.thecvf.com/content_CVPR_2019/papers/Zhou_On_the_Continuity_of_Rotation_Representations_in_Neural_Networks_CVPR_2019_paper.pdf) Yi Zhou, Connelly Barnes, Jingwan Lu, Jimei Yang, and Hao Li. *CVPR 2019* </br>
**One Sentence Summary**: A new continuous representations for joint rotation.

* **\[GraphCMR\]** [Convolutional Mesh Regression for Single-Image Human Shape Reconstruction.](https://github.com/nkolot/GraphCMR) Nikos Kolotouros, Georgios Pavlakos, Kostas Daniilidis. *CVPR 2019* </br>
**One Sentence Summary**: Directly regressing the meshes of human body with graph convolutions, then using meshes to regress SMPL parameters.

* **\[\]** [Delving Deep into Hybrid Annotations for 3D Human Recovery in the Wild.](https://penincillin.github.io/dct_iccv2019) Yu Rong, Ziwei Liu, Cheng Li, Kaidi Cao, Chen Change Loy. *ICCV 2019* </br>
**One Sentence Summary**: A comprehensive study on the cost and effectiveness of different annotations for in-the-wild images. (Dense correspondence is effective.)

* **\[HoloPose\]** [HoloPose: Holistic 3D Human Reconstruction In-The-Wild.](https://www.arielai.com/holopose/) Rıza Alp Guler, and Iasonas Kokkinos. *CVPR 2019* </br>
**One Sentence Summary**: Regressing the body model parameters from body-part features with reprojection loss of densepose and key points.

* **\[DecoMR\]** [3D Human Mesh Regression with Dense Correspondence.](https://openaccess.thecvf.com/content_CVPR_2020/papers/Zeng_3D_Human_Mesh_Regression_With_Dense_Correspondence_CVPR_2020_paper.pdf) Wang Zeng, Wanli Ouyang, Ping Luo, Wentao Liu, and Xiaogang Wang. *CVPR 2020* </br>
**One Sentence Summary**: Recovering human mesh using the aligned features in UV space.

* **\[HKMR\]** [Hierarchical Kinematic Human Mesh Recovery.](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123620749.pdf) Georgios Georgakis, Ren Li, Srikrishna Karanam, Terrence Chen, Jana Ko seck a, and Ziyan Wu. *ECCV 2020* </br>
**One Sentence Summary**: Optimizing the SMPL body pose parameters seperatly based on different parts of body.

* **\[PARE\]** [PARE: Part Attention Regressor for 3D Human Body Estimation.](https://pare.is.tue.mpg.de) Muhammed Kocabas, Chun-Hao P. Huang, Otmar Hilliges, and Michael J. Black. *ICCV 2021* </br>
**One Sentence Summary**: Handling the occlusion problem using part attention. (Attention maps are inilized by segmentation mask and trained with the 3D branch jointly.)

* **\[DSR\]** [Learning to Regress Bodies from Images using Differentiable Semantic Rendering.](https://dsr.is.tue.mpg.de) Sai Kumar Dwivedi, Nikos Athanasiou, Muhammed Kocabas, Michael J. Black. *ICCV 2021* </br>
**One Sentence Summary**: Using differentiable rendering to supervise the training of HMR with the semantic prior of clothes (calculated from AGORA).

* **\[Skeleton2Mesh\]** [Skeleton2Mesh: Kinematics Prior Injected Unsupervised Human Mesh Recovery.](https://dsr.is.tue.mpg.de) Zhenbo Yu, Junjie Wang, Jingwei Xu, Bingbing Ni, Chenglong Zhao, Minsi Wang, Wenjun Zhang. *ICCV 2021* </br>
**One Sentence Summary**: 3D human pose estimation using differentiable IK.

* **\[PyMAF\]** [PyMAF: 3D Human Pose and Shape Regression with Pyramidal Mesh Alignment Feedback Loop.](https://github.com/HongwenZhang/PyMAF) Zhang, Hongwen and Tian, Yating and Zhou, Xinchi and Ouyang, Wanli and Liu, Yebin and Wang, Limin and Sun, Zhenan. *ICCV 2021* </br>
**One Sentence Summary**: HMR (body model parameters regression network) using mesh-aligned multi-scale features & densepose supervisions.

* **\[METRO\]** [End-to-End Human Pose and Mesh Reconstruction with Transformers.]() Kevin Lin Lijuan Wang Zicheng Liu. *CVPR 2021* </br>
**One Sentence Summary**: 3D human pose estimation using transformer. (3D joints/vertics locations are used as position embedding.)

### From Videos 

* [Learning 3D Human Dynamics from Video.](https://openaccess.thecvf.com/content_CVPR_2019/papers/Kanazawa_Learning_3D_Human_Dynamics_From_Video_CVPR_2019_paper.pdf) Angjoo Kanazawa, Jason Y. Zhang, Panna Felsen, Jitendra Malik *CVPR 2019* </br>
**One Sentence Summary**: A temporal encoder with sliding windows and a hallucinator for the current time step to predict the pose of current and adjacent frames. 

* **\[VIBE\]** [VIBE: Video Inference for Human Body Pose and Shape Estimation.](https://openaccess.thecvf.com/content_CVPR_2020/papers/Kocabas_VIBE_Video_Inference_for_Human_Body_Pose_and_Shape_Estimation_CVPR_2020_paper.pdf) Muhammed Kocabas, Nikos Athanasiou, Michael J. Black. *CVPR 2020* </br>
**One Sentence Summary**: Temporal HMR (CNN+GRU for parameters regression & adversary training to avoid unreasonable temporal actions.)

* **\[SmoothNet\]** [SmoothNet: A Plug-and-Play Network for Refining Human Poses in Videos.](https://github.com/cure-lab/SmoothNet) Zeng, Ailing and Yang, Lei and Ju, Xuan and Li, Jiefeng and Wang, Jianyi and Xu, Qiang. *ECCV 2022* </br>
**One Sentence Summary**: A plugin module to reduce the temporal jitter noise.

* **\[GLAMR\]** [GLAMR: Global Occlusion-Aware Human Mesh Recovery with Dynamic Cameras.](https://github.com/NVlabs/GLAMR) Ye Yuan, Umar Iqbal, Pavlo Molchanov, Kris Kitani, Jan Kautz. *CVPR 2022* </br>
**One Sentence Summary**: 

* [Human Mesh Recovery from Multiple Shots.](https://geopavlakos.github.io/multishot/) Georgios Pavlakos Jitendra Malik Angjoo Kanazawa *CVPR 2022* </br>
**One Sentence Summary**: Using SPIN with smoothness term of canonical frame to get the ground-truth. Then using the ground-truth pose to train temporal HMR with transformer.

### Considering Environment

* **\[PHOSA\]** [Perceiving 3D Human-Object Spatial Arrangements from a Single Image in the Wild.](https://jasonyzhang.com/phosa/) Zhang, Jason Y. and Pepose, Sam and Joo, Hanbyul and Ramanan, Deva and Malik, Jitendra and Kanazawa, Angjoo. *ECCV 2020* </br>
**One Sentence Summary**: .

* [The One Where They Reconstructed 3D Humans and Environments in TV Shows.](https://ethanweber.me/sitcoms3D/) Georgios Pavlakos and Ethan Weber and and Matthew Tancik and Angjoo Kanazawa *ECCV 2022* </br>
**One Sentence Summary**: Improving the recovery of 3D human for TV shows by reconstructing the environment and estimating the camera and body scale information. 

### For Multiple Persons

* **\[BMP\]** [Body Meshes as Points.](https://github.com/jfzhang95/BMP) Zhang, Jianfeng and Yu, Dongdong and Liew, Jun Hao and Nie, Xuecheng and Feng, Jiashi *CVPR 2021* </br>
**One Sentence Summary**: One-stage model to estimate multiple persons' 3D body (using the similar idea as CenterNet)

 * **\[ROMP\]** [Monocular, One-stage, Regression of Multiple 3D People.](https://github.com/Arthur151/ROMP) Sun, Yu and Bao, Qian and Liu, Wu and Fu, Yili and Michael J., Black and Mei, Tao. *ICCV 2021* </br>
**One Sentence Summary**: .

 * **\[BEV\]** [Putting People in their Place: Monocular Regression of 3D People in Depth.](https://arthur151.github.io/BEV/BEV.html) Sun, Yu and Liu, Wu and Bao, Qian and Fu, Yili and Mei, Tao and Black, Michael J.. *CVPR 2022* </br>
**One Sentence Summary**: .

### Beyond Body Models (meshes, voxel, and etc.)

* **\[BodyNet\]** [BodyNet: Volumetric Inference of 3D Human Body Shapes.](https://github.com/gulvarol/bodynet) Gül Varol, Duygu Ceylan, Bryan Russell, Jimei Yang, Ersin Yumer, Ivan Laptev and Cordelia Schmid. *ECCV 2018* </br>
**One Sentence Summary**: Volumetric Inference with the supervision of 2d & 3d keypoints, segmentations and voxelized SMPL model.

* **\[I2L-MeshNet\]** [I2L-MeshNet: Image-to-Lixel Prediction Network for Accurate 3D Human Pose and Mesh Estimation from a Single RGB Image.](https://github.com/mks0601/I2L-MeshNet_RELEASE) Gyeongsik Moon and Kyoung Mu Lee. *ECCV 2020* </br>
**One Sentence Summary**: Regressing 2d & 3d key points firstly, and then regressing the mesh directly.


## Applications

### Tracking
* **\[HMAR\]** [Tracking People with 3D Representations.](http://people.eecs.berkeley.edu/~jathushan/T3DP/) Jathushan Rajasegaran, Georgios Pavlakos, Angjoo Kanazawa, and Jitendra Malik *NeurIPS 2021* </br>
**One Sentence Summary**: HMR + texture recovery (using appearance flow) for human tracking. 

* **\[PHALP\]** [Tracking People by Predicting 3D Appearance, Location and Pose.](http://people.eecs.berkeley.edu/~jathushan/PHALP/) Rajasegaran, Jathushan and Pavlakos, Georgios and Kanazawa, Angjoo and Malik, Jitendra *CVPR 2022* </br>
**One Sentence Summary**: Predicting and matching HMAR estimation for tracking in videos. 

### Motion prediction
* **\[PHD\]** [Predicting 3D Human Dynamics from Video.](https://openaccess.thecvf.com/content_ICCV_2019/papers/Zhang_Predicting_3D_Human_Dynamics_From_Video_ICCV_2019_paper.pdf) Jason Y. Zhang, Panna Felsen, Angjoo Kanazawa, Jitendra Malik *ICCV 2019* </br>
**One Sentence Summary**: "Learning 3D Human Dynamics from Video" for 3D motion prediction.


## Related Topic
* Detailed 3D Human Recovery (Clothing)
* 3D Face
* 3D Hand 
* Dense Pose Estimation
* Human Pose Prediction

## Related Research Groups
[Michael Black](http://ps.is.mpg.de) (Max Planck Institute for Intelligent Systems) </br>
[Yebin Liu](http://www.liuyebin.com) (Tsinghua University) </br>
[Kyoung Mu Lee](https://cv.snu.ac.kr/index.php/~kmlee/) (Seoul National University) </br>
[Yaser Sheikh](https://scholar.google.com/citations?user=Yd4KvooAAAAJ&hl=en) (Carnegie Mellon University, Facebook Reality Labs) </br>
[Angjoo Kanazawa](https://people.eecs.berkeley.edu/~kanazawa/) (University of California, Berkeley) </br>
[Kostas Daniilidis](http://www.cis.upenn.edu/~kostas) (University of Pennsylvania) </br>
[Xiaowei Zhou](https://xzhou.me) (Zhejiang University) </br>
