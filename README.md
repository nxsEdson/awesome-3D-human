# Awesome-3D-Human-Pose
A summary on 3D human pose estimation

## Survey
* [Recovering 3D Human Mesh from Monocular Images: A Survey](https://arxiv.org/abs/2203.01923)</br>
Yating Tian, Hongwen Zhang, Yebin Liu, Limin Wang</br>

## Body Models
* **\[SCAPE\]** [SCAPE: Shape Completion and Animation of People.](https://ai.stanford.edu/~drago/Papers/shapecomp.pdf) D. Anguelov, P. Srinivasan, D. Koller, S. Thrun, J. Rodgers, and J. Davis. *ACM Trans. Graphics, 2005* </br>
**Brief Summary**: First body model disentangling human body into rigid transformation of pose, id-related shape, and pose-related shape.

* **\[SMPL\]** [SMPL: A Skinned Multi-Person Linear Model.](https://smpl.is.tue.mpg.de) Loper, Matthew and Mahmood, Naureen and Romero, Javier and Pons-Moll, Gerard and Black, Michael J. *ACM Trans. Graphics, 2015* </br>
**Brief Summary**: The most widely-used body model which can be easily used in rendering engines for animation (with bones).

* **\[SMPL-X\]** [Expressive Body Capture: 3D Hands, Face, and Body from a Single Image.](https://smpl-x.is.tue.mpg.de/) Pavlakos, Georgios and Choutas, Vasileios and Ghorbani, Nima and Bolkart, Timo and Osman, Ahmed A. A. and Tzionas, Dimitrios and Black, Michael J. *CVPR 2019* </br>
**Brief Summary**: SMPL + MANO (hand model) + FLAME (head model)

* **\[STAR\]** [STAR: A Sparse Trained Articulated Human Body Regressor.](https://star.is.tue.mpg.de) Osman, Ahmed A A and Bolkart, Timo and Black, Michael J. *ECCV 2020* </br>
**Brief Summary**: Disentangling the pose-related blend shapes in SMPL to per-joint pose-related blend shapes

* **\[DeepDaz\]** [UltraPose: Synthesizing Dense Pose with 1 Billion Points by Human-body Decoupling 3D Model.](https://github.com/MomoAILab/ultrapose) Haonan Yan, Jiaqi Chen, Xujie Zhang, Shengkai Zhang, Nianhong Jiao, Xiaodan Liang, Tianxiang Zheng. *ICCV 2021* </br>
**Brief Summary**: Human body model with parameters having a specific physical meaning and decoupled with each other (based on [Daz](https://www.daz3d.com/) model)

* **\[GHUM\]** [GHUM & GHUML: Generative 3D Human Shape and Articulated Pose Models.](https://openaccess.thecvf.com/content_CVPR_2020/papers/Xu_GHUM__GHUML_Generative_3D_Human_Shape_and_Articulated_Pose_CVPR_2020_paper.pdf) Hongyi Xu, Eduard Gabriel Bazavan, Andrei Zanfir, William T. Freeman, Rahul Sukthankar, Cristian Sminchisescu. *CVPR 2020* </br>
**Brief Summary**: Human body model with non-linear (VAEs) id-related shape and face expression embedding spaces.


## Human pose estimation 

### Optimization-based Method 

## Research Groups
[Michael Black](http://ps.is.mpg.de) (Max Planck Institute for Intelligent Systems)
[Yebin Liu](http://www.liuyebin.com) (Tsinghua University) 
[Kyoung Mu Lee](https://cv.snu.ac.kr/index.php/~kmlee/) (Seoul National University)
