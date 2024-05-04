# HATNet
* The pytorch implementation for HATNet in paper "Hybrid Attention-aware Transformer Network Collaborative Multiscale Feature Alignment for Building Change Detection".

# Requirements
* Python 3.6
* Pytorch 1.7.0

# Datasets Preparation
The path list in the datasest folder is as follows:

|—train

* ||—A

* ||—B

* ||—OUT

|—val

* ||—A

* ||—B

* ||—OUT

|—test

* ||—A

* ||—B

* ||—OUT


where A contains pre-temporal images, B contains post-temporal images, and OUT contains ground truth images.
# Train
* python train.py --dataset-dir dataset-path
# Test
* python eval.py --ckp-paths weight-path --dataset-dir dataset-path
# Visualization
* python visualization visualization.py --ckp-paths weight-path --dataset-dir dataset-path (Note that batch-size must be 1 when using visualization.py)
* Besides, you can adjust the parameter of full_to_color to change the color

# Citation
If this work is helpful to you, please cite it as:
```
@article{xu2024hybrid,
  title={Hybrid Attention-Aware Transformer Network Collaborative Multiscale Feature Alignment for Building Change Detection},
  author={Xu, Chuan and Ye, Zhaoyi and Mei, Liye and Yu, Haonan and Liu, Jianchen and Yalikun, Yaxiaer and Jin, Shuangtong and Liu, Sheng and Yang, Wei and Lei, Cheng},
  journal={IEEE Transactions on Instrumentation and Measurement},
  volume={73},
  pages={1--14},
  year={2024},
  publisher={IEEE}
}
```
