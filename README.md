## Introduction

This repository provides an analysis-by-synthesis framework to fit a textured FLAME model to multiview face images or an image. FLAME is a lightweight generic 3D head model learned from over 33,000 head scans, but it does not come with an appearance space.

该工作的初衷是由于人脸图像的特殊性，试图通过引入人脸先验估计图片对应的相机外参。该工作主要利用 [FLAME](https://flame.is.tue.mpg.de/) 模型，故适用于基于 FLAME 的框架，用于其他框架下或需要额外学习一个全局的转换。该工作也可以看做一个初步的低精度的人脸重建工作，输入仅为多视角或单视角人脸图片，多视角图片对应相机内参利用 COLMAP 得到（相关代码集成于框架内），主要输出除了外参和FLAME等相关参数之外，也包括利用  PyTorch3d 渲染得到的人脸图片。

## Dependencies
- PyTorch3d
- configargparse
- face-alignment
- opencv-python

建议使用anaconda配置环境，部分包可使用pip安装，以下配置命令（针对cuda10.2版本）亲测有效，仅供参考：
```
conda create -n pytorch3d python=3.7
conda activate pytorch3d
conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch
conda install -c fvcore -c iopath -c conda-forge fvcore iopath
conda install -c bottler nvidiacub
conda install pytorch3d -c pytorch3d
conda install configargparse
pip install opencv-python==4.1.2.30 face-alignment chumpy
```
## How To Run?

对于多视角图片，输入图片放在''./data/scene-name/images''文件夹下。

图片预处理：1、pad image，将图片扩充为正方形；2、得到真实face landmark；3、得到真实head mask。具体命令如下：
```
python process_data/imgsPad.py --id='str'
```
```
python process_data/imgs2landmarks.py --id='str'
```
```
python process_data/face_parsing/test.py --id='str'
```
然后 fitting ，主要优化过程见photometric_fitting.py：
```
python run_fitting.py --config configs/str.txt
```
note1: 上述命令中str替换为输入图片的scene-name，例如示例的2-346；

note2：更改对应的‘./configs/str.txt’可以设置config参数；

note3：opencv-python版本这里选择4.1.2.30；anaconda环境‘’envs/env-name/lib/python3.7/site-packages/pytorch3d/renderer/cameras.py‘’文件报错“two devices”时，可考虑将文件中报错语句对应参数torch.tensor(0,0)后加上'.cpu()'。



对于单视角图片，见"./SingView"目录下，详情可参考  [photometric_optimization](https://github.com/HavenFeng/photometric_optimization)
示例命令：

```
python photometric_fitting.py 00000
```

## Notes：

这里仅包括主要框架，相关大文件需要自行下载并放在指定位置。

[FLAME_texture.npz](https://flame.is.tue.mpg.de/download.php)：放在'./flame-data/'下；

[generic_model.pkl](https://flame.is.tue.mpg.de/download.php)：放在'./flame-data/'下。