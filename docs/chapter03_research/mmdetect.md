## 环境及代码

mmdetect检测相关的代码有几个：

**1.nanodet**
电脑：cv@supermicro
代码路径：`/home/cv/wu/github/nanodet`
conda环境：nanodet

**2. FCOS**
电脑：cv@supermicro
代码路径：`/home/cv/wu/github/FCOS
conda环境：FCOS，还没调通

**3.GFocal**

电脑：cv@supermicro
代码路径：`/home/cv/wu/github/GFocal
环境：





https://github.com/implus/GFocalV2 出来了



主要讲讲GFocal的环境问题

1. 安装官网教程直接使用docker
2. 按照官方教程直接构建镜像：`docker build -t mmdetection docker/`，用`docker images`可以发现有个`mmdetection`的镜像。
3. 构建容器`docker run --gpus all -it -p 5224:22 --ipc=host -v /mnt/data/tracking/wu_m:/root/wu -v /mnt/data:/root/data -v /mnt/data1:/root/data1 --name mmdetect2 cf3e7d00ccbe`

坑爹的事发生了。。。

可以发现构建的容器里用pip list发现在环境里已经有各种mmdetection的库了，路径在/mmdetection下。
然后GFocal使用的mmdetect的库直接用的是代码路径下的mmdet的库，这个库更老。
理论上是先有GFocal，然后/mmdetection在自己的框架里封装了GFocal，所以mmdetection的库更新。然后在使用GFocal时，系统会自动找系统的mmdetection库，没有找本地的mmdet。

首先，在launch.json里加入："env": {"PYTHONPATH":"${workspaceFolder}"}，优先找本地的mmdet

然后，装mmcv
这个库有两个东西要装，一个是mmcv，一个是mmcv-full，然后mmcv-full的版本非常奇特，而且最后直接下载mmcv-full的包来装，直接用pip装可能会出问题

然后在装mmcv时依赖的opencv-python有问题，先下载opencv-python装，，无语。。

步骤：
在`https://pypi.tuna.tsinghua.edu.cn/simple/opencv-python/`里下载`opencv_python-4.0.1.24-cp36-cp36m-manylinux1_x86_64.whl`
安装`pip install opencv_python-4.0.1.24-cp36-cp36m-manylinux1_x86_64.whl`

安装mmcv，注意版本：pip install mmcv==0.4.3

在`https://download.openmmlab.com/mmcv/dist/index.html`里下载`mmcv_full-1.2.7+torch1.3.0+cu101-cp36-cp36m-manylinux1_x86_64.whl`
同样注意版本，版本容易出错
安装 `pip install mmcv_full-1.2.7+torch1.3.0+cu101-cp36-cp36m-manylinux1_x86_64.whl`

在代码目录下，执行：python setup.py develop，等很久。。

在launch.json里
```shell
"name": "Python: 当前文件",
"type": "python",
"request": "launch",
"cwd": "${workspaceFolder}",
"program": "${workspaceFolder}/tools/train.py",
"console": "integratedTerminal",
"env": {"PYTHONPATH":"${workspaceFolder}"},
"args": ["configs/gfl_r50_1x.py"]
```

在`configs/gfl_r50_1x.py`里修改一下配置，这个世界终于清静了。。





## mmdetection代码参考网站：

CSDN:

EwanRenton: https://blog.csdn.net/sinat_29963957

Shane Zhao: https://blog.csdn.net/silence2015/article/details/105175019?spm=1001.2014.3001.5501



mmdetection 框架安装与使用！:https://zhuanlan.zhihu.com/p/75825433
