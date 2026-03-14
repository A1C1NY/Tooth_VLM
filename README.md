# 图像相似度去重
## 介绍
本项目旨在通过计算图像之间的相似度来实现重复图像的去除。我们使用深度学习模型提取图像特征，并基于这些特征计算相似度。

计算相似度时使用余弦相似度来衡量图像特征之间的相似程度。通过设定一个相似度阈值，我们可以识别出相似度高于该阈值的图像，并将其视为重复图像组，保留其中一个图像，删除其他重复图像。并且将相似的cluster关系保存在txt文件中，方便后续分析。‘

## 使用方法
发给你的应该是一个脚本，包含python代码。你可以按照以下步骤使用该脚本：

### 环境配置
建议用VSCode开一个venv环境，conda也行但是容易误报环境冲突。最好根据自己GPU大致的发布时间选择相应的python版本，或直接查阅pytorch官网推荐的版本。
安装必要的Python库，如`numpy`、`PIL`、`torch`等。你可以使用以下命令安装所需库：
```bash
pip install numpy Pillow 
pip install scikit-learn
pip install tqdm
pip install shutil
# 如果缺少照着报错安装就行
```
对于pytorch，参考[官方文档](https://pytorch.org/)进行安装。
首先使用指令获取自己的GPU型号，然后根据型号选择合适的安装命令。


在cmd或者powershell中输入以下命令：

```bash
nvidia-smi
```

从GPU型号首行可以看到自己的cuda版本，这个版本是你的GPU能兼容的最高版本，然后根据版本在官网上选择符要求的命令。

例如我使用RTX 5060ti，CUDA Version: 12.9 

在官网上应该选择：
```bash
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

如果用的是Intel的GPU，我不确定pytorch是否兼容。如果怕麻烦可以直接下载cpu版本。

### 运行脚本
在data.py所在目录下打开命令行，运行以下命令：
```bash
mkdir output
mkdir images
```

然后将需要去重的图像放入`images`文件夹中， **保证自己已经进入了环境！！！** 运行以下命令：

```bash
D:\File\Programming\Tooth\VLM_tooth\Scripts\python.exe d:/File/Programming/Tooth/data.py
# 虚拟环境下的那个python.exe路径 + data.py的路径
```

脚本会自动处理`images`文件夹中的图像，并将去重后的图像保存在`output`文件夹中。同时，脚本会在`output`文件夹生成一个名为`duplicates.txt`的文件，记录相似图像的cluster关系。


### 目前的效果
阈值建议在0.94 - 0.95之间进行调整，过高可能会漏掉一些相似图像，过低可能会误判一些不同的图像为相似。根据实际情况调整阈值以获得最佳去重效果。

对于颜色加强，一些基本的旋转效果不错。但是对于一些大规模平移，以及图像外延效果不好。