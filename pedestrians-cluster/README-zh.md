# Face Cluster Framework (人脸聚类框架)
[English Version](https://github.com/xiaoxiong74/face-cluster-framework/blob/master/README.md) | 中文版 [Blog](https://blog.csdn.net/qq_42189083/article/details/110449238)


## Intorduction

一个人脸图片聚类框架

对于给定的大量待聚类人脸图片，利用人脸特征抽取组件(face_feature_extract)进行人脸特征抽取，并对用抽取的人脸特征进行人脸聚类并进行图片归档。
采用的人脸聚类算法较当前主流人脸聚类算法效果更优，具体测评效果详见[基于infomap的人脸聚类](https://blog.csdn.net/qq_42189083/article/details/110002878)

## Cluster Result

* 输入数据：
![](data/tmp/input.png)

* 部分聚类效果：

![](data/tmp/result_0.png)
![](data/tmp/result_1.png)

![](data/tmp/result_2.png)
![](data/tmp/result_3.png)


## Requirements
* Python >= 3.6
* sklearn
* infomap
* numpy
* faiss-gpu(or faiss-cpu)
* torch >= 1.2
* torchvision

## Datasets and Pretrain_models
* 可用测试人脸图片数据10000张(data_sample), [下载地址 BaiduYun](https://pan.baidu.com/s/19Ho011j_ZpIT93aS1gSdrg)(passwd: trka)
* 人脸特征抽取的预训练模型(pretrain_model), [下载地址 BaiduYun](https://pan.baidu.com/s/19Ho011j_ZpIT93aS1gSdrg)(passwd: trka)
* 归档标注后的人脸图片数据10000张(labeled_data_sample), [下载地址 BaiduYun](https://pan.baidu.com/s/19Ho011j_ZpIT93aS1gSdrg)(passwd: trka)


## Run
1. 将待聚档图片放入到 'data/input_pictures' 目录下
2. 下载人脸特征抽取的预训练模型，将2个tar文件放到 'pretrain_models' 目录下
3. 运行：
```bash
python main.py
```
4. 人脸图片聚类结果目录 'data/output_pictures'，每个数字子目录下为同一个人的人脸图片，格式如下：

![](data/tmp/output_all.png)


## Evaluate

如果想测评聚类效果，可以利用归档标注后的人脸图片数据(如上述下载数据的labeled_data_sample)，放到'data/input_pictures'目录下，
并在main.py中设置is_evaluate=True即可测评聚类效果。不同人脸数据集可以通过调整main.py中的min_sim与k值获得最优参数

labeled_data_sample数据的聚类指标(调整参数还能提高)：
![](data/tmp/evaluation.png)

此外，可以通过利用自己的数据(如戴口罩的人脸数据)进行人脸特征解析模型训练，训练可以参考[hfsoftmax](https://github.com/yl-1993/hfsoftmax)

## References

* [face-cluster](https://github.com/xiaoxiong74/face-cluster-by-infomap)
* [face_feature_extract](https://github.com/yl-1993/hfsoftmax)