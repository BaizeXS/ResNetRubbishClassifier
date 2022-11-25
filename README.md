# Rubbish Classification

> 要求：
>
> - 两人一组完成作业，最终以 **5-10** 页的实验报告形式呈现；
> - 实验报告包括但不限于：**实验设计、方法描述、模型设计、实验结果及分析、总结**。
> - 作业 1 最终提交时间为 11 月 30 日，包括**实验报告和打包的源代码**。
> - 作业评价考查：实验报告结构和逻辑清晰合理、方法和模型设计结果较好、有充分的结合方法设计的实验结果分析等。

## 实验设计

- Dataset dataloader 设计

  http://t.csdn.cn/KNH6Q

  http://t.csdn.cn/MwX0t

  改进：

  - 路径、transform、loader、resize

- BN

  ![image-20221125152739263](../Typora/image-20221125152739263.png)

- relu inplace参数

- res34（主要）

- Res50

## 方法描述

![image-20221124232729126](../Typora/image-20221124232729126.png)

![../_images/resnet18.svg](https://zh-v2.d2l.ai/_images/resnet18.svg)



## 模型设计



## 实验结果及分析



## 总结





- Dataset设计

- Image Transform

  由于图片像素不同，有两种方案

  - Resize and Crop

    ImageNet

    - Resize(256)短边

    - Crop(224)

      Single Crop / Multiple Crop？

      [炼丹常识](https://www.cnblogs.com/zjutzz/articles/8733044.html)

      [Five and Ten](https://blog.csdn.net/kuweicai/article/details/106734398)

      [Another](https://blog.csdn.net/pengchengliu/article/details/118856713)

      [Another](https://blog.csdn.net/IT_flying625/article/details/104900050)

    Our Decision:

    Use Resize and Single Central Crop while training, and use Multiple Crop while testing.

    Btw, we choose while training so that we can learn more features from the center of a pic as we choose pic containing item that is in the center.  

  - normalize

    [Normalize](https://developer.aliyun.com/article/928968)

    常用([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])

    > Normalize 对每个通道执行以下操作：
    >
    > ```
    > image = (image - mean) / std
    > ```
    >
    > 在您的情况下，参数`mean, std`以 0.5、0.5 的形式传递。这将在 [-1,1] 范围内标准化图像。例如，最小值 0 将转换为`(0-0.5)/0.5=-1`，最大值 1 将转换为`(1-0.5)/0.5=1.`

