# Report

### 1. 直接拼接

首先，尝试直接以中间图像为基准进行拼接，主要步骤是：

1. 调用OpenCV中SIFT算法找出两张图像中的特征点；

2. 用BFMatcher进行特征点的匹配，按3倍最小距离+100的阈值筛选掉比较差的点（加100的原因是实验中发现有时候匹配出的点distance会为0）；

3. 调用OpenCV中的findHomography找出两图像间的单应矩阵M；

4. 根据M计算出变换后图像的MBR，根据该MBR扩充画布大小；

5. 使用OpenCV中warpPerspective函数将图像进行投影变换，使用渐进渐出算法进行两图像间的融合

6. 如果还有图像没拼接，则回到1，以第5步的结果为基准进行拼接；否则结束

最终结果如下：

![img](https://github.com/Liu-Yuzhen/image-stitching/blob/master/pic/wps1.png) 

图像之间没有明显裂缝，但由于是以中间图像为基准，两边变形非常严重，于是希望找到一个使每个图像变形能较为均匀的方法。

 

### 2.圆柱投影后拼接

1. 圆柱投影

   现在的目的是找到一种方法，让所有照片的变形能够尽可能均匀的分配。

   如果不考虑拍摄时的遮挡和相机量测等误差的影像，且各张照片拍摄时的投影中心不变，那么理论上投影到以焦距为半径的圆柱上再进行拼接应该是不会产生变形的（这里的“不会变形”指的是已经投影后再进行透视变换不会变形），此时只有圆柱投影这一过程产生的变形，因此每张照片形变量相等。

   下方是圆柱投影的示意图：

![img](https://github.com/Liu-Yuzhen/image-stitching/blob/master/pic/wps2.jpg) 

​		经推导，圆柱投影时满足如下关系式：

![img](https://github.com/Liu-Yuzhen/image-stitching/blob/master/pic/wps3.png)

​		其中：

![img](https://github.com/Liu-Yuzhen/image-stitching/blob/master/pic/wps4.png)

 

​		其中R取相机的焦距时，拼接出来每张图片的变形应该是最均匀的，但我们不知道f究竟是多少。下面定义变量***\*f\****：

![img](https://github.com/Liu-Yuzhen/image-stitching/blob/master/pic/wps5.png)

​	下图时f取0.5时圆柱投影后的结果：

![img](https://github.com/Liu-Yuzhen/image-stitching/blob/master/pic/wps6.jpg) 

 

2. 单应矩阵

   先尝试通过不断剔除大于10倍中误差的点来进行粗差探测，但效果很差：

![img](https://github.com/Liu-Yuzhen/image-stitching/blob/master/pic/wps7.jpg) 

​		最终采取间接平差+RanSAC的方法找出单应矩阵。

​		数学模型为：

![img](https://github.com/Liu-Yuzhen/image-stitching/blob/master/pic/wps9.png)

​		其中，X,Y为基准图中的像点坐标，x，y为变换图的像点坐标。

​		如果进行严密的带参数的间接平差，以x,y为观测值，移项后进行泰勒展开可以得到：

![img](https://github.com/Liu-Yuzhen/image-stitching/blob/master/pic/wps10.png)

![img](https://github.com/Liu-Yuzhen/image-stitching/blob/master/pic/wps11.png)

​		由于这种方法需要初值，且需要迭代计算，计算量将会很大。可以采取以方程值作为观测值的间接平差简化计算:

![img](https://github.com/Liu-Yuzhen/image-stitching/blob/master/pic/wps12.png)

​		其中：

![img](https://github.com/Liu-Yuzhen/image-stitching/blob/master/pic/wps13.png)

RanSAC步骤：

​	1. 初始化最大投票数为0；

​	2. 先随机挑选5个点进行间接平差，得到结果后统计满足阈值的点的个数，如果大于最大投票数，则将其覆盖；

​	3.  重复以上步骤N次。N可以由下式计算：

![img](https://github.com/Liu-Yuzhen/image-stitching/blob/master/pic/wps14.png)

​	4. 用所有内点进行平差，计算单应矩阵。

 

3.影像融合

​	根据韦伯定律，人眼对亮度的响应是非线性的，所以这里融合采用余弦函数加权:

![img](https://github.com/Liu-Yuzhen/image-stitching/blob/master/pic/wps15.jpg) 

​	下图是f=0.5时拼接的最终结果（仍旧以中间图像为基准，最下角从小到大为图片拼接的顺序）：

![img](https://github.com/Liu-Yuzhen/image-stitching/blob/master/pic/wps16.png) 

f = 0.5

​	可以看出来，两边的变形仍旧非常大，但不同的是此时两边是变小了。

​	接下来尝试不同的f取值，可以得到不同的拼接结果：

![img](https://github.com/Liu-Yuzhen/image-stitching/blob/master/pic/wps17.png) 

f = 0.6

​									 

![img](https://github.com/Liu-Yuzhen/image-stitching/blob/master/pic/wps18.png) 

f = 0.7

![img](https://github.com/Liu-Yuzhen/image-stitching/blob/master/pic/wps19.png) 

f = 0.8

![img](https://github.com/Liu-Yuzhen/image-stitching/blob/master/pic/wps20.png) 

f = 0.9

![img](https://github.com/Liu-Yuzhen/image-stitching/blob/master/pic/wps21.png) 

f = 1

 

4.寻找最优f

​	从上面这一些列图片可以发现，从f=0.6开始，两边变形是向着变小的方向进行的；然后两边逐渐变大，最终变形向着扩大的方向前进。

​	这是很显然的结果，因为当f越大，投影前后变化就越小，当f取∞时，就是原图，那么进行拼接时和（一）中的直接拼接相比就没什么两样了。

​	为了找到最合适的f，需要不断尝试。但为了省去人工判断，下面定义**形变系数ξ**：

![img](https://github.com/Liu-Yuzhen/image-stitching/blob/master/pic/wps22.png)

 

​	其中：![img](https://github.com/Liu-Yuzhen/image-stitching/blob/master/pic/wps23.png)。

​	当然，ξ只能衡量左右变长的变形大小，但在此实验中已经足够。

​	为了衡量拼接的几何误差和融合的效果，采取以下两项指标（附录）：

1. 点位中误差**sigma**(n为内点个数，以内点的中误差对整体的点位中误差进行无偏估计)：

![img](https://github.com/Liu-Yuzhen/image-stitching/blob/master/pic/wps24.png)

 

2. 结构相似性**SSIM**能反映对比度和亮度的差别，本次实验中计算全局的SSIM：

![img](https://github.com/Liu-Yuzhen/image-stitching/blob/master/pic/wps25.png)

 

​	接下来从取f=0.50-1.26步长为0.01进行实验（采用间接平差+RANSAC+余弦函数加权融合），得到如下曲线图：

![img](https://github.com/Liu-Yuzhen/image-stitching/blob/master/pic/wps26.jpg) 

 

![img](https://github.com/Liu-Yuzhen/image-stitching/blob/master/pic/wps27.jpg) 

 

 

### 3. 结果评价与分析

- 从几何误差上来看，点位标准差较小，没有明显规律性，都在0.85个像素以内；

- 从匹配程度上来看，SSIM稳定在1附近，说明结构相似性良好；

- 从变形上看，在区间[0.5,0.84]上，形变呈递减趋势；在f>0.84时，形变系数呈现递增趋势。由此可推断当f=0.84时拼接的变形最小，对应的结果如图所示:

![img](https://github.com/Liu-Yuzhen/image-stitching/blob/master/pic/wps28.png) 

​	此外，还可以大致推断相机的主距：

![img](https://github.com/Liu-Yuzhen/image-stitching/blob/master/pic/wps29.png)

