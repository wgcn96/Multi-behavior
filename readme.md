# 基于用户多行为的个性化商品推荐

## 案例简介

本案例是基于国内最大的电商零售平台天猫上用户真实的行为数据log，挖掘浏览、点击、收藏以及购买等不同行为背后反映用户的兴趣方向。并通过这些多行为的信号更好地为用户进行精准推荐。



## 数据集简介

天猫数据集中包括浏览、加购物车和购买三种行为。我们采样了子集，用户商品数量以及每种行为的记录数如下：

- users: 12,921

- items: 22,570

- \# view:  531,640

- \# Add-to-cart: 24,681

- \# Purchase: 160,840

  

## 代码文件说明

+ BatchGenUser.py: 生成随机梯度下降的batch样本
+ Dataset.py: 读文本格式文件，构建数据
+ EvaluateUser.py: 对模型进行性能测试
+ Main.py: 运行主函数
+ Models.py: 训练的模型



## 模型介绍

Collective MF（CMF）:  MF（矩阵分解）是协同过滤中的经典算法，其核心是通过对稀疏的用户-商品矩阵进行矩阵分解，然后补充空缺值。CMF是在MF的基础上进行的改进，其最大的改进之处在于多种目标的协同优化。具体地说：共享商品侧的信息，在每一种行为信息下单独分解用户侧的矩阵，进行多种目标行为的拟合。



## 代码环境以及运行方法

需要tensorflow和其他基础包：

`pip install -r requirements.txt`



在环境满足要求后：

`python Main.py --model CMF`



运行结果：

![image-20200928160624757](C:\Users\netlab\AppData\Roaming\Typora\typora-user-images\image-20200928160624757.png)



