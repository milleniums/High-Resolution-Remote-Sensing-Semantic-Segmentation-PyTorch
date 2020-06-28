# High-Resolution-Remote-Sensing-Semantic-Segmentation-PyTorch
遥感也许还不能预测未来，但它一定会告诉你过去和现在，以及一步步见证着未来。

## Update soon：
   - [x] 膨胀预测
   - [ ] 后处理
   - [ ] 半监督方法：伪标签
   - [ ] 数据增强：MixUp
   - [x] 加入tensorboardX可视化输出
   - [ ] 类别不均衡解决办法：focal loss
   - [ ] 引入辅助loss：边缘损失、KL散度、向量场损失


## TODO:
   - [ ] 总结训练步骤、技巧
   - [ ] 分享预训练模型


## 项目预测图展示
   ![avatar](pic/predict.png)



## 使用TensorBoard查看各项指标以及可视化（img, GT, Pre）
   - 本地：
   
    tensorboard --logdir=<your_log_dir>
   - 服务器：
    
    #首先使用ssh登陆服务器并转发端口:
    ssh -L 6006:127.0.0.1:6006 远程服务器用户名@远程服务器Ip
    (ssh -L 本地端口:本地IP:远程端口 远程服务器用户名@远程服务器Ip)
    
    #在服务器上运行tensorboard:
    tensorboard --logdir=<your_log_dir>
    
    #在浏览器里输入:
    远程服务器Ip:6006
   - 各项指标样例
   
   ![avatar](pic/tensorboard.png)
   
   - 可视化测试集样例（img, GT, Pre）
   
   ![avatar](pic/img_gt_pre.png)

    
   
## 数据分享

- **场景分类**
    ##### 2019年遥感图像稀疏表征智能分析竞赛
    百度网盘 链接:https://pan.baidu.com/s/1NFvL8KV4pRwumBah05GG8A  密码:x5k0
    
- **变化检测**
    ##### 2019年遥感图像稀疏表征智能分析竞赛
    百度网盘 链接:https://pan.baidu.com/s/1u4WKF_rRysvc0tEnrYq0Sg  密码:4udm

- **语义分割**
    ##### [2019年县域农业大脑AI挑战赛](https://tianchi.aliyun.com/competition/entrance/231717/information)
    百度网盘 链接:https://pan.baidu.com/s/1VXMLTCZM10quebX6Wf169w  密码:cmt7
    ##### 2019年遥感图像稀疏表征智能分析竞赛
    百度网盘 链接:https://pan.baidu.com/s/1LiMjwotcnQus0HFntEiZog  密码:3obz
    ##### [GID](https://arxiv.org/abs/1807.05713)
    [官方下载链接](https://x-ytong.github.io/project/GID.html)
    
    百度网盘 链接:https://pan.baidu.com/s/1LM6WX6zgihzIlY5OJWGEFA  密码:n8n0
    
    - 数据集信息
    
        ![avatar](pic/Fifteen_info.PNG)
        
        ![avatar](pic/Five_info.PNG)
    
    - 样例展示
        ![avatar](pic/classes_sample.jpg)


## Reference

    https://github.com/kazuto1011/deeplab-pytorch
    https://github.com/fregu856/deeplabv3
    https://github.com/zijundeng/pytorch-semantic-segmentation
    https://github.com/jwyang/dcn.pytorch


## Contact
    
    author: hk
    qq：3025531252
    email: 3025531252@qq.com