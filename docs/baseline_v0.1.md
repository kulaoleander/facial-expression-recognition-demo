# Baseline v0.1 - SimpleCNN 基础版本记录

## 1. 这个文档的作用

这个文档用于记录当前项目的第一个完整 baseline 版本。

当前项目已经完成了一个基础的 Facial Expression Recognition 流程：

- 数据集检查
- PyTorch DataLoader
- SimpleCNN 模型
- 训练
- accuracy 评估
- 保存模型权重
- 加载模型权重
- 单张图片预测
- Streamlit 网页 demo
- pytest 自动化测试

这个版本不是为了追求最高准确率，而是为了固定一个可以解释、可以复现、可以对比的基础版本。

后面所有改进，例如 detailed evaluation、training curves、validation split、data augmentation、ImprovedCNN、ResNet18，都应该和这个 baseline 进行比较。

---

## 2. 当前项目目标

项目名称：

- facial-expression-recognition-demo

项目任务：

- Facial Expression Recognition
- 中文可以理解为：人脸表情识别

当前模型要做的事情：

- 输入一张 48x48 灰度人脸图片
- 输出它属于哪一种表情类别

当前支持 7 个表情类别：

- angry
- disgust
- fear
- happy
- neutral
- sad
- surprise

---

## 3. 当前数据集状态

当前使用的数据集类型：

- FER2013-style dataset

当前本地数据结构：

- data/raw/train/angry
- data/raw/train/disgust
- data/raw/train/fear
- data/raw/train/happy
- data/raw/train/neutral
- data/raw/train/sad
- data/raw/train/surprise
- data/raw/test/angry
- data/raw/test/disgust
- data/raw/test/fear
- data/raw/test/happy
- data/raw/test/neutral
- data/raw/test/sad
- data/raw/test/surprise

当前图片格式：

- 48x48
- grayscale
- mode: L

当前模型输入 shape：

- [batch_size, 1, 48, 48]

当前模型输出 shape：

- [batch_size, 7]

这里的 7 对应 7 个表情类别。

---

## 4. 当前模型

当前模型：

- SimpleCNN

模型文件：

- src/model.py

SimpleCNN 在当前项目里的作用：

- 它是第一个基础 CNN 模型
- 它用于跑通完整深度学习流程
- 它是后续模型改进的对照组

这里要注意：

SimpleCNN 不一定是最好的模型。

它的主要价值是：

- 简单
- 能跑通
- 能解释
- 能作为 baseline

以后如果 ImprovedCNN 或 ResNet18 准确率更高，我们才可以说它们相对 SimpleCNN 有提升。

---

## 5. 当前训练设置

当前训练文件：

- src/train.py

当前训练设置：

- model: SimpleCNN
- epochs: 3
- loss function: CrossEntropyLoss
- optimizer: Adam
- evaluation: 每个 epoch 后计算 test accuracy

当前模型权重保存位置：

- outputs/models/simple_cnn.pth

注意：

simple_cnn.pth 是本地训练后生成的模型权重文件。

它不上传 GitHub。

这是正确做法，因为模型权重文件通常比较大，而且每个人可以在本地重新训练生成。

---

## 6. 当前 baseline 示例结果

一次训练示例结果：

- Epoch 1/3 | loss: 1.6252 | test accuracy: 0.4055
- Epoch 2/3 | loss: 1.4232 | test accuracy: 0.4749
- Epoch 3/3 | loss: 1.3043 | test accuracy: 0.5052

当前 baseline 大概表现：

- test accuracy 约 47% 到 50%

注意：

深度学习训练有随机性。

不同电脑、不同初始化、不同数据 shuffle 顺序，都可能导致结果略有不同。

所以后面写测试时，不应该把 accuracy 固定死，比如不能要求 accuracy 必须等于 0.5052。

---

## 7. 当前已经完成的项目流程

当前 v0.1 baseline 已经完成：

1. 检查环境
2. 检查数据集结构
3. 用 ImageFolder 读取图片数据
4. 用 DataLoader 生成 batch
5. 定义 SimpleCNN
6. 训练模型
7. 计算 overall accuracy
8. 保存模型权重
9. 加载模型权重
10. 对单张图片进行预测
11. 用 Streamlit 做网页 demo
12. 用 pytest 做自动化测试

当前主要文件：

- src/environment_check.py
- src/inspect_dataset.py
- src/data_loader.py
- src/model.py
- src/train.py
- src/evaluate.py
- src/load_model.py
- src/predict.py
- app/streamlit_app.py
- tests/

---

## 8. 当前测试状态

当前 pytest 状态：

- 29 passed

当前测试主要用于确认：

- 环境可以正常导入依赖
- 数据加载流程能工作
- 模型 forward 输出 shape 正确
- 训练函数能运行
- 评估函数能返回结果
- 模型能保存和加载
- 单图预测流程能工作

后续注意：

深度学习项目里的测试不应该强行检查某一次训练的具体准确率。

因为训练结果有随机性。

更合理的测试方式是检查：

- 函数能不能运行
- 输出格式对不对
- shape 对不对
- 文件能不能生成
- 返回值类型是否正确

---

## 9. 当前版本的限制

当前 v0.1 baseline 已经跑通，但还比较基础。

目前还没有：

- confusion matrix
- per-class accuracy
- precision
- recall
- F1-score
- classification report
- training loss curve
- accuracy curve
- validation split
- data augmentation
- ImprovedCNN
- ResNet18 transfer learning
- top-3 prediction
- low confidence warning
- OpenCV face detection
- face crop pipeline

这些不是错误。

这些就是后面阶段要逐步补上的内容。

---

## 10. 为什么要冻结 baseline

冻结 baseline 的原因是：

如果没有 baseline，后面模型改了之后就不知道有没有真的变好。

例如后面我们会做：

- 加 detailed evaluation
- 加 training curves
- 加 validation split
- 加 data augmentation
- 加 ImprovedCNN
- 加 ResNet18

如果没有记录当前 SimpleCNN 的状态，后面就会出现问题：

- 不知道 ImprovedCNN 是不是真的更好
- 不知道 augmentation 有没有帮助
- 不知道 ResNet18 提升了多少
- 不知道模型变差是因为代码问题还是实验设置变化
- 面试时解释不清楚自己怎么做实验对比

所以 baseline 的作用就是：

- 固定当前版本
- 作为后续改进的对照组
- 让项目从“能跑”变成“能分析、能比较、能解释”

---

## 11. 面试中可以怎么解释

如果面试官问：

你怎么知道后面的模型改进是有效的？

可以回答：

我先完成并记录了一个 SimpleCNN baseline。这个 baseline 包含数据加载、训练、评估、模型保存、模型加载、单图预测、Streamlit demo 和 pytest 测试。后续我再加入 detailed evaluation、training curves、validation split、data augmentation、ImprovedCNN 和 ResNet18，并且把它们和 baseline 进行比较。这样可以避免只凭感觉判断模型是否变好。

如果面试官问：

为什么当前 accuracy 只有大约 50%？

可以回答：

因为当前版本只是一个基础 SimpleCNN baseline，只训练了少量 epoch，也还没有加入 validation split、data augmentation、ImprovedCNN、transfer learning 和 face detection。这个阶段的目标不是追求最高准确率，而是先建立一个完整、可复现、可解释的基础 pipeline，然后再逐步改进。

---

## 12. 下一阶段

下一阶段是：

- Stage 13: Detailed Evaluation

下一阶段要做的事情：

- confusion matrix
- classification report
- precision
- recall
- F1-score
- per-class accuracy
- basic error analysis

下一阶段暂时不改模型。

原因是：

在改进模型之前，应该先更清楚地知道当前模型错在哪里。