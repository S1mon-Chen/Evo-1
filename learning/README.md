# Evo-1 三天共学路线

目标：能解释论文的核心想法、从图像和指令到动作的代码路径，并亲手运行一个使用仓库真实动作生成类的 CPU 练习。

## 第 1 阶段：问题与总体结构

读 [论文 v2](https://arxiv.org/html/2511.04555v2) 的摘要、引言和第 3.1 节。回答三个问题：输入是什么？输出是什么？为什么作者要同时追求轻量化和保留视觉语言语义？

## 第 2 阶段：论文与代码对照

读论文第 3.2、3.3 节，并沿着这些文件追踪数据：

1. `Evo_1/scripts/Evo1.py`：组装视觉语言编码器和动作生成模块。
2. `Evo_1/model/internvl3/internvl3_embedder.py`：把图像、文本转为融合 token。
3. `Evo_1/model/action_head/flow_matching.py`：训练时预测速度；推理时从噪声逐步生成动作。
4. `Evo_1/scripts/train.py`：计算训练目标和损失。

## 第 3 阶段：CPU 动手练习

练习文件是 `learning/flow_matching_exercise.py`。它直接调用仓库的 `FlowmatchingActionHead`，使用合成 token、机器人状态和动作。这个练习帮助理解机制，不代表预训练 Evo-1 的机器人表现。

先从仓库根目录检查环境：

```bash
conda run -n cs231n python learning/flow_matching_exercise.py --check
```

完成文件中的两个 TODO 后运行：

```bash
conda run -n cs231n python learning/flow_matching_exercise.py
```

完成标准：解释 `actions_gt`、`noise`、`target_velocity` 的形状和关系，并说明 `get_action` 为什么要反复更新动作。
