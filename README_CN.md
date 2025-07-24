# GRCritic: 面向真实世界具身强化学习的通用评判模型

<div align="center">

**语言**: [English](README.md) | [中文](README_CN.md)

</div>

## 🚀 交互式演示与主页

<div align="center">

### [🎮 **试用交互式演示与主页**](https://c99de865b42aff6577.gradio.live)
> **在线演示现已在主页上线，尽情试用！！！**

</div>

> **📅 模型和报告将在8月开源**
<div align="center">
  <img src="evo_vlac/examples/images/demo1-mid.gif" alt="title" width="800"/>
</div>

## GRCritic

GRCritic是一个专为真实世界具身强化学习任务设计的通用成对评判模型。它为轨迹评估和任务完成验证提供了强大的评估能力。

GRCritic基于3000小时+人类数据、1200小时+综合公开机器人操作数据和15小时+自收集操作数据进行训练。提供2b和8b两个版本。

## ✨ 核心特性

• **成对比较机制** - 提升渐进式密集评判准确性，更好地识别状态变化，可以为通用真实世界强化学习提供准确密集的评价。

• **多模态能力** - 支持过程跟踪、任务完成判断、任务描述估计、视觉问答，可以进行具身动作输出，具备VLA能力。

• **灵活的零样本和单样本** -上下文能力，在不同机器人实体、场景和任务间保持优异性能。

• **人类-任务共感** - 基于ego4D人类数据集，模型理解常见任务并建立真实世界人类任务与具身任务的共感。

• **轨迹质量筛选** - GRCritic评估收集的轨迹并基于VOC值过滤低分轨迹，即过滤流畅性和质量较低的数据，提升模仿学习的效果和效率。

## 框架

<div align="center">
  <img src="evo_vlac/examples/images/framework.png" alt="GRCritic框架" width="800"/>
</div>

*GRCritic基于综合公开机器人操作数据集、人类演示数据、自收集操作数据和各种图像理解数据集的组合进行训练。视频数据被处理成成对样本以学习任意两个状态之间的差异，辅以任务描述以实现任务完成评估和动作输出，如左下角所示。该模型对训练集中未涵盖的实体、场景和任务表现出强大的泛化能力。它评估任务进度和动作，为真实世界强化学习提供密集的评判反馈，并为数据精炼提供指导。此外，该模型可以作为视觉-语言-动作(VLA)系统直接执行操作任务，展现零样本能力来处理不同场景中新的、未见过的任务，如右侧图表所示。*

## 性能表现

关于模型性能和评估指标的详细信息可在主页中找到。

## 快速开始

> **📅 模型和报告将在8月开源**

**在线演示现已在主页上线，尽情试用！！！**

## 引用

```bibtex
@article{grcritic2025,
  title={GRCritic: General Pair-wise Critic Model for Real World Embodied Reinforcement Learning},
  author={Qi Zhang and Shaopeng Zhai},
  year={2025},
  note={Paper coming soon}
}
```