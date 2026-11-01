# SOD：面向小模型工具推理的逐步骤在线蒸馏

本仓库基于论文 [*SOD: Step-wise On-policy Distillation for Small Language Model Agents*](https://arxiv.org/abs/2605.07725) 及其[原始实现](https://github.com/YoungZ365/SOD)，复现 Qwen3 小模型的冷启动 SFT 与 SOD 训练。下文依次介绍研究背景、问题、方法、论文效果和复现流程。

**本次复现模型：**[Qwen3-1.7B 冷启动 SFT（Hugging Face）](https://huggingface.co/Taoger-Xu/SOD-Qwen3-1.7B-SFT) · [SFT 训练曲线](https://swanlab.cn/@xujintao/SOD/runs/baxdrcv2/chart) · [SOD 训练曲线](https://swanlab.cn/@xujintao/SOD/runs/sod1p7b1/chart)

## 背景

工具增强推理（Tool-Integrated Reasoning，TIR）要求模型在多轮推理中调用代码执行等外部工具，并根据工具返回的观察继续作答。小模型在长轨迹中探索能力较弱，而强化学习通常只在任务结束时给出稀疏奖励。在线策略蒸馏（On-Policy Distillation，OPD）让学生在自己生成的轨迹上接受教师的逐 token 指导，提供更密集的训练信号。

## 问题

在 TIR 中，一次错误的工具调用可能把错误观察写入后续上下文，使学生与教师的状态分布迅速拉开，错误还会沿后续步骤累积。普通 OPD 对各步骤施加相同强度的教师指导，可能在学生已明显偏离教师的状态上继续放大不可靠的信号。

![论文中的问题示意图](assets/intro.png)

## 研究方法

SOD（Step-wise On-policy Distillation）在学生自己生成的多轮轨迹上联合使用任务奖励与教师指导：

1. 学生生成回答和工具调用；工具环境返回观察，形成多步骤轨迹。
2. GRPO 根据任务结果提供轨迹级训练信号。
3. 学生和教师分别计算学生轨迹的 token 概率；每个推理步骤用两者的平均 log 概率差异衡量偏离程度。外部工具产生的观察 token 不计入这一步骤的差异。
4. 偏离扩大时降低该步骤的蒸馏权重；学生重新接近教师时恢复权重，并设置上限。
5. 联合优化 GRPO 损失和逐步骤加权的 OPD 损失。SOD 阶段更新的是学生模型，教师只提供指导。

权重计算复用蒸馏前向过程已有的概率。方法定义与公式见[论文第 4 节](https://arxiv.org/html/2605.07725v3#S4)。

![SOD 框架图](assets/framework.png)

## 论文效果

下表是**原论文**的 average@32 结果，并非本次复现的评测成绩。这里只列出与方法对比最直接的 SFT、普通 OPD 和 SOD；完整基线见[论文第 5 节](https://arxiv.org/html/2605.07725v3#S5)。

| 学生 | 方法 | AIME 2024 | AIME 2025 | GPQA | LiveCodeBench | 四项平均 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 0.6B | SFT | 5.67 | 5.42 | 15.20 | 9.61 | 8.97 |
| 0.6B | OPD | 16.82 | 22.95 | 17.76 | 22.65 | 20.04 |
| 0.6B | **SOD** | **20.84** | **26.13** | **22.19** | **27.72** | **24.22** |
| 1.7B | SFT | 26.77 | 22.40 | 29.85 | 24.63 | 25.91 |
| 1.7B | OPD | 43.86 | 37.04 | 31.73 | 32.45 | 36.27 |
| 1.7B | **SOD** | **50.83** | **41.72** | **38.72** | **40.63** | **42.98** |

### 本次复现状态

本次使用 Qwen3-1.7B 学生和公开的 [SOD-GRPO_teacher-4B](https://huggingface.co/youngzhong/SOD-GRPO_teacher-4B) 教师。3K SFT 已完成 5 个 epoch、共 115 步；最终训练损失约为 0.143，合并后的模型位于 `outputs/qwen3_1p7b_sft_hf/`。[SFT 训练曲线](https://swanlab.cn/@xujintao/SOD/runs/baxdrcv2/chart)和[正在进行的 SOD 训练曲线](https://swanlab.cn/@xujintao/SOD/runs/sod1p7b1/chart)保存在 SwanLab。

SFT 脚本目前将同一份 3K 数据用于训练和损失验证，因此其 `val/loss` **不是独立测试集成绩**。本次复现尚未得到可与论文结果直接比较的独立基准成绩。

## 复现流程

### 1. 环境、数据和工具

```bash
git clone https://github.com/Taoger-Xu/Step-wise-OPD.git
cd Step-wise-OPD
conda create -n SOD python=3.11
conda activate SOD
bash scripts/install_vllm_sglang_mcore.sh
pip install -e '.[vllm]'
```

下载 [3K SFT 数据](https://huggingface.co/datasets/Gen-Verse/Open-AgentRL-SFT-3K)、[30K RL 数据](https://huggingface.co/datasets/Gen-Verse/Open-AgentRL-30K)、[评测数据](https://huggingface.co/datasets/Gen-Verse/Open-AgentRL-Eval)，以及 [Qwen3-1.7B](https://huggingface.co/Qwen/Qwen3-1.7B) 学生基础模型和 [4B 教师](https://huggingface.co/youngzhong/SOD-GRPO_teacher-4B)。原仓库也支持 [Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B) 学生。

工具执行需要 [SandboxFusion](https://github.com/bytedance/SandboxFusion)。部署后将地址写入 `recipe/demystify/sandbox_fusion_tool_config.yaml`；代码评测地址也可用 `SOD_SANDBOX_FUSION_URL` 设置。本机配置使用 `http://127.0.0.1:8080/run_code`。

### 2. 冷启动 SFT

```bash
export MODEL_PATH=/path/to/Qwen3-1.7B
export TRAIN_DATA=/path/to/full_sft_3k_shuffled_v4.parquet
export SAVE_PATH=/path/to/sft_checkpoints
bash examples/SOD/run_sft.sh optim.lr=5e-5
```

本次 1.7B 复现使用 8 张 GPU、全局 batch 128、每卡 micro batch 16、序列并行 4、最大长度 32,768、右截断、学习率 `5e-5`、5 个 epoch。当前脚本每 10 步保存一次且只保留最近 1 个检查点；多卡通信在本机设置了 `NCCL_P2P_DISABLE=1`。

### 3. 合并 SFT 检查点

SFT 产生 FSDP 分片检查点；SOD 加载 Hugging Face 格式的学生模型：

```bash
python -m verl.model_merger merge --backend fsdp \
  --local_dir /path/to/sft_checkpoints/global_step_115 \
  --target_dir /path/to/qwen3_1p7b_sft_hf
```

### 4. SOD 训练

```bash
export OPEN_AGENT_RL=/path/to/Open-AgentRL-30K.parquet
export AIME_2024=/path/to/aime_2024_problems.parquet
export AIME_2025=/path/to/aime_2025_problems.parquet
export STUDENT_MODEL_PATH=/path/to/qwen3_1p7b_sft_hf
export TEACHER_MODEL_PATH=/path/to/SOD-GRPO_teacher-4B
export TOOL_CONFIG_PATH=recipe/demystify/sandbox_fusion_tool_config.yaml
export DEFAULT_LOCAL_DIR=/path/to/sod_checkpoints
bash examples/SOD/run_sod.sh
```

`run_sod.sh` 默认使用 8 张 GPU、训练 batch 64、每题采样 16 条轨迹、学生学习率 `1e-6`、最多 16 轮工具交互、训练 1 个 epoch，并每 10 步评测和保存。原 README 的参考硬件为 8× H20 96GB；本次在 8× A100 80GB 上运行。教师是项目提供的 GRPO 检查点，学生训练脚本不会重新训练教师。

### 5. 独立评测

```bash
bash examples/SOD/eval/run_eval_aime.sh
bash examples/SOD/eval/run_eval_livecodebench.sh
```

评测数据还包含 GPQA-Diamond。比较 SFT 与 SOD 检查点时，应说明采样设置与论文 average@32 口径是否一致。

## 模型与原论文引用

论文作者提供的模型：[SOD-0.6B](https://huggingface.co/youngzhong/SOD-0.6B)、[SOD-1.7B](https://huggingface.co/youngzhong/SOD-1.7B)、[SOD-GRPO_teacher-4B](https://huggingface.co/youngzhong/SOD-GRPO_teacher-4B)。本次复现的冷启动模型：[Qwen3-1.7B SFT](https://huggingface.co/Taoger-Xu/SOD-Qwen3-1.7B-SFT)；它尚未经独立基准评测。

原 README 中的论文引用保留如下：

```bibtex
@article{zhong2026sod,
  title={SOD: Step-wise On-policy Distillation for Small Language Model Agents},
  author={Zhong, Qiyong and Zheng, Mao and Song, Mingyang and Lin, Xin and Sun, Jie and Jiang, Houcheng and Wang, Xiang and Fang, Junfeng},
  journal={arXiv preprint arXiv:2605.07725},
  year={2026}
}
```

本项目基于 [VeRL](https://github.com/volcengine/verl)、[Open-AgentRL](https://github.com/Gen-Verse/Open-AgentRL) 和 [ReTool](https://github.com/ReTool-RL/ReTool) 的工作。
