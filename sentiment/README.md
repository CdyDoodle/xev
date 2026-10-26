# xev-sentiment

从 **Qwen3.5-4B-Base** 开始，用自己写的代码 post-train 一个中文评论 sentiment classifier（positive / negative），输出 calibrated probability。读出方式借用 [Kev](https://github.com/jaredpalmer/kev) 的 pointer head 思路，但不使用 Kev 的代码和权重。

- 实验报告：<https://cdydoodle.github.io/xev/sentiment/report.html>（[report.html](report.html)）
- 实验计划：<https://cdydoodle.github.io/xev/sentiment/plan.html>（[plan.html](plan.html)）

## 结果

| | Zero-shot baseline | Ours（T=1.23） |
|---|---|---|
| ChnSentiCorp test accuracy（848 条） | 0.905 | **0.956**（+5.2 pp，95% CI [+3.5, +7.0]） |
| ChnSentiCorp test NLL / ECE | 0.236 / 0.024 | **0.123 / 0.016** |
| 错误率 ≤1% 时可自动处理的比例 | 42.7% | **85.7%** |
| 外卖评论 out-of-domain accuracy（11,973 条） | **0.899** | 0.886（−1.3 pp，95% CI [−1.8, −0.9]） |

- 原始 ChnSentiCorp 切分有严重 leakage：test 里 324 条（27%）和 train 完全一样，另有 36 条评论标签自相矛盾，全部删除后再评估。
- 领域内明显变好；领域外（外卖评论很短、负面居多）比不训练的 base model 略差。
- 单 seed、没有做 hyperparameter search。

## 方法

```
<state> 评论 <q> 这条评论的情感倾向是？ <opt> 负面：… </opt> <opt> 正面：… </opt> <decide>
```

- Backbone：Qwen3.5-4B-Base 去掉 LM head，bf16，冻结；LoRA r=16（attention、MLP、Gated DeltaNet 的投影）
- Pointer head：score_j = (W_q h_decide)·(W_k h_opt_j)/√256，W_k 零初始化
- Loss：两个选项上的 cross-entropy，每个 epoch 随机交换选项顺序
- 训练：AdamW，lr 1e-4，batch 16，2 个 epoch（1,028 步），在 RTX 4090 上 36 分钟
- Calibration：在 validation 上拟合 temperature（T=1.23）

## 文件

| 文件 | 作用 |
|---|---|
| `prepare_data.py` | 下载 ChnSentiCorp，删除标签矛盾、重复和 leakage |
| `baseline.py` | zero-shot baseline：base model 读 A/B 的 next-token 概率 |
| `model.py` | 输入格式、pointer head、LoRA |
| `train.py` | 训练循环 |
| `calibrate.py` | 在 validation 上拟合 temperature |
| `evaluate.py` / `metrics.py` | accuracy、NLL、Brier、ECE、coverage、order flip、bootstrap CI |
| `prepare_ood.py` | 外卖评论 out-of-domain 数据 |
| `serve.py` | FastAPI 服务：`POST /v1/sentiment` |
| `tools/` | 辅助脚本（smoke run 对比、分析、服务测试） |

## 运行

需要 Linux（或 WSL）和一块 CUDA GPU（实测 RTX 4090，显存峰值约 10 GB）。

```bash
source env.sh          # venv 和 Hugging Face cache 放在 Linux 文件系统
uv sync
uv run python prepare_data.py
uv run python baseline.py --split validation
uv run python train.py --out runs/main
uv run python calibrate.py --run runs/main
uv run python baseline.py --split test --temperature 1.2030
uv run python evaluate.py --run runs/main
uv run uvicorn serve:app --port 8010
```

## 数据与 License

- [ChnSentiCorp](https://huggingface.co/datasets/lansinuote/ChnSentiCorp)（revision `b0c4c119`）和 [waimai_10k](https://huggingface.co/datasets/XiangPan/waimai_10k)（revision `dbb12540`）都没有写明 license，这里只用于学习和研究；数据不在本仓库里，运行脚本时从 Hugging Face 下载。报告里引用了少量评论作为分析例子。
- Base model [Qwen3.5-4B-Base](https://huggingface.co/Qwen/Qwen3.5-4B-Base) 是 Apache-2.0。训练好的权重没有上传。
