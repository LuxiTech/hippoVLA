# CALVIN 课程学习训练说明

本文档说明如何为 CALVIN 的 `RynnBrainOFT` memory 训练使用课程学习采样器。

这里的“划分数据”不是把原始 LeRobot 数据集复制成多个新目录，也不会改写原始 parquet/video 文件。它只是离线生成一个 `calvin_curriculum.json`，里面记录哪些 episode 属于 `easy`、`medium`、`hard`。训练时 dataloader 读取这个 JSON，并按不同训练阶段动态改变采样概率。

## 目标

课程学习采样器会让训练数据分布随训练步数变化：

- 训练前期更多采样简单任务和更干净的轨迹；
- 训练中期混合简单、中等、困难任务；
- 训练后期提高困难任务比例，同时保留一部分简单任务，避免遗忘。

这样做的原因是：CALVIN 是多任务评测，如果从一开始完全均匀随机采样，模型可能在困难任务、长轨迹任务、低频任务上学得不够稳定，从而影响各任务平均成功率。

## 相关文件

- `build_calvin_curriculum.py`
  - 离线分析脚本；
  - 读取转换后的 CALVIN LeRobot 数据；
  - 生成课程学习划分文件和可读 summary。

- `curriculum_calvin_abcd/calvin_curriculum.json`
  - 训练真正读取的课程学习配置；
  - 里面包含 `easy`、`medium`、`hard` 的任务和 episode 列表；
  - 也包含不同训练阶段的采样权重。

- `curriculum_calvin_abcd/calvin_curriculum_summary.md`
  - 给人看的 summary；
  - 可以检查哪些任务被分到了 easy/medium/hard。

- `run_calvin_train_rynnbrain_memory.sh`
  - 当前 memory 训练脚本；
  - 已经接入了 curriculum 开关和 curriculum 路径。

## 数据是怎么分的

脚本按“任务 task”划分，而不是按单帧划分。

也就是说，同一个自然语言任务下的所有 episode 会被放到同一个难度组里，避免同一类任务一部分在 easy、一部分在 hard，导致课程学习不稳定。

每个任务会得到一个 `difficulty_score`，范围是 `0.0-1.0`：

- 分数越低，越简单；
- 分数越高，越困难。

默认难度公式是：

```text
0.55 * episode_length_percentile
+ 0.25 * action_magnitude_percentile
+ 0.20 * action_delta_percentile
+ keyword_adjustment
```

含义：

- `episode_length_percentile`
  - 平均轨迹越长，通常任务越难；
  - 比如 drawer、slider、door、pick/place 类任务通常更长。

- `action_magnitude_percentile`
  - 动作幅度越大，可能越难；
  - 这个需要扫描 parquet 里的 action。

- `action_delta_percentile`
  - 动作变化越剧烈，轨迹可能越不平滑；
  - 这个也需要扫描 parquet。

- `keyword_adjustment`
  - 根据任务文本做轻微修正；
  - 包含 `drawer`、`sliding`、`slider`、`door`、`place`、`put`、`pick`、`grasp`、`lift`、`rotate` 的任务会更偏 hard；
  - 包含 `switch`、`button`、`light`、`lamp`、`led` 的任务会更偏 easy。

最后脚本把所有 task 按 difficulty score 排序，再大致分成三等份：

- 前 1/3：`easy`
- 中间 1/3：`medium`
- 后 1/3：`hard`

当前已经生成的一版全量划分结果是：

```text
easy: 130 tasks, 6246 episodes
medium: 130 tasks, 6291 episodes
hard: 129 tasks, 5333 episodes
```

## 如何执行代码划分数据

先进入项目根目录：

```bash
cd /home/user01/jiangnan/starVLA
```

### 快速模式

快速模式只使用 episode 长度和任务文本，不扫描所有 action parquet，速度快：

```bash
python examples/calvin/train_files/build_calvin_curriculum.py \
  --dataset-root /mnt/data/jiangnan/lerobot/task_ABC_D_lerobot \
  --output-dir examples/calvin/train_files/curriculum_calvin_abcd \
  --no-scan-actions
```

### 完整模式

完整模式会额外扫描 parquet 中的动作幅度和动作变化，更适合做最终训练：

```bash
python examples/calvin/train_files/build_calvin_curriculum.py \
  --dataset-root /mnt/data/jiangnan/lerobot/task_ABC_D_lerobot \
  --output-dir examples/calvin/train_files/curriculum_calvin_abcd
```

运行后会生成：

```text
examples/calvin/train_files/curriculum_calvin_abcd/calvin_curriculum.json
examples/calvin/train_files/curriculum_calvin_abcd/calvin_curriculum_summary.md
```

你可以先打开 summary 检查划分是否符合直觉：

```bash
less examples/calvin/train_files/curriculum_calvin_abcd/calvin_curriculum_summary.md
```

## 当前训练阶段采样策略

当前 JSON 中包含三个阶段：

| 阶段 | step 范围 | easy | medium | hard |
|---|---:|---:|---:|---:|
| `warmup_easy` | `0-20000` | `0.70` | `0.30` | `0.00` |
| `mixed` | `20000-60000` | `0.30` | `0.50` | `0.20` |
| `hard_focus` | `60000-100000` | `0.20` | `0.35` | `0.45` |

训练时 dataloader 会先根据当前 step/index 判断处于哪个阶段，然后按该阶段的权重选择 `easy`、`medium` 或 `hard`，再从对应组里的 episode 采样训练样本。

## 如何开启 curriculum

当前训练脚本已经开启了 curriculum：

```bash
/home/user01/jiangnan/starVLA/examples/calvin/train_files/run_calvin_train_rynnbrain_memory.sh
```

里面有：

```bash
curriculum_path=/home/user01/jiangnan/starVLA/examples/calvin/train_files/curriculum_calvin_abcd/calvin_curriculum.json
```

并且传入：

```bash
--datasets.vla_data.curriculum_enabled True \
--datasets.vla_data.curriculum_path ${curriculum_path} \
```

所以直接运行：

```bash
cd /home/user01/jiangnan/starVLA
bash examples/calvin/train_files/run_calvin_train_rynnbrain_memory.sh
```

如果启动日志里看到类似：

```text
Loaded curriculum from ...
curriculum group `easy` step counts per dataset: ...
curriculum group `medium` step counts per dataset: ...
curriculum group `hard` step counts per dataset: ...
```

就说明课程学习采样已经生效。

## 如何关闭 curriculum

方法一：在训练脚本中把开关改成 `False`：

```bash
--datasets.vla_data.curriculum_enabled False \
```

方法二：直接删掉或注释这两行：

```bash
--datasets.vla_data.curriculum_enabled True \
--datasets.vla_data.curriculum_path ${curriculum_path} \
```

关闭后 dataloader 会回到原来的随机采样逻辑，不再按 easy/medium/hard 调整采样比例。

## 是否真的“划分了数据”

没有真正切分、复制、移动或改写原始数据。

原始数据仍然在：

```text
/mnt/data/jiangnan/lerobot/task_ABC_D_lerobot
```

课程学习只生成一个索引文件：

```text
examples/calvin/train_files/curriculum_calvin_abcd/calvin_curriculum.json
```

这个 JSON 记录：

- 哪些 episode 属于 easy；
- 哪些 episode 属于 medium；
- 哪些 episode 属于 hard；
- 每个训练阶段 easy/medium/hard 的采样权重。

训练时只是“按索引采样”，不会改变 dataset 本身。

## 推荐实验

建议至少做这些对比：

1. Baseline：关闭 curriculum。
2. 当前 schedule：

```text
0-20000: easy=0.70, medium=0.30, hard=0.00
20000-60000: easy=0.30, medium=0.50, hard=0.20
60000-100000: easy=0.20, medium=0.35, hard=0.45
```

3. 更偏困难任务：

```text
final stage: easy=0.10, medium=0.30, hard=0.60
```

4. 更保守：

```text
final stage: easy=0.25, medium=0.45, hard=0.30
```

评测时不要只看 `final_model`。CALVIN 成功率可能在中间 checkpoint 更高，建议评测多个 checkpoint，例如：

```text
steps_50000
steps_75000
steps_100000
final_model
```

## 修改划分规则

如果你觉得某些任务分组不合理，可以修改：

```text
examples/calvin/train_files/build_calvin_curriculum.py
```

重点看：

```python
HARD_KEYWORDS = (...)
EASY_KEYWORDS = (...)
```

也可以修改 `build_curriculum()` 里的阶段权重。

改完后重新运行划分脚本，生成新的 `calvin_curriculum.json`，再重新启动训练即可。
