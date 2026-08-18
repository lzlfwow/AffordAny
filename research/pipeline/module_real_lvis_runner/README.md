# Real LVIS Runner

该模块把真实 `LVIS` 数据集接到当前 M0-M9 管道上，并提供两个阶段的直接入口：

- `stage1`：先按 `category_name` 做文本级交互筛选，再对通过类别逐个做实例筛选。
  该实例筛选顺序是：几何预过滤 -> LLM 图像质量检验。
  每个类别只保留 1 个最终样本，并导出可审核清单。
- `stage2`：读取 `stage1` 的审核结果，只对批准对象复制到新工作区并继续执行 M3-M9，最后导出聚合数据集索引。

默认真实数据入口：

- 标注：`data/lvis/valid/annotation/lvis_v1_val.json`
- 图像：`data/lvis/valid/image/val2017/`

直接使用：

```bash
python research/pipeline/module_real_lvis_runner/run_real_lvis_pipeline.py stage1 --run-name demo --limit 20
python research/pipeline/module_real_lvis_runner/run_real_lvis_pipeline.py stage2 --run-name demo
```

说明：

- `stage1` 的 `--start-index/--limit` 现在作用在“类别批次”上，不是对象批次。
- `stage1` 的几何预过滤会先剔除：掩码占比过小、长宽比过大、贴边过多的实例。
- `stage1` 会输出终端进度，包括：类别文本筛选进度、几何拒绝、LLM 质量通过/拒绝、缺图跳过。
- `stage1` 关键文件新增：`category_filter_records.json`。
- `stage1` 现在支持断点恢复：同一个 `run_name` 下重新执行且不加 `--overwrite`，会读取 `stage1_progress.json`，跳过已完成类别并继续往后跑。
- `stage2` 仍然是对象级断点恢复：已完成对象跳过，未完成对象重跑。

关键输出：

- `research/pipeline/outputs/datasets/lvis_real/<run_name>/stage1_candidates/`
- `research/pipeline/outputs/datasets/lvis_real/<run_name>/stage2_dataset/`
