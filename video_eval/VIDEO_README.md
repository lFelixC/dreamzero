
```bash
cd /data/dreamzero
source .venv/bin/activate

NO_ALBUMENTATIONS_UPDATE=1 \
CUDA_VISIBLE_DEVICES=0,1 \
python -m torch.distributed.run --standalone --nproc_per_node=2 \
socket_test_optimized_AR.py \
  --port 5001 \
  --enable-dit-cache \
  --model-path /data/checkpoints/dreamzero/DreamZero-DROID

cd /data/dreamzero
source .venv/bin/activate

env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u all_proxy -u no_proxy -u NO_PROXY \
python test_client_AR.py --host 127.0.0.1 --port 5001 --num-chunks 1
```



请在 `/data/dreamzero` 这个 repo 内完成以下工作。要求尽量**精炼、最小改动、可复用官方逻辑**，不要自己重新实现 DreamZero 的核心推理流程，优先复用 repo 中官方 server / client 已有的函数和调用方式。

## 总体要求

1. 所有新增代码都放在：
   `/data/dreamzero/video_eval`
2. 不要修改原有训练和推理主流程，除非确实必须；如果必须修改，请保持修改最小，并在最终汇报里说明。
3. 所有路径都使用**绝对路径**。
4. 所有脚本都要能直接从命令行运行。
5. 所有新脚本都要带清晰注释，并支持 `--help`。
6. 每一步做完后，都要先做一次**最小可运行验证**，确认能跑通再进入下一步。
7. 如果遇到路径名不一致，比如 `robotset` / `roboset`，先自动检查实际目录并适配，不要卡住。
8. 最终给我一份简短的运行说明，告诉我每个脚本怎么用。

---

## 第一步：写一个精炼版推理 server / client，并实际启动验证

### 目标

参考官方给出的命令和实现，写一个更精炼的推理版本，放在：

* `/data/dreamzero/video_eval/server.py`
* `/data/dreamzero/video_eval/client.py`

要求：

1. 复用官方 server/client 的核心函数，不要重写模型推理逻辑。
2. 支持使用 checkpoint：
   `/data/checkpoints/dreamzero/dreamzero_droid_wan22_full_finetune/checkpoint-5000`
3. client 输出的视频保存到：
   `/data/dreamzero/video_eval/video_results`
4. 每次运行单独创建一个时间戳文件夹，格式例如：
   `YYYYMMDD_HHMMSS`
5. 每次推理结果至少保存：

   * 生成视频
   * 输入信息或请求内容
   * 运行日志
   * 一个简单的 metadata 文件，记录 checkpoint、时间、输入样本标识

### 建议输出结构

例如：

```bash
/data/dreamzero/video_eval/video_results/20260408_153000/
  ├── pred.mp4
  ├── request.json
  ├── metadata.json
  ├── run.log
```

### 验证要求

写完后，请你自己实际尝试：

1. 用上述 checkpoint 启动 server
2. 用 client 发送一次最小测试请求
3. 确认成功生成并保存视频到对应时间戳目录

如果失败：

* 不要跳过
* 先查看报错并尽量修复
* 最终汇报里写明失败点和当前状态

---

## 第二步：解压 RoboSet 数据

### 目标

在目录：

`/data/datasets/robotset`

下找到其中的两个 RoboSet 数据集压缩包并解压。

### 要求

1. 自动识别压缩格式，比如 `.zip`、`.tar`、`.tar.gz`、`.tgz`。
2. 解压后保持目录清晰，不要把文件打散到根目录。
3. 解压完成后输出每个数据集的实际路径和目录结构前两层。
4. 如果 `/data/datasets/robotset` 实际不存在，请检查是否为 `/data/datasets/roboset`，并自动适配。

---

## 第三步：写转换脚本，把 RoboSet 转成 DreamZero 可直接推理的格式，并做抽样验证

### 目标

写一个转换脚本：

`/data/dreamzero/video_eval/convert_roboset_to_dreamzero.py`

把第二步解压出来的两个 RoboSet 数据集转换成 **DreamZero 当前推理流程可以直接读取的最小输入格式**。

### 要求

1. 不要凭空定义格式，先去检查 DreamZero 官方 client / 推理入口真正需要的输入格式，然后按那个格式生成。
2. 两个数据集**各自抽取 50 条**。
3. 抽样时使用固定随机种子，例如 `42`，保证可复现。
4. 保存一份 manifest，记录抽到的是哪些轨迹/样本。
5. 转换后的数据输出到指定/data/datasets/robotset下，格式严格要求dreamzero的格式。

### 转换内容要求

请根据 DreamZero 推理实际需要，尽量保留并组织好：

* 历史观测帧
* 动作
* GT future video 或 future frames
* 样本 id
* 必要的相机信息或元数据

如果某些字段 RoboSet 没有，而 DreamZero 推理又必须要，请在代码里做最小兼容处理，并把处理方式写进 conversion log。

### 验证要求

转换完成后：

1. 从每个数据集各取 1 条样本
2. 用第一步写好的 server/client 或等价调用方式做一次真实推理
3. 把生成视频保存到 `video_results` 的时间戳目录下
4. 检查该样本是否真的能完整走通推理流程

如果只能先跑通 1 个数据集，也要明确写出来，不要假装两个都成功。

---

## 第四步：写视频评测脚本，计算 LPIPS / SSIM / PSNR

### 目标

写一个评测脚本：

`/data/dreamzero/video_eval/evaluate_video_metrics.py`

它可以读取推理完成的视频，并与对应 GT 视频逐帧对齐，计算以下三个指标：

* LPIPS
* SSIM
* PSNR

### 要求

1. 支持输入：

   * 预测视频路径
   * GT 视频路径，或者 GT 帧目录
   * 输出目录
2. 支持把视频拆成图片帧后再计算，也可以直接逐帧读取，但要确保和 GT 正确对齐。
3. 默认按帧一一对应比较。
4. 如果长度不一致：

   * 默认截断到两者最短长度
   * 在结果文件里记录原始长度和实际评测长度
5. 如果分辨率不一致：

   * 优先以 GT 分辨率为准对预测帧做 resize
   * 并在结果中记录这一点
6. 结果写到对应视频输出目录下，例如：

```bash
/data/dreamzero/video_eval/video_results/20260408_153000/
  ├── pred.mp4（这是之前推理得到的）
  ├── metrics_results.txt
  ├── metrics_results.json
  └── frame_pairs/   # 如果你实现了中间帧导出，可选
```

### 指标实现要求

1. LPIPS 用常见实现即可，优先稳定、易安装的版本。
2. SSIM 和 PSNR 使用标准实现。
3. 输出内容至少包含：

   * 每帧 LPIPS / SSIM / PSNR
   * 平均值
   * 帧数
   * 样本 id
4. 最好额外输出一个汇总格式，便于后续多样本批量统计。

### 验证要求

请至少对第三步成功推理出来的一条样本，完成：

1. 预测视频与 GT 的对齐
2. 指标计算
3. 在结果目录下写出 metrics 文件

---

## 最终交付物

请确保最终至少包含这些文件：

```bash
/data/dreamzero/video_eval/
  ├── server.py
  ├── client.py
  ├── convert_roboset_to_dreamzero.py
  ├── evaluate_video_metrics.py
  ├── README.md
  └── video_results/
```

README.md 里请写清楚：

1. 每个脚本的用途
2. 依赖安装方式
3. 启动 server 的命令
4. 发送 client 请求的命令
5. 数据转换命令
6. 指标评测命令
7. 你实际验证过的样例路径
8. 当前已知问题和限制

---

## 额外要求

1. 优先保证**先跑通单条样本**，再考虑批量化。
2. 代码风格保持简单，不要过度封装。
3. 关键步骤都打印日志。
4. 如果某一步跑不通，不要直接跳过，请保留中间产物并说明原因。
5. 最终请给出一份简短总结，说明：

   * 哪些脚本已完成
   * 哪些验证已通过
   * 生成视频保存在什么位置
   * 指标文件保存在什么位置



可以，这样更合理。你真正需要的不是“固定两个 50 条数据集”，而是一个**通用的单数据集 batch eval 入口**：

* 传入一个 `gt` 数据集路径
* 自动扫描这个数据集里的样本
* 批量推理生成一批视频
* 自动和对应 GT 配对
* 统一计算指标
* 输出 summary


新增改动：

请你在现有 `/data/dreamzero/video_eval` pipeline 基础上做一次**通用化的单数据集 batch eval 改造**。

我后续会不断加入新的测试数据集，所以这里**不要把逻辑写死成两个固定数据集**。
我希望你实现的是一个**通用入口**：

* 传入一个 GT 数据集根路径
* 自动读取其中的样本
* 用指定 checkpoint 批量推理生成预测视频
* 自动与对应 GT 配对
* 批量计算 LPIPS / SSIM / PSNR
* 输出精简 summary

注意：
我只需要**单独 eval 一个数据集**。
如果以后我要同时 eval 两个数据集，我会自己写外层脚本调用两次。
所以请你把这次实现的重点放在：

**“给一个 gt_root，就能完成这个数据集的一整套 batch 推理和评测。”**

---

# 目标

请新增一个通用批量入口，例如：

`/data/dreamzero/video_eval/run_batch_eval.py`

它应该完成从“读取一个数据集”到“生成 summary”的整条链路。

---

# 输入参数

请至少支持以下参数：

```bash
python /data/dreamzero/video_eval/run_batch_eval.py \
  --gt_root /path/to/one/eval_dataset \
  --checkpoint /data/checkpoints/dreamzero/dreamzero_droid_wan22_full_finetune/checkpoint-5000 \
  --output_root /data/dreamzero/video_eval/video_results
```

建议支持的参数包括：

* `--gt_root`
  要评测的单个数据集根目录。后续会换不同数据集，所以这是核心参数。

* `--checkpoint`
  DreamZero 推理使用的 checkpoint。

* `--output_root`
  batch eval 的输出根目录。

* `--max_samples`
  可选。用于 smoke test，限制只跑前 N 条。

* `--manifest_path`
  可选。如果给了 manifest，就优先从 manifest 读；如果没给，就自动扫描 `gt_root`。

* `--resume`
  可选。若某些样本已经完成推理或评测，可以跳过，避免重复跑。

* `--server_mode`
  可选。保留当前 server/client 推理方式时可用。

---

# 设计要求

## 1. 单数据集通用，不要写死具体数据集名字

不要把代码写死成只支持两个 RoboSet 数据集，或者只支持某个固定目录结构。

你要做的是：

* 给一个 `gt_root`
* 自动发现可评测样本
* 组装出 DreamZero 推理所需输入
* 统一完成 batch eval

也就是说，这个脚本应该尽量**数据集无关**。
如果某个数据集有 manifest，就优先用 manifest；
如果没有 manifest，就写一个通用 scanner，至少能扫描已有转换后的 DreamZero eval 格式。

---

## 2. 允许内部逐条处理，但对外必须是批处理

可以内部逐条 for-loop 推理和评测。
但对我来说，我只想执行**一条命令**，而不是手工对每个视频单独跑。

大白话就是：

**内部怎么实现都行，但外部体验必须是一键跑完整个数据集。**

---

## 3. 尽量复用现有可用代码

请尽量复用现在已经打通的：

* server/client 逻辑
* 单条推理逻辑
* `evaluate_video_metrics.py`

不要推翻重写。
优先做成一个**batch orchestrator**，把现有流程串起来。

---

# 样本发现与读取

请你优先支持以下两种模式：

## 模式 A：manifest 驱动

如果 `gt_root` 下存在 manifest 或用户显式传入 `--manifest_path`，就优先从 manifest 读取样本。
每条样本至少要能定位到：

* `sample_id`
* 推理输入所需的 context / action / prompt / metadata
* `gt_video` 或 `gt_frames`

## 模式 B：目录扫描

如果没有 manifest，就自动扫描 `gt_root`，识别是否是当前 DreamZero 可推理格式。
请尽量支持已有转换后的标准目录。

如果扫描失败，不要静默跳过，要清楚报错说明：

* 没找到 manifest
* 也没识别出标准目录结构

---

# 输出目录规范

请把一次 batch run 的结果保存到一个时间戳目录，例如：

`/data/dreamzero/video_eval/video_results/batch_YYYYMMDD_HHMMSS/`

建议目录结构如下：

```bash
/data/dreamzero/video_eval/video_results/batch_YYYYMMDD_HHMMSS/
  ├── samples/
  │   ├── sample_000/
  │   │   ├── metadata.json
  │   │   ├── pred.mp4
  │   │   ├── gt.mp4
  │   │   ├── metrics_results.json
  │   │   └── metrics_results.txt
  │   ├── sample_001/
  │   └── ...
  ├── batch_manifest.json
  ├── overall_summary.txt
  ├── overall_summary.json
  ├── per_task_lpips.csv
  ├── worst_5_cases.csv
  └── run.log
```

要求：

* 每条样本一个独立子目录
* 每条样本保留自己的 pred / gt / metrics
* 根目录保留总体 summary
* `batch_manifest.json` 记录所有样本状态

---

# 批量推理要求

请把现有单条推理逻辑封装进 batch runner。

要求：

* 自动遍历数据集中的样本
* 每条样本都保留自己的 `sample_id`
* 每条样本都生成自己的 `pred.mp4`
* 每条样本都正确绑定自己的 GT

如果当前最稳定的方式是 server/client：

* 可以继续使用 server/client
* 但请由 `run_batch_eval.py` 自动完成整批请求
* 不要让我手工对每条样本运行 client

如果直接 Python 内部调用更稳：

* 也可以在 batch runner 里直接调用
* 但尽量复用现有逻辑，不要自己重写推理核心

---

# 批量评测要求

请对 batch 中每条成功推理的样本，自动完成：

* LPIPS
* SSIM
* PSNR

每条样本都输出：

* `metrics_results.json`
* `metrics_results.txt`

要求：

* 自动找到对应 GT
* 长度不一致时默认截断到最短长度
* 分辨率不一致时默认 resize 到 GT 分辨率
* 这些行为必须写进 `metrics_results.json`

---

# 只保留精华 summary

我不要太复杂的指标。
请最终只输出这几项。

## A. `overall_summary`

包括：

* `num_total_samples`
* `num_successful_inference`
* `num_successful_metrics`
* `num_failed_samples`
* `mean_lpips`
* `mean_ssim`
* `mean_psnr`
* `early_lpips`
* `late_lpips`
* `lpips_drift = late_lpips - early_lpips`
* `avg_evaluated_frames`
* `truncation_rate`
* `resize_rate`

大白话解释：

* `mean_lpips / ssim / psnr`：平均来看，这个数据集上预测得怎么样
* `early vs late lpips`：模型会不会越预测越飘
* `avg_evaluated_frames / truncation_rate / resize_rate`：这些分数是不是在比较公平的条件下算出来的

## B. `per_task_lpips.csv`

如果样本里能解析出 task，就输出：

* `task`
* `N`
* `mean_lpips`

如果没有 task 字段，就跳过，不要报错。

大白话解释：

* 这是为了看模型最怕哪类任务

## C. `worst_5_cases.csv`

输出 LPIPS 最差的 5 条：

* `sample_id`
* `task`（如果有）
* `pred_video`
* `gt_video`
* `mean_lpips`
* `evaluated_frames`

大白话解释：

* 这是为了快速定位模型最典型的失败样本

---

# 容错要求

1. 如果某个样本推理失败：

* 记录到 `run.log`
* 在 `batch_manifest.json` 里标记状态
* 继续跑后续样本

2. 如果某个样本评测失败：

* 同样记录错误
* 不影响其他样本

3. 最终 summary 只基于成功完成 metrics 的样本统计

4. 如果 `--resume` 打开：

* 已完成推理且已有 `pred.mp4` 的样本可跳过推理
* 已完成 `metrics_results.json` 的样本可跳过评测
* 但最终 summary 仍要重新生成

---

# README 更新要求

请更新 `/data/dreamzero/video_eval/README.md`，写清楚：

1. `run_batch_eval.py` 的用途
2. `--gt_root` 的作用
3. 输入数据集应该具备什么结构
4. manifest 模式和自动扫描模式怎么工作
5. 如何运行一整个数据集的 batch eval
6. 输出目录里每个文件是什么意思
7. overall summary 里每个值的大白话解释
8. 你实际 smoke test 过的命令

---

# 最后要求

请你实际做一次最小验证。
如果完整数据集太慢，可以用 `--max_samples 2` 或 `--max_samples 5` 做 smoke test。
但代码必须支持完整数据集批量运行。

最终请告诉我：

* 这个单数据集 batch eval 入口是否完成
* 运行命令是什么
* 输出目录在哪里
* summary 文件在哪里
* 当前已知限制有哪些


