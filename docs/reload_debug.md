# DreamZero Reload Debug

## 1. 重复加载现象定义

### 1.1 本轮对“真的重复加载”的定义

同一个进程里，同一份逻辑资源被实际读入两次及以上，且第二次加载不是业务必需：

- 同一份 encoder 权重再次 `torch.load`
- 同一份 DiT safetensors shard 再次 `load_file`
- 同一路径 tokenizer 再次 `AutoTokenizer.from_pretrained`
- 同一个 checkpoint 在模型初始化后又被整包叠加一次

这类问题会真实增加启动时间、CPU 内存峰值，部分场景还会放大显存抖动。

### 1.2 本轮对“日志重复打印但实际只加载一次”的定义

以下情况先不当作 bug：

- 分 shard 打印多条 `Loading shard: ...`，但每条对应不同 shard 文件
- 多 rank 同时打印同一条加载日志，但每个 rank 只执行了自己那一次加载
- 外层 wrapper 打印“开始加载模型”，内层真实加载只发生一次

这类情况需要靠计数器和耗时，而不是靠肉眼看日志次数判断。

### 1.3 当前已确认的高优先级真实重复加载

#### A. 全量 checkpoint 加载链路会重复加载 encoder 和可能的 DiT

调用链 1，训练侧：

- `groot/vla/experiment/base.py::BaseExperiment.create_model`
- `instantiate(cfg.model)`
- `groot/vla/model/dreamzero/base_vla.py::VLA.__init__`
- `groot/vla/model/dreamzero/action_head/wan_flow_matching_action_tf.py::WANPolicyHead.__init__`
- 这里先加载：
  - `text_encoder_pretrained_path`
  - `image_encoder_pretrained_path`
  - `vae_pretrained_path`
  - 以及在 `skip_component_loading=false` 时加载 DiT shard
- 然后 `BaseExperiment.create_model` 再从 `cfg.pretrained_model_path` 把完整 checkpoint state_dict 叠加一次

结论：

- text encoder / image encoder / VAE 是明确的真实重复加载
- DiT 是否重复，取决于 `skip_component_loading`

#### B. 推理侧从 full checkpoint 加载时也会重复加载

调用链 2，推理侧：

- `groot/vla/model/n1_5/sim_policy.py::GrootSimPolicy.__init__`
- `cls.from_pretrained(model_path)`
- `groot/vla/model/dreamzero/base_vla.py::VLA.from_pretrained`
- `model = cls(config)`
- `WANPolicyHead.__init__` 再次先加载 text/image/VAE/DiT
- 然后 `VLA.from_pretrained` 再把 checkpoint state_dict 载入一次

结论：

- full checkpoint 推理路径同样存在真实重复加载

### 1.4 当前已确认的次优先级重复加载

#### C. tokenizer 在训练数据链路里被重复构造

调用链：

- `groot/vla/model/dreamzero/transform/dreamzero_cotrain.py::DreamTransform`
- `groot/vla/model/dreamzero/transform/dreamzero_cotrain.py::DefaultDataCollator`

两处都会用同一个 `tokenizer_path` 调用 `AutoTokenizer.from_pretrained`。

说明：

- 这是同进程内真实重复构造
- 但权重文件大多已经在 HF cache，本轮优先级低于 encoder / DiT 重复加载

## 2. 保存时缺少 encoder 权重的现象定义

### 2.1 LoRA-only 保存不会带上 encoder 权重

调用链：

- `groot/vla/experiment/base.py::BaseTrainer.save_model`
- 当 `save_lora_only=true` 时，只保留 `requires_grad=True` 的参数
- `groot/vla/model/dreamzero/action_head/wan_flow_matching_action_tf.py`
  里 `text_encoder` / `image_encoder` / `vae` 都被显式冻结

结果：

- LoRA-only checkpoint 不会包含 encoder 权重
- 之后加载这类 checkpoint 时，仍然依赖外部 `text_encoder_pretrained_path` / `image_encoder_pretrained_path` / `vae_pretrained_path`

### 2.2 即使 full checkpoint 含有 encoder 参数，当前加载顺序仍然可能先依赖外部路径

调用链：

- `VLA.from_pretrained`
- 先读 `config.json`
- 再 `cls(config)`，其中 `WANPolicyHead.__init__` 会立即去读外部 encoder 路径
- 最后才 `load_state_dict(checkpoint)`

结果：

- 如果 checkpoint 被挪到新机器、原始 encoder 路径不存在，加载会在读 checkpoint 之前先失败
- 这也是“保存结果不自包含”的根因之一

## 3. 验证方法

### 3.1 这轮已落地的最小验证手段

验证阶段曾临时加入一个纯 debug 计数器，用来确认重复加载来源。

说明：

- 该计数器在修复确认后已从运行时代码移除
- 当前仓库保留的是修复后的稳定逻辑，不再保留 `DREAMZERO_RELOAD_DEBUG` 开关
- 下述内容保留为历史验证记录

### 3.2 判断标准

验证阶段采用的判断标准是：

- 同一个进程里，同一份资源被再次真实读取，就按重复加载处理
- 不同 shard 文件分别读取，不算重复加载
- 多 rank 各自读取自己的副本，不算“单进程重复加载”

### 3.3 建议的前后对比方式

当前推荐的验证方式：

- 直接跑真实 `from_pretrained` / 训练创建模型路径
- 观察是否打印 `Skipping external component loading`
- 检查 checkpoint 目录是否已包含本地 sidecar
- 在需要时用临时脚本统计 checkpoint index 中是否已自带 `action_head.text_encoder / image_encoder / vae / model`

### 3.4 预期看到的“只是日志重复”

- 不同 shard 文件分别出现多次，但 key 不同
- 不同 rank 各自出现一次同路径加载

这两类不应直接算成 bug，需要按单进程单 key 计数判断。

## 4. 最小改动方案

### 4.1 第一阶段：先做临时 debug 验证

目标：

- 不改训练逻辑
- 不改分布式
- 不改 forward
- 先把重复加载定位准确

状态：

- 已完成，且 debug 观测层已在修复确认后移除

### 4.2 第二阶段：修复“保存时缺少 encoder 权重”

最小方案：

1. 保存 checkpoint 时，把实际使用到的 encoder 权重文件一并拷贝到 checkpoint 目录
2. 加载 checkpoint 时，优先查找 checkpoint 内携带的 encoder 文件
3. 只有 checkpoint 内没有时，才回退到原始外部路径

说明：

- 这一步不改 checkpoint 结构里的核心模型张量
- 只补齐 checkpoint 的外部依赖

### 4.3 第三阶段：修复 full checkpoint 的重复加载

最小方案：

1. 区分“full checkpoint 加载”与“LoRA-only checkpoint 加载”
2. 对 full checkpoint：
   - 在实例化 `WANPolicyHead` 前禁用 text/image/VAE/DiT 的 eager component loading
   - 让完整 checkpoint state_dict 成为唯一权重来源
3. 对 LoRA-only checkpoint：
   - 保持现在的 base component loading
   - 只叠加 LoRA 权重

说明：

- 这一步只改初始化顺序，不改训练/推理计算逻辑
- 也是减少真实重复加载的主修复点

### 4.4 第四阶段：低优先级收尾

可选优化：

- 复用 tokenizer 实例，避免 `DreamTransform` 和 `DefaultDataCollator` 各自再构造一次

这项收益比 encoder / DiT 小，放在后面。

## 5. 当前结论

- 最高优先级问题不是“日志看起来多”，而是 full checkpoint 路径里 text/image/VAE 先按 config 加载一次、再按 checkpoint 叠一次。
- `save_lora_only=true` 下 checkpoint 不自带 encoder 权重是当前“保存后缺少所需 encoder 权重”的直接原因。
- 本轮最安全的推进顺序是：
  1. 用 debug 计数器确认次数
  2. 先让 checkpoint 自带 encoder 依赖
  3. 再对 full checkpoint 加载路径做一次性去重

## 6. 实测结果

### 6.1 测试对象

- checkpoint: `/data/checkpoints/dreamzero/DreamZero-DROID`
- 外部 base component 路径:
  - `/data/checkpoints/dreamzero/Wan2.1-I2V-14B-480P/models_t5_umt5-xxl-enc-bf16.pth`
  - `/data/checkpoints/dreamzero/Wan2.1-I2V-14B-480P/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth`
  - `/data/checkpoints/dreamzero/Wan2.1-I2V-14B-480P/Wan2.1_VAE.pth`

### 6.2 checkpoint 内容检查

对 `model.safetensors.index.json` 的检查结果：

- `action_head.text_encoder`: 242 个参数，分布在 3 个 checkpoint shard
- `action_head.image_encoder`: 393 个参数，分布在 1 个 checkpoint shard
- `action_head.vae`: 194 个参数，分布在 1 个 checkpoint shard
- `action_head.model`: 1317 个参数，分布在 8 个 checkpoint shard

结论：

- 这个 full checkpoint 本身就已经包含 text/image/VAE/DiT 参数
- 因此如果初始化阶段又先去读外部 Wan2.1 路径，那就是“真实重复加载”，不是单纯日志问题

### 6.3 真实加载测试

受控复现方式：

1. 从 checkpoint `config.json` 实例化 `VLA(config)`
2. 让当前代码按现状先走 `WANPolicyHead.__init__` 的外部 component loading
3. 再读取 `model-00003-of-00010.safetensors`
4. 把该 shard 的 state_dict 应用到已经实例化的模型上

实测日志要点：

- text encoder 外部加载: 5424.83 ms
- image encoder 外部加载: 2558.94 ms
- VAE 外部加载: 98.42 ms
- 7 个 Wan2.1 DiT shard 在初始化阶段都被加载
- `VLA(config)` 总初始化耗时: 209.44 s
- 初始化后进程峰值 `ru_maxrss`: 156409216 KB
- `model-00003-of-00010.safetensors` 内含：
  - `action_head.text_encoder`: 37 个 tensor
  - `action_head.image_encoder`: 393 个 tensor
  - `action_head.vae`: 194 个 tensor
  - `action_head.model`: 86 个 tensor
- 把这个 checkpoint shard 应用到模型仅耗时: 0.13 s

结论：

- 这次实测已经确认：当前 full checkpoint 路径下，模型先读取外部 Wan2.1 encoder / DiT 权重，再被 DreamZero-DROID checkpoint shard 覆盖
- 这属于真实重复加载

### 6.4 保存自包含性检查

对 `/data/checkpoints/dreamzero/DreamZero-DROID` 根目录的检查结果：

- 未发现 `models_t5_umt5-xxl-enc-bf16.pth`
- 未发现 `models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth`
- 未发现 `Wan2.1_VAE.pth`

结论：

- 该 checkpoint 不是自包含的
- 当前加载仍依赖外部 Wan2.1 encoder 文件

## 7. 建议实施方案

### 7.1 根因拆解

当前其实是两个独立问题叠在一起：

#### 问题 A：full checkpoint 加载顺序错误

现状：

- 先按 config 里的外部路径加载 text/image/VAE/DiT
- 再把 full checkpoint state_dict 叠加进来

问题本质：

- full checkpoint 已经自带这些参数
- 外部 component loading 在 full checkpoint 路径上是多余的

#### 问题 B：LoRA-only checkpoint 不自包含

现状：

- `save_lora_only=true` 只保存可训练参数
- encoder 权重不进 checkpoint
- 加载时必须依赖外部 Wan 路径

问题本质：

- checkpoint 能恢复 LoRA，但不能独立恢复它依赖的 base encoder 文件

### 7.2 最小且安全的修复策略

#### 方案 1：把 `skip_component_loading` 变成真正的“跳过所有外部 component loading”

落点文件：

- `groot/vla/model/dreamzero/action_head/wan_flow_matching_action_tf.py`

当前行为：

- `skip_component_loading` 只跳过 DiT
- text/image/VAE 仍然会加载

建议行为：

- 当 `skip_component_loading=true` 时，统一跳过：
  - `text_encoder_pretrained_path`
  - `image_encoder_pretrained_path`
  - `vae_pretrained_path`
  - DiT safetensors / HF fallback

为什么这是最小改动：

- 这个 flag 的名字和注释本来就在表达“从 full pretrained model 加载时跳过单组件加载”
- 只是把它补齐为和名字一致的行为

#### 方案 2：在 full checkpoint 加载入口自动打开 `skip_component_loading`

落点文件：

- `groot/vla/model/dreamzero/base_vla.py`
- `groot/vla/experiment/base.py`

建议做法：

- 在 `VLA.from_pretrained` / `VLA.from_pretrained_for_tuning` 里，先检查 checkpoint 是否自带：
  - `action_head.text_encoder`
  - `action_head.image_encoder`
  - `action_head.vae`
  - `action_head.model`
- 若自带，则在实例化前把 `config.action_head_cfg.config.skip_component_loading = true`

训练侧补充：

- 在 `BaseExperiment.create_model` 里，如果 `cfg.pretrained_model_path` 指向的是 self-contained full checkpoint，也在 `instantiate(cfg.model)` 前把同一个 flag 打开

为什么这样安全：

- 只改变初始化顺序
- 不改变最终 `state_dict -> model` 的结果
- full checkpoint 仍然是唯一权重来源

#### 方案 3：保存时把 encoder sidecar 文件复制进 checkpoint

落点文件：

- `groot/vla/experiment/base.py::BaseTrainer.save_model`

建议做法：

- 每次 `save_pretrained(output_dir, ...)` 之后，把当前 action head 正在使用的：
  - text encoder 文件
  - image encoder 文件
  - VAE 文件
- 复制到当前 `output_dir`

建议目录：

- 直接放在 checkpoint 根目录下即可，先不引入新层级
- 保持原 basename，不改文件名

为什么这样最小：

- 不碰核心 checkpoint 张量
- 不需要重做 safetensors 打包
- 只补外部依赖

#### 方案 4：加载时优先使用 checkpoint 内的 local sidecar

落点文件：

- `groot/vla/model/dreamzero/base_vla.py`
- 如有必要，再补 `groot/vla/model/n1_5/sim_policy.py`

建议做法：

- 在 `VLA.from_pretrained` / `VLA.load_lora` 读取 `config.json` 后、实例化前：
  - 先检查 checkpoint 根目录下有没有同 basename 的 sidecar 文件
  - 有就覆盖 config 中的 `*_pretrained_path`
  - 没有再回退到原 config 里的外部绝对路径

为什么这样安全：

- 优先本地 sidecar 只影响加载来源，不改模型行为
- 旧 checkpoint 仍然能继续工作，因为没有 sidecar 时会回退

### 7.3 不建议的方案

- 不建议把 encoder 全部重新打进 `model.safetensors` 的新格式
  原因：会碰 checkpoint 结构，风险高，不符合本轮最小改动要求。

- 不建议在加载时“猜测” checkpoint 类型但不做内容检查
  原因：LoRA-only 与 full checkpoint 都可能叫类似名字，靠路径名判断不稳。

- 不建议直接删除现有外部路径字段
  原因：会破坏老 checkpoint 和现有脚本兼容性。

### 7.4 推荐实施顺序

1. 先修 `skip_component_loading` 的语义，让它真正跳过 text/image/VAE/DiT 全部外部加载。
2. 再在 `from_pretrained` / `create_model` 入口自动识别 self-contained full checkpoint，并打开这个 flag。
3. 然后在 `save_model` 时复制 encoder sidecar，解决“保存后缺文件”。
4. 最后在 `load_lora` / `from_pretrained` 中优先使用 checkpoint 内 sidecar，保证 checkpoint 自包含。

### 7.5 修复后预期验证结果

对于 `/data/checkpoints/dreamzero/DreamZero-DROID`：

- `encoder_weight_load` 应降为 0
- `dit_shard_load` 应降为 0
- 只剩 `checkpoint_shard_load` 指向 DreamZero-DROID 自己的 `model-xxxxx.safetensors`

对于 LoRA-only checkpoint：

- checkpoint 根目录中应能看到 copied sidecar 文件
- 在没有外部 Wan2.1 目录时，仍能通过 checkpoint 本地 sidecar 完成加载

## 8. 已实施修复

### 8.1 代码修复

已完成以下代码改动：

- `groot/vla/model/dreamzero/action_head/wan_flow_matching_action_tf.py`
  - `skip_component_loading=true` 时，现已同时跳过：
    - text encoder 外部加载
    - image encoder 外部加载
    - VAE 外部加载
    - DiT 外部加载
- `groot/vla/utils/checkpoint_sidecar.py`
  - 新增 checkpoint sidecar 工具层
  - 支持：
    - 检测 full checkpoint 是否自带 text/image/VAE/DiT
    - 优先发现 checkpoint 根目录内的 local sidecar
    - 覆盖 nested `text_encoder_cfg / image_encoder_cfg / vae_cfg`
    - 在保存时把 sidecar materialize 到 checkpoint 目录
- `groot/vla/model/dreamzero/base_vla.py`
  - `from_pretrained`
  - `from_pretrained_for_tuning`
  - `load_lora`
  以上入口都会在实例化前先做 checkpoint 识别和 sidecar 覆盖
- `groot/vla/experiment/base.py`
  - `create_model` 在训练入口也会先识别 checkpoint 类型
  - `save_model` 在保存后会把 sidecar 文件 materialize 到输出目录

### 8.2 权重迁移

已将以下外部 sidecar 文件迁入：

- `/data/checkpoints/dreamzero/DreamZero-DROID/models_t5_umt5-xxl-enc-bf16.pth`
- `/data/checkpoints/dreamzero/DreamZero-DROID/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth`
- `/data/checkpoints/dreamzero/DreamZero-DROID/Wan2.1_VAE.pth`

同时，为保持旧路径兼容，原目录保留了 symlink：

- `/data/checkpoints/dreamzero/Wan2.1-I2V-14B-480P/models_t5_umt5-xxl-enc-bf16.pth -> ../DreamZero-DROID/models_t5_umt5-xxl-enc-bf16.pth`
- `/data/checkpoints/dreamzero/Wan2.1-I2V-14B-480P/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth -> ../DreamZero-DROID/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth`
- `/data/checkpoints/dreamzero/Wan2.1-I2V-14B-480P/Wan2.1_VAE.pth -> ../DreamZero-DROID/Wan2.1_VAE.pth`

说明：

- 这次没有迁移整套 `diffusion_pytorch_model-*.safetensors`
- 原因是 `DreamZero-DROID` full checkpoint 已自带 `action_head.model.*`
- 修复后 full checkpoint 不再依赖外部 Wan2.1 DiT 基座

## 9. 修复后实测

### 9.1 checkpoint 识别结果

对 `/data/checkpoints/dreamzero/DreamZero-DROID` 执行 `prepare_action_head_cfg_for_checkpoint(...)` 后：

- 检测到 `self_contained_full = true`
- 检测到 checkpoint 根目录内已有 local sidecars
- nested config 路径已被覆盖到：
  - `/data/checkpoints/dreamzero/DreamZero-DROID/models_t5_umt5-xxl-enc-bf16.pth`
  - `/data/checkpoints/dreamzero/DreamZero-DROID/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth`
  - `/data/checkpoints/dreamzero/DreamZero-DROID/Wan2.1_VAE.pth`
- `skip_component_loading = true`

### 9.2 真实 `from_pretrained` 测试结果

测试命令：

```bash
PYTHONPATH=/data/dreamzero /data/dreamzero/.venv/bin/python -u - <<'PY'
from groot.vla.model.dreamzero.base_vla import VLA
model = VLA.from_pretrained('/data/checkpoints/dreamzero/DreamZero-DROID')
print(type(model).__name__)
PY
```

结果：

- 实际加载的只有：
  - `/data/checkpoints/dreamzero/DreamZero-DROID/model-00001-of-00010.safetensors`
  - ...
  - `/data/checkpoints/dreamzero/DreamZero-DROID/model-00010-of-00010.safetensors`
- 不再触发外部 encoder / DiT eager loading
- 也不再依赖外部 Wan2.1 DiT 基座

关键日志：

- `Skipping external component loading (expecting checkpoint state_dict to provide full weights)`
- `Successfully loaded pretrained weights`

结论：

- full checkpoint 的重复外部加载已经修复

### 9.3 保存侧 materialize 测试结果

对 `materialize_component_sidecars(...)` 做了轻量验证：

- 能在目标输出目录生成 3 个本地 sidecar
- 当前实现优先使用 hardlink
- 测试中 3 个 sidecar 的 `nlink=2`

结论：

- 保存侧现在可以把 encoder sidecar 一起落到 checkpoint 目录
- 不需要每次真的复制 16G+ 数据
