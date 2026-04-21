# AGENTS.md

## 项目当前目标
当前任务是给 DreamZero 做“瘦身”，不是重写成新框架。

## 明确不做
- 不修改底层训练逻辑
- 不修改分布式训练逻辑
- 不修改 torch compile / 性能优化路径
- 不修改 checkpoint 格式
- 不修改核心 model forward 行为
- 忽略/data/dreamzero/example和/data/dreamzero/video_eval还有/data/dreamzero/third_party
- /data/dreamzero/scripts/train/droid_training_full_finetune_wan22_local.sh这是我之前写的脚本

## 当前允许修改的范围
- 训练 / 推理 / eval 的入口层
- wrapper / launcher / scripts
- 配置整理
- 路径硬编码清理
- 死代码识别与删除
- 文档与目录结构整理

## 工程要求
- 改动优先小而可回滚
- 先保留兼容层，再考虑删除旧入口
- 每次改动都要说明：
  - 改了什么
  - 为什么安全
  - 如何验证
- 不做为了“未来通用性”的额外抽象

## 完成标准
- 主链路更清晰
- 入口数量减少
- 不改变已有训练/推理核心行为
- 有最小 smoke test 或运行命令