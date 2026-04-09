# H200 远程推理部署说明

## 目标

在远程 H200 机器上启动 policy server，本地采集图片后，将图片发送到远程 server，并拿到返回的 action。

当前目标拆成两步：

1. 先确保本地可以打通远程 server 的端口；
2. 端口打通后，再从本地上传图片，请求 action。

---

## 当前结论

目前整条链路已经打通，可以直接在 `main dreamzero` 里把 server 地址配置为：

- `ip = 127.0.0.1`
- `port = 5001`

即可访问远程 policy server。

---

## 服务信息

远程服务运行在：

- 远程容器：`docker-nc69`
- 服务类型：`websockets server`
- 接口类型：`roboarena policy server interface`
- 监听地址：`0.0.0.0:5001`

服务日志已经确认：

- `server listening on 0.0.0.0:5001`

这说明服务进程已经成功启动，并且正在容器内监听 5001 端口。

---

## 已验证的链路

### 1. 远程 docker-nc69 内部访问 5001 成功

在远程容器内已经验证：

```bash
nc -vz 127.0.0.1 5001