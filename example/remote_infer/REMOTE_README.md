# DreamZero 远程推理说明

这份文档说明如何把远端 `DreamZero policy server` 接到本地 `main_dreamzero.py`，以及如何在不暴露你自己的 `ssh config`、跳板机细节、容器细节的前提下，把访问权限最小化地给外部协作者或评测方。

适用对象：

- 你自己在本地跑 [main_dreamzero.py](/data/dreamzero/example/remote_infer/main_dreamzero.py)
- 需要让外部协作者通过跳板机访问远端 websocket policy server

当前约定：

- 远端服务跑在容器 `docker-nc69`
- 远端服务监听 `0.0.0.0:5001`
- 服务接口是 `roboarena policy server interface`
- 本地客户端是 [main_dreamzero.py](/data/dreamzero/example/remote_infer/main_dreamzero.py)

## 1. 总体链路

目标链路是：

```text
本地 main_dreamzero.py
    -> 本地 127.0.0.1:5001
    -> SSH 转发
    -> docker-nc69:5001
    -> DreamZero websocket server
```

也就是说，`main_dreamzero.py` 不直接感知远端机器和跳板机细节，只需要访问本地端口。

## 2. 本地自用方式

如果你自己已经能从当前机器 SSH 到 `docker-nc69`，最简单的方式就是本地端口转发：

```bash
ssh -N -L 5001:127.0.0.1:5001 docker-nc69
```

转发建立后，你的本地程序直接连：

- `remote_host = 127.0.0.1`
- `remote_port = 5001`

这也是 [main_dreamzero.py](/data/dreamzero/example/remote_infer/main_dreamzero.py) 最适合的配置方式。

## 3. main_dreamzero.py 的推荐参数

在 [main_dreamzero.py](/data/dreamzero/example/remote_infer/main_dreamzero.py) 中，远端 server 参数建议配置为：

```python
remote_host = "127.0.0.1"
remote_port = 5001
```

也就是让脚本永远访问“你本地的转发端口”，而不是直接写远端地址。

## 4. 给外部协作者的安全接入方式

如果对方也需要访问这个 policy server，但你不想把自己的：

- `~/.ssh/config`
- 跳板机跳转链路
- `docker-nc69` 这个内部目标

都暴露给对方，推荐用“两层转发”。

### 4.1 你自己先在跳板机上建立内部转发

先在跳板机上执行：

```bash
ssh -N -L 127.0.0.1:15001:127.0.0.1:5001 docker-nc69
```

这条命令的含义是：

- 跳板机本机的 `127.0.0.1:15001`
- 转发到 `docker-nc69` 内部的 `127.0.0.1:5001`

这样以后，跳板机只暴露一个“本机回环端口” `15001`，而不会把 `docker-nc69` 直接暴露给对方。

### 4.2 给对方的公钥只开最小权限

如果你能控制跳板机用户的 `~/.ssh/authorized_keys`，建议给对方的公钥加限制。

示例：

```text
restrict,port-forwarding,permitopen="127.0.0.1:15001",no-pty,no-agent-forwarding,no-X11-forwarding ssh-ed25519 AAAA...
```

这表示：

- 只允许端口转发
- 只允许转发到 `127.0.0.1:15001`
- 不给 shell
- 不给 agent forwarding
- 不给 X11

这样对方即使有 SSH 权限，也拿不到你完整的跳板机访问能力。

### 4.3 对方只需要执行一条简单命令

对方机器上执行：

```bash
ssh -N -L 5001:127.0.0.1:15001 <user>@<jump_host>
```

然后对方本地访问：

```text
127.0.0.1:5001
```

就等价于访问你容器里的 DreamZero server。

这套方式的好处是：

- 你不用把自己的 `ssh config` 给对方
- 你不用把 `docker-nc69` 这个内部主机名给对方
- 对方只知道跳板机地址和一条固定命令

## 5. 给 RoboArena / 外部评测方时要注意什么

上面的 SSH 转发方式适合：

- 你自己本地访问
- 指定协作者手动接入
- 评测方愿意自己建立 SSH 隧道

但如果对方要求的是“直接从他们的服务器访问你的 policy server”，那 SSH 本地转发本身还不够，因为：

- `127.0.0.1:5001` 只对建立隧道的那台机器本地可见
- 它不是公网地址

换句话说：

- 如果 RoboArena 愿意持有一把受限公钥并自己 SSH 到跳板机建隧道，这个方案可用
- 如果 RoboArena 只接受一个直接可访问的公网 `host:port`，那还需要额外的公网入口或 relay

## 6. 连通性自检

### 6.1 在容器里确认服务已监听

```bash
nc -vz 127.0.0.1 5001
```

### 6.2 在建立本地转发后，本机检查

```bash
nc -vz 127.0.0.1 5001
```

或直接跑本地客户端。

### 6.3 main_dreamzero.py 使用前检查

确认：

- SSH 转发窗口处于保持连接状态
- 本地 `127.0.0.1:5001` 可访问
- `remote_host` 和 `remote_port` 配的是 `127.0.0.1:5001`

## 7. 常见问题

### Q1. 为什么不直接把 docker-nc69 暴露给对方？

因为这样会暴露你的内部网络拓扑，而且对方可能拿到比所需更多的访问面。推荐始终由你自己在跳板机上做一层内部转发。

### Q2. 为什么不直接把 ssh config 发给对方？

因为 `ssh config` 往往带有：

- 跳板关系
- 用户名
- Host 别名
- 内网主机名
- 认证方式

这些都不适合直接外发。对方最好只拿到“最终可执行的一条 SSH 命令”。

### Q3. 如果对方不能 SSH，只能要求一个公网地址怎么办？

那就不能只靠 `ssh -L` 了，需要额外的公网入口，比如：

- 一台可公网入站的 relay 机器
- 反向隧道到公网机器
- 网络管理员做端口映射
- FRP / cloudflared / ngrok 一类内网穿透

### Q4. 现在这条文档最推荐的方式是什么？

如果你只是给少量协作者或评测方使用，我最推荐：

1. 你自己在跳板机上执行：

```bash
ssh -N -L 127.0.0.1:15001:127.0.0.1:5001 docker-nc69
```

2. 给对方一把受限 SSH key，只允许：

```text
permitopen="127.0.0.1:15001"
```

3. 对方执行：

```bash
ssh -N -L 5001:127.0.0.1:15001 <user>@<jump_host>
```

4. 对方本地把客户端指向：

```text
127.0.0.1:5001
```

这就是“最小暴露、最小权限、最少信息泄露”的做法。
