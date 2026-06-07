# client.py
import zerorpc

# ***************************************************************
# !! 重要 !!
# 将 "NUC_IP_ADDRESS" 替换成你上一步查到的 NUC 的实际 IP 地址
# ***************************************************************
NUC_IP = "172.16.0.4" # <--- 在这里填入你 NUC 的 IP

# 创建一个 ZeroRPC 客户端
client = zerorpc.Client()

# 连接到 NUC 上的服务器
# 地址格式为 "tcp://<IP地址>:<端口号>"
try:
    client.connect(f"tcp://{NUC_IP}:4242")
    print(f"Successfully connected to the robot server at {NUC_IP}:4242")

    # 现在你可以像调用本地对象一样调用 FrankaRobot 的方法了
    # 注意: 你需要知道 FrankaRobot 类有哪些可以被远程调用的方法。
    # 这里我们假设它有一个名为 `get_state` 的方法来获取机器人状态。
    # 请根据 DROID 库中 FrankaRobot 类的实际方法进行替换。
    
    # print("Attempting to get robot state...")
    # robot_state = client.launch_robot() # 示例：调用 get_state() 方法
    # print("Received robot state:")
    # print(robot_state)

    # 另一个例子：假设有一个控制夹爪的方法 set_gripper(width, speed)
    # print("\nAttempting to open the gripper...")
    # response = client.set_gripper(0.08, 0.1) # 示例参数
    # print(f"Gripper command response: {response}")

except zerorpc.exceptions.LostRemote as e:
    print(f"Connection lost or could not connect to the server: {e}")
except Exception as e:
    print(f"An unexpected error occurred: {e}")
finally:
    # 关闭客户端连接
    client.close()
    print("\nClient connection closed.")