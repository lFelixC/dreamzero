#!/usr/bin/env python3
"""
远程DROID控制脚本
从laptop远程控制NUC上的DROID机器人
"""

import time
import numpy as np
from droid.misc.server_interface import ServerInterface


def test_connection(robot):
    """测试与DROID机器人的连接"""
    print("🔗 测试机器人连接...")

    try:
        # 获取机器人状态
        state = robot.get_robot_state()
        print(f"✅ 连接成功! 机器人状态键: {list(state.keys())}")

        # 获取当前位姿
        ee_pose = robot.get_ee_pose()
        print(f"📍 当前末端位姿位置: {ee_pose[0][:3]}")
        print(f"🔄 当前末端位姿姿态: {ee_pose[1]}")

        # 获取关节状态
        joint_pos = robot.get_joint_positions()
        print(f"🦾 关节数量: {len(joint_pos)}")
        print(f"📊 当前关节位置: {np.round(joint_pos, 3)}")

        # 获取夹爪状态
        gripper_state = robot.get_gripper_state()
        print(f"🤏 夹爪状态: {gripper_state}")

        return True

    except Exception as e:
        print(f"❌ 连接失败: {e}")
        return False


def test_gripper(robot):
    """测试夹爪功能"""
    print("\n🤏 测试夹爪功能...")

    try:
        print("📖 打开夹爪...")
        robot.update_gripper(1.0, velocity=False, blocking=True)  # 完全打开
        time.sleep(1)

        gripper_state = robot.get_gripper_state()
        print(f"   夹爪宽度: {gripper_state.width:.3f}")

        print("📚 关闭夹爪...")
        robot.update_gripper(0.0, velocity=False, blocking=True)  # 完全关闭
        time.sleep(1)

        gripper_state = robot.get_gripper_state()
        print(f"   夹爪宽度: {gripper_state.width:.3f}")

        print("📖 再次打开夹爪...")
        robot.update_gripper(1.0, velocity=False, blocking=True)

        print("✅ 夹爪测试完成!")
        return True

    except Exception as e:
        print(f"❌ 夹爪测试失败: {e}")
        return False


def test_robot_movement(robot):
    """测试机器人运动"""
    print("\n🚀 测试机器人运动...")

    try:
        # 获取当前位置
        current_pose = robot.get_ee_pose()
        current_pos = np.array(current_pose[0])
        current_quat = np.array(current_pose[1])

        print(f"📍 当前位置: {np.round(current_pos, 3)}")

        # 小幅度移动测试 (向前移动5cm)
        delta_pos = np.array([0.05, 0.0, 0.0])  # 向前5cm
        new_pos = current_pos + delta_pos

        print(f"➡️ 目标位置: {np.round(new_pos, 3)}")

        # 创建位姿命令 [位置(x,y,z), 姿态(quat), 夹爪]
        action = np.concatenate([new_pos, current_quat, [1.0]])  # 保持夹爪打开

        # 发送命令
        robot.update_command(action, action_space="cartesian_position", blocking=True)

        time.sleep(2)

        # 验证移动结果
        new_pose = robot.get_ee_pose()
        actual_pos = np.array(new_pose[0])

        print(f"🎯 实际到达位置: {np.round(actual_pos, 3)}")
        print(f"📏 位置误差: {np.linalg.norm(actual_pos - new_pos):.4f} m")

        # 移动回原位
        print("🔙 移动回原位...")
        home_action = np.concatenate([current_pos, current_quat, [1.0]])
        robot.update_command(home_action, action_space="cartesian_position", blocking=True)

        print("✅ 运动测试完成!")
        return True

    except Exception as e:
        print(f"❌ 运动测试失败: {e}")
        return False


def test_joint_movement(robot):
    """测试关节空间运动"""
    print("\n🦾 测试关节空间运动...")

    try:
        # 获取当前关节位置
        current_joints = robot.get_joint_positions()
        print(f"📊 当前关节位置: {np.round(current_joints, 3)}")

        # 小幅度关节运动 (只移动第7个关节)
        delta_joints = np.zeros(7)
        delta_joints[-1] = 0.1  # 第7关节移动0.1弧度

        new_joints = current_joints + delta_joints

        print(f"🎯 目标关节位置: {np.round(new_joints, 3)}")

        # 创建关节命令 [7个关节角度, 夹爪]
        action = np.concatenate([new_joints, [1.0]])  # 保持夹爪打开

        # 发送命令
        robot.update_command(action, action_space="joint_position", blocking=True)

        time.sleep(2)

        # 验证移动结果
        actual_joints = robot.get_joint_positions()
        print(f"🎯 实际关节位置: {np.round(actual_joints, 3)}")
        print(f"📏 关节误差: {np.linalg.norm(actual_joints - new_joints):.4f} rad")

        # 移动回原位
        print("🔙 移动回原位...")
        home_action = np.concatenate([current_joints, [1.0]])
        robot.update_command(home_action, action_space="joint_position", blocking=True)

        print("✅ 关节运动测试完成!")
        return True

    except Exception as e:
        print(f"❌ 关节运动测试失败: {e}")
        return False


def main():
    """主函数"""
    print("🤖 DROID远程控制测试脚本")
    print("=" * 50)

    # NUC IP配置
    nuc_ip = "172.16.0.4"

    try:
        # 连接到NUC上的DROID
        print(f"📡 正在连接到NUC: {nuc_ip}")
        robot = ServerInterface(ip_address=nuc_ip)

        print("✅ 成功连接到DROID机器人!")

        # 运行测试套件
        tests = [
            ("连接测试", test_connection),
            ("夹爪测试", test_gripper),
            ("机器人运动测试", test_robot_movement),
            ("关节运动测试", test_joint_movement),
        ]

        results = []
        for test_name, test_func in tests:
            print(f"\n{'='*20} {test_name} {'='*20}")
            try:
                result = test_func(robot)
                results.append((test_name, result))
            except Exception as e:
                print(f"❌ {test_name}出现异常: {e}")
                results.append((test_name, False))

        # 测试结果汇总
        print(f"\n{'='*50}")
        print("📊 测试结果汇总:")

        passed = 0
        for test_name, result in results:
            status = "✅ 通过" if result else "❌ 失败"
            print(f"   {test_name}: {status}")
            if result:
                passed += 1

        print(f"\n🎯 总体结果: {passed}/{len(results)} 测试通过")

        if passed == len(results):
            print("🎉 所有测试通过! DROID机器人工作正常。")
        else:
            print("⚠️ 部分测试失败，请检查机器人状态。")

    except Exception as e:
        print(f"❌ 无法连接到DROID机器人: {e}")
        print("💡 请检查:")
        print("   1. NUC是否开机并连接到网络")
        print("   2. DROID服务是否正在运行")
        print("   3. 网络连接是否正常")
        print(f"   4. IP地址 {nuc_ip} 是否正确")


if __name__ == "__main__":
    main()