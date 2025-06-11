# Ignore lint errors because this file is mostly copied from ACT (https://github.com/tonyzhaozh/act).
# ruff: noqa
import collections
import time
from typing import Optional, List
import dm_env
from interbotix_xs_modules.arm import InterbotixManipulatorXS
from interbotix_xs_msgs.msg import JointSingleCommand
import numpy as np

from examples.aloha_real import constants
from examples.aloha_real import robot_utils

# This is the reset position that is used by the standard Aloha runtime.
DEFAULT_RESET_POSITION = [0, -0.96, 1.16, 0, -0.3, 0]


class RealEnv:
    """
    真实机器人双臂操作环境
    Action space:      [左臂关节位置(6),            # 绝对关节位置
                        左夹爪归一化位置(1),         # 归一化夹爪位置 (0: 闭合, 1: 打开)
                        右臂关节位置(6),            # 绝对关节位置
                        右夹爪归一化位置(1),]        # 归一化夹爪位置 (0: 闭合, 1: 打开)

    Observation space: {"qpos": 拼接[左臂关节位置(6),      # 绝对关节位置
                                    左夹爪归一化位置(1),   # 归一化夹爪位置 (0: 闭合, 1: 打开)
                                    右臂关节位置(6),      # 绝对关节位置
                                    右夹爪归一化位置(1)]  # 归一化夹爪位置 (0: 闭合, 1: 打开)
                        "qvel": 拼接[左臂关节速度(6),      # 绝对关节速度 (弧度)
                                    左夹爪归一化速度(1),   # 归一化夹爪速度 (正: 打开中, 负: 闭合中)
                                    右臂关节速度(6),      # 绝对关节速度 (弧度)
                                    右夹爪归一化速度(1)]  # 归一化夹爪速度 (正: 打开中, 负: 闭合中)
                        "images": {"顶部摄像头": (480x640x3),    # 高度, 宽度, 通道, uint8格式
                                   "底部摄像头": (480x640x3),
                                   "左腕部摄像头": (480x640x3),
                                   "右腕部摄像头": (480x640x3)}
    """

    def __init__(self, init_node, *, reset_position: Optional[List[float]] = None, setup_robots: bool = True):
        """
        初始化真实机器人环境
        :param init_node: 是否初始化ROS节点
        :param reset_position: 自定义重置位置（6维关节角度列表）
        :param setup_robots: 是否进行机器人初始化设置
        """
        # reset_position = START_ARM_POSE[:6]
        self._reset_position = reset_position[:6] if reset_position else DEFAULT_RESET_POSITION

        self.puppet_bot_left = InterbotixManipulatorXS(
            robot_model="vx300s",
            group_name="arm",
            gripper_name="gripper",
            robot_name="puppet_left",
            init_node=init_node,
        )
        self.puppet_bot_right = InterbotixManipulatorXS(
            robot_model="vx300s", group_name="arm", gripper_name="gripper", robot_name="puppet_right", init_node=False
        )
        if setup_robots:
            self.setup_robots()

        self.recorder_left = robot_utils.Recorder("left", init_node=False)
        self.recorder_right = robot_utils.Recorder("right", init_node=False)
        self.image_recorder = robot_utils.ImageRecorder(init_node=False)
        self.gripper_command = JointSingleCommand(name="gripper")

    def setup_robots(self):
        robot_utils.setup_puppet_bot(self.puppet_bot_left)
        robot_utils.setup_puppet_bot(self.puppet_bot_right)

    def get_qpos(self):
        """获取归一化的关节位置观测（包含双臂和夹爪）"""
        left_qpos_raw = self.recorder_left.qpos
        right_qpos_raw = self.recorder_right.qpos
        left_arm_qpos = left_qpos_raw[:6]
        right_arm_qpos = right_qpos_raw[:6]
        left_gripper_qpos = [
            constants.PUPPET_GRIPPER_POSITION_NORMALIZE_FN(left_qpos_raw[7])
        ]  # this is position not joint
        right_gripper_qpos = [
            constants.PUPPET_GRIPPER_POSITION_NORMALIZE_FN(right_qpos_raw[7])
        ]  # this is position not joint
        return np.concatenate([left_arm_qpos, left_gripper_qpos, right_arm_qpos, right_gripper_qpos])

    def get_qvel(self):
        """获取归一化的关节速度观测（包含双臂和夹爪）"""
        left_qvel_raw = self.recorder_left.qvel
        right_qvel_raw = self.recorder_right.qvel
        left_arm_qvel = left_qvel_raw[:6]
        right_arm_qvel = right_qvel_raw[:6]
        left_gripper_qvel = [constants.PUPPET_GRIPPER_VELOCITY_NORMALIZE_FN(left_qvel_raw[7])]
        right_gripper_qvel = [constants.PUPPET_GRIPPER_VELOCITY_NORMALIZE_FN(right_qvel_raw[7])]
        return np.concatenate([left_arm_qvel, left_gripper_qvel, right_arm_qvel, right_gripper_qvel])

    def get_effort(self):
        """获取关节力矩观测（包含双臂）"""
        left_effort_raw = self.recorder_left.effort
        right_effort_raw = self.recorder_right.effort
        left_robot_effort = left_effort_raw[:7]
        right_robot_effort = right_effort_raw[:7]
        return np.concatenate([left_robot_effort, right_robot_effort])

    def get_images(self):
        return self.image_recorder.get_images()

    def set_gripper_pose(self, left_gripper_desired_pos_normalized, right_gripper_desired_pos_normalized):
        left_gripper_desired_joint = constants.PUPPET_GRIPPER_JOINT_UNNORMALIZE_FN(left_gripper_desired_pos_normalized)
        self.gripper_command.cmd = left_gripper_desired_joint
        self.puppet_bot_left.gripper.core.pub_single.publish(self.gripper_command)

        right_gripper_desired_joint = constants.PUPPET_GRIPPER_JOINT_UNNORMALIZE_FN(
            right_gripper_desired_pos_normalized
        )
        self.gripper_command.cmd = right_gripper_desired_joint
        self.puppet_bot_right.gripper.core.pub_single.publish(self.gripper_command)

    def _reset_joints(self):
        """将双臂关节重置到初始位置"""
        robot_utils.move_arms(
            [self.puppet_bot_left, self.puppet_bot_right], [self._reset_position, self._reset_position], move_time=1
        )

    def _reset_gripper(self):
        """重置夹爪状态：先完全打开再闭合，确保夹爪状态正确"""
        robot_utils.move_grippers(
            [self.puppet_bot_left, self.puppet_bot_right], [constants.PUPPET_GRIPPER_JOINT_OPEN] * 2, move_time=0.5
        )
        robot_utils.move_grippers(
            [self.puppet_bot_left, self.puppet_bot_right], [constants.PUPPET_GRIPPER_JOINT_CLOSE] * 2, move_time=1
        )

    def get_observation(self):
        obs = collections.OrderedDict()
        obs["qpos"] = self.get_qpos()
        obs["qvel"] = self.get_qvel()
        obs["effort"] = self.get_effort()
        obs["images"] = self.get_images()
        return obs

    def get_reward(self):
        return 0

    def reset(self, *, fake=False):
        """
        重置环境到初始状态
        :param fake: 是否虚拟重置（跳过实际硬件操作）
        :return: 包含初始观测的TimeStep对象
        """
        if not fake:
            # Reboot puppet robot gripper motors
            self.puppet_bot_left.dxl.robot_reboot_motors("single", "gripper", True)
            self.puppet_bot_right.dxl.robot_reboot_motors("single", "gripper", True)
            self._reset_joints()
            self._reset_gripper()
        return dm_env.TimeStep(
            step_type=dm_env.StepType.FIRST, reward=self.get_reward(), discount=None, observation=self.get_observation()
        )

    def step(self, action):
        """
        执行一个时间步长的动作
        :param action: 14维动作向量 [左臂(6)+左夹爪(1)+右臂(6)+右夹爪(1)]
        :return: 包含新观测的TimeStep对象
        """
        state_len = int(len(action) / 2)
        left_action = action[:state_len]
        right_action = action[state_len:]
        self.puppet_bot_left.arm.set_joint_positions(left_action[:6], blocking=False)
        self.puppet_bot_right.arm.set_joint_positions(right_action[:6], blocking=False)
        self.set_gripper_pose(left_action[-1], right_action[-1])
        time.sleep(constants.DT)
        return dm_env.TimeStep(
            step_type=dm_env.StepType.MID, reward=self.get_reward(), discount=None, observation=self.get_observation()
        )


def get_action(master_bot_left, master_bot_right):
    """
    从主控机械臂获取动作指令
    :param master_bot_left: 左主控机械臂实例
    :param master_bot_right: 右主控机械臂实例 
    :return: 14维动作向量（与RealEnv的action space结构匹配）
    """
    action = np.zeros(14)  # 6 joint + 1 gripper, for two arms
    # Arm actions
    action[:6] = master_bot_left.dxl.joint_states.position[:6]
    action[7 : 7 + 6] = master_bot_right.dxl.joint_states.position[:6]
    # Gripper actions
    action[6] = constants.MASTER_GRIPPER_JOINT_NORMALIZE_FN(master_bot_left.dxl.joint_states.position[6])
    action[7 + 6] = constants.MASTER_GRIPPER_JOINT_NORMALIZE_FN(master_bot_right.dxl.joint_states.position[6])

    return action


def make_real_env(init_node, *, reset_position: Optional[List[float]] = None, setup_robots: bool = True) -> RealEnv:
    """创建真实机器人环境实例的工厂函数"""
    return RealEnv(init_node, reset_position=reset_position, setup_robots=setup_robots)
