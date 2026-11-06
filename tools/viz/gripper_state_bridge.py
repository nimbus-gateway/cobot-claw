#!/usr/bin/env python3
"""
Publish the Robotiq emulator's finger position as ROS joint states, so RViz
can draw the gripper opening and closing.

Polls the emulator in the URSim container (GET POS on port 63352) and
publishes /gripper/joint_states for the 2F-85 model in gripper_on_ur.urdf.xacro.
Every moving joint is published explicitly (the model's mimic joints are
expanded here) so robot_state_publisher doesn't need mimic support.

    EMU_HOST  emulator address (default 192.168.56.101)
    EMU_PORT  emulator port    (default 63352)
"""
from __future__ import annotations

import os
import socket

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState

EMU_HOST = os.getenv("EMU_HOST", "192.168.56.101")
EMU_PORT = int(os.getenv("EMU_PORT", "63352"))
RATE_HZ = 20.0
CLOSED_RAD = 0.7929          # robotiq_2f_85 gripper_closed_position

# (joint, multiplier on the left knuckle angle), from the model's mimic tags
JOINTS = [
    ("robotiq_85_left_knuckle_joint", 1.0),
    ("robotiq_85_right_knuckle_joint", -1.0),
    ("robotiq_85_left_inner_knuckle_joint", 1.0),
    ("robotiq_85_right_inner_knuckle_joint", -1.0),
    ("robotiq_85_left_finger_tip_joint", -1.0),
    ("robotiq_85_right_finger_tip_joint", 1.0),
]


class GripperStateBridge(Node):
    def __init__(self) -> None:
        super().__init__("gripper_state_bridge")
        self.pub = self.create_publisher(JointState, "joint_states", 10)
        self.sock: socket.socket | None = None
        self.pos = 0
        self.warned = False
        self.create_timer(1.0 / RATE_HZ, self.tick)

    def _read_pos(self) -> int | None:
        try:
            if self.sock is None:
                self.sock = socket.create_connection((EMU_HOST, EMU_PORT), timeout=0.5)
            self.sock.sendall(b"GET POS\n")
            reply = self.sock.recv(64).decode(errors="replace").split()
            if len(reply) >= 2 and reply[0] == "POS":
                self.warned = False
                return int(reply[1])
        except (OSError, ValueError):
            pass
        if self.sock is not None:
            self.sock.close()
            self.sock = None
        if not self.warned:
            self.get_logger().warn(f"gripper emulator not reachable at {EMU_HOST}:{EMU_PORT}; retrying")
            self.warned = True
        return None

    def tick(self) -> None:
        pos = self._read_pos()
        if pos is not None:
            self.pos = max(0, min(255, pos))
        angle = CLOSED_RAD * self.pos / 255
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = [name for name, _ in JOINTS]
        msg.position = [angle * k for _, k in JOINTS]
        self.pub.publish(msg)


def main() -> None:
    rclpy.init()
    node = GripperStateBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
