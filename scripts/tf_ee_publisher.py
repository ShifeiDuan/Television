#!/usr/bin/env python3
"""
Run this node INSIDE the ROS2 Docker container.
It reads TF transforms for both EE frames and republishes them as
PoseStamped topics that roslibpy (on the host) can subscribe to.

Usage (inside Docker):
    python tf_ee_publisher.py \
        --arm0-base arm_0/base_link \
        --arm0-tool arm_0/tool \
        --arm1-base arm_1/base_link \
        --arm1-tool arm_1/tool \
        --hz 50
"""

import argparse
import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from geometry_msgs.msg import PoseStamped
import tf2_ros


class TFEEPublisher(Node):
    def __init__(self, args):
        super().__init__('tf_ee_publisher')

        self.arm0_base = args['arm0_base']
        self.arm0_tool = args['arm0_tool']
        self.arm1_base = args['arm1_base']
        self.arm1_tool = args['arm1_tool']

        self.tf_buffer   = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.pub_arm0 = self.create_publisher(PoseStamped, '/robot/arm0/ee_pose', 10)
        self.pub_arm1 = self.create_publisher(PoseStamped, '/robot/arm1/ee_pose', 10)

        period = 1.0 / args['hz']
        self.create_timer(period, self._timer_cb)
        self.get_logger().info(
            f'TF EE publisher ready @ {args["hz"]} Hz\n'
            f'  {self.arm0_base} → {self.arm0_tool}  →  /robot/arm0/ee_pose\n'
            f'  {self.arm1_base} → {self.arm1_tool}  →  /robot/arm1/ee_pose')

    def _lookup(self, base_frame: str, child_frame: str):
        try:
            tf = self.tf_buffer.lookup_transform(
                base_frame, child_frame,
                rclpy.time.Time(),
                Duration(seconds=0.05))
            return tf
        except (tf2_ros.LookupException,
                tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException) as e:
            self.get_logger().warn(f'TF [{base_frame} → {child_frame}]: {e}',
                                   throttle_duration_sec=2.0)
            return None

    def _timer_cb(self):
        now = self.get_clock().now().to_msg()

        for tf_result, pub, base_frame in [
            (self._lookup(self.arm0_base, self.arm0_tool), self.pub_arm0, self.arm0_base),
            (self._lookup(self.arm1_base, self.arm1_tool), self.pub_arm1, self.arm1_base),
        ]:
            if tf_result is None:
                continue
            msg = PoseStamped()
            msg.header.stamp    = now
            msg.header.frame_id = base_frame
            t = tf_result.transform.translation
            q = tf_result.transform.rotation
            msg.pose.position.x    = t.x
            msg.pose.position.y    = t.y
            msg.pose.position.z    = t.z
            msg.pose.orientation.x = q.x
            msg.pose.orientation.y = q.y
            msg.pose.orientation.z = q.z
            msg.pose.orientation.w = q.w
            pub.publish(msg)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--arm0-base', type=str, default='arm_0/base_link')
    parser.add_argument('--arm0-tool', type=str, default='arm_0/tool')
    parser.add_argument('--arm1-base', type=str, default='arm_1/base_link')
    parser.add_argument('--arm1-tool', type=str, default='arm_1/tool')
    parser.add_argument('--hz',        type=float, default=50.0)
    args = vars(parser.parse_args())

    rclpy.init()
    node = TFEEPublisher(args)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
