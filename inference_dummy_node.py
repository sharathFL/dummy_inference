#!/usr/bin/env python3
import random
import requests
import rclpy
from rclpy.node import Node
from std_msgs.msg import String


class RandomBitsNode(Node):
    def __init__(self):
        super().__init__('random_bits_node')
        self.declare_parameter('length', 64)
        self.declare_parameter('rate_hz', 10.0)
        self.declare_parameter('api_url', 'http://api-server:8080/predict')
        self.declare_parameter('api_interval_sec', 5.0)

        self._length = self.get_parameter('length').value
        rate_hz = self.get_parameter('rate_hz').value
        self._api_url = self.get_parameter('api_url').value

        self._pub = self.create_publisher(String, 'predictions', 10)
        self.create_timer(1.0 / rate_hz, self._publish)
        self.create_timer(
            self.get_parameter('api_interval_sec').value,
            self._call_api,
        )

        self._latest_bits = ''

    def _publish(self):
        self._latest_bits = ''.join(random.choice('01') for _ in range(self._length))
        msg = String()
        msg.data = self._latest_bits
        self._pub.publish(msg)

    def _call_api(self):
        if not self._latest_bits:
            return
        try:
            resp = requests.post(
                self._api_url,
                json={'json_data': self._latest_bits},
                timeout=4.0,
            )
            self.get_logger().info(f'API response: {resp.status_code}')
        except requests.exceptions.RequestException as e:
            self.get_logger().warn(f'API call failed: {e}')


def main(args=None):
    rclpy.init(args=args)
    node = RandomBitsNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
