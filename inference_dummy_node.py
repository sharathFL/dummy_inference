#!/usr/bin/env python3
import random
import requests
from datetime import datetime, timezone
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

API_URL = 'http://192.168.10.126:8000/api/heartbeat'


class InferenceDummyNode(Node):
    def __init__(self):
        super().__init__('inference_dummy_node')
        self.declare_parameter('length', 64)
        self.declare_parameter('rate_hz', 10.0)
        self.declare_parameter('api_interval_sec', 5.0)

        self._length = self.get_parameter('length').value
        rate_hz = self.get_parameter('rate_hz').value

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
        payload = {
            'machine_id': 'CNC-001',
            'timestamp': datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
            'status': 'running',
            'health_metrics': {
                'temperature': round(random.uniform(60.0, 90.0), 1),
                'vibration': round(random.uniform(1.0, 5.0), 1),
                'power_consumption': round(random.uniform(10.0, 20.0), 1),
            },
            'metadata': {
                'name': 'CNC Lathe Alpha',
                'location': 'Floor A',
                'operator': 'Jane Smith',
            },
        }
        try:
            resp = requests.post(API_URL, json=payload, timeout=4.0)
            self.get_logger().info(f'Heartbeat sent: {resp.status_code}')
        except requests.exceptions.RequestException as e:
            self.get_logger().warn(f'Heartbeat failed: {e}')


def main(args=None):
    rclpy.init(args=args)
    node = InferenceDummyNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
