#!/usr/bin/env python3

import os
import sys
import time
import asyncio
import random
import configparser
import requests
import numpy as np

from datetime import datetime, timezone

import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from std_msgs.msg import String

from audio_interfaces.msg import Signals


sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "src")))
from config.log_config import get_logger


logger = get_logger(__name__)

API_URL = "https://1c41-106-51-87-203.ngrok-free.app/api/heartbeat"

config = configparser.ConfigParser()
config.read("src/data_collection/data_collection/data_collector_node.ini")


async def get_topics_list():
    logger.info("Getting topics list...")

    node_dummy = Node("_ros2cli_dummy_to_show_topic_list")
    topic_list = []
    topics_found = False

    try:
        while not topics_found:
            print("[DISCOVERY] Searching for /telescopii topics...")
            logger.info("Searching for /telescopii topics...")

            topics = node_dummy.get_topic_names_and_types()
            await asyncio.sleep(5)

            for topic_name, topic_types in topics:
                if "/telescopii" in topic_name:
                    print(f"[DISCOVERY] Found topic: {topic_name}")
                    logger.info(f"Topic found: {topic_name}")
                    topic_list.append(topic_name)
                    topics_found = True

        return list(set(topic_list))

    except Exception as e:
        print(f"[DISCOVERY ERROR] {e}")
        logger.error(f"Unable to get topics list: {e}")
        return []

    finally:
        node_dummy.destroy_node()


class RealtimeInferenceHeartbeatNode(Node):

    def __init__(self, node_name, topic):
        super().__init__(node_name)

        self.config = config
        self.topic = topic

        self.frames = []
        self.starttime = time.time()
        self.latest_prediction = None

        self.score_buffer = []
        self.last_stream_print_time = 0

        self.clip_duration = int(
            self.config["DEFAULT"]["clip_duration"].strip()
        )

        self.threshold = 0.5
        self.score_window_size = 10

        x = self.topic.split("/")

        self.machine_name = x[2]
        self.sensor_type = x[3]
        self.sensor_name = x[4]

        self.subscription = self.create_subscription(
            Signals,
            self.topic,
            self.listener_callback,
            10
        )

        self.prediction_pub = self.create_publisher(
            String,
            "predictions",
            10
        )

        self.api_interval_sec = 5.0

        self.create_timer(
            self.api_interval_sec,
            self.send_heartbeat
        )

        print(
            f"[SUBSCRIBED] {self.topic} -> "
            f"Machine: {self.machine_name}, "
            f"Sensor Type: {self.sensor_type}, "
            f"Sensor: {self.sensor_name}"
        )

        logger.info(f"Subscribed to topic: {self.topic}")

    def listener_callback(self, msg):
        self.frames.extend(msg.signals_vect)

        current_time = time.time()
        laptime = round(current_time - self.starttime, 2)

        if current_time - self.last_stream_print_time > 1:
            print(
                f"[STREAMING] {self.topic} | "
                f"buffer={len(self.frames)} samples | "
                f"elapsed={laptime}s"
            )
            self.last_stream_print_time = current_time

        if laptime >= self.clip_duration:
            print(
                f"\n[CLIP READY] {self.topic} | "
                f"{len(self.frames)} samples collected"
            )

            logger.info(
                f"Collected {self.clip_duration} sec data from topic {self.topic}"
            )

            signal_data = self.frames.copy()
            sampling_rate = msg.fs

            prediction_result = self.run_prediction(
                signal_data,
                sampling_rate
            )

            self.latest_prediction = prediction_result
            self.publish_prediction(prediction_result)

            self.starttime = time.time()
            self.frames.clear()

    def run_prediction(self, signal_data, sampling_rate):
        """
        Threshold-based prediction logic.

        Replace this method later with actual ML inference.
        """

        signal = np.array(signal_data, dtype=np.float32)

        if len(signal) == 0:
            result = {
                "machine_id": self.machine_name,
                "sensor_type": self.sensor_type,
                "sensor_name": self.sensor_name,
                "status": "no_data",
                "prediction": "no_data",
                "score": 0.0,
                "rms": 0.0,
                "variance": 0.0,
                "sampling_rate": sampling_rate,
                "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            }

            print(f"[PREDICTION] {self.machine_name} | NO DATA")
            return result

        rms = np.sqrt(np.mean(signal ** 2))
        variance = np.var(signal)

        current_score = rms + 0.5 * variance

        self.score_buffer.append(current_score)

        if len(self.score_buffer) > self.score_window_size:
            self.score_buffer.pop(0)

        average_score = np.mean(self.score_buffer)

        if average_score > self.threshold:
            status = "anomaly"
            prediction = "anomaly_detected"
        else:
            status = "running"
            prediction = "running_ok"

        print(
            f"[PREDICTION] {self.machine_name} | "
            f"Score={average_score:.4f} | "
            f"Current={current_score:.4f} | "
            f"RMS={rms:.4f} | "
            f"Variance={variance:.4f} | "
            f"Threshold={self.threshold} | "
            f"Status={status.upper()}"
        )

        result = {
            "machine_id": self.machine_name,
            "sensor_type": self.sensor_type,
            "sensor_name": self.sensor_name,
            "status": status,
            "prediction": prediction,
            "score": float(average_score),
            "current_score": float(current_score),
            "rms": float(rms),
            "variance": float(variance),
            "threshold": float(self.threshold),
            "sampling_rate": sampling_rate,
            "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        }

        logger.info(f"Prediction result: {result}")

        return result

    def publish_prediction(self, prediction_result):
        msg = String()
        msg.data = str(prediction_result)

        self.prediction_pub.publish(msg)

        print(
            f"[PUBLISHED] /predictions | "
            f"{prediction_result['machine_id']} | "
            f"{prediction_result['prediction']}"
        )

        logger.info(f"Published prediction: {msg.data}")

    def send_heartbeat(self):
        if self.latest_prediction is None:
            print(f"[HEARTBEAT] Waiting for first prediction from {self.machine_name}")
            return

        payload = {
            "machine_id": self.machine_name,
            "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "status": self.latest_prediction.get("status", "unknown"),
            "health_metrics": {
                "temperature": round(random.uniform(60.0, 90.0), 1),
                "vibration": round(random.uniform(1.0, 5.0), 1),
                "power_consumption": round(random.uniform(10.0, 20.0), 1),
                "anomaly_score": self.latest_prediction.get("score", 0.0),
                "rms": self.latest_prediction.get("rms", 0.0),
                "variance": self.latest_prediction.get("variance", 0.0),
            },
            "metadata": {
                "topic": self.topic,
                "machine_name": self.machine_name,
                "sensor_type": self.sensor_type,
                "sensor_name": self.sensor_name,
                "prediction": self.latest_prediction.get("prediction", "unknown"),
                "threshold": self.latest_prediction.get("threshold", self.threshold),
            },
        }

        try:
            print(
                f"[HEARTBEAT] Sending | "
                f"Machine={self.machine_name} | "
                f"Status={self.latest_prediction.get('status')} | "
                f"Score={self.latest_prediction.get('score'):.4f}"
            )

            response = requests.post(
                API_URL,
                json=payload,
                timeout=4.0
            )

            print(f"[HEARTBEAT SUCCESS] HTTP {response.status_code}")
            logger.info(f"Heartbeat sent: {response.status_code}")

        except requests.exceptions.RequestException as e:
            print(f"[HEARTBEAT FAILED] {e}")
            logger.warning(f"Heartbeat failed: {e}")


def main(args=None):
    rclpy.init(args=args)

    executor = None

    try:
        topics_list = asyncio.run(get_topics_list())

        if not topics_list:
            print("[ERROR] No /telescopii topics found. Exiting.")
            logger.error("No /telescopii topics found. Exiting.")
            return

        print(f"[STARTUP] Total topics found: {len(topics_list)}")
        logger.info("Initializing realtime inference + heartbeat nodes...")

        executor = MultiThreadedExecutor()

        collector_nodes = [
            RealtimeInferenceHeartbeatNode(
                f"rt_inference_heartbeat_node_{index}",
                topic
            )
            for index, topic in enumerate(topics_list)
        ]

        for node in collector_nodes:
            executor.add_node(node)

        print("[STARTUP] Realtime inference + heartbeat node started.")
        logger.info("All nodes started.")

        executor.spin()

    except KeyboardInterrupt:
        print("[SHUTDOWN] Keyboard interrupt received.")
        logger.info("Keyboard interrupt received. Shutting down.")

    except Exception as e:
        print(f"[ERROR] {e}")
        logger.error(f"Error: {e}")

    finally:
        if executor is not None:
            executor.shutdown()

        rclpy.shutdown()
        print("[SHUTDOWN] ROS 2 shutdown complete.")


if __name__ == "__main__":
    main()