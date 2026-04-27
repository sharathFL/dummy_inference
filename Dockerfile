FROM ros:iron-ros-base

RUN echo "source /opt/ros/iron/setup.bash" >> /root/.bashrc

WORKDIR /app

RUN pip3 install requests

COPY random_bits_node.py .

ENTRYPOINT ["/ros_entrypoint.sh"]
CMD ["python3", "random_bits_node.py"]
