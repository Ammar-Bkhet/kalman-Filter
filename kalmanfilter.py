import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
import numpy as np
from numpy.linalg import inv

class KalmanFilter(Node):
    def __init__(self):
        super().__init__('kalman_filter_node')

        # Initialize Kalman variables
        self.x = np.zeros((2, 1))  # State vector [position_x, position_y]
        self.P = np.eye(2)  # Covariance matrix
        self.Q = np.eye(2)  # Process noise covariance matrix
        self.R = np.eye(2)  # Measurement noise covariance matrix

        # The constant linear speed of the robot
        self.linear_speed = 0.1  

        # Subscribe to the /odom_noise topic
        self.subscription = self.create_subscription(
            Odometry,
            '/odom_noise',
            self.odom_callback,
            1
        )

        # Subscribe to the /cmd_vel topic
        self.cmd_vel_subscription = self.create_subscription(
            Twist,
            '/cmd_vel',
            self.cmd_vel_callback,
            1
        )

        # Publish the estimated reading
        self.estimated_pub = self.create_publisher(
            Odometry,
            "/odom_estimated",
            1
        )

    def odom_callback(self, msg):
        # Extract the position measurements from the Odometry message
        z = np.array([
            [msg.pose.pose.position.x],
            [msg.pose.pose.position.y]
        ])

        # Prediction step
        # Update the state estimate using the motion model
        dt = 0.1  # Time step
        u = np.array([
            [self.linear_speed * dt],  # Control input for x-position
            [0.0]  # Control input for y-position (assuming constant speed)
        ])
        A = np.eye(2)  # State transition matrix
        self.x = A @ self.x + u  # State prediction
        self.P = A @ self.P @ A.T + self.Q  # Covariance prediction

        # Update step
        # Compute Kalman gain
        H = np.eye(2)  # Measurement matrix
        S = H @ self.P @ H.T + self.R  # Innovation covariance
        K = self.P @ H.T @ inv(S)  # Kalman gain

        # Update state estimate and covariance
        self.x = self.x + K @ (z - H @ self.x)  # State update
        self.P = (np.eye(2) - K @ H) @ self.P  # Covariance update

        # Publish the estimated reading
        odom_estimated = Odometry()
        odom_estimated.pose.pose.position.x = self.x[0, 0]
        odom_estimated.pose.pose.position.y = self.x[1, 0]
        self.estimated_pub.publish(odom_estimated)

    def cmd_vel_callback(self, msg):
        # Update the linear speed based on the cmd_vel message
        self.linear_speed = msg.linear.x

def main(args=None):
    rclpy.init(args=args)
    node = KalmanFilter()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
