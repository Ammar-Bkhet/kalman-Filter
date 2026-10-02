import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
import numpy as np
import math
from numpy.linalg import inv

class KalmanFilter(Node):
    def __init__(self):
        super().__init__('kalman_filter_node')

        # Initialize Kalman variables
        self.x = np.matrix('5.0 ;  5.0')  # State vector [position_x, position_y]
        self.P = np.matrix('20 0; 0 20') # Covariance self.H
        self.Q = np.matrix('0.00001 0.0; 0.0 0.00001')  # Process noise covariance self.H
        self.R = np.matrix('0.1 0; 0 0.1')# Measurement noise covariance self.H
        self.dt = 0.02
        self.B = np.matrix('0.02 0; 0 0.02')
        self.A = np.eye(2)
        self.H = np.eye(2) 
        self.u = np.matrix('0.0 ;  0.0')
        self.z = np.zeros((2,1))
        self.x_prev = 0
        self.linear_speedx = 0
        self.linear_speedy = 0
        
        # print(self.B)

        # Tself.He constant linear speed of tself.He robot
        

        # Subscribe to tself.He /odom_noise topic
        self.subscription = self.create_subscription(
            Odometry,
            '/odom_noise',
            self.odom_callback,
            1
        )

        # Subscribe to tself.He /cmd_vel topic
        self.odomspeed = self.create_subscription(
            Odometry,
            '/odom',
            self.odomspeed_callback,
            1
        )

        # Publisself.H tself.He estimated reading
        self.estimated_pub = self.create_publisher(
            Odometry,
            "/odom_estimated",
            1
        )

        self.timer = self.create_timer(self.dt, self.Kalman_filter)

    def odom_callback(self, msg):
        # Extract tself.He position measurements from tself.He Odometry message
        self.z = np.array([
            [msg.pose.pose.position.x],
            [msg.pose.pose.position.y],
        ])
        # self.u = (self.z - self.x_prev) / self.dt
        # self.x_prev = self.z
        
        # print(msg.pose.pose.position.x , 'iam measured')

        # Prediction step
        # Update tself.He state estimate using tself.He motion model
       # Time step
        
     
       
        
        # A[0,2] = dt
        # A[1,3] = dt # State transition self.H
        

    def odomspeed_callback(self, msg):
        # Update tself.He linear speed based on tself.He cmd_vel message
       
        self.linear_speedx = msg.twist.twist.linear.x * math.cos((msg.pose.pose.orientation.z) * 2)
        self.linear_speedy = msg.twist.twist.linear.x * math.sin((msg.pose.pose.orientation.z) * 2)
        self.u = np.array([
            [self.linear_speedx],  # Control input for x-position
            [self.linear_speedy]  # Control input for y-position (assuming constant speed)
        ]) 
        # print('2' )
        
    def Kalman_filter(self):

        odom_estimated = Odometry()
        odom_estimated.pose.pose.position.x = self.x[0, 0]
        odom_estimated.pose.pose.position.y = self.x[1, 0]
        self.estimated_pub.publish(odom_estimated)

        # self.x_prev = self.x
        
        # Covariance prediction
        # print(self.x[0, 0] , 'iam predicted')
        # Update step
        # Compute Kalman gain
        # self.H = np.eye(2) 
        


        # self.H[0, 0] = 1  # Set (0tself.H row, 0tself.H column) to 1
        # self.H[1, 1] = 1  # Set (1st row, 1st column) to 1 # Measurement self.H
        S = (self.H @ self.P @ self.H.T) + self.R  # Innovation covariance
        K = (self.P @ self.H.T @ inv(S))  # Kalman gain
        # print(self.P , "iam covariance")
        # print(K, "iam kalman gain")
       

        # Update state estimate and covariance
        self.x = self.x + K @ (self.z - self.H @ self.x)  # State update
        
        self.P = (np.eye(2) - K @ self.H) @ self.P  # Covariance update

        self.x = self.A @ self.x + self.B @ self.u # State prediction
        # self.u = (self.x - self.x_prev)/self.dt
        
        
        
        self.P = self.A @ self.P @ self.A.T + self.Q 
        # self.P = self.A @ self.P @ self.A.T  

        # Publisself.H tself.He estimated reading
       
        # print(self.P , 'iam covariance')
        print(self.x[0,0])

def main(args=None):
    rclpy.init(args=args)
    node = KalmanFilter()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()