import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
import numpy as np
import math
from numpy.linalg import inv
from tf_transformations import euler_from_quaternion
from tf_transformations import quaternion_from_euler

class ExtendedKalmanFilter(Node):
    def __init__(self):
        super().__init__('kalman_filter_node')

        # Initialize Kalman variables
        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0 
        self.f = np.zeros((3,1))
        self.states = np.zeros((3,1))
        self.F = np.eye(3)
        # State vector [position_x, position_y]
        self.z_pred = np.zeros((2,1))
        self.P = np.matrix('20.0 0.0 0.0; 0.0 20.0 0.0; 0.0 0.0 20.0') # Covariance self.H
        self.Q = np.matrix('0.001 0.0 0.0; 0.0 0.001 0.0; 0.0 0.0 0.001')  # Process noise covariance self.H
        self.R = np.matrix('0.1 0; 0 0.1')# Measurement noise covariance self.H
        self.dt = 0.0
        self.H = np.array([
                [1, 0, 0],
                [0, 1, 0]
            ]) 
        self.z = np.zeros((2,1))
        self.last_time = None
        self.linear_speedx = 0.0
        self.linear_speedy = 0.0
        self.angular_velocity = 0.0
        
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

        # self.timer = self.create_timer(self.dt, self.Kalman_filter)

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


        current_time = self.get_clock().now()
        if self.last_time is None:
            self.last_time = current_time
            return
        dt = (current_time - self.last_time).nanoseconds * 1e-9
        self.last_time = current_time
        self.dt = dt
        print(self.dt)

         # Update tself.He linear speed based on tself.He cmd_vel message
        quat = msg.pose.pose.orientation
    
        # Convert the quaternion to Euler angles (roll, pitch, yaw)
        _, _, yaw = euler_from_quaternion(
            [quat.x, quat.y, quat.z, quat.w]
        )
       
        self.linear_speedx = msg.twist.twist.linear.x * math.cos((yaw) )
        self.linear_speedy = msg.twist.twist.linear.x * math.sin((yaw) )
        self.angular_velocity = msg.twist.twist.angular.z
        # print('2' )

        
        # run EKF
        self.Kalman_filter()
       
        
    def Kalman_filter(self):

        odom_estimated = Odometry()
        odom_estimated.pose.pose.position.x = self.x
        odom_estimated.pose.pose.position.y = self.y
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
       
        
        self.z_pred = np.array([self.f[0], self.f[1]]) # State update

        self.res = self.z -self.z_pred

        self.states = self.states + (K @ self.res)



        
        self.P = (np.eye(3) - K @ self.H) @ self.P  # Covariance update

        self.f = np.array([
                [ (self.x + self.linear_speedx * self.dt) ],
                [ (self.y + self.linear_speedy * self.dt) ],
                [ self.theta + self.angular_velocity * self.dt]
        ])

        # print(self.linear_speedx)
        self.x = float(self.f[0])
        self.y = float(self.f[1])
        self.theta = float(self.f[2])

        self.states = np.array([
            [self.x],
            [self.y],
            [self.theta]
        ])

        self.F = np.array([
            [1,0, -self.linear_speedy * self.dt],
            [0,1, self.linear_speedx * self.dt],
            [0,0,1]
        ])
        
        
        
        
        self.P = self.F @ self.P @ self.F.T + self.Q 

        # Publisself.H tself.He estimated reading
       
        # print(self.P , 'iam covariance')

def main(args=None):
    rclpy.init(args=args)
    node = ExtendedKalmanFilter()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()