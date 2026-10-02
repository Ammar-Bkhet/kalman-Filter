import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
import numpy as np
from numpy.linalg import inv
from tf_transformations import euler_from_quaternion
import time

class KalmanFilter(Node):
    def __init__(self):
        super().__init__('kalman_filter_node')
        # Initialize kalman variables
        self.dt = 0
        self.t2 = 0
        self.t1 = 0
        self.A = np.eye(2)
        self.H = np.eye(2)
        self.R = np.diag([0.1, 0.1])
        self.X = np.matrix('5.0 ; 5.0')
        self.P = np.matrix('20 0; 0 20')
        self.U = np.zeros((2,1))
        self.z = np.zeros((2,1))
        self.last_time = None
        self.Q = self.Q = np.matrix('0.00001 0.0; 0.0 0.00001')
        # Subscribe to the /odom_noise topic
        self.subscription = self.create_subscription(Odometry,'/odom_noise', self.odom_noise_callback,1)
        
        # Subscribe to the /odom topic
        self.subscription = self.create_subscription(Odometry,'/odom',self.odom_callback,1)
        
        #publish the estimated reading
        self.estimated_pub=self.create_publisher(Odometry,"/odom_estimated",1)

    def odom_noise_callback(self, msg):
        # Extract the position measurements from the Odometry message
        x_noised = msg.pose.pose.position.x
        y_noised = msg.pose.pose.position.y
        self.z = np.array([[x_noised],[y_noised]])
        # Prediction step
        # Update step
        #publish the estimated reading

    def odom_callback(self, msg):
    
        v_bot = msg.twist.twist.linear.x
        q = msg.pose.pose.orientation
        quat = [q.x, q.y, q.z, q.w]
        _, _, yaw = euler_from_quaternion(quat)
        self.U = np.array([[v_bot*np.cos(yaw)],[v_bot*np.sin(yaw)]])
        # print(self.U)
        # self.get_logger().info(f"x : {x_actual}, y: {y_actual}, yaw: {yaw:.3f} rad")
        # self.get_logger().info(f"vel : {v_bot}")
        #time
        current_time = self.get_clock().now()
        if self.last_time is None:
            self.last_time = current_time
            return
        dt = (current_time - self.last_time).nanoseconds * 1e-9
        self.last_time = current_time
        self.dt = dt
        self.B = np.diag([self.dt, self.dt])
        # print(self.dt)
        
        self.X = np.dot(self.A, self.X) + np.dot(self.B, self.U)
        # print(np.dot(self.B, self.U))c
        # print(self.X[0,0])
        self.P = np.dot(self.A,np.dot(self.P,self.A.T)) + self.Q 
        S = np.dot(self.H,np.dot(self.P,self.H.T))+self.R
        K = np.dot(np.dot(self.P,self.H.T),inv(S))
        self.X = self.X+np.dot(K,(self.z-np.dot(self.H,self.X)))
        print(self.X[0,0])
        I = np.eye(2)
        self.P = (I-(K @ self.H))@ self.P
        #dt
        # print("time :", self.dt)

        estimated = Odometry()
        estimated.pose.pose.position.x = self.X[0,0]
        estimated.pose.pose.position.y = self.X[1,0]
        # print(estimated.pose.pose.position.x)
        self.estimated_pub.publish(estimated)
       

def main(args=None):
    rclpy.init(args=args)
    node = KalmanFilter()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()