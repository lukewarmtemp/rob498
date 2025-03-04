import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped


class Vicon(Node):
    def __init__(self):
        super().__init__('vicon')
        
        # Initialize storage variables
        self.position = None
        self.orientation = None
        self.timestamp = None
        self.frame_id = None

        # Initialize SENDING storage variables
        self.set_position = None
        self.set_orientation = None

        # Subscriber to Vicon pose data
        self.vicon_subscriber = self.create_subscription(PoseStamped, '/vicon/ROB498_Drone/ROB498_Drone', self.vicon_callback, 1)
        self.get_logger().info('Subscribing to Vicon!')
        
        # Publisher for VisionPose topic
        self.vision_pose_publisher = self.create_publisher(PoseStamped, '/mavros/vision_pose/pose', 1)
        self.get_logger().info('Publishing to VisionPose')
        
        # Publisher for SetPoint topic
        self.setpoint_publisher = self.create_publisher(PoseStamped, '/mavros/setpoint_position/local', 1)
        self.get_logger().info('Publishing to SetPoint')

        # Statement to end the inits
        self.get_logger().info('Vicon Node All Setup and Started!')


    def vicon_callback(self, msg):
        self.position = msg.pose.position
        self.orientation = msg.pose.orientation
        self.timestamp = msg.header.stamp
        self.frame_id = msg.header.frame_id

        # Print values normally
        print(f"Position: x={self.position.x}, y={self.position.y}, z={self.position.z}")
        print(f"Orientation: x={self.orientation.x}, y={self.orientation.y}, z={self.orientation.z}, w={self.orientation.w}")
        print(f"Timestamp: {self.timestamp.sec}.{self.timestamp.nanosec}")
        print(f"Frame ID: {self.frame_id}")
    
        # Everytime we get stuff, write both immediately
        self.send_vision_pose()
        self.send_setpoint()


    def send_vision_pose(self):
        # Create a new PoseStamped message to publish to vision_pose topic
        vision_pose_msg = PoseStamped()
        vision_pose_msg.header.stamp = self.timestamp
        vision_pose_msg.header.frame_id = self.frame_id
        vision_pose_msg.pose.position = self.position
        vision_pose_msg.pose.orientation = self.orientation
        # Publish the message to the /mavros/vision_pose/pose topic
        self.vision_pose_publisher.publish(vision_pose_msg)
 

    def send_setpoint(self):
        # Create a new PoseStamped message to publish to setpoint topic
        setpoint_msg = PoseStamped()
        setpoint_msg.pose.position = self.set_position
        setpoint_msg.pose.orientation = self.set_orientation
        # These are the same
        setpoint_msg.header.stamp = self.timestamp
        setpoint_msg.header.frame_id = self.frame_id
        # Publish the message to the /mavros/setpoint_position/local topic
        self.setpoint_publisher.publish(setpoint_msg)


    def set_non_z_pose(self):
        # Maintain the non y related parts of the flight
        self.set_position.x = self.position.x
        self.set_position.y = self.position.y
        self.set_orientation = self.orientation


# def main(args=None):
#     rclpy.init(args=args)
#     node = Vicon()
#     rclpy.spin(node)
#     node.destroy_node()
#     rclpy.shutdown()

# if __name__ == '__main__':
#     main()
