# TurtleBot3 Localization with Kalman Filtering

This ROS 2 workspace demonstrates 2D TurtleBot3 localization in Gazebo. It treats Gazebo odometry as the reference pose, corrupts its position with Gaussian noise, and estimates the robot position again with either a linear 2D Kalman Filter (KF) or an Extended Kalman Filter (EKF).

The result is easy to inspect in PlotJuggler: compare the reference `/odom` signal, the deliberately noisy `/odom_noise` signal, and the filter output `/odom_estimated`.

> **Demo video:** :
> [EKF estimated position versus noisy position](https://drive.google.com/file/d/1BZ8uKG9LQV13VPVUCcMkbu100MhqeR9a/view?usp=sharing)

## Pipeline

```text
Gazebo TurtleBot3
      │
      ├── /odom ──────────────────────────────┐
      │                                        │ control / motion input
      ▼                                        ▼
Noise node ── /odom_noise ──► KF or EKF ──► /odom_estimated
      │                         ▲
      └─ Gaussian x/y noise     └─ noisy position measurement

PlotJuggler: /odom, /odom_noise, /odom_estimated
```

## What is in the workspace?

| Component | Purpose |
| --- | --- |
| `src/turtlebot3_simulations/` | TurtleBot3 Gazebo simulation packages and worlds. |
| `src/kalman_filter/kalman_filter/noise.py` | Adds zero-mean Gaussian noise (standard deviation `0.1 m`) to `/odom` position and publishes `/odom_noise`. |
| `src/kalman_filter/kalman_filter/2d_kalmanfilter.py` | Linear 2D KF that estimates `[x, y]`. |
| `src/kalman_filter/kalman_filter/EKF.py` | EKF that estimates `[x, y, yaw]` using the robot motion model and noisy position measurements. |

## Prerequisites

This project is intended for a desktop ROS 2 installation with Gazebo Classic. It has been structured like the TurtleBot3 ROS 2 simulation packages; use the ROS 2 distribution that matches your installed TurtleBot3/Gazebo packages.

Install the core ROS dependencies, replacing `<ros-distro>` with your distribution name (for example, `humble`):

```bash
sudo apt update
sudo apt install ros-<ros-distro>-gazebo-ros-pkgs \
  ros-<ros-distro>-turtlebot3-description \
  ros-<ros-distro>-turtlebot3-teleop \
  ros-<ros-distro>-plotjuggler-ros
```

The filter package also needs Python NumPy and `tf_transformations`, which are normally installed with the ROS desktop stack. If they are absent, install the matching ROS package for `tf_transformations` and `python3-numpy` through your system package manager.

## Build the workspace

From the workspace root:

```bash
source /opt/ros/<ros-distro>/setup.bash
colcon build --symlink-install
source install/setup.bash
```

Set the TurtleBot3 model before every launch. This guide uses the Burger model:

```bash
export TURTLEBOT3_MODEL=burger
```

To avoid repeating the command, add the export to your shell profile if Burger is your usual model. `waffle` and `waffle_pi` are also available when the matching model assets are installed.

## Run the complete experiment

Open separate terminals. In every terminal, source ROS 2 and the workspace, then set the model:

```bash
source /opt/ros/<ros-distro>/setup.bash
source ~/turtlebot3_localization/install/setup.bash
export TURTLEBOT3_MODEL=burger
```

Replace `~/turtlebot3_localization` if you cloned the workspace somewhere else.

### 1. Spawn TurtleBot3 in Gazebo

```bash
ros2 launch turtlebot3_gazebo empty_world.launch.py
```

The robot spawns at `(0, 0)`. To spawn it elsewhere, pass the launch arguments:

```bash
ros2 launch turtlebot3_gazebo empty_world.launch.py x_pose:=1.0 y_pose:=-0.5
```

Other included worlds can be launched in the same way, such as `turtlebot3_world.launch.py` or `turtlebot3_house.launch.py`.

### 2. Drive the robot

In another terminal:

```bash
ros2 run turtlebot3_teleop teleop_keyboard
```

Keep this terminal focused and use the displayed keyboard controls to move the robot. Motion makes the difference between raw noisy measurements and the estimated trajectory visible.

### 3. Add measurement noise

In a third terminal:

```bash
ros2 run kalman_filter Noise
```

This node subscribes to `/odom` and publishes `/odom_noise`. It adds independent Gaussian noise to `pose.pose.position.x` and `pose.pose.position.y`; the configured noise standard deviation is `0.1 m`.

### 4. Start **one** filter

Choose exactly one estimator. Both implementations use the same output topic and node name, so do not run them together.

For the linear 2D Kalman Filter:

```bash
ros2 run kalman_filter KalmanFilter
```

Or, for the Extended Kalman Filter:

```bash
ros2 run kalman_filter EKF
```

Both filters subscribe to `/odom_noise` for the position measurement and `/odom` for the simulated robot motion information, then publish their estimate on `/odom_estimated`.

### 5. Visualize with PlotJuggler

Launch PlotJuggler in a final terminal:

```bash
ros2 run plotjuggler plotjuggler
```

1. Create a **ROS 2 Topic Subscriber** (often available from the Streaming menu) and start streaming.
2. In the topic tree, find and drag these signals onto the same time-series plot:
   - `/odom/pose/pose/position/x`
   - `/odom_noise/pose/pose/position/x`
   - `/odom_estimated/pose/pose/position/x`
3. Repeat for the corresponding `.../position/y` fields, either in a second plot or overlaid with the x signals.
4. Drive the robot. The noisy trace should fluctuate around `/odom`, while `/odom_estimated` should be smoother and follow the robot motion.

For a path view, use an XY plot and assign the matching x and y series from one topic to each curve. Overlay the noisy and estimated paths to see how filtering reduces measurement jitter.

## Topic reference

| Topic | Type | Publisher | Meaning |
| --- | --- | --- | --- |
| `/odom` | `nav_msgs/msg/Odometry` | TurtleBot3 Gazebo | Reference simulated pose and velocity. |
| `/odom_noise` | `nav_msgs/msg/Odometry` | `Noise` | Position measurement after Gaussian x/y noise. |
| `/odom_estimated` | `nav_msgs/msg/Odometry` | `KalmanFilter` or `EKF` | Filtered pose estimate. |
| `/cmd_vel` | `geometry_msgs/msg/Twist` | Teleoperation node | Velocity command sent while driving. |

Useful checks while debugging:

```bash
ros2 topic list | grep odom
ros2 topic echo /odom_noise --once
ros2 topic echo /odom_estimated --once
ros2 node list
```

## Filter choice

Use `KalmanFilter` for the simple constant-velocity, planar position experiment. It maintains a two-element state `[x, y]` and runs at a fixed 50 Hz timer.

Use `EKF` when you want the motion model to retain heading. Its state is `[x, y, yaw]`, and it uses the odometry velocity and elapsed ROS time to predict the next state before incorporating the noisy x/y measurement.

Neither node estimates a full 3D pose; the published `Odometry` message is used here as a convenient ROS container for the planar position result. For a fair comparison, begin a fresh simulation before each run because the current filter initial states are defined in the scripts.

## Troubleshooting

| Symptom | Check |
| --- | --- |
| Launch fails with `KeyError: 'TURTLEBOT3_MODEL'` | Run `export TURTLEBOT3_MODEL=burger` in that terminal before launching Gazebo. |
| `Package 'kalman_filter' not found` | Build with `colcon build --symlink-install`, then source `install/setup.bash` in the current terminal. |
| No `/odom_noise` | Confirm Gazebo is running and use `ros2 topic echo /odom --once`; then start `ros2 run kalman_filter Noise`. |
| No estimate or two competing estimates | Run only one of `KalmanFilter` and `EKF`; both publish `/odom_estimated`. |
| PlotJuggler has no signals | Start the ROS 2 streaming subscriber, then move the robot so messages arrive. Verify that PlotJuggler was launched after sourcing the same ROS environment. |
| Robot does not move | Keep the teleop terminal focused and use the controls shown there; verify `/cmd_vel` with `ros2 topic echo /cmd_vel`. |

## Stop and reset

Press `Ctrl+C` in each terminal. To reset the experiment, stop all nodes, start Gazebo again, then repeat the noise, filter, teleoperation, and PlotJuggler steps. Restarting Gazebo respawns the robot and returns it to the requested spawn pose.
