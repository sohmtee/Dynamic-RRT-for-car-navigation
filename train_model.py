import pybullet as p
import pybullet_envs
import gym
import random
import math
import numpy as np
import time
import matplotlib.pyplot as plt

from stable_baselines3 import PPO

# Connect to the physics server
p.connect(p.GUI)

# Definir altura del suelo y radio del objetivo
ground_height = 0.01
goal_radius = 0.1

# Set gravity
p.setGravity(0, 0, -10)
time.sleep(1. / 240.)

# Function to generate a random point on the ground
def generate_random_point_on_ground(bounds, ground_height):
    return [random.uniform(bounds[0][0], bounds[0][1]),
            random.uniform(bounds[1][0], bounds[1][1]),
            ground_height]

# Crear la forma visual para el objetivo
goal_sphere_id = p.createVisualShape(shapeType=p.GEOM_SPHERE, radius=goal_radius, rgbaColor=[0, 1, 0, 1])

# Define PyBullet environment
class PyBulletEnv(gym.Env):

    def __init__(self):
        super(PyBulletEnv, self).__init__()
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)  # Continuous action space: [-1, 1]
        self.observation_space = gym.spaces.Box(low=-float('inf'), high=float('inf'), shape=(7,), dtype=np.float32)  # 7 observations: [goal_x, goal_y, turtle_x, turtle_y, turtle_theta, linear_velocity, angular_velocity]
        self.turtle = p.loadURDF("turtlebot.urdf", [0, 0, 0])
        plane = p.loadURDF("plane.urdf")
        self.bounds = [[-1, 1], [-1, 1]]
        self.goal_point = generate_random_point_on_ground(self.bounds, ground_height)
        self.goal_body_id = p.createMultiBody(baseVisualShapeIndex=goal_sphere_id, basePosition=self.goal_point)
        self.prev_distance_to_goal = None

    def step(self, action):
        # Get current position and orientation of the Turtlebot
        turtle_position, turtle_orientation = p.getBasePositionAndOrientation(self.turtle)
        turtle_orientation_euler = p.getEulerFromQuaternion(turtle_orientation)

        # Get the position of the goal
        goal_position = p.getBasePositionAndOrientation(self.goal_body_id)[0]

        # Calculate vector from turtle to goal
        goal_vector = np.array([goal_position[0] - turtle_position[0], goal_position[1] - turtle_position[1]])
        # Calculate vector pointing towards the front of the turtle (assuming orientation is along z-axis)
        forward_vector = np.array([math.cos(turtle_orientation_euler[2]), math.sin(turtle_orientation_euler[2])])

        # Calculate angle between forward vector and goal vector
        angle_difference = math.atan2(goal_vector[1], goal_vector[0]) - math.atan2(forward_vector[1], forward_vector[0])

        # Wrap angle difference to [-pi, pi] range
        if angle_difference > math.pi:
            angle_difference -= 2 * math.pi
        elif angle_difference < -math.pi:
            angle_difference += 2 * math.pi

        # Determine action
        if abs(angle_difference) < 0.1:  # If facing goal
            forward = action[0]
            turn = 0
        else:  # If not facing goal
            forward = 0
            turn = np.sign(angle_difference)

        # Convert action to velocities
        speed = 10
        rightWheelVelocity = (forward + turn) * speed
        leftWheelVelocity = (forward - turn) * speed

        # Apply velocities to the wheels
        p.setJointMotorControl2(self.turtle, 0, p.VELOCITY_CONTROL, targetVelocity=leftWheelVelocity, force=1000)
        p.setJointMotorControl2(self.turtle, 1, p.VELOCITY_CONTROL, targetVelocity=rightWheelVelocity, force=1000)

        p.stepSimulation()

        # Reward calculation
        distance_to_goal = np.linalg.norm(np.array(goal_position[:2]) - np.array(turtle_position[:2]))

        if self.prev_distance_to_goal is None:
            reward = 150
        else:
            if distance_to_goal < self.prev_distance_to_goal:
                reward = 0.1  # Reward for getting closer to the goal
            elif distance_to_goal > self.prev_distance_to_goal:
                reward = -0.5  # Penalty for moving away from the goal
            else:
                reward = -0.08  # Penalty for staying in the same place

        self.prev_distance_to_goal = distance_to_goal

        # Check if Turtlebot reached the goal or went out of bounds
        done = distance_to_goal < goal_radius or distance_to_goal > 1.5

        # If the episode ends, reset the environment
        if done:
            self.reset()

        # Observations
        obs = np.array([
            goal_position[0],
            goal_position[1],
            turtle_position[0],
            turtle_position[1],
            turtle_orientation_euler[2],
            forward,
            turn
        ])

        return obs, reward, done, {}



    def reset(self):
        # Reset Turtlebot position
        p.resetBasePositionAndOrientation(self.turtle, [0, 0, 0], [0, 0, 0, 1])

        # Reset goal position
        self.goal_point = generate_random_point_on_ground(self.bounds, ground_height)
        p.resetBasePositionAndOrientation(self.goal_body_id, self.goal_point, [0, 0, 0, 1])

        # Reset prev_distance_to_goal
        self.prev_distance_to_goal = None

        # Return initial observation
        return np.array([
            self.goal_point[0],
            self.goal_point[1],
            0,
            0,
            0,
            0,
            0
        ])
    

    def render(self, mode='human'):
        if mode == 'human':
            p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
        elif mode == 'rgb_array':
            # Implement rendering to RGB array if needed
            pass
        else:
            super(PyBulletEnv, self).render(mode=mode)


# Create PyBullet environment
env = PyBulletEnv()

model = PPO("MlpPolicy", env, verbose=1, learning_rate=0.001, n_steps=2048, batch_size=64, n_epochs=20, gamma=0.99, gae_lambda=0.95, clip_range=0.2, ent_coef=0.01)

# Train the RL model

model.learn(total_timesteps=500000, log_interval=10)

# Save the trained model
model.save("./pybullet_rl_model")

