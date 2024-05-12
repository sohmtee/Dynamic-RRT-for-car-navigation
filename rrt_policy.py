import pybullet as p
import pybullet_envs
import gym
import random
import math
import numpy as np
import time
import subprocess
import sys

from stable_baselines3 import PPO

# subprocess.run([sys.executable, 'base3.py'])
from rrt import path_points 

# Connect to the physics server
p.connect(p.GUI)
p.getCameraImage(320, 200)

# Define ground height and goal radius
ground_height = 0.01
goal_radius = 0.1

# Function to generate a random point on the ground
def generate_random_point_on_ground(bounds, ground_height):
    return [random.uniform(bounds[0][0], bounds[0][1]),
            random.uniform(bounds[1][0], bounds[1][1]),
            ground_height]

# Create the visual shape for the goal
goal_sphere_id = p.createVisualShape(shapeType=p.GEOM_SPHERE, radius=goal_radius, rgbaColor=[0, 1, 0, 1])



# Define PyBullet environment
class PyBulletEnv(gym.Env):
    p.setGravity(0, 0, -10)
    time.sleep(1. / 240.)
    p.getCameraImage(320, 200)
    
    def __init__(self):
        super(PyBulletEnv, self).__init__()
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=-float('inf'), high=float('inf'), shape=(7,), dtype=np.float32)
        self.turtle = p.loadURDF("turtlebot.urdf", [0, 0, 0])
        plane = p.loadURDF("plane.urdf")
        self.bounds = [[-1, 1], [-1, 1]]

        
        self.goals = path_points 
        self.goal_body_ids = [p.createMultiBody(baseVisualShapeIndex=goal_sphere_id, basePosition=goal) for goal in self.goals]
        self.current_goal_index = 0
        self.prev_distance_to_goal = None

    def step(self, action):
        turtle_position, turtle_orientation = p.getBasePositionAndOrientation(self.turtle)
        turtle_orientation_euler = p.getEulerFromQuaternion(turtle_orientation)
        goal_position = p.getBasePositionAndOrientation(self.goal_body_ids[self.current_goal_index])[0]
        goal_vector = np.array([goal_position[0] - turtle_position[0], goal_position[1] - turtle_position[1]])
        forward_vector = np.array([math.cos(turtle_orientation_euler[2]), math.sin(turtle_orientation_euler[2])])
        angle_difference = math.atan2(goal_vector[1], goal_vector[0]) - math.atan2(forward_vector[1], forward_vector[0])
        # angle_difference = math.atan2(goal_vector[1], goal_vector[0]) - math.atan2(forward_vector[1], forward_vector[0])

        if angle_difference > math.pi:
            angle_difference -= 2 * math.pi
        elif angle_difference < -math.pi:
            angle_difference += 2 * math.pi

        if abs(angle_difference) < 0.1:
            forward = action[0]
            turn = 0
        else:
            forward = 0
            turn = np.sign(angle_difference)

        speed = 20
        rightWheelVelocity = (forward + turn) * speed
        leftWheelVelocity = (forward - turn) * speed

        p.setJointMotorControl2(self.turtle, 0, p.VELOCITY_CONTROL, targetVelocity=leftWheelVelocity, force=1000)
        p.setJointMotorControl2(self.turtle, 1, p.VELOCITY_CONTROL, targetVelocity=rightWheelVelocity, force=1000)

        p.stepSimulation()

        distance_to_goal = np.linalg.norm(np.array(goal_position[:2]) - np.array(turtle_position[:2]))

        if self.prev_distance_to_goal is None:
            reward = 150
        else:
            if distance_to_goal < self.prev_distance_to_goal:
                reward = 0.1
            elif distance_to_goal > self.prev_distance_to_goal:
                reward = -0.5
            else:
                reward = -0.08

        self.prev_distance_to_goal = distance_to_goal

        done = distance_to_goal < goal_radius

        if done:
            self.current_goal_index += 1
            if self.current_goal_index >= len(self.goals):
                self.reset()

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
        p.resetBasePositionAndOrientation(self.turtle, [-5, 5, 0.01], [0, 0, 0, 1])
        # self.goals = [generate_random_point_on_ground(self.bounds, ground_height) for _ in range(4)]
        self.goal_body_ids = [p.createMultiBody(baseVisualShapeIndex=goal_sphere_id, basePosition=goal) for goal in self.goals]
        self.current_goal_index = 0
        self.prev_distance_to_goal = None
        return np.array([
            self.goals[0][0],
            self.goals[0][1],
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
            pass
        else:
            super(PyBulletEnv, self).render(mode=mode)

# Create PyBullet environment
env = PyBulletEnv()

# Evaluate the policy
model = PPO.load("./pybullet_rl_model")
obs = env.reset()
total_reward = 0
num_episodes = 1000

# Generate the initial point at the top left corner
initial_point = [-5, 5, ground_height]

obstacle_half_extents = [0.5, 0.5, 0.1]  # Half-lengths of the obstacle
obstacle_positions = [[0, 0, 0.1], [-2, -2, 0.1], [2, -1, 0.1], [-4, -1, 0.1], [3, -2, 0.1],[-5, 0, 0.1],[5, -5, 0.1],[3, -5, 0.1],[0, -5, 0.1],[3, -5, 0.1],[-3, -5, 0.1],
                    [-3, 3, 0.1] ,[-4, 3, 0.1], [ 4, 3, 0.1], [ 2, 3, 0.1] ,[ 2, 2, 0.1],[ 3, 3, 0.1],[-2, 3, 0.1] , [-3, 1, 0.1],[1, 1, 0.1],[-5, -5, 0.1],[5, -3, 0.1],
                    [-3, -3, 0.1],[-5, -3, 0.1],[-2, -3, 0.1],[-1, 5, 0.1],[5, 0, 0.1],[5, 1, 0.1],[5, 3, 0.1],[4, 5, 0.1],[4.5, 4.5, 0.1] ,[5, 5, 0.1],[5, -4, 0.1]]
obstacle_ids = []
for obstacle_position in obstacle_positions:
    obstacle_id = p.createCollisionShape(p.GEOM_BOX, halfExtents=obstacle_half_extents)
    obstacle_visual_id = p.createVisualShape(p.GEOM_BOX, halfExtents=obstacle_half_extents, rgbaColor=[1, 0.5, 0.5, 1])
    obstacle_body_id = p.createMultiBody(basePosition=obstacle_position, baseCollisionShapeIndex=obstacle_id, baseVisualShapeIndex=obstacle_visual_id)
    obstacle_ids.append(obstacle_body_id)

goal_point = [2, -3, ground_height]
# Visualizar la meta como una esfera verde
goal_sphere_id = p.createVisualShape(shapeType=p.GEOM_SPHERE, radius=goal_radius, rgbaColor=[1, 0, 0, 1])
goal_body_id = p.createMultiBody(baseVisualShapeIndex=goal_sphere_id, basePosition=goal_point)

# Create the start point as a green circle on the ground
start_sphere_id = p.createVisualShape(shapeType=p.GEOM_SPHERE, radius=0.05, rgbaColor=[0, 1, 0, 1])
start_body_id = p.createMultiBody(baseVisualShapeIndex=start_sphere_id, basePosition=initial_point)  

for episode in range(num_episodes):
    episode_reward = 0
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _ = env.step(action)
        episode_reward += reward
        env.render()
    print(f"Episode {episode + 1}: Total reward = {episode_reward}")
    total_reward += episode_reward

average_reward = total_reward / num_episodes
print(f"Average reward over {num_episodes} episodes: {average_reward}")

env.close()

# Wait until the simulation window is closed
while p.isConnected():
    p.getCameraImage(320, 200)




