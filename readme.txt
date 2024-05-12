To run the basic RRT to generate a feasible path for the turtlebot, run the command below:
	python rrt.py

To run the RRT for path-planning with the RL policy for path-following, run the command below:
	python rrt_policy.py



Note: once the feasible path is generated, the policy to follow the path. We are using a policy model we trained and saved in the project folder as pybullet_rl_model

To train the policy from scratch, run the command below:
	python train.py
