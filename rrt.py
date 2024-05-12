import pybullet as p
import numpy as np
import random
import time

class Node:
    def __init__(self, position, node_id, parent_node):
        self.position = position
        self.node_id = node_id
        self.parent_node = parent_node
        self.visual_id = None
        self.child_nodes = []  # Lista para almacenar los hijos de este nodo

        self.line_id = None  # Almacena el identificador de la línea azul

    @staticmethod
    def find_node_by_id(tree, node_id):
        for node in tree:
            if node.node_id == node_id:
                return node
        return None

    @staticmethod
    def remove_node_by_id(tree, node_id):
        node_to_remove = Node.find_node_by_id(tree, node_id)
        if node_to_remove:
            # Eliminar todas las líneas azules asociadas al nodo
            if node_to_remove.line_id is not None:
                p.removeUserDebugItem(node_to_remove.line_id)
            tree.remove(node_to_remove)
            if node_to_remove.visual_id is not None:
                p.removeBody(node_to_remove.visual_id)
            return True
        else:
            return False
    
    @staticmethod
    def update_obstacles(obstacle_positions, obstacle_ids, bounds, obstacle_speed, obstacle_directions):
        for i in range(len(obstacle_positions)):
            # Verificar límites del plano para cambiar dirección
            if obstacle_positions[i][0] >= bounds[0][1]:
                obstacle_directions[i] = -1
            elif obstacle_positions[i][0] <= bounds[0][0]:
                obstacle_directions[i] = 1
            
            # Actualizar posición según la dirección y la velocidad
            obstacle_positions[i][0] += obstacle_speed * obstacle_directions[i]

            # Actualizar posición en PyBullet
            p.resetBasePositionAndOrientation(obstacle_ids[i], obstacle_positions[i], [0, 0, 0, 1])

    def remove_nodes_in_contact_with_obstacles(self, obstacle_ids):
        nodes_to_remove = []  # Lista para almacenar los nodos que deben eliminarse

        # Verificar si el nodo está en contacto con algún obstáculo
        for obstacle_id in obstacle_ids:
            aabb_min, aabb_max = p.getAABB(obstacle_id)
            if (aabb_min[0] <= self.position[0] <= aabb_max[0] and
                aabb_min[1] <= self.position[1] <= aabb_max[1] and
                aabb_min[2] <= self.position[2] <= aabb_max[2]):
                # El nodo está en contacto con este obstáculo, añadirlo a la lista de nodos a eliminar
                nodes_to_remove.append(self)
                break  # No es necesario seguir verificando con otros obstáculos

        # Recorrer los hijos del nodo actual y llamar recursivamente al método
        for child_node in self.child_nodes:
            nodes_to_remove.extend(child_node.remove_nodes_in_contact_with_obstacles(obstacle_ids))

        return nodes_to_remove


# Function to generate a random point on the ground
def generate_random_point_on_ground(bounds, ground_height):
    return [random.uniform(bounds[0][0], bounds[0][1]),
            random.uniform(bounds[1][0], bounds[1][1]),
            ground_height]


def find_nearest_node(tree, point):
    if not tree:
        # Si el árbol está vacío, devuelve el nodo inicial como el más cercano
        return Node(position=initial_point, node_id=0, parent_node=None)

    distances = [np.linalg.norm(np.array(point) - np.array(node.position)) for node in tree]
    nearest_node = tree[np.argmin(distances)]
    return nearest_node

# Function to apply the steer function between two points
def steer(from_node, to_point, max_distance, obstacle_ids, obstacle_ids1, step_size=0.1):
    direction = np.array(to_point) - np.array(from_node.position)
    distance = np.linalg.norm(direction)
    if distance > max_distance:
        direction = direction / distance * max_distance

    num_steps = int(distance / step_size)
    for i in range(num_steps):
        # Get the intermediate point
        intermediate_point = list(np.array(from_node.position) + (i+1) * step_size * direction / distance)

        # Check for collision with obstacles
        for obstacle_id in obstacle_ids:
            aabb_min, aabb_max = p.getAABB(obstacle_id)
            if (aabb_min[0] <= intermediate_point[0] <= aabb_max[0] and
                aabb_min[1] <= intermediate_point[1] <= aabb_max[1] and
                aabb_min[2] <= intermediate_point[2] <= aabb_max[2]):
                return None  # Collision detected, do not continue
        for obstacle_id in obstacle_ids1:
            aabb_min, aabb_max = p.getAABB(obstacle_id)
            if (aabb_min[0] <= intermediate_point[0] <= aabb_max[0] and
                aabb_min[1] <= intermediate_point[1] <= aabb_max[1] and
                aabb_min[2] <= intermediate_point[2] <= aabb_max[2]):
                return None  # Collision detected, do not continue

    return Node(position=list(np.array(from_node.position) + direction), node_id=len(rrt_tree), parent_node=from_node)

# Simulation parameters
bounds = [[-5, 5], [-5, 5]]  # Bounds of the plane in x, y
ground_height = 0.01  # Height of the ground
max_distance = 0.5  # Maximum distance between nodes
max_iterations = 1000  # Maximum number of iterations
goal_radius = 0.5  # Radius of the goal circle
obstacle_half_extents = [0.6, 0.6, 0.1]  # Half-lengths of the obstacle

# Initialize PyBullet simulation
# p.connect(p.DIRECT)
p.connect(p.GUI)
p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
plane = p.loadURDF("plane.urdf")
offset = [0, 0, 0]

# RRT tree
rrt_tree = []

# Generate the initial point at the top left corner
initial_point = [-5, 5, ground_height]
rrt_tree.append(Node(position=initial_point, node_id=0, parent_node=None))

# Place the obstacle at the center of the workspace square
# obstacle_position = [0, 0, 0.5]
# obstacle_id = p.createCollisionShape(p.GEOM_BOX, halfExtents=obstacle_half_extents)
# obstacle_visual_id = p.createVisualShape(p.GEOM_BOX, halfExtents=obstacle_half_extents, rgbaColor=[0.5, 0.5, 0.5, 1])
# obstacle_body_id = p.createMultiBody(basePosition=obstacle_position, baseCollisionShapeIndex=obstacle_id,
#                                       baseVisualShapeIndex=obstacle_visual_id)

obstacle_positions = [[0, 0, 0.1], [-2, -2, 0.1], [2, -1, 0.1], [-4, -1, 0.1], [3, -2, 0.1],[-5, 0, 0.1],[5, -5, 0.1],[3, -5, 0.1],[0, -5, 0.1],[3, -5, 0.1],[-3, -5, 0.1],
                    [-3, 3, 0.1] ,[-4, 3, 0.1], [ 4, 3, 0.1], [ 2, 3, 0.1] ,[ 2, 2, 0.1],[ 3, 3, 0.1],[-2, 3, 0.1] , [-3, 1, 0.1],[1, 1, 0.1],[-5, -5, 0.1],[5, -3, 0.1],
                    [-3, -3, 0.1],[-5, -3, 0.1],[-2, -3, 0.1],[-1, 5, 0.1],[5, 0, 0.1],[5, 1, 0.1],[5, 3, 0.1],[4, 5, 0.1],[4.5, 4.5, 0.1] ,[5, 5, 0.1],[5, -4, 0.1]]

obstacle_ids = []
for obstacle_position in obstacle_positions:
    obstacle_id = p.createCollisionShape(p.GEOM_BOX, halfExtents=obstacle_half_extents)
    obstacle_visual_id = p.createVisualShape(p.GEOM_BOX, halfExtents=obstacle_half_extents, rgbaColor=[0.5, 0.5, 0.5, 1])
    obstacle_body_id = p.createMultiBody(basePosition=obstacle_position, baseCollisionShapeIndex=obstacle_id, baseVisualShapeIndex=obstacle_visual_id)
    obstacle_ids.append(obstacle_body_id)

obstacle_positions1 =  [[0, 0, 0.1], [-2, -2, 0.1], [2, -1, 0.1], [-4, -1, 0.1], [3, -2, 0.1],[-5, 0, 0.1],[5, -5, 0.1],[3, -5, 0.1],[0, -5, 0.1],[3, -5, 0.1],[-3, -5, 0.1],
                    [-3, 3, 0.1] ,[-4, 3, 0.1], [ 4, 3, 0.1], [ 2, 3, 0.1] ,[ 2, 2, 0.1],[ 3, 3, 0.1],[-2, 3, 0.1] , [-3, 1, 0.1],[1, 1, 0.1],[-5, -5, 0.1],[5, -3, 0.1],
                    [-3, -3, 0.1],[-5, -3, 0.1],[-2, -3, 0.1],[-1, 5, 0.1],[5, 0, 0.1],[5, 1, 0.1],[5, 3, 0.1],[4, 5, 0.1],[4.5, 4.5, 0.1],[5, 5, 0.1], [5, -4, 0.1]]
obstacle_ids1 = []
for obstacle_position1 in obstacle_positions1:
    obstacle_id1 = p.createCollisionShape(p.GEOM_BOX, halfExtents=obstacle_half_extents)
    obstacle_visual_id1 = p.createVisualShape(p.GEOM_BOX, halfExtents=obstacle_half_extents, rgbaColor=[0.5, 0.5, 0.5, 1])
    obstacle_body_id1 = p.createMultiBody(basePosition=obstacle_position1, baseCollisionShapeIndex=obstacle_id1, baseVisualShapeIndex=obstacle_visual_id1)
    obstacle_ids1.append(obstacle_body_id1)

# Create the start point as a green circle on the ground
start_sphere_id = p.createVisualShape(shapeType=p.GEOM_SPHERE, radius=0.05, rgbaColor=[0, 1, 0, 1])
start_body_id = p.createMultiBody(baseVisualShapeIndex=start_sphere_id, basePosition=initial_point)

# Generate a random point on the ground for the goal
goal_point = [2, -3, ground_height]

# Create the goal circle as a green sphere on the ground
goal_sphere_id = p.createVisualShape(shapeType=p.GEOM_SPHERE, radius=goal_radius, rgbaColor=[0, 1, 0, 1])
goal_body_id = p.createMultiBody(baseVisualShapeIndex=goal_sphere_id, basePosition=goal_point)

# Visualizar la meta como una esfera verde
goal_sphere_id = p.createVisualShape(shapeType=p.GEOM_SPHERE, radius=goal_radius, rgbaColor=[0, 1, 0, 1])
goal_body_id = p.createMultiBody(baseVisualShapeIndex=goal_sphere_id, basePosition=goal_point)

# Velocidad de movimiento de los obstáculos
obstacle_speed = 0.0020

# Dirección inicial de movimiento de los obstáculos
obstacle_directions = [1] * len(obstacle_positions)

# Main simulation loop
iteration = 0
running = True
reached_goal = False
goal_node = None  # Variable para almacenar el nodo de la meta
while running:
    # Generar un punto aleatorio en el suelo
    random_point = generate_random_point_on_ground(bounds, ground_height)

    time.sleep(0.01)

    Node.update_obstacles(obstacle_positions, obstacle_ids, bounds, obstacle_speed, obstacle_directions)

    # Eliminar nodos en contacto con obstáculos
    nodes_to_remove = []
    for node in rrt_tree:
        nodes_to_remove.extend(node.remove_nodes_in_contact_with_obstacles(obstacle_ids))
    for node in nodes_to_remove:
        Node.remove_node_by_id(rrt_tree, node.node_id)


    # time.sleep(1)
    # Encontrar el nodo más cercano en el árbol RRT
    nearest_node = find_nearest_node(rrt_tree, random_point)

    # Aplicar la función steer para llegar desde el nodo más cercano hasta el punto aleatorio
    new_node = steer(nearest_node, random_point, max_distance, obstacle_ids, obstacle_ids1)

    # Si no se detecta colisión, agregar el nuevo punto al árbol RRT
    if new_node is not None:
        rrt_tree.append(new_node)

        # Visualizar la esfera que representa el nuevo punto
        new_node.visual_id = p.createVisualShape(shapeType=p.GEOM_SPHERE,
                                                 radius=0.02,
                                                 visualFramePosition=new_node.position,
                                                 rgbaColor=[1, 0, 0, 1])
        p.createMultiBody(baseVisualShapeIndex=new_node.visual_id)

        # Visualizar la línea conectando el nuevo punto con el nodo más cercano
        p.addUserDebugLine(nearest_node.position, new_node.position, [0, 0, 1], 2)

        # Comprobar si se alcanzó la meta
        if np.linalg.norm(np.array(new_node.position) - np.array(goal_point)) <= goal_radius:
            reached_goal = True
            goal_node = new_node  # Guardar el nodo de la meta
            break  # Salir del bucle
 
    # Condition to clear nodes and connectors after 20 iterations
    if iteration == 20:
        for node in rrt_tree:
            if node.visual_id is not None:
                p.removeBody(node.visual_id)
        rrt_tree.clear()

    iteration += 1
    if iteration >= max_iterations:
        running = False

if reached_goal:
    path_points = []  # Lista para almacenar los puntos del camino
    current_node = goal_node
    path_points.append(current_node.position)
 
    while current_node is not None:
        path_points.append(current_node.position)
        current_node = current_node.parent_node
    # print(path_points)

    # Invertir la lista de puntos para que estén en orden desde el inicio hasta la meta
    path_points.reverse()

    # Dibujar una línea roja conectando los puntos del camino
    for i in range(len(path_points) - 1):
        p.addUserDebugLine(path_points[i], path_points[i + 1], [1, 0, 0], lineWidth=10)
    # Desconectar PyBullet y cerrar la simulación

    time.sleep(5)
    p.disconnect()
        


# Wait until the simulation window is closed
while p.isConnected():
    p.getCameraImage(320, 200)
