import numpy as np
import pygame
import time
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import traci
import os
import sys
import random
from collections import deque
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import xml.etree.ElementTree as ET

# Add SUMO to Python path
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")

# RL Agent for Traffic Light Control
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0   # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()
        
    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        
    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])
    
    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = reward + self.gamma * np.amax(self.model.predict(next_state)[0])
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

# Traffic Light Simulator
class TrafficLightSimulator:
    def __init__(self, use_gui = True):
        # Initialize SUMO
        self.sumo_binary = "sumo-gui" if use_gui else "sumo"
        self.sumo_cmd = [self.sumo_binary, "-c", "intersection.sumocfg"]
        
        # State and action definitions
        self.state_size = 8  # Waiting vehicles in each direction
        self.action_size = 4  # 4 phases - NS, EW, SN, WE
        
        # Create RL agent
        self.agent = DQNAgent(self.state_size, self.action_size)
        
        # Traffic light phases (mapping from action to SUMO phase)
        self.phases = {
            0: "GGGgrrrrrrrrrrrr",  # North-South
            1: "rrrrGGGgrrrrrrrr",  # East-West
            2: "rrrrrrrrGGGgrrrr",  # South-North
            3: "rrrrrrrrrrrrGGGg"   # West-East
        }
        
        # Min and max phase duration
        self.min_phase_duration = 5
        self.max_phase_duration = 30
        
        # Current state
        self.current_phase = 0
        self.current_phase_duration = 0
        self.yellow_phase_active = False
        self.yellow_time_left = 0
        
        # Performance metrics
        self.total_waiting_time = 0
        self.avg_waiting_times = []
        
        # Parse node and edge data for visualization
        self.parse_network_data()
        
    def parse_network_data(self):
        # Extract node and edge data from the provided XML
        self.nodes = {}
        self.edges = {}
        
        # Parse nodes from the provided XML string
        nodes_root = ET.fromstring("<nodes>\n    <node id=\"1\" x=\"0\" y=\"0\" type=\"traffic_light\"/>\n    <node id=\"2\" x=\"100\" y=\"0\"/>\n    <node id=\"3\" x=\"0\" y=\"100\"/>\n    <node id=\"4\" x=\"-100\" y=\"0\"/>\n    <node id=\"5\" x=\"0\" y=\"-100\"/>\n</nodes>")
        
        for node in nodes_root.findall('node'):
            node_id = node.get('id')
            x = float(node.get('x'))
            y = float(node.get('y'))
            node_type = node.get('type', '')
            self.nodes[node_id] = {'x': x, 'y': y, 'type': node_type}
        
        # Parse edges from the provided XML string
        edges_root = ET.fromstring("<edges>\n    <!-- North-South -->\n    <edge id=\"3to1\" from=\"3\" to=\"1\" numLanes=\"2\" speed=\"13.89\"/>\n    <edge id=\"1to5\" from=\"1\" to=\"5\" numLanes=\"2\" speed=\"13.89\"/>\n    \n    <!-- South-North -->\n    <edge id=\"5to1\" from=\"5\" to=\"1\" numLanes=\"2\" speed=\"13.89\"/>\n    <edge id=\"1to3\" from=\"1\" to=\"3\" numLanes=\"2\" speed=\"13.89\"/>\n    \n    <!-- East-West -->\n    <edge id=\"2to1\" from=\"2\" to=\"1\" numLanes=\"2\" speed=\"13.89\"/>\n    <edge id=\"1to4\" from=\"1\" to=\"4\" numLanes=\"2\" speed=\"13.89\"/>\n    \n    <!-- West-East -->\n    <edge id=\"4to1\" from=\"4\" to=\"1\" numLanes=\"2\" speed=\"13.89\"/>\n    <edge id=\"1to2\" from=\"1\" to=\"2\" numLanes=\"2\" speed=\"13.89\"/>\n</edges>")
        
        for edge in edges_root.findall('edge'):
            edge_id = edge.get('id')
            from_node = edge.get('from')
            to_node = edge.get('to')
            num_lanes = int(edge.get('numLanes'))
            speed = float(edge.get('speed'))
            self.edges[edge_id] = {
                'from': from_node,
                'to': to_node,
                'numLanes': num_lanes,
                'speed': speed
            }
    
    def get_state(self):
        # Observe traffic state
        state = np.zeros(self.state_size)
        
        # Count waiting vehicles for each incoming edge
        incoming_edges = ['3to1', '2to1', '5to1', '4to1']
        lane_indices = [0, 1, 2, 3, 4, 5, 6, 7]
        
        for i, edge in enumerate(incoming_edges):
            for lane in range(self.edges[edge]['numLanes']):
                lane_id = f"{edge}_{lane}"
                waiting_vehicles = traci.lane.getLastStepHaltingNumber(lane_id)
                queue_length = traci.lane.getLastStepVehicleNumber(lane_id)
                mean_speed = traci.lane.getLastStepMeanSpeed(lane_id)
                
                # Waiting vehicles have more influence on state
                state[i*2] += waiting_vehicles
                # Queue length and speed also provide information
                state[i*2+1] = queue_length - waiting_vehicles
        
        return state
    
    def step(self, action):
        # Apply action
        if not self.yellow_phase_active:
            # If we change phase, first set yellow phase
            if action != self.current_phase:
                # Set yellow phase
                yellow_state = self.get_yellow_state(self.phases[self.current_phase])
                traci.trafficlight.setRedYellowGreenState("1", yellow_state)
                self.yellow_phase_active = True
                self.yellow_time_left = 3
            else:
                # Keep current phase
                traci.trafficlight.setRedYellowGreenState("1", self.phases[action])
                self.current_phase = action
                self.current_phase_duration += 1
        else:
            # In yellow phase
            self.yellow_time_left -= 1
            if self.yellow_time_left <= 0:
                # Yellow phase complete, set new phase
                traci.trafficlight.setRedYellowGreenState("1", self.phases[action])
                self.current_phase = action
                self.current_phase_duration = 0
                self.yellow_phase_active = False
        
        # Let the simulation run for one step
        traci.simulationStep()
        
        # Get new state and calculate reward
        next_state = self.get_state()
        reward = self.calculate_reward(next_state)
        
        # Check if simulation is done
        done = False
        if traci.simulation.getMinExpectedNumber() <= 0:
            done = True
        
        return next_state, reward, done
    
    def get_yellow_state(self, current_state):
        """Convert green signals to yellow"""
        yellow_state = ""
        for c in current_state:
            if c == 'G':
                yellow_state += 'y'
            else:
                yellow_state += c
        return yellow_state
    
    def calculate_reward(self, state):
        # Calculate reward based on reduction in waiting vehicles
        total_waiting = sum(state[::2])  # Sum only waiting vehicle counts
        
        # Calculate reward: negative of waiting vehicles
        reward = -total_waiting
        
        # Penalize very long phases
        if self.current_phase_duration > self.max_phase_duration:
            reward -= 10
        
        # Bonus for short queues
        if total_waiting < 5:
            reward += 5
            
        # Track performance
        self.total_waiting_time += total_waiting
        self.avg_waiting_times.append(total_waiting)
        
        return reward
    
    def reset(self):
        # End current simulation and start a new one
        traci.close()
        traci.start(self.sumo_cmd)
        
        # Reset state variables
        self.current_phase = 0
        self.current_phase_duration = 0
        self.yellow_phase_active = False
        self.yellow_time_left = 0
        self.total_waiting_time = 0
        self.avg_waiting_times = []
        
        # Return initial state
        return self.get_state()

# GUI for traffic light visualization
class TrafficLightGUI:
    def __init__(self, simulator):
        self.simulator = simulator
        
        # Pygame initialization
        pygame.init()
        self.width, self.height = 800, 600
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Traffic Light Reinforcement Learning Controller")
        
        # Colors
        self.colors = {
            'red': (255, 0, 0),
            'yellow': (255, 255, 0),
            'green': (0, 255, 0),
            'black': (0, 0, 0),
            'white': (255, 255, 255),
            'gray': (150, 150, 150),
            'blue': (0, 0, 255),
            'dark_gray': (50, 50, 50)
        }
        
        # Fonts
        self.font = pygame.font.SysFont('Arial', 20)
        self.title_font = pygame.font.SysFont('Arial', 24, bold=True)
        
        # Road dimensions
        self.road_width = 40
        self.road_color = self.colors['gray']
        self.lane_color = self.colors['white']
        
        # Traffic light dimensions
        self.tl_radius = 10
        self.tl_spacing = 25
        self.tl_pos = {
            'N': {'x': self.width//2, 'y': self.height//2 - 60},
            'E': {'x': self.width//2 + 60, 'y': self.height//2},
            'S': {'x': self.width//2, 'y': self.height//2 + 60},
            'W': {'x': self.width//2 - 60, 'y': self.height//2}
        }
        
        # Performance chart settings
        self.chart_rect = pygame.Rect(self.width - 300, 10, 290, 200)
        self.chart_data = []
        self.max_chart_points = 100
        
        # RL metrics panel
        self.metrics_rect = pygame.Rect(10, 10, 280, 200)
        
        # Clock for controlling framerate
        self.clock = pygame.time.Clock()
        self.fps = 30
        
        # Simulation time
        self.sim_time = 0
        self.last_action_time = 0
        self.action_duration = 0
        
    def get_light_colors(self, phase_string):
        # Map phase string to traffic light colors
        colors = {
            'N': self.colors['red'],
            'E': self.colors['red'],
            'S': self.colors['red'],
            'W': self.colors['red']
        }
        
        if 'G' in phase_string[:4]:  # North direction
            colors['N'] = self.colors['green']
        elif 'y' in phase_string[:4]:
            colors['N'] = self.colors['yellow']
            
        if 'G' in phase_string[4:8]:  # East direction
            colors['E'] = self.colors['green']
        elif 'y' in phase_string[4:8]:
            colors['E'] = self.colors['yellow']
            
        if 'G' in phase_string[8:12]:  # South direction
            colors['S'] = self.colors['green']
        elif 'y' in phase_string[8:12]:
            colors['S'] = self.colors['yellow']
            
        if 'G' in phase_string[12:]:  # West direction
            colors['W'] = self.colors['green']
        elif 'y' in phase_string[12:]:
            colors['W'] = self.colors['yellow']
            
        return colors
    
    def draw_traffic_lights(self, phase_string):
        # Draw the intersection
        pygame.draw.rect(self.screen, self.road_color, 
                         (self.width//2 - self.road_width, 
                          self.height//2 - self.road_width, 
                          self.road_width*2, self.road_width*2))
        
        # Draw roads
        # Vertical road
        pygame.draw.rect(self.screen, self.road_color, 
                         (self.width//2 - self.road_width//2, 
                          0, self.road_width, self.height))
        
        # Horizontal road
        pygame.draw.rect(self.screen, self.road_color, 
                         (0, self.height//2 - self.road_width//2, 
                          self.width, self.road_width))
        
        # Draw lane markings
        for i in range(10, self.height, 30):
            if i < self.height//2 - self.road_width//2 or i > self.height//2 + self.road_width//2:
                pygame.draw.rect(self.screen, self.lane_color, 
                                 (self.width//2, i, 10, 20))
        for i in range(10, self.width, 30):
            if i < self.width//2 - self.road_width//2 or i > self.width//2 + self.road_width//2:
                pygame.draw.rect(self.screen, self.lane_color, 
                                 (i, self.height//2, 20, 10))
        
        # Get traffic light colors
        colors = self.get_light_colors(phase_string)
        
        # Draw traffic lights
        for direction, pos in self.tl_pos.items():
            pygame.draw.circle(self.screen, colors[direction], (pos['x'], pos['y']), self.tl_radius)
            
            # Draw direction label
            direction_label = self.font.render(direction, True, self.colors['white'])
            self.screen.blit(direction_label, 
                            (pos['x'] - direction_label.get_width()//2, 
                             pos['y'] - direction_label.get_height()//2))
    
    def draw_performance_chart(self):
        # Draw chart background
        pygame.draw.rect(self.screen, self.colors['dark_gray'], self.chart_rect)
        pygame.draw.rect(self.screen, self.colors['white'], self.chart_rect, 2)
        
        # Draw chart title
        title = self.title_font.render("Waiting Vehicles Over Time", True, self.colors['white'])
        self.screen.blit(title, (self.chart_rect.centerx - title.get_width()//2, self.chart_rect.y + 5))
        
        # Draw chart data
        if len(self.chart_data) > 1:
            max_value = max(self.chart_data) if max(self.chart_data) > 0 else 1
            data_points = min(len(self.chart_data), self.max_chart_points)
            points = []
            
            for i in range(data_points):
                idx = len(self.chart_data) - data_points + i
                x = self.chart_rect.x + 10 + i * (self.chart_rect.width - 20) / (data_points - 1 if data_points > 1 else 1)
                y = self.chart_rect.bottom - 30 - (self.chart_data[idx] / max_value) * (self.chart_rect.height - 60)
                points.append((x, y))
                
            if len(points) > 1:
                pygame.draw.lines(self.screen, self.colors['green'], False, points, 2)
        
        # Draw axes
        pygame.draw.line(self.screen, self.colors['white'], 
                         (self.chart_rect.x + 10, self.chart_rect.bottom - 30),
                         (self.chart_rect.right - 10, self.chart_rect.bottom - 30), 2)
        pygame.draw.line(self.screen, self.colors['white'], 
                         (self.chart_rect.x + 10, self.chart_rect.top + 30),
                         (self.chart_rect.x + 10, self.chart_rect.bottom - 30), 2)
        
        # X-axis label
        x_label = self.font.render("Time", True, self.colors['white'])
        self.screen.blit(x_label, (self.chart_rect.centerx - x_label.get_width()//2, self.chart_rect.bottom - 25))
        
        # Y-axis label
        y_label = self.font.render("Waiting Vehicles", True, self.colors['white'])
        y_label = pygame.transform.rotate(y_label, 90)
        self.screen.blit(y_label, (self.chart_rect.x + 5, self.chart_rect.centery - y_label.get_width()//2))
    
    def draw_rl_metrics(self, epsilon, reward, action, phase_duration):
        # Draw metrics background
        pygame.draw.rect(self.screen, self.colors['dark_gray'], self.metrics_rect)
        pygame.draw.rect(self.screen, self.colors['white'], self.metrics_rect, 2)
        
        # Draw metrics title
        title = self.title_font.render("RL Traffic Controller", True, self.colors['white'])
        self.screen.blit(title, (self.metrics_rect.centerx - title.get_width()//2, self.metrics_rect.y + 5))
        
        # Draw metrics
        y_pos = self.metrics_rect.y + 40
        spacing = 30
        
        metrics = [
            f"Simulation Time: {self.sim_time:.1f}s",
            f"Exploration (ε): {epsilon:.3f}",
            f"Last Reward: {reward:.1f}",
            f"Current Action: Phase {action}",
            f"Phase Duration: {phase_duration}s",
        ]
        
        for metric in metrics:
            text = self.font.render(metric, True, self.colors['white'])
            self.screen.blit(text, (self.metrics_rect.x + 10, y_pos))
            y_pos += spacing
    
    def update(self, phase_string, state, epsilon, reward, action, phase_duration):
        # Clear screen
        self.screen.fill(self.colors['black'])
        
        # Draw traffic lights
        self.draw_traffic_lights(phase_string)
        
        # Update and draw performance chart
        avg_waiting = sum(state[::2])  # Sum waiting vehicles
        self.chart_data.append(avg_waiting)
        if len(self.chart_data) > self.max_chart_points * 2:
            self.chart_data = self.chart_data[-self.max_chart_points*2:]
        self.draw_performance_chart()
        
        # Draw RL metrics
        self.draw_rl_metrics(epsilon, reward, action, phase_duration)
        
        # Update display
        pygame.display.flip()
        
        # Control framerate
        self.clock.tick(self.fps)
        
        # Check for quit event
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return False
            
        return True

# Main function
def main():
    # Parameters
    episodes = 100
    batch_size = 32
    gui = True  # Use GUI
    
    # Create simulator
    simulator = TrafficLightSimulator(gui = gui)
    
    # Create GUI if enabled
    if gui:
        gui = TrafficLightGUI(simulator)
    
    # Training loop
    for episode in range(episodes):
        # Start SUMO
        traci.start(simulator.sumo_cmd)
        
        # Reset simulator
        state = simulator.get_state()
        state = np.reshape(state, [1, simulator.state_size])
        
        # Variables for this episode
        done = False
        step_count = 0
        total_reward = 0
        last_reward = 0
        
        while not done:
            # Choose action
            action = simulator.agent.act(state)
            
            # Take action and observe next state and reward
            next_state, reward, done = simulator.step(action)
            next_state = np.reshape(next_state, [1, simulator.state_size])
            
            # Store in memory
            simulator.agent.remember(state, action, reward, next_state, done)
            
            # Learn
            if len(simulator.agent.memory) > batch_size:
                simulator.agent.replay(batch_size)
            
            # Update state
            state = next_state
            last_reward = reward
            total_reward += reward
            step_count += 1
            
            # Update GUI
            phase_string = ""
            if simulator.yellow_phase_active:
                phase_string = simulator.get_yellow_state(simulator.phases[simulator.current_phase])
            else:
                phase_string = simulator.phases[simulator.current_phase]
            
            if gui:
                gui.sim_time += 1/gui.fps
                continue_sim = gui.update(
                    phase_string, 
                    np.reshape(state, simulator.state_size),
                    simulator.agent.epsilon,
                    last_reward,
                    simulator.current_phase,
                    simulator.current_phase_duration
                )
                if not continue_sim:
                    traci.close()
                    return
            
            # Small delay to see what's happening
            if gui:
                time.sleep(0.01)
        
        # End of episode
        print(f"Episode: {episode+1}/{episodes}, Steps: {step_count}, " \
              f"Total Reward: {total_reward}, Epsilon: {simulator.agent.epsilon:.4f}")
        
        # Close SUMO
        traci.close()
    
    # Cleanup
    if gui:
        pygame.quit()

if __name__ == "__main__":
    main()