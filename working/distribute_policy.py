import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import random
import gc
import copy
from pack import Pack, DummyPack
from robot import Robot
from task import Task
from SETTING import NODE_DISTANCES, DEVICE
from Net import Net

class DistributePolicy:
  def __init__(self):
    pass
    
  def reset(self, pack_queue:list[Pack], robots:list[Robot]):
    pass
    
  def distribute(self, pack_queue:list[Pack], robots:list[Robot])->None:
    pass

class NaiveDistributePolicy(DistributePolicy):
  def __init__(self):
    super().__init__()
    
  def distribute(self, pack_queue:list[Pack], robots:list[Robot])->None:
    len_robots = len(robots)
    robot_id = 0
    for pack in pack_queue:
      robot = robots[robot_id]
      robot.pack_list.append(pack)
      pack.robot_id = robot_id
            
      robot_id = 0 if robot_id + 1 == len_robots else robot_id + 1

class DQN_DistributePolicy(DistributePolicy):
  def __init__(self):
    super().__init__()
    self.epsilon = 1.0
    self.epsilon_decay = 0.995
    self.epsilon_min = 0.01
    self.learning_rate = 0.001
    self.batch_size = 32
    self.memory_size = 2048 * 32
    self.memory_idx = 0
    self.criterion = nn.MSELoss()
    self.memory = []
    self.gamma = 0.95
    self.loss = None
    self.first_try = True
    self.node_distances = NODE_DISTANCES
    
  def reset(self, pack_queue, robots):
    # 一応書いたが、ネットワーク系のパラメータは本来あまりここに書くべきじゃない気がする
    output_size = len(robots)
    self.input_array[:] = self.make_observation(pack_queue[0], robots, pack_queue)[:]
    self.first_try = False
        
    input_size = len(self.input_array)
    hidden_size = 128
    self.model = Net(input_size, hidden_size, output_size).to(device=DEVICE)
    self.target_model = Net(input_size, hidden_size, output_size).to(device=DEVICE)
    self.update_target_model()
    self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        
    self.action_size = len(robots)
    self.input_size = input_size
    
  def save_model(self, filepath):
    torch.save(self.model.state_dict(), filepath)
    
  def load_model(self, filepath):
    self.model.load_state_dict(torch.load(filepath))
    self.update_target_model()
    
  def distribute(self, pack_queue:list[Pack], robots:list[Robot]):
    for pack in pack_queue:
      _ = self.step_distribute(pack, robots, pack_queue)
    
  def act(self, pack:Pack, robots:list[Robot], pack_queue:list[Pack]):
    if np.random.rand() <= self.epsilon:
      return random.randrange(self.action_size)
    else:
      return self.step_distribute(pack, robots, pack_queue)
    
  def candidate_distribute(self, pack:Pack, robots:list[Robot], pack_queue:list[Pack]):
    observation = self.make_observation(pack, robots, pack_queue)
    an_input = torch.tensor(observation).to(device=DEVICE)
    action_robot_id = torch.argmax(self.model(an_input)).cpu().numpy()
    return action_robot_id
    
  def step_distribute(self, pack:Pack, robots:list[Robot], pack_queue:list[Pack]):
    robot_id = self.candidate_distribute(pack, robots, pack_queue)
    robots[robot_id].pack_list.append(pack)
    pack.robot_id = robot_id
    pack.current_node = pack.target_node
    return robot_id
    
  #@profile
  def train(self):
    # minibatchを使ってはいるが、実際には一つ一つ学習している。
    # batchごとにしたい場合は、最初にobservationへの変換をまとめて行ってからやるようにする
    # 実装的にみにくいので、いったんここまで
        
    if len(self.memory) < self.batch_size:
      return
    mini_batch = random.sample(self.memory, self.batch_size)
        
    for state, action, reward, next_state, terminated, truncated, info in mini_batch:
      observation = torch.tensor(self.make_observation(*state)).to(device=DEVICE).unsqueeze(0)
      next_observation = torch.tensor(self.make_observation(*next_state)).to(device=DEVICE).unsqueeze(0)
      reward = torch.tensor(reward).to(device=DEVICE)
            
      with torch.no_grad():
        target_action = torch.argmax(self.model(next_observation))
        target = reward + (1-int(terminated))*self.gamma*self.target_model(next_observation)[0][target_action]

      q_values = self.model(observation)
      target_f = q_values.clone().detach()
      target_f[0][action] = target

      self.optimizer.zero_grad()
      loss = self.criterion(q_values, target_f)
      loss.backward()
      self.optimizer.step()
      self.loss = loss.item()
        
    if self.epsilon > self.epsilon_min:
      self.epsilon *= self.epsilon_decay
    
  def update_target_model(self):
    self.target_model.load_state_dict(self.model.state_dict())
    
  #@profile
  def remember(self, state, action, reward, next_state, terminated, truncated, info):
    if len(self.memory) < self.memory_size:
      self.memory.append(copy.deepcopy((state, action, reward, next_state, terminated, truncated, info)))
    else:
      self.memory[self.memory_idx] = copy.deepcopy((state, action, reward, next_state, terminated, truncated, info))
      self.memory_idx = 0 if self.memory_idx == self.memory_size else self.memory_idx + 1
    
  #@profile
  def make_observation(self, pack:Pack, robots:list[Robot], pack_queue:list[Pack]): 
    """
    可変引数な形状を固定ベクトルに変える  

    状況に応じて、DummyPackの中身を変更したければ都合のいい範囲で変更して良い
    robotのノード、packのノードに関して相対位置か近さの指標を考える
    pack_queueは可能な範囲で情報を入れる
    現在の観測情報は4次元に固定しているが、長さはなんでもいいので固定すること
    """
    start_node = pack.start_node
    target_node = pack.target_node
    is_dummy_pack = isinstance(pack, DummyPack)
        
    is_dummy_info = [float(is_dummy_pack)]    
        
    robot_info = []
    fast_time = np.inf
    for robot in robots:
      fast_time = min(fast_time, robot.queue_time_from_start(self.node_distances))
            
    for robot in robots:
      robot_info.append(0.0 if is_dummy_pack else 1/self.node_distances[robot.current_node, start_node] if robot.current_node != start_node else 1.0)
      robot_info.append(2/(fast_time-robot.queue_time_from_start(self.node_distances)) if fast_time!=robot.queue_time_from_start(self.node_distances) else 1.0)
        
    pack_info = []
    for other_pack in pack_queue:
      pack_info.append(1.0 if other_pack.start_node == target_node else 1/self.node_distances[target_node, other_pack.start_node])
      pack_info.append(float(pack.robot_id == -1))
      for robot in robots:
        pack_info.append(float(robot.id==pack.robot_id))
                
        
    ret_list = robot_info + pack_info + is_dummy_info
    if self.first_try:
      self.input_array= np.array(ret_list, dtype=np.float32)[:]
    else:
      self.input_array = np.array(ret_list, dtype=np.float32)
        
    return self.input_array
        
if __name__ == "__main__":
  robots = [Robot(i, i) for i in range(2)]
  pack_list = [Pack(i, [Task(3*i, 3*i, 3*i+1, 0, NODE_DISTANCES), Task(3*i, 3*i+1, 3*i+2, 0, NODE_DISTANCES), Task(3*i, 3*i+2, 3*i, 0, NODE_DISTANCES)]) for i in range(3)]
  #policy = NaiveDistributePolicy()
  policy = DQN_DistributePolicy()
  policy.reset(pack_list, robots)
  policy.distribute(pack_list, robots)
  print(robots)
  print(pack_list)