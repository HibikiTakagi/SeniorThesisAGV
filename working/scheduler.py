import numpy as np
from numpy.typing import NDArray
from task import Task
from robot import Robot
from pack_policy import PackPolicy, NaivePackPolicy
from distribute_policy import DistributePolicy, NaiveDistributePolicy, DQN_DistributePolicy
from test_map import TEST_NODE_DISTANCES

from SETTING import PACK_POLICY_NAME, DISTRIBUTE_POLICY_NAME

PACK_POLICY_DICT = {
  "Naive" : NaivePackPolicy
}

DISTRIBUTE_POLICY_DICT = {
  "Naive" : NaiveDistributePolicy,
  "DQN"   : DQN_DistributePolicy,
}

PACK_POLICY = PACK_POLICY_DICT[PACK_POLICY_NAME]
DISTRIBUTE_POLICY = DISTRIBUTE_POLICY_DICT[DISTRIBUTE_POLICY_NAME]

class Scheduler:
  def __init__(self, task_queue: list[Task], robots: list[Robot]):
    self.task_queue = task_queue
    self.pack_policy = PACK_POLICY()
    self.distribute_policy = DISTRIBUTE_POLICY()
    self.pack_queue = self.pack_policy.pack(self.task_queue)
    self.robots = robots
    self.distribute_policy.reset(self.pack_queue, self.robots)
  
  def reset(self, task_queue: list[Task], robots: list[Robot]):
    self.task_queue = task_queue
    self.pack_queue = self.pack_policy.pack(self.task_queue)
    self.robots = robots
    # エピソード間でrobots数が変わることは現状想定していない（それはネットワークが変わることを意味する）
    # あくまでrobot内の状態をもとに戻すことをここでは行っている
  
  def pack(self):
    """
    このかんすうでやること
    →task_queueが渡されるのでpackに変換してpackのキューをつくること
    """
    self.pack_queue = self.pack_policy.pack(self.task_queue)
  
  def distribute(self):
    """
    このかんすうでやること
    →どのロボットに入れるかを考えてpackのrobot_idを張り替える
    順番はどうする？
    →Robotにpack_queueをつくってpack_idを突っ込んで順番とする
    """
    self.distribute_policy.distribute(self.pack_queue, self.robots)
  
  def dump_schedules(self, graph:NDArray[np.int32]):
    resources = [robot.name for robot in self.robots]
    tasks = []
    resource_allocation = []
    start_times = []
    durations = []
    
    robots_current_position = [robot.start_node for robot in self.robots]
    robots_current_end_time = [0 for _ in self.robots]
    robot_pack_count = [0 for _ in self.robots]
    
    for robot in self.robots:
      r_id = robot.id
      for pack in robot.pack_list:
        position = robots_current_position[r_id]
        end_time = robots_current_end_time[r_id]
        destination = pack.start_node
        
        move_task_name = f'M{r_id}-{robot_pack_count[r_id]}'
        move_task_resource_allocation = self.robots[r_id].name
        move_task_start_times = end_time
        move_task_durations = graph[position, destination]
        
        tasks.append(move_task_name)
        resource_allocation.append(move_task_resource_allocation)
        start_times.append(move_task_start_times)
        durations.append(move_task_durations)
        
        robot_pack_count[r_id] += 1
        end_time += move_task_durations
        position = pack.start_node
        destination = pack.target_node
        pack_durations = pack.time_assume(graph)
        
        tasks.append(pack.name)
        resource_allocation.append(self.robots[r_id].name)
        start_times.append(end_time)
        durations.append(pack_durations)
        
        robots_current_position[r_id] = pack.target_node
        robots_current_end_time[r_id] = end_time + pack_durations
    
    """
    for pack in self.pack_queue:
      r_id = pack.robot_id
      position = robots_current_position[r_id]
      end_time = robots_current_end_time[r_id]
      destination = pack.start_node
      
      move_task_name = f'M{r_id}-{robot_pack_count[r_id]}'
      move_task_resource_allocation = self.robots[r_id].name
      move_task_start_times = end_time
      move_task_durations = graph[position, destination]
      
      tasks.append(move_task_name)
      resource_allocation.append(move_task_resource_allocation)
      start_times.append(move_task_start_times)
      durations.append(move_task_durations)
      
      robot_pack_count[r_id] += 1
      end_time += move_task_durations
      position = pack.start_node
      destination = pack.target_node
      pack_durations = pack.time_assume(graph)
      
      tasks.append(pack.name)
      resource_allocation.append(self.robots[r_id].name)
      start_times.append(end_time)
      durations.append(pack_durations)
      
      robots_current_position[r_id] = pack.target_node
      robots_current_end_time[r_id] = end_time + pack_durations

    Returns:
      _type_: _description_
    """
    
    end_times = [start + duration for start, duration in zip(start_times, durations)]
    return resources, tasks, resource_allocation, start_times, durations, end_times

if __name__ == "__main__":
  robots = [Robot(i, i) for i in range(2)]
  print(robots)
  task_list = [
    Task(i,i,i+1,0,TEST_NODE_DISTANCES) for i in range(7)
  ]
  scheduler = Scheduler(task_list, robots)
  scheduler.pack()
  scheduler.distribute()
  resources, tasks, resource_allocation, start_times, durations, end_time = scheduler.dump_schedules(TEST_NODE_DISTANCES)
  print(f'resource:{resources}')
  print(f'tasks:{tasks}')
  print(f'allocation:{resource_allocation}')
  print(f'start_time:{start_times}')
  print(f'duration:{durations}')
  print(f'end_time:{end_time}')
  print(f'scheduler:{scheduler.pack_queue}')