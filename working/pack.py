import numpy as np
from numpy.typing import NDArray
from task import Task, DummyTask
from test_map import TEST_NODE_DISTANCES

class Pack:
  max_len_task = 3
    
  def __init__(self, p_id, task_list: list[Task]):
    self.id = p_id
    self.task_list = task_list
    self.robot_id = -1
    assert len(self.task_list) <= self.max_len_task
    
  def __repr__(self):
    task_id_list = []
    for task in self.task_list:
      task_id_list.append(task.id)
    return f'Pack(id:{self.id}, task_list:{task_id_list}, robot_id:{self.robot_id})'
    
  @property
  def name(self):
    return f'P{self.id}:{self.start_node}-{self.target_node}'
    
  @property
  def is_complete(self):
    flag = True
    for task in self.task_list:
      flag = flag and task.is_complete
    return flag
    
  @property
  def start_node(self):
    return self.task_list[0].start_node
    
  @property
  def target_node(self):
    return self.task_list[-1].target_node
    
  @property
  def node_list(self):
    node_list = []
    for task in self.task_list:
      node_list.append(task.start_node)
      node_list.append(task.target_node)
    return node_list
    
  @property
  def is_empty(self):
    return len(self.task_list)==0
    
  def set_pack(self, task_list:list[Task]):
    self.task_list = task_list
    
  def set_robot_id(self, robot_id:int):
    self.robot_id = robot_id
    
  def mean_waiting_time(self, current_time:int):
    tmp = 0
    for task in self.task_list:
      tmp += task.get_waiting_time(current_time)
    return tmp/len(self.task_list)
    
  def worst_waiting_time(self, current_time:int):
    tmp = 0
    for task in self.task_list:
      tmp = max(tmp, task.get_waiting_time(current_time))
    return tmp
    
  def sort_pack(self, sort_idx:list[int]):
    assert len(sort_idx) == len(self.task_list) and len(set(sort_idx)) == len(sort_idx)
    rst_list = []
    for idx in sort_idx:
      rst_list.append(self.task_list[idx])
    return rst_list
    
  def time_assume(self, graph:NDArray[np.int32]):
    before_target_node = None
    initial_flag = True
    time = 0
    for task in self.task_list:
      if initial_flag:
        initial_flag = False
      else:
        time += graph[before_target_node, task.start_node]
        pass
      time += graph[task.start_node, task.target_node]
      before_target_node = task.target_node
    return time

class DummyPack(Pack):
  def __init__(self, p_id, task_list, graph):
    dummy_task_list = [DummyTask(0,0,0,graph) for _ in range(self.max_len_task)]
    super().__init__(p_id, dummy_task_list)
    
if __name__ == "__main__":
  pack_a = Pack(0, [Task(0, 1, 8, 0, TEST_NODE_DISTANCES), Task(1, 2, 4, 2, TEST_NODE_DISTANCES), Task(2, 3, 5, 5, TEST_NODE_DISTANCES)])
  print(pack_a)
  print(pack_a.mean_waiting_time(100))
  print(pack_a.worst_waiting_time(100))
  print(pack_a.is_complete)
  print(pack_a.sort_pack([1,2,0]))
  print(pack_a.time_assume(TEST_NODE_DISTANCES))