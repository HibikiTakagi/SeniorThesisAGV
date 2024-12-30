import copy
import random
import numpy as np
from robot import Robot
from task import Task
from pack import DummyPack, Pack
from task import Task, DummyTask
from SETTING import NODE_DISTANCES, ROBOTS, TASK_LIST, LEN_NODE
from scheduler import Scheduler
from utils import extract_circular_sublist

class JobSchedulerEnv:
  def __init__(self):
    self.robots = ROBOTS
    task_list = TASK_LIST
    self.scheduler = Scheduler(task_queue=task_list, robots=self.robots)
    pass

  #@profile
  def reset(self):
    for robot in self.robots:
      #print(f"bpack:{robot.pack_list}")
      robot.reset()
      #print(f"apack{robot.pack_list}")
    task_list = random.sample(TASK_LIST, len(TASK_LIST))
    self.scheduler.reset(task_list, self.robots)
    self.current_pack_id = 0
    #state = (self.scheduler.pack_queue[self.current_pack_id], self.scheduler.robots, extract_circular_sublist(self.scheduler.pack_queue, self.current_pack_id))
    state = (self.scheduler.pack_queue[self.current_pack_id], self.scheduler.robots, self.scheduler.pack_queue)        
    info = None
    self.num_step = 0

    self.current_state_viewer = copy.deepcopy(state)
    return state, info

  def render(self, mode='human'):
    """
    理想：現時点での状態を画像で表示する
    →未実装
    """
    return None

  def close(self):
    pass

  def seed(self, seed=None):
    pass

  @property
  def is_terminal(self):
    """
    現時点：packが終わったらTrue
    理想：packは終わらないので基本False
    """
    return self.current_pack_id == len(self.scheduler.pack_queue)

  @property
  def is_truncated(self):
    """
    現時点では永久False
    場合によってはstep数10でTrue
    """
    #return self.num_step>=10
    return False

  #@profile
  def step(self, action: int):
    self.num_step += 1
    self.current_pack_id += 1
    if self.is_terminal:
      state = (DummyPack(len(self.scheduler.pack_queue), [DummyTask(0,0,0,NODE_DISTANCES) for _ in range(DummyPack.max_len_task)], NODE_DISTANCES), self.scheduler.robots, self.scheduler.pack_queue) 
      #state = (DummyPack(len(self.scheduler.pack_queue), [DummyTask(0,0,0,NODE_DISTANCES) for _ in range(DummyPack.max_len_task)], NODE_DISTANCES), self.scheduler.robots, extract_circular_sublist(self.scheduler.pack_queue, 0)) 
    else:
      state = (self.scheduler.pack_queue[self.current_pack_id], self.scheduler.robots, self.scheduler.pack_queue)
      #state = (self.scheduler.pack_queue[self.current_pack_id], self.scheduler.robots, extract_circular_sublist(self.scheduler.pack_queue, self.current_pack_id))

    reward = self.reward(action, self.current_state_viewer, state)
    terminated = self.is_terminal
    truncated = self.is_truncated
    info = None

    self.current_state_viewer = copy.deepcopy(state) # heavy Code.
    #print(f"S:{state}, A:{action}, Term:{terminated}, trunc:{truncated}")
    return state, reward, terminated, truncated, info

  def reward(self, action: int, current_state: tuple[Pack, list[Robot], list[Pack]], next_state: tuple[Pack, list[Robot], list[Pack]]):
    """
    やってはいけないこと：current_state, next_stateの中身をこの関数のなかで変更してはいけない
    (値を変更した瞬間にmemoryに保存すべき情報との整合性がとれなくなるため)

    報酬に用いていい情報は全てこの引数の中にあるもののみと考えよ

    考えられる例: Packに含まれるノードと現在のrobotのノード間の距離・時間に関するもの
    感覚的には、next_packの情報は報酬計算では要らない
    """
    current_pack, current_robots, current_queue = current_state
    next_pack, next_robots, next_queue = next_state

    #current_time = current_robots[action].queue_time_from_start(NODE_DISTANCES)
    #next_time = next_robots[action].queue_time_from_start(NODE_DISTANCES)
    #add_time_reward = - (next_time - current_time)

    ctime_worst = 0
    for c_robot in current_robots:
      ctime_worst = max(c_robot.queue_time_from_start(NODE_DISTANCES), ctime_worst)

    ntime_worst = 0
    for n_robot in next_robots:
      ntime_worst = max(n_robot.queue_time_from_start(NODE_DISTANCES), ntime_worst)

    diff_worst = ntime_worst - ctime_worst
    #worst_time_reward = 1 / (ntime_worst - ctime_worst) if ntime_worst != ctime_worst else 1.0

    ctime_avg = 0
    for c_robot in current_robots:
      ctime_avg += c_robot.queue_time_from_start(NODE_DISTANCES)
    ctime_avg = ctime_avg / len(current_robots)

    ntime_avg = 0
    for n_robot in next_robots:
      ntime_avg += n_robot.queue_time_from_start(NODE_DISTANCES)
    ntime_avg = ntime_avg / len(next_robots)

    diff_avg = ntime_avg - ctime_avg
    #reward_dif_worst_avg = 1 / (ntime_worst - ntime_avg) if ntime_worst != ntime_avg else 1.0

    return - ((diff_worst+diff_avg) / np.sqrt(LEN_NODE)) ** 2
  
    ntime_late_job_agv = 0
    for n_robot in next_robots:
      ntime_late_job_agv += n_robot.queue_job_late_time_from_start(NODE_DISTANCES)
    return - ntime_late_job_agv
