import copy
import random
import numpy as np
from robot import Robot
from task import Task
from pack import DummyPack, Pack
from task import Task, DummyTask
from SETTING import NODE_DISTANCES, ROBOTS, TASK_LIST, LEN_NODE
from scheduler import Scheduler

class JobSchedulerEnv:
  def __init__(self):
    self.robots = ROBOTS
    task_list = TASK_LIST
    self.scheduler = Scheduler(task_queue=task_list, robots=self.robots)
    pass

  #@profile
  def reset(self):
    for robot in self.robots:
      robot.reset()
    task_list = random.sample(TASK_LIST, len(TASK_LIST))
    self.scheduler.reset(task_list, self.robots)
    self.current_pack_id = 0
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
    現時点ではstep数10でTrue
    場合によっては永久False
    """
    return self.num_step >= 10


  def step(self, action: int):
    self.num_step += 1
    self.current_pack_id += 1
    if self.is_terminal:
      state = (DummyPack(len(self.scheduler.pack_queue), [DummyTask(0, 0, 0, NODE_DISTANCES) for _ in range(DummyPack.max_len_task)]), self.scheduler.robots, self.scheduler.pack_queue) 
    else:
      state = (self.scheduler.pack_queue[self.current_pack_id], self.scheduler.robots, self.scheduler.pack_queue)
    reward = self.reward(action, self.current_state_viewer, state)
    terminated = self.is_terminal
    truncated = self.is_truncated
    info = None

    del self.current_state_viewer
    self.current_state_viewer = copy.deepcopy(state)
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

    current_time = current_robots[action].queue_time_from_start(NODE_DISTANCES)
    next_time = next_robots[action].queue_time_from_start(NODE_DISTANCES)
    add_time_reward = - (next_time - current_time)

    ctime_worst = 0
    for c_robot in current_robots:
      ctime_worst = max(c_robot.queue_time_from_start(NODE_DISTANCES), ctime_worst)

    ntime_worst = 0
    for n_robot in next_robots:
      ntime_worst = max(n_robot.queue_time_from_start(NODE_DISTANCES), ntime_worst)
    worst_time_reward = 1 / (ntime_worst - ctime_worst) if ntime_worst != ctime_worst else 1.0

    ntime_avg = 0
    for n_robot in next_robots:
      ntime_avg += n_robot.queue_time_from_start(NODE_DISTANCES)
    ntime_avg = ntime_avg / len(next_robots)

    reward_dif_worst_avg = 1 / (ntime_worst - ntime_avg) if ntime_worst != ntime_avg else 1.0


    return - (ntime_worst - ctime_worst) / LEN_NODE
