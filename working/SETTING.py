import pickle
import torch

from map_maker import generate_mesh_base_scc_map, shortest_path_matrix
from robot import generate_robot_list_random
from task import generate_task_list_random

#DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu" )
MODEL_PATH = "./save_data/model.pth"
DEVICE = torch.device("cpu")

## 保存・読み込みに関する設定
SAVE_FRAG = False
LOAD_FRAG = True
SAVE_FILE = './save_data/data.pkl'

## 結果に関する設定
FIG_LOSS_PATH = './save_data/loss.png'
FIG_RET_PATH = './save_data/ret.png'
FIG_REWARD_PATH = './save_data/reward.png'
FIG_GANTT_RESOURCE_PATH = './save_data/gantt_resource.html'
FIG_GANTT_TASK_PATH = './save_data/gantt_task.html'
CSV_THROUGHPUT_PATH = './save_data/throuput.csv'
CSV_PACK_LATENCY_PATH = './save_data/pack_latency.csv'

if LOAD_FRAG:
  ## 保存されたデータを読み込む場合
  print("load from "+SAVE_FILE)
  with open(SAVE_FILE, 'rb') as f:
    load_data = pickle.load(f)
  CONNECT_DICT    = load_data["connect_dict"]
  LEN_NODE        = len(CONNECT_DICT)
  NODE_DISTANCES  = shortest_path_matrix(CONNECT_DICT)
  ROBOTS          = load_data["robots"]
  NUM_ROBOT       = len(ROBOTS)
  TASK_LIST       = load_data["task_list"]
  NUM_TASK        = len(TASK_LIST)
  PACK_POLICY_NAME = load_data["pack_policy_name"]
  DISTRIBUTE_POLICY_NAME = load_data["distribute_policy_name"]
else:
  ## 新たにデータを生成する場合
  CONNECT_DICT = generate_mesh_base_scc_map()

  LEN_NODE = len(CONNECT_DICT)
  NODE_DISTANCES = shortest_path_matrix(CONNECT_DICT)

  NUM_ROBOT = 10
  ROBOTS = generate_robot_list_random(NUM_ROBOT, LEN_NODE)
  NUM_TASK = 100
  #NUM_TASK = 1200
  TASK_LIST = generate_task_list_random(NUM_TASK, LEN_NODE, NODE_DISTANCES)
  PACK_POLICY_NAME = "Naive"
  DISTRIBUTE_POLICY_NAME = "DQN"
  #DISTRIBUTE_POLICY = "Naive"

## 保存フラグが立っている場合、データを保存
SAVE_DATA = {
  "connect_dict"  : CONNECT_DICT,
  "robots"        : ROBOTS,
  "task_list"     : TASK_LIST,
  "pack_policy_name" : PACK_POLICY_NAME,
  "distribute_policy_name" : DISTRIBUTE_POLICY_NAME
}
if SAVE_FRAG:
  print("save to "+SAVE_FILE)
  with open(SAVE_FILE, 'wb') as f:
    pickle.dump(SAVE_DATA, f)


if __name__ == "__main__":
  print(CONNECT_DICT)
  print(ROBOTS)
  print(TASK_LIST)
  print(DEVICE)
  print(PACK_POLICY_NAME)
  print(DISTRIBUTE_POLICY_NAME)