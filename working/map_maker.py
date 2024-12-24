import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path

def shortest_path_matrix(direct):
  data = []
  row = []
  col = []
  for start_node, dest_nodes in direct.items():
    for dest_node in dest_nodes:
      row.append(start_node)
      col.append(dest_node)
      data.append(1)

  # 最大ノード番号を取得して行列のサイズを決定
  max_node = max(max(row, default=0), max(col, default=0))
  n = max_node + 1  # ノードは0から始まるため、最大ノード番号に1を加える
  
  csr = csr_matrix((data, (row, col)), shape=(n, n)).toarray()
    
  my_round_int = lambda x: np.round((x*2+1)//2)
  return my_round_int(shortest_path(csr)).astype(int)
  #return torch.tensor(my_rount_int(shortest_path(csr)).astype(int)).to(DEVICE)

LEN_NODE = 9
CONNECT_DICT = {
  0:[1,3],
  1:[0,2,4],
  2:[1,5],
  3:[0,4,6],
  4:[1,3,5,7],
  5:[2,4,8],
  6:[3,7],
  7:[4,6,8],
  8:[5,7]
}
NODE_DISTANCES = shortest_path_matrix(CONNECT_DICT)

if __name__ == "__main__":
  print(shortest_path_matrix(CONNECT_DICT))
  print(type(NODE_DISTANCES))