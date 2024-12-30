import numpy as np
from numpy.typing import NDArray
import random
import networkx as nx
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path

def shortest_path_matrix(direct):
  """
  指定された隣接リスト形式のグラフ（direct）から最短経路行列を作成。
  """
  data = []
  row = []
  col = []
  for start_node, dest_nodes in direct.items():
    for dest_node in dest_nodes:
      row.append(start_node)
      col.append(dest_node)
      data.append(1)

  ## 最大ノード番号を取得して行列のサイズを決定
  max_node = max(max(row, default=0), max(col, default=0))
  n = max_node + 1  # ノードは0から始まるため、最大ノード番号に1を加える
  
  ## 隣接リスト形式のグラフを疎行列（CSR形式）に変換
  csr = csr_matrix((data, (row, col)), shape=(n, n)).toarray()
    
  my_round_int = lambda x: np.round((x*2+1)//2)
  ## SciPyのshortest_path関数を用いて最短経路行列を計算
  return my_round_int(shortest_path(csr)).astype(int)

def generate_mesh_base_scc_fantom_map():
  """
  メッシュ構造を持つグリッドグラフを生成し、その中から最も大きい強連結成分（SCC）を抽出。
  """
  removes = 60
  rows = 40
  cols = 40
  max_len_node = rows*cols
  min_len_node = 10
  
  removed_nodes=random.sample(range(rows*cols), removes)
    
  graph = {}
    
  def get_neighbor(node, direction):
    row, col = divmod(node, cols)
    if direction == 'up' and row > 0:
      return node - cols
    elif direction == 'down' and row < rows - 1:
      return node + cols
    elif direction == 'left' and col > 0:
      return node - 1
    elif direction == 'right' and col < cols - 1:
      return node + 1
    return None
    
  def dict_to_networkx(adj_list):
    """
    隣接リストをNetworkXの有向グラフに変換
    """
    G = nx.DiGraph()
    for node, neighbors in adj_list.items():
      for neighbor in neighbors:
        G.add_edge(node, neighbor)
    return G

  ## 各ノードの上下左右の隣接ノードを計算。（削除されたノードは無視）
  for row in range(rows):
    for col in range(cols):
      node = row * cols + col
      if node in removed_nodes:
        continue
            
      graph[node] = []
      ## （端や穴あきを除いて）上下左右に辺を張っているため、実質無向グラフになる。      
      right_neighbor = get_neighbor(node, 'right')
      if right_neighbor is not None and right_neighbor not in removed_nodes:
        graph[node].append(right_neighbor)
                
      left_neighbor = get_neighbor(node, 'left')
      if left_neighbor is not None and left_neighbor not in removed_nodes:
        graph[node].append(left_neighbor)
                
      down_neighbor = get_neighbor(node, 'down')
      if down_neighbor is not None and down_neighbor not in removed_nodes:
        graph[node].append(down_neighbor)

      up_neighbor = get_neighbor(node, 'up')
      if up_neighbor is not None and up_neighbor not in removed_nodes:
        graph[node].append(up_neighbor)

  ## SCCをリスト化し、最も大きいSCCを選択。
  x_graph = dict_to_networkx(graph)
  x_sccs = list(nx.strongly_connected_components(x_graph))
  scc = []
  len_scc = 0
  for x_scc in x_sccs:
    tmp_scc = list(x_scc)
    tmp_len_scc = len(tmp_scc)
    if len_scc < tmp_len_scc:
      scc = tmp_scc
      len_scc = tmp_len_scc

  ## sccのノードのみを含むサブグラフを生成。  
  new_graph = {}
  for start_node in scc:
    new_graph[start_node] = []
    for c_dest_node in graph[start_node]:
      if c_dest_node in scc:
        new_graph[start_node].append(c_dest_node)
    
  graph = new_graph

  ## グラフが一定のノード数を満たさない場合は再帰的に再生成
  if len(graph) < min_len_node or len(graph) > max_len_node:
    return generate_mesh_base_scc_map()
    
  return graph

def generate_mesh_base_scc_map():
  """
  SCCグラフを生成し、
  ノード番号を連続するインデックス（0から始まる）に変換。
  """
  fantom_dict = generate_mesh_base_scc_fantom_map()
  route_nodes = np.array(list(fantom_dict.keys()))
  ## 新しいマッピング先のノード番号
  label_nodes = {route_nodes[i]:i for i in range(len(route_nodes))}
  
  def change_to_connect_dict(connect_dict, label_node):
    """
    ラベル変換を行い、元のノード番号を新しい番号にマッピング。
    """
    ret_dict = {}
    for start_node, dest_nodes in connect_dict.items():
      ret_dict[label_node[start_node]] = []
      for dest in dest_nodes:
        ret_dict[label_node[start_node]].append(label_node[dest])
    return ret_dict

  return change_to_connect_dict(fantom_dict, label_nodes)

#CONNECT_DICT = generate_mesh_base_scc_map()
#LEN_NODE = len(CONNECT_DICT)
#NODE_DISTANCES = shortest_path_matrix(CONNECT_DICT)

if __name__ == "__main__":
  connect_dict = generate_mesh_base_scc_map()
  len_node = len(connect_dict)
  node_distances = shortest_path_matrix(connect_dict)
  print(connect_dict)
  print(len_node)
  print(node_distances)