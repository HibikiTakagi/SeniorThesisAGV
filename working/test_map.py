from map_maker import shortest_path_matrix

TEST_CONNECT_DICT = {
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

TEST_LEN_NODE = len(TEST_CONNECT_DICT)

TEST_NODE_DISTANCES = shortest_path_matrix(TEST_CONNECT_DICT)

if __name__ == "__main__":
  print(TEST_CONNECT_DICT)
  print(TEST_LEN_NODE)
  print(TEST_NODE_DISTANCES)