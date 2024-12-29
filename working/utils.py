def extract_circular_sublist(lst, index, length=5):
  n = len(lst)
  if n == 0:
    return []  # 空リストの場合は空を返す

  # 指定範囲の開始と終了を計算
  #start = index - length // 2
  #end = index + (length // 2) + 1
  start = index
  end = index + length

  # 循環して要素を取得
  result = [lst[i % n] for i in range(start, end)]

  return result

if __name__ == "__main__":
  sample_list = [1, 2, 3, 4, 5]
  index = 3
  length = 7
  
  result = extract_circular_sublist(sample_list, index, length)
  print(f"Input List: {sample_list}")
  print(f"Start Index: {index}, Length: {length}")
  print(f"Circular Sublist: {result}") 
