import csv
import numpy as np
from SETTING import CSV_THROUGHPUT_PATH, CSV_PACK_LATENCY_PATH

def save_csv_throughput(throughputs):
  avg_throughput = np.mean(throughputs)
  var_throughput = np.var(throughputs)
  max_throughput = np.max(throughputs)
  min_throughput = np.min(throughputs)

  with open(CSV_THROUGHPUT_PATH, mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["Robot", "Throughput"])
    for i, value in enumerate(throughputs):
      writer.writerow([f"R{i}", value])
    writer.writerow([])
    writer.writerow(["Average", avg_throughput])
    writer.writerow(["Variance", var_throughput])
    writer.writerow(["Max", max_throughput])
    writer.writerow(["Min", min_throughput])

def save_csv_pack_latency(pack_latency):
  avg_pack_latency = np.mean(pack_latency)
  var_pack_latency = np.var(pack_latency)
  max_pack_latency = np.max(pack_latency)
  min_pack_latency = np.min(pack_latency)

  with open(CSV_PACK_LATENCY_PATH, mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["Pack", "PackLatency"])
    for i, value in enumerate(pack_latency):
      writer.writerow([f"Pack{i}", value])
    writer.writerow([])
    writer.writerow(["Average", avg_pack_latency])
    writer.writerow(["Variance", var_pack_latency])
    writer.writerow(["Max", max_pack_latency])
    writer.writerow(["Min", min_pack_latency])

if __name__ == "__main__":
  throughputs = [2.1, 4.5, 6.8]
  save_csv_throughput(throughputs)