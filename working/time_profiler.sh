kernprof -l -v -o ./perf/perftime.lprof learning.py
python3.10 -m line_profiler ./perf/perftime.lprof > ./perf/learning.txt