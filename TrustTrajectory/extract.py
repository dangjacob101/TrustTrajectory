import json

with open("Final-Benchmark.ipynb", "r") as f:
    data = json.load(f)

# The second cell seems to have the most recent code
source_lines = data["cells"][1]["source"]
with open("benchmark.py", "w") as f:
    f.writelines(source_lines)
