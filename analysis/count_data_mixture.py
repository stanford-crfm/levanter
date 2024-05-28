from collections import Counter


counter = Counter()
log_file = "log.txt"
with open(log_file, "r") as f:
    for line in f:
        counter.update(line.strip().split())

# normalize the counts
total = sum(counter.values())
for key in counter:
    counter[key] /= total

for key, value in counter.items():
    print(f"{key}: {value:.3f}")