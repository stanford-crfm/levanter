with open("output.log", "r") as f:
    lines = f.readlines()
    new_lines = []
    for line in lines:
        if "cc" not in line:
            new_lines.append(line)

with open("output_new.log", "w") as f:
    f.write("\n".join(new_lines))