import json

json_file = (
    "/home/lff/bigdata1/cjw/projs/qwen3vl/qwen-vl-finetune/data/sp-mllm-502k.jsonl"
)

problem_types = []
problem_sub_types = []

with open(json_file, "r") as f:
    lines = f.readlines()

total_count = len(lines)
loss_count = 0

for line in lines:
    data = json.loads(line)
    metadata = data.get("metadata", None)
    if not metadata:
        loss_count += 1
        continue

    problem_type = metadata["problem_type"]
    problem_sub_type = metadata["problem_sub_type"]
    if problem_type not in problem_types:
        problem_types.append(problem_type)
        problem_sub_types.append([problem_sub_type])
    else:
        if (
            problem_sub_type
            and problem_sub_type
            not in problem_sub_types[problem_types.index(problem_type)]
        ):
            problem_sub_types[problem_types.index(problem_type)].append(
                problem_sub_type
            )


with open("problem_types.txt", "w") as f:
    for i, problem_type in enumerate(problem_types):
        f.write(f"{problem_type}: {problem_sub_types[i]}\n")
    f.write(f"Total Count: {total_count}\n")
    f.write(f"Loss Count: {loss_count}\n")
