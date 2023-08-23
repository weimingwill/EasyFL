import argparse
import os
import shutil

folders = [
    "allensville",
    "beechwood",
    "benevolence",
    "coffeen",
    "collierville",
    "corozal",
    "cosmos",
    "darden",
    "forkland",
    "hanson",
    "hiteman",
    "ihlen",
    "klickitat",
    "lakeville",
    "leonardo",
    "lindenwood",
    "markleeville",
    "marstons",
    "mcdade",
    "merom",
    "mifflinburg",
    "muleshoe",
    "newfields",
    "noxapater",
    "onaga",
    "pinesdale",
    "pomaria",
    "ranchester",
    "shelbyville",
    "stockman",
    "tolstoy",
    "uvalda",
]

TASKS = {
    's': 'segment_semantic',
}


def parse_tasks(task_str):
    tasks = []
    for char in task_str:
        tasks.append(TASKS[char])
    return tasks


def run():
    parser = argparse.ArgumentParser(description='Extract')
    parser.add_argument("--dir", type=str)

    args = parser.parse_args()

    for f in folders:
        p = os.path.join(args.dir, "segment_semantic", f)
        files = os.listdir(p)
        for file in files:
            if "segmentsemantic" in file:
                old_file = os.path.join(p, file)
                new_file = old_file.replace("segmentsemantic", "segment_semantic")
                shutil.move(old_file, new_file)


if __name__ == '__main__':
    run()
