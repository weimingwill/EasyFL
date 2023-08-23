import argparse
import os

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
    'd': 'depth_zbuffer',
    'n': 'normal',
    'N': 'normal2',
    'k': 'keypoints2d',
    'e': 'edge_occlusion',
    'r': 'reshading',
    't': 'edge_texture',
    'a': 'rgb',
    'c': 'principal_curvature'
}


def parse_tasks(task_str):
    tasks = []
    for char in task_str:
        tasks.append(TASKS[char])
    return tasks


def run():
    parser = argparse.ArgumentParser(description='Extract')
    parser.add_argument("--dir", type=str)
    parser.add_argument('--tasks', type=str)

    args = parser.parse_args()

    tasks = parse_tasks(args.tasks)

    for f in folders:
        for t in tasks:
            p = os.path.join(args.dir, t, f)
            try:
                print(f"{t}-{f}: {len(os.listdir(p))}")
            except Exception as e:
                print(e)
        print()


if __name__ == '__main__':
    run()
