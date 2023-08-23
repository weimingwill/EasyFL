import argparse
import os
import shutil
import tarfile


def run():
    parser = argparse.ArgumentParser(description='Extract')
    parser.add_argument("--source", type=str)
    parser.add_argument('--target', type=str)
    parser.add_argument('--task', type=str)

    args = parser.parse_args()

    files = os.listdir(args.source)
    for file in files:
        if args.task in file:
            print(f"Processing {file}")
            try:
                source_path = os.path.join(args.source, file)
                target_path = os.path.join(args.target, args.task)
                file_obj = tarfile.open(source_path, "r")
                file_obj.extractall(target_path)
                file_obj.close()
                old_name = os.path.join(target_path, args.task)
                place = file.replace(args.task, "").replace("_.tar", "")
                new_name = os.path.join(target_path, place)
                shutil.move(old_name, new_name)
                print(f"Extracted {file}")
            except Exception as e:
                print()
                print(f"Failed to extract {file}")
                print(e)
                print()


if __name__ == '__main__':
    run()
