import jsonlines
import random
import argparse

random.seed(42)


def shuffle_jsonl(args):
    with jsonlines.open(args.file_path) as reader:
        data = [line for line in reader]
    random.shuffle(data)

    with jsonlines.open(args.output_path, mode='w') as writer:
        writer.write_all(data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file_path", type=str)
    parser.add_argument("--output_path", type=str)

    args = parser.parse_args()
    shuffle_jsonl(args)