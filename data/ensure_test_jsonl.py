"""Download the pile-uncopyrighted test split to `--test_dir/test.jsonl`."""

import argparse

from hullft.embedder import ensure_test_jsonl


def main():
    parser = argparse.ArgumentParser(
        description="Download pile-uncopyrighted test split."
    )
    parser.add_argument(
        "--test_dir",
        default="data/pile/test",
        help="Directory to store test.jsonl (default: data/pile/test).",
    )
    args = parser.parse_args()
    path = ensure_test_jsonl(args.test_dir)
    print(f"Test JSONL ready at: {path}")


if __name__ == "__main__":
    main()
