#!/usr/bin/env python

import argparse
import logging
import sys
from duet.duetconfig import DuetBenchConfig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "config", type=str, help="YAML config file for the duet benchmark"
    )
    args = parser.parse_args()

    try:
        config = DuetBenchConfig(args.config)
    except Exception as e:
        logging.critical(f"Critical config error: {e}")
        sys.exit(1)

    duet_names = [duet.benchmark for duet in config.duets]
    print(" ".join(duet_names))


if __name__ == "__main__":
    main()
