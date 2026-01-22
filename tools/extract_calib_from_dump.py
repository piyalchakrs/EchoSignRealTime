import argparse
import os
import re
import numpy as np

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")

LINE_RE = re.compile(
    r"^FLEX:\s+([-\d\.]+)\s+([-\d\.]+)\s+([-\d\.]+)\s+([-\d\.]+)\s+([-\d\.]+)"
)

def main():
    parser = argparse.ArgumentParser(
        description="Compute FLEX_MIN/FLEX_MAX from a dump file."
    )
    parser.add_argument(
        "--file",
        default=os.path.join(DATA_DIR, "calib_dump.txt"),
        help="Path to dump file containing FLEX lines",
    )
    args = parser.parse_args()

    path = args.file
    if not os.path.exists(path):
        raise SystemExit(f"File not found: {path}")

    vals = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            m = LINE_RE.match(line)
            if not m:
                continue
            flex = [int(float(x)) for x in m.groups()]
            vals.append(flex)

    if not vals:
        raise SystemExit("No FLEX lines found in file.")

    arr = np.array(vals, dtype=int)  # shape (N,5)
    flex_min = arr.min(axis=0)
    flex_max = arr.max(axis=0)

    print("Suggested FLEX_MIN / FLEX_MAX (paste into Calib.h):\n")
    print("static const int FLEX_MIN[5] = { "
          + ", ".join(str(int(x)) for x in flex_min) + " };")
    print("static const int FLEX_MAX[5] = { "
          + ", ".join(str(int(x)) for x in flex_max) + " };")
    print("\nPer-sensor stats:")
    for i in range(5):
        print(f"F{i+1}: min={flex_min[i]}, max={flex_max[i]}")

if __name__ == "__main__":
    main()
