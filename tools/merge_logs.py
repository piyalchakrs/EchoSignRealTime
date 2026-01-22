import glob
import os
import re
import csv

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
OUT_CSV = os.path.join(DATA_DIR, "dataset.csv")

# FLEX: f1 f2 f3 f4 f5 | ACC: ax ay az | GYRO: gx gy gz | GDP=val
LINE_RE = re.compile(
    r"^FLEX:\s+([-\d\.]+)\s+([-\d\.]+)\s+([-\d\.]+)\s+([-\d\.]+)\s+([-\d\.]+)"
    r"\s+\|\s+ACC:\s+([-\d\.]+)\s+([-\d\.]+)\s+([-\d\.]+)"
    r"\s+\|\s+GYRO:\s+([-\d\.]+)\s+([-\d\.]+)\s+([-\d\.]+)"
    r"\s+\|\s+GDP=([-\d\.]+)"
)

# We keep GDP in the feature set:
# f1,f2,f3,f4,f5,gdp,ax,ay,az,gx,gy,gz,label
HEADER = [
    "f1", "f2", "f3", "f4", "f5",
    "gdp",
    "ax", "ay", "az",
    "gx", "gy", "gz",
    "label"
]


def extract_label_from_filename(path: str) -> str:
    base = os.path.basename(path)  # raw_<label>_<id>.txt
    if not base.startswith("raw_") or not base.endswith(".txt"):
        return "unknown"
    core = base[4:-4]
    parts = core.split("_")
    if len(parts) >= 2 and parts[-1].isdigit():
        label_parts = parts[:-1]
    else:
        label_parts = parts
    label = " ".join(label_parts)
    return label


def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    paths = sorted(glob.glob(os.path.join(DATA_DIR, "raw_*.txt")))
    if not paths:
        print("No raw_*.txt files found in data/. Nothing to do.")
        return

    rows = []

    for path in paths:
        label = extract_label_from_filename(path)
        print(f"Parsing {os.path.basename(path)} (label='{label}')")

        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                m = LINE_RE.match(line)
                if not m:
                    continue
                vals = [float(x) for x in m.groups()]
                f1, f2, f3, f4, f5, ax, ay, az, gx, gy, gz, gdp = vals
                # Reorder into our canonical feature order:
                # f1,f2,f3,f4,f5,gdp,ax,ay,az,gx,gy,gz,label
                row = [
                    f1, f2, f3, f4, f5,
                    gdp,
                    ax, ay, az,
                    gx, gy, gz,
                    label,
                ]
                rows.append(row)

    if not rows:
        print("No valid data lines found in any raw_*.txt files.")
        return

    print(f"Writing {len(rows)} samples to {OUT_CSV}")
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(HEADER)
        writer.writerows(rows)

    # Quick summary per class
    counts = {}
    for r in rows:
        lbl = r[-1]
        counts[lbl] = counts.get(lbl, 0) + 1

    print("\nSamples per label:")
    for lbl, n in sorted(counts.items(), key=lambda kv: kv[0]):
        print(f"  {lbl}: {n}")

    print("\nDone.")

if __name__ == "__main__":
    main()
