import argparse
import os
import sys
import time
import threading
import serial

BANNER = """
EchoSign – Manual-Control Gesture Data Collector
------------------------------------------------
Device output must look like:

  FLEX: f1 f2 f3 f4 f5 | ACC: ax ay az | GYRO: gx gy gz | GDP=val

CONTROL:
  • Type label name → press ENTER
  • 3-second live PREVIEW shows sensor lines (not recorded)
  • Press ENTER to START recording
  • Press ENTER again to STOP recording
  • File saved to: data/raw_<label>_<id>.txt

ESP32:
  • On START, PC sends 'S' → ESP beeps + LED ON
  • On STOP,  PC sends 'E' → ESP beeps + LED OFF
"""

TRIM_FRONT = 10
TRIM_BACK  = 10


def make_slug(label: str) -> str:
    slug = label.strip().replace(" ", "_")
    slug = "".join(c for c in slug if c.isalnum() or c in "_-")
    return slug or "gesture"


def preview_pose(ser: serial.Serial, seconds: float = 3.0) -> None:
    print(f"\nPreviewing for {seconds:.1f} seconds – adjust your hand pose.")
    print("(These lines are NOT saved.)\n")

    end_time = time.time() + seconds
    count = 0
    while time.time() < end_time:
        line = ser.readline().decode("utf-8", errors="ignore").strip()
        if not line:
            continue
        if line.startswith("FLEX:"):
            count += 1
            # Print every 4th line to avoid flooding
            if count % 4 == 0:
                print(line)
    print("Preview done.\n")


def wait_for_enter(prompt: str) -> None:
    input(prompt)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", default="COM15", help="Serial port (default COM15)")
    parser.add_argument("--baud", type=int, default=115200)
    parser.add_argument("--timeout", type=float, default=1.0)
    args = parser.parse_args()

    print(BANNER)

    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    os.makedirs(data_dir, exist_ok=True)

    # Open serial port
    try:
        ser = serial.Serial(args.port, args.baud, timeout=args.timeout)
    except Exception as e:
        print(f"ERROR opening serial port {args.port}: {e}")
        sys.exit(1)

    time.sleep(2)
    ser.reset_input_buffer()

    while True:
        print("\n------------------------------------------")
        label = input("Enter gesture label (blank to quit): ").strip()
        if not label:
            print("Exiting.")
            break

        slug = make_slug(label)

        # Determine session ID
        session_id = 1
        while True:
            fname = f"raw_{slug}_{session_id:02d}.txt"
            path = os.path.join(data_dir, fname)
            if not os.path.exists(path):
                break
            session_id += 1

        print(f"\nUpcoming recording: label='{label}' → File: {fname}")

        # Short live preview (not recorded)
        preview_pose(ser, seconds=3.0)

        # Wait for user to start
        wait_for_enter("Press ENTER to START recording...")

        print("\nRecording... Press ENTER again to STOP.\n")
        ser.reset_input_buffer()

        # Tell ESP recording started (for LED + beep)
        try:
            ser.write(b"S")
        except Exception:
            pass

        lines = []

        stop_flag = {"stop": False}

        def wait_for_stop():
            input()   # user presses ENTER to stop
            stop_flag["stop"] = True

        stopper = threading.Thread(target=wait_for_stop, daemon=True)
        stopper.start()

        # Main recording loop
        while not stop_flag["stop"]:
            line = ser.readline().decode("utf-8", errors="ignore").strip()
            if not line:
                continue
            if line.startswith("FLEX:"):
                lines.append(line)
                if len(lines) % 25 == 0:
                    print(f"Collected {len(lines)} samples...", end="\r")

        # Tell ESP recording stopped
        try:
            ser.write(b"E")
        except Exception:
            pass

        print(f"\nStopped. Raw samples: {len(lines)}")

        # Auto-trim noisy edges (entering/leaving pose)
        if len(lines) > (TRIM_FRONT + TRIM_BACK):
            trimmed = lines[TRIM_FRONT: len(lines) - TRIM_BACK]
            print(f"Auto-trimmed {TRIM_FRONT} from start and {TRIM_BACK} from end.")
            print(f"Final samples kept: {len(trimmed)}")
            lines = trimmed
        else:
            print("Not enough samples to trim; keeping all.")

        # Save file
        with open(path, "w", encoding="utf-8") as f:
            f.write(f"# label={label}\n")
            f.write(f"# timestamp={time.time()}\n")
            f.write("# format: FLEX: f1 f2 f3 f4 f5 | ACC: ax ay az | "
                    "GYRO: gx gy gz | GDP=val\n")
            for l in lines:
                f.write(l + "\n")

        print(f"Saved: {fname}\n")

        again = input("Record another session? (y/n): ").strip().lower()
        if again != "y":
            break

    ser.close()
    print("Done.")

if __name__ == "__main__":
    main()
