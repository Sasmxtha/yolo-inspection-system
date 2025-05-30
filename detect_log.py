import cv2
import os
import json
from datetime import datetime
from ultralytics import YOLO

# Load YOLO model
model = YOLO(r"content\runs\detect\train\weights\best.pt")

# Folder to save snapshots and logs
log_dir = r"YOLO_Confusion_Snaps"
os.makedirs(log_dir, exist_ok=True)

# Step 1: Extract base element names (excluding _w and _n suffixes) 
#       element name with no suffix ---------    OK
#       element name _w -> element misaligned    NOT OK
#       element name _n -> element missing       NOT OK
all_names = model.names.values() if isinstance(model.names, dict) else model.names
base_elements = set()

for name in all_names:
    if name.endswith("_w") or name.endswith("_n"):
        base = name.rsplit("_", 1)[0]
        base_elements.add(base)
    else:
        base_elements.add(name)

base_elements = sorted(base_elements)    #or extract the element names form total_counts

total_counts = {      #list all the elements on the board and its total count 
    "B_MOV": 1,
    "G_MOV": 1,
    "MOV": 6,
    "b_pin": 2,
    "bs_pin": 1,
    "cap_b": 2,
    "cap_s": 1,
    "dual_F": 1,
    "fuse": 1,
    "jumper_cap": 1,
    "r_pin": 1,
    "w_pin": 1,
    "ws_pin": 1,
}

# Step 2: Prompt user for expected OK counts 
def ac_count():
    expected_counts = {}
    print("Detected elements (excluding '_w' and '_n' suffixes):")
    for elem in base_elements:
        while True:
            try:
                count = int(input(f"Enter expected OK count for '{elem}': "))
                expected_counts[elem] = count
                break
            except ValueError:
                print("Please enter a valid integer.")
    return expected_counts
  
# Run webcam detection 
def run_detection_loop(expected_counts):
    
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    if not cap.isOpened():
        print("Error: Webcam not found.")
        return False

    print("\n Press 's' to take snapshot. Press 'q' to quit detection.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame.")
            break

        results = model.predict(source=frame, conf=0.3, save=False, verbose=False)
        result = results[0]

        # Copy frame to draw on
        annotated = frame.copy()

        # Draw boxes and labels
        for box in result.boxes:
            cls_id = int(box.cls)
            label = result.names[cls_id]

            # Get bounding box coordinates (xyxy)
            xyxy = box.xyxy[0].cpu().numpy().astype(int)
            x1, y1, x2, y2 = xyxy

            # Determine box color and label text
            if label.endswith("_w") or label.endswith("_n"):    
                color = (0, 0, 255)  # Red
                text = "NOT OK"
            else:
                color = (0, 255, 0)  # Green
                text = "OK"

            # Draw rectangle
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

            # Draw filled rectangle for text background
            (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.rectangle(annotated, (x1, y1 - text_h - 10), (x1 + text_w, y1), color, -1)

            # Put text
            cv2.putText(annotated, text, (x1, y1 - 5),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Instruction text
        cv2.putText(annotated, "Press 's' to snap, 'q' to quit.", (10, 30),cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

        cv2.imshow("YOLO Detection", annotated)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):
            take_snapshot(result, frame, expected_counts)
            break
        elif key == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            return False

    cap.release()
    cv2.destroyAllWindows()
    return True

# Take snapshot and save log 
def take_snapshot(result, frame, expected_counts):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    ok_counts = {key: 0 for key in expected_counts}
    not_ok_counts = {key: 0 for key in expected_counts}

    for box in result.boxes:
        cls_id = int(box.cls)
        label = result.names[cls_id]

        if label.endswith("_w") or label.endswith("_n"):
            base = label[:-2]
            if base in not_ok_counts:
                not_ok_counts[base] += 1
        elif label in ok_counts:
            ok_counts[label] += 1

    # Calculate expected_not_ok = total_counts - expected_counts
    expected_not_ok = {}
    for key in expected_counts:
        total = total_counts.get(key, 0)
        expected_ok = expected_counts.get(key, 0)
        expected_not_ok[key] = total - expected_ok

    # Save JSON log (without saving image)
    log = {
        "timestamp": timestamp,
        "expected": expected_counts,
        "expected_not_ok": expected_not_ok,
        "ok_detected": ok_counts,
        "not_ok_detected": not_ok_counts
    }

    json_path = os.path.join(log_dir, f"{timestamp}.json")
    with open(json_path, "w") as f:
        json.dump(log, f, indent=4)

    # Print results summary
    print(f"\nLog saved: {json_path}")
    print("OK counts:")
    for name, count in ok_counts.items():
        print(f"  {name}: {count} / {expected_counts[name]}")
    print("NOT OK counts:")
    for name, count in not_ok_counts.items():
        print(f"  {name}: {count}")
        
# --- Main loop ---
if __name__ == "__main__":
    print("\nStarting YOLO detection with snapshot logging...")

    while True:
        expected_counts=ac_count()
        cont = run_detection_loop(expected_counts)
        if not cont:
            print("Exiting program.")
            break

        user_input = input("\nType 'ok' to take another snapshot or 'done' to finish: ").strip().lower()
        if user_input == "done":
            print("Program ended. Logs saved at:")
            print(log_dir)
            break
