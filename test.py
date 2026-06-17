import cv2
import numpy as np
import csv
import os
import torch
from ultralytics import YOLO
from shapely.geometry import Point, Polygon, box as shapely_box
from statistics import mean

# ======================
# Config - adjust paths and parameters here
# ======================
# NOTE: Using absolute paths based on your previous input.
VIDEO_PATH = r"C:\Users\reswant\ML-5thSemProj\ML-FOG-CONDITION\fog78.mp4"
MODEL_PATH = r"C:\Users\reswant\ML-5thSemProj\ML-FOG-CONDITION\v8l.pt"
CSV_PATH = r"C:\Users\reswant\ML-5thSemProj\ML-FOG-CONDITION\data.csv"

# Draw exactly two polygons
polygons = []
current_pts = []
drawing_mode = "polygon"

# Tracking structures
vehicle_positions = {}   # tid -> list of (x, y, t)
vehicle_last_speed = {}  # tid -> last speed_kmh

# ======================
# Device Setup (UPDATED FOR CUDA/NVIDIA GPU)
# ======================
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("✅ Using NVIDIA GPU (CUDA)")
else:
    device = torch.device("cpu")
    print("⚠️ CUDA GPU not available, using CPU")

# ======================
# CSV Setup
# video_name,row_type,interval_idx,lane_id,interval_count,cumulative_count,interval_density,interval_speed,Congestion level
# ======================
def init_csv():
    file_exists = os.path.isfile(CSV_PATH)
    with open(CSV_PATH, mode="a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow([
                "video_name", "row_type", "interval_idx", "lane_id",
                "interval_count", "cumulative_count",
                "interval_density", "interval_speed", "Congestion level"
            ])

def write_csv_row(video_name, row_type, interval_idx, lane_id, interval_count,
                  cumulative_count, interval_density, interval_speed, congestion=""):
    with open(CSV_PATH, mode="a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            video_name, row_type, interval_idx, lane_id,
            interval_count, cumulative_count,
            f"{interval_density:.6f}", f"{interval_speed:.2f}", congestion
        ])

# ======================
# Mouse callback for polygon drawing (two polygons enforced)
# ======================
def mouse_callback(event, x, y, flags, param):
    global current_pts, polygons, drawing_mode
    if drawing_mode != "polygon":
        return # Ignore clicks if not in polygon drawing mode

    if event == cv2.EVENT_LBUTTONDOWN:
        current_pts.append((x, y))
    elif event == cv2.EVENT_RBUTTONDOWN:
        # finish polygon if 3+ points
        if len(current_pts) >= 3:
            polygons.append(current_pts.copy())
            current_pts = []
            print(f"✅ Polygon added ({len(polygons)} total).")
            if len(polygons) == 2:
                # Automatic switch to "done" after 2 polygons
                drawing_mode = "done" 
                print("🟢 Two polygons added. Press 'q' to start processing.")

# ======================
# Draw UI annotations (polygons + in-progress points)
# ======================
def draw_annotations(frame):
    for poly in polygons:
        cv2.polylines(frame, [np.array(poly, np.int32)], True, (0, 255, 0), 2)
    for pt in current_pts:
        cv2.circle(frame, pt, 4, (0, 0, 255), -1)

# ======================
# Speed calculation (pixel -> km/h)
# Note: The meters_per_pixel factor is crucial for accuracy.
# ======================
def calculate_speed_from_track(positions, meters_per_pixel=0.08):
    """
    positions: list of (x, y, t) for one vehicle (t in seconds)
    returns: speed in km/h (float)
    """
    if len(positions) < 2:
        return 0.0
    x0, y0, t0 = positions[0]
    xn, yn, tn = positions[-1]
    pixel_dist = np.hypot(xn - x0, yn - y0)
    time_elapsed = tn - t0
    if time_elapsed <= 0.01: # Avoid division by zero/near-zero
        return 0.0
    dist_m = pixel_dist * meters_per_pixel
    speed_m_s = dist_m / time_elapsed
    speed_kmh = speed_m_s * 3.6
    return float(speed_kmh)

# ======================
# Per-frame occupancy fraction (intersection area / polygon area)
# ======================
def frame_occupancy_fraction(boxes, poly):
    """
    Returns fraction in [0,1]: total intersection area of boxes with poly / poly_area.
    boxes is a list of box coordinates [x1,y1,x2,y2].
    """
    polygon_shape = Polygon(poly)
    poly_area = polygon_shape.area
    if poly_area <= 0:
        return 0.0

    total_intersection = 0.0
    for b in boxes:
        x1, y1, x2, y2 = b
        vehicle_shape = shapely_box(x1, y1, x2, y2)
        inter = polygon_shape.intersection(vehicle_shape).area
        total_intersection += inter

    frac = total_intersection / poly_area
    # numeric safety cap
    return max(0.0, min(1.0, frac))

# ======================
# Main processing pipeline
# ======================
def process_video(video_path, model_path, fps, report_interval=10.0):
    global vehicle_positions, vehicle_last_speed

    cap = cv2.VideoCapture(video_path)
    model = YOLO(model_path)

    video_name = os.path.basename(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = (total_frames / fps) if total_frames > 0 else None
    print(f"Processing {video_name} - fps={fps:.2f}, frames={total_frames}, duration={duration:.1f}s")

    # Per-lane accumulators
    num_lanes = len(polygons)
    cumulative_ids = [set() for _ in range(num_lanes)]
    interval_ids = [set() for _ in range(num_lanes)]
    interval_speeds = [[] for _ in range(num_lanes)]
    interval_frame_densities = [[] for _ in range(num_lanes)]

    all_interval_densities = [[] for _ in range(num_lanes)]
    all_speeds = [[] for _ in range(num_lanes)]

    frame_count = 0
    interval_index = 0
    init_csv()

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        current_time = frame_count / fps

        # Run tracking on GPU (device=0 for first GPU)
        results = model.track(frame, persist=True, verbose=False, device=0)[0]

        per_frame_lane_boxes = [[] for _ in range(num_lanes)]

        if results.boxes.id is not None:
            for box, tid in zip(results.boxes.xyxy, results.boxes.id):
                tid = int(tid.item())
                x1, y1, x2, y2 = map(int, box[:4])
                # Centroid is used for geometric inclusion check
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

                # Draw bbox & id on frame
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)
                cv2.putText(frame, f"ID {tid}", (x1, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)

                # Update track history for speed calculation
                if tid not in vehicle_positions:
                    vehicle_positions[tid] = []
                vehicle_positions[tid].append((cx, cy, current_time))
                # Keep only recent history (last 2 seconds)
                vehicle_positions[tid] = [p for p in vehicle_positions[tid] if current_time - p[2] <= 2.0]

                # Compute current speed
                speed_kmh = calculate_speed_from_track(vehicle_positions[tid], meters_per_pixel=0.08)
                vehicle_last_speed[tid] = speed_kmh

                # Check inclusion in user-defined polygons
                for lane_idx, poly in enumerate(polygons):
                    if Polygon(poly).contains(Point(cx, cy)):
                        # Data collection for this lane
                        per_frame_lane_boxes[lane_idx].append([x1, y1, x2, y2])
                        interval_ids[lane_idx].add(tid)
                        cumulative_ids[lane_idx].add(tid)
                        
                        if speed_kmh > 0:
                            interval_speeds[lane_idx].append(speed_kmh)
                            all_speeds[lane_idx].append(speed_kmh)

        # Compute and accumulate per-frame density
        for i in range(num_lanes):
            frac = frame_occupancy_fraction(per_frame_lane_boxes[i], polygons[i])
            interval_frame_densities[i].append(frac)

        # Draw annotations and overlay live stats
        draw_annotations(frame)
        for i in range(num_lanes):
            live_count = len(per_frame_lane_boxes[i])
            avg_speed = mean(interval_speeds[i]) if interval_speeds[i] else 0.0
            last_frac = interval_frame_densities[i][-1] if interval_frame_densities[i] else 0.0
            
            # Simple Congestion Labeling (0=Low, 1=Medium, 2=High based on Density)
            if last_frac < 0.20:
                congestion_text = "LOW"
                text_color = (0, 255, 0) # Green
            elif 0.20 <= last_frac <= 0.60:
                congestion_text = "MEDIUM"
                text_color = (0, 255, 255) # Yellow
            else:
                congestion_text = "HIGH"
                text_color = (0, 0, 255) # Red
                
            cv2.putText(frame, f"Zone{i+1}: Live={live_count}, D={last_frac:.2f}, V={avg_speed:.1f}km/h, Cong: {congestion_text}",
                        (10, 30 + i*30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)

        cv2.imshow("Traffic Density & Speed", frame)
        key = cv2.waitKey(1)
        if key == ord("q"):
            break

        # Check for interval completion
        if current_time >= (interval_index + 1) * report_interval:
            interval_index += 1
            print(f"\n=== Interval {interval_index} ===")
            
            for i in range(num_lanes):
                int_count = len(interval_ids[i])
                cum_count = len(cumulative_ids[i])
                
                # Calculate interval metrics
                interval_density = float(mean(interval_frame_densities[i])) if interval_frame_densities[i] else 0.0
                interval_speed = float(mean(interval_speeds[i])) if interval_speeds[i] else 0.0

                # Determine Congestion Level for CSV (0, 1, or 2)
                if interval_density < 0.20:
                    congestion_level = 0
                elif interval_density <= 0.60:
                    congestion_level = 1
                else:
                    congestion_level = 2
                
                print(f"Zone {i+1}: Count={int_count}, Density={interval_density:.6f}, Speed={interval_speed:.2f} km/h, Congestion={congestion_level}")
                
                write_csv_row(video_name, "interval", interval_index, i+1,
                              int_count, cum_count, interval_density, interval_speed, congestion_level)

                all_interval_densities[i].append(interval_density)

                # Clear interval accumulators
                interval_ids[i].clear()
                interval_speeds[i].clear()
                interval_frame_densities[i].clear()

    # --- FINAL STATS & PARTIAL INTERVAL ---
    has_unreported = any(interval_frame_densities[i] or interval_ids[i] or interval_speeds[i] for i in range(num_lanes))
    if has_unreported:
        interval_index += 1
        print(f"\n=== Final Partial Interval {interval_index} ===")
        
        for i in range(num_lanes):
            int_count = len(interval_ids[i])
            cum_count = len(cumulative_ids[i])
            interval_density = float(mean(interval_frame_densities[i])) if interval_frame_densities[i] else 0.0
            interval_speed = float(mean(interval_speeds[i])) if interval_speeds[i] else 0.0

            if interval_density < 0.20:
                congestion_level = 0
            elif interval_density <= 0.60:
                congestion_level = 1
            else:
                congestion_level = 2
            
            print(f"Zone {i+1}: Count={int_count}, Density={interval_density:.6f}, Speed={interval_speed:.2f} km/h")
            write_csv_row(video_name, "interval", interval_index, i+1,
                          int_count, cum_count, interval_density, interval_speed, congestion_level)
            all_interval_densities[i].append(interval_density)

    # Overall final statistics per lane
    print("\n=== OVERALL FINAL STATS ===")
    for i in range(num_lanes):
        total_unique = len(cumulative_ids[i])
        avg_density = float(mean(all_interval_densities[i])) if all_interval_densities[i] else 0.0
        avg_speed = float(mean(all_speeds[i])) if all_speeds[i] else 0.0

        if avg_density < 0.20:
            congestion_level = 0
        elif avg_density <= 0.60:
            congestion_level = 1
        else:
            congestion_level = 2

        print(f"Zone {i+1}: Total unique vehicles={total_unique}, Avg Density={avg_density:.6f}, Avg Speed={avg_speed:.2f} km/h, Overall Congestion={congestion_level}")
        
        write_csv_row(video_name, "overall", "overall", i+1,
                      total_unique, total_unique, avg_density, avg_speed, congestion_level)

    cap.release()
    cv2.destroyAllWindows()
    print("Processing complete. CSV updated at:", CSV_PATH)

# ======================
# Entry point: draw polygons then process
# ======================
if __name__ == "__main__":
    # Path checks
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model file not found at {MODEL_PATH}")
        exit(1)
    if not os.path.exists(VIDEO_PATH):
        print(f"Error: Video file not found at {VIDEO_PATH}")
        exit(1)

    # Load first frame for annotation
    cap = cv2.VideoCapture(VIDEO_PATH)
    ret, first_frame = cap.read()
    if not ret:
        print("Error reading video or video is empty.")
        exit(1)

    cv2.namedWindow("Annotate")
    cv2.setMouseCallback("Annotate", mouse_callback)
    print("Draw exactly two polygons (left-click to add points, right-click to finish each polygon).")
    print("When 'Two polygons added' appears, press 'q' to proceed.")

    while True:
        display = first_frame.copy()
        draw_annotations(display)
        
        mode_status = "DRAWING POLYGON" if drawing_mode == "polygon" else "DONE ANNOTATING"
        cv2.putText(display, f"Status: {mode_status} (L-click add, R-click finish, q=start processing)",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        cv2.imshow("Annotate", display)
        key = cv2.waitKey(1)
        
        # Only break if 'q' is pressed AND two polygons have been defined
        if key == ord("q") and len(polygons) == 2:
            break

    cap.release()
    cv2.destroyAllWindows()
    
    if len(polygons) != 2:
        print("Error: Two polygons were not successfully defined. Exiting.")
        exit(1)

    print(f"Polygons drawn: {len(polygons)}. Starting processing...")

    # Determine fps
    cap = cv2.VideoCapture(VIDEO_PATH)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0 # Use 25.0 as fallback
    cap.release()

    process_video(VIDEO_PATH, MODEL_PATH, fps, report_interval=10.0)                                                                                                                                                                                                     
