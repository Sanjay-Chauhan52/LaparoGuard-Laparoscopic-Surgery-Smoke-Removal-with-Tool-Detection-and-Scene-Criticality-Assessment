"""
diagnostic_detection.py — Visualize smoke and bleeding detection
Shows what the algorithm is actually detecting to debug issues
"""

import cv2
import numpy as np
import os


def detect_smoke_visual(frame):
    """Improved smoke detection with local contrast analysis"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Global contrast
    global_contrast = np.std(gray) / 255.0
    
    # Local contrast (divide image into regions)
    h, w = gray.shape
    regions = 4
    region_h, region_w = h // regions, w // regions
    local_contrasts = []
    
    for i in range(regions):
        for j in range(regions):
            region = gray[i*region_h:(i+1)*region_h, j*region_w:(j+1)*region_w]
            local_contrasts.append(np.std(region) / 255.0)
    
    avg_local_contrast = np.mean(local_contrasts)
    
    # Brightness
    brightness = np.mean(gray) / 255.0
    
    # Laplacian variance (sharpness)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    sharpness = np.var(laplacian) / 10000.0
    
    # Edge density
    edges = cv2.Canny(gray, 30, 100)
    edge_density = np.count_nonzero(edges) / (gray.shape[0] * gray.shape[1])
    
    # Combined visibility score (higher = better visibility)
    visibility_score = (
        avg_local_contrast * 0.35 +
        global_contrast * 0.20 +
        sharpness * 0.20 +
        edge_density * 0.15 +
        brightness * 0.10
    )
    
    is_smoky = visibility_score < 0.18
    
    # Create visualization
    vis = frame.copy()
    
    # Show metrics
    info = f"LocalContrast: {avg_local_contrast:.3f} | Global: {global_contrast:.3f} | Sharp: {sharpness:.3f} | Edge: {edge_density:.3f}"
    cv2.putText(vis, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 2)
    
    status = f"SMOKE DETECTED: {is_smoky} | Visibility: {visibility_score:.3f}"
    color = (0, 0, 255) if is_smoky else (0, 255, 0)
    cv2.putText(vis, status, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    
    # Show edge map
    edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    
    return vis, edges_bgr, visibility_score, is_smoky


def detect_bleeding_visual(frame):
    """Improved bleeding detection with concentrated region analysis"""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # More precise red color ranges
    lower_red1 = np.array([0, 100, 70])
    upper_red1 = np.array([8, 255, 255])
    lower_red2 = np.array([165, 50, 50])
    upper_red2 = np.array([180, 255, 180])
    
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    red_mask = cv2.add(mask1, mask2)
    
    # Morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel)
    
    # Find contours
    contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter small contours
    min_contour_area = 500
    large_contours = [c for c in contours if cv2.contourArea(c) > min_contour_area]
    
    # Calculate red pixel ratio from significant regions
    red_mask_filtered = np.zeros_like(red_mask)
    cv2.drawContours(red_mask_filtered, large_contours, -1, 255, -1)
    
    total_pixels = frame.shape[0] * frame.shape[1]
    red_pixels = np.count_nonzero(red_mask_filtered)
    red_ratio = red_pixels / total_pixels
    num_red_regions = len(large_contours)
    
    # Bleeding requires both sufficient red pixels AND concentrated regions
    is_bleeding = (red_ratio > 0.30) and (num_red_regions >= 2)
    
    # Create visualization
    vis = frame.copy()
    
    # Draw contours on original frame
    cv2.drawContours(vis, large_contours, -1, (0, 0, 255), 2)
    
    # Show metrics
    status = f"BLEEDING: {is_bleeding} | RedRatio: {red_ratio:.4f} | Regions: {num_red_regions}"
    color = (0, 0, 255) if is_bleeding else (0, 255, 0)
    cv2.putText(vis, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
    # Create mask visualization
    mask_bgr = cv2.cvtColor(red_mask_filtered, cv2.COLOR_GRAY2BGR)
    
    return vis, mask_bgr, red_ratio, is_bleeding


def main():
    video_path = r"C:\Users\thesa\Desktop\Sem-7\Healthcare\LaparoscopicDetection\surgery.mp4"
    
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) if cap.get(cv2.CAP_PROP_FPS) > 0 else 30
    
    print(f"\nDiagnostic Mode: Visualizing Smoke and Bleeding Detection")
    print("Press 'q' to quit, 's' to save current frame")
    print("=" * 70)
    
    frame_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        timestamp = frame_count / fps
        
        # Resize for display
        frame_small = cv2.resize(frame, (640, 480))
        
        # Detect smoke
        smoke_vis, smoke_edges, smoke_score, is_smoky = detect_smoke_visual(frame_small)
        
        # Detect bleeding
        bleed_vis, bleed_mask, red_ratio, is_bleeding = detect_bleeding_visual(frame_small)
        
        # Combine into grid
        # Top row: Original, Smoke Detection, Smoke Edges
        # Bottom row: Original, Bleeding Detection, Bleeding Mask
        
        row1 = np.hstack([frame_small, smoke_vis, smoke_edges])
        row2 = np.hstack([frame_small, bleed_vis, bleed_mask])
        
        combined = np.vstack([row1, row2])
        
        # Add labels
        cv2.putText(combined, "Original", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        cv2.putText(combined, "Smoke Detection", (650, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        cv2.putText(combined, "Smoke Edges", (1290, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        cv2.putText(combined, "Bleeding Detection", (650, 500), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        cv2.putText(combined, "Bleeding Mask", (1290, 500), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        cv2.putText(combined, f"Frame: {frame_count} | Time: {timestamp:.2f}s", (10, 970), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        cv2.imshow('Diagnostic: Smoke & Bleeding Detection', combined)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            os.makedirs("diagnostic_output", exist_ok=True)
            cv2.imwrite(f"diagnostic_output/frame_{frame_count}_smoke_{smoke_score:.3f}_bleed_{red_ratio:.4f}.png", combined)
            print(f"Saved frame: {frame_count}")
        
        if frame_count % 30 == 0:
            print(f"[Frame {frame_count}] Smoke: {is_smoky} ({smoke_score:.3f}) | Bleeding: {is_bleeding} ({red_ratio:.4f})")
    
    cap.release()
    cv2.destroyAllWindows()
    print("\nDiagnostic complete!")


if __name__ == "__main__":
    main()
