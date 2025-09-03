import cv2
import numpy as np
from typing import Optional


class TrackerDebugger:
    """Separate debugging utilities for troop tracking"""

    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self.frame_count = 0

    def debug_features(self, roi, corners, track_id):
        """Quick feature visualization"""
        if not self.enabled or corners is None:
            return

        debug_roi = cv2.cvtColor(roi, cv2.COLOR_GRAY2BGR)

        for i, corner in enumerate(corners):
            x, y = corner.ravel()
            cv2.circle(debug_roi, (int(x), int(y)), 2, (0, 255, 0), -1)

        cv2.putText(debug_roi, f"T{track_id}: {len(corners)} features",
                    (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)

        cv2.imshow(f"Debug_Track_{track_id}", debug_roi)

    def debug_displacement(self, good_old, good_new, displacement, track_id):
        """Quick displacement analysis"""
        if not self.enabled or len(good_new) == 0:
            return 0, 0

        try:
            displacements = good_new - good_old
            magnitudes = np.linalg.norm(displacements, axis=1)
            avg_displacement = np.mean(displacements, axis=0)

            # Safely convert to scalars using .item() method
            avg_mag = np.mean(magnitudes).item()

            # Ensure avg_displacement is a 1D array and extract x,y components
            avg_displacement = np.asarray(avg_displacement).flatten()
            disp_x = avg_displacement[0].item() if len(
                avg_displacement) > 0 else 0.0
            disp_y = avg_displacement[1].item() if len(
                avg_displacement) > 1 else 0.0

            print(
                f"T{track_id}: {len(good_new)} features, avg_mag={avg_mag:.2f}, disp=({disp_x:.1f},{disp_y:.1f})")

            return disp_x, disp_y

        except Exception as e:
            print(f"Debug displacement error for track {track_id}: {e}")
            return 0.0, 0.0

    def test_lk_parameters(self, roi_prev, roi_curr, corners, track_id):
        """Test LK parameters every 30 frames"""
        if not self.enabled or self.frame_count % 30 != 0:
            return

        configs = [
            ((15, 15), 2, "current"),
            ((25, 25), 2, "larger"),
            ((15, 15), 3, "more_levels")
        ]

        print(f"\n--- T{track_id} LK Parameter Test ---")
        for (win_size, max_level, name) in configs:
            try:
                lk_params = dict(winSize=win_size, maxLevel=max_level,
                                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

                new_points, status, error = cv2.calcOpticalFlowPyrLK(
                    roi_prev, roi_curr, corners, None, **lk_params)

                good_count = np.sum(status.flatten() ==
                                    1) if status is not None else 0
                print(f"  {name}: {good_count}/{len(corners)} tracked")
            except:
                print(f"  {name}: FAILED")

    def increment_frame(self):
        """Call this once per frame"""
        if self.enabled:
            self.frame_count += 1
