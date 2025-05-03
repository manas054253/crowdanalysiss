# detector.py (Corrected)
import os
import cv2
import torch
import time
import numpy as np
import pygame
from ultralytics import YOLO
from deep_sort.utils.parser import get_config
from deep_sort.deep_sort import DeepSort

class OvercrowdingDetector:
    def __init__(self, video_source=0, model_path="yolov8n.pt", deep_sort_weights="deep_sort/deep/checkpoint/ckpt.t7", use_camera=True):
        self.video_source = video_source
        self.model = YOLO(model_path)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.tracker = DeepSort(model_path=deep_sort_weights, max_age=70)
        self.alert_cooldown = 5  # seconds
        self.alert_active = False
        self.last_alert_time = 0
        self.sound_available = self._init_audio()
        
        # Only initialize camera if requested
        self.cap = None
        if use_camera:
            self.cap = cv2.VideoCapture(video_source)
            if self.cap.isOpened():
                self.cap.set(3, 1280)
                self.cap.set(4, 720)
                self.frame_width = int(self.cap.get(3))
                self.frame_height = int(self.cap.get(4))
                print(f"Camera initialized: {self.frame_width}x{self.frame_height}")
            else:
                print(f"Warning: Could not open camera source {video_source}")
                # Set default dimensions if camera fails but use_camera was true
                self.frame_width = 1280 
                self.frame_height = 720
        else:
            # Default dimensions for when no camera is used
            self.frame_width = 1280
            self.frame_height = 720
            
        self.person_class_id = 0  # 'person' class ID in YOLO
        self.start_time = time.perf_counter()
        self.counter = 0
        self.fps = 0
        self.threshold = 5 # Default threshold

    def _init_audio(self):
        try:
            pygame.init()
            pygame.mixer.init()
            # Ensure the path is correct relative to where you run the script
            siren_path = "siren.wav" 
            if not os.path.exists(siren_path):
                 print(f"Warning: Siren sound file not found at {os.path.abspath(siren_path)}")
                 return False
            self.siren_sound = pygame.mixer.Sound(siren_path)
            print("Audio initialized successfully.")
            return True
        except Exception as e:
            print(f"Warning: Siren sound file not found or audio system initialization failed. Error: {e}")
            return False

    # **** THIS IS THE FIX: Indent this method ****
    def process_single_frame(self, frame):
        """Processes a single frame for person detection and tracking."""
        try:
            # Ensure we have a valid frame
            if frame is None or frame.size == 0:
                # Return a placeholder or raise a specific error
                print("Error: Received invalid or empty frame.")
                # Create a black frame with an error message
                error_frame = np.zeros((self.frame_height, self.frame_width, 3), dtype=np.uint8)
                cv2.putText(error_frame, "Invalid Frame Received", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                return error_frame, 0, False # Return default values
            
            # Ensure frame dimensions are known (might happen if camera wasn't used initially)
            if not hasattr(self, 'frame_height') or self.frame_height <= 0:
                self.frame_height, self.frame_width = frame.shape[:2]
                if self.frame_height <= 0 or self.frame_width <= 0:
                     print("Error: Frame has invalid dimensions after read.")
                     # Return error frame again
                     error_frame = np.zeros((480, 640, 3), dtype=np.uint8) # Use generic default
                     cv2.putText(error_frame, "Invalid Frame Dimensions", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                     return error_frame, 0, False

            # Process frame with YOLO
            # Added explicit check for person class and confidence
            results = self.model(frame, device=self.device, classes=[self.person_class_id], conf=0.5, verbose=False)
            
            # Track people using DeepSORT
            current_tracks = self._track_people(results, frame)
            current_count = len(current_tracks)
            
            # Handle alert status
            self._handle_alert(current_count, self.threshold)
            
            # Create a display frame with UI elements
            display_frame = frame.copy() # Work on a copy
            self._draw_ui(display_frame, current_tracks, current_count, self.threshold)
            
            # Update FPS calculation
            self.counter += 1
            current_time = time.perf_counter()
            elapsed = current_time - self.start_time
            if elapsed >= 1.0:
                self.fps = self.counter / elapsed
                self.counter = 0
                self.start_time = current_time # Reset start time

            return display_frame, current_count, self.alert_active
        
        except Exception as e:
            import traceback
            print(f"Error processing frame: {e}")
            traceback.print_exc() # Print detailed traceback
            # Return original frame with an error message drawn on it
            error_display_frame = frame.copy() if frame is not None else np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(error_display_frame, "Processing Error", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            return error_display_frame, 0, False # Return defaults on error


    def _track_people(self, results, frame):
        """Updates the tracker with YOLO results and returns active tracks."""
        tracks = []
        detections = []
        
        # Process results from YOLO
        for result in results:
            boxes = result.boxes  # Access the Boxes object
            if boxes is not None and len(boxes) > 0:
                xywhs = boxes.xywh.cpu().numpy()
                confs = boxes.conf.cpu().numpy()
                # Prepare detections in the format DeepSORT expects: [x1, y1, x2, y2, confidence]
                # Need conversion from xywh to xyxy first
                for i in range(len(xywhs)):
                    x_center, y_center, w, h = xywhs[i]
                    x1 = x_center - w / 2
                    y1 = y_center - h / 2
                    x2 = x_center + w / 2
                    y2 = y_center + h / 2
                    # Append as [x1, y1, x2, y2, conf] - although deepsort might use xywh internally
                    # Let's stick to the common format often used with DeepSORT examples
                    # detections.append([x1, y1, x2, y2, confs[i]]) 
                    
                    # **Correction:** DeepSORT's update method expects xywh and confidences separately.
                    pass # We'll use xywhs and confs directly below.

                # Make sure frame dimensions are valid
                if frame is not None and frame.shape[0] > 0 and frame.shape[1] > 0:
                   # Update tracker using xywh format
                   _ = self.tracker.update(xywhs, confs, frame)
                else:
                   print("Warning: Skipping tracker update due to invalid frame.")


        # Get confirmed tracks from the tracker
        # Check if self.tracker and self.tracker.tracker exist
        if hasattr(self, 'tracker') and hasattr(self.tracker, 'tracker') and self.tracker.tracker is not None:
            for track in self.tracker.tracker.tracks:
                if not track.is_confirmed() or track.time_since_update > 1:
                    continue
                tracks.append(track)
        else:
             print("Warning: Tracker object not fully initialized or accessible.")
             
        return tracks

    def _handle_alert(self, current_count, threshold):
        """Manages the alert state based on count, threshold, and cooldown."""
        current_time = time.time()
        # Activate alert if count meets threshold AND cooldown has passed
        if current_count >= threshold and not self.alert_active:
             if (current_time - self.last_alert_time) > self.alert_cooldown:
                 self.alert_active = True
                 self.last_alert_time = current_time
                 print(f"ALERT Triggered: Count {current_count} >= Threshold {threshold}")
                 if self.sound_available and self.siren_sound:
                     try:
                        self.siren_sound.play()
                     except Exception as audio_err:
                         print(f"Error playing sound: {audio_err}")
        # Deactivate alert if count drops below threshold (add a small buffer time perhaps?)
        elif current_count < threshold and self.alert_active:
             # Optional: Add a delay before deactivating alert?
             # if current_time - self.last_alert_time > 3: # Example: Deactivate after 3s below threshold
             self.alert_active = False
             print("Alert Deactivated")
             if self.sound_available and self.siren_sound:
                 try:
                     self.siren_sound.stop() # Stop the sound if it was playing continuously
                 except Exception as audio_err:
                      print(f"Error stopping sound: {audio_err}")


    def _draw_ui(self, frame, tracks, count, threshold):
        """Draws bounding boxes, stats, and alert messages on the frame."""
        # Make sure frame dimensions are valid
        if frame is None or frame.shape[0] <= 0 or frame.shape[1] <= 0:
             print("Warning: Cannot draw UI on invalid frame.")
             return 

        frame_height, frame_width = frame.shape[:2]

        # Draw bounding boxes for tracked people
        for idx, track in enumerate(tracks, 1):
             # Check if track has the necessary attribute
             if hasattr(track, 'to_tlbr'):
                 try:
                     x1, y1, x2, y2 = map(int, track.to_tlbr()) # Top-left, bottom-right
                     # Ensure coordinates are within frame boundaries
                     x1, y1 = max(0, x1), max(0, y1)
                     x2, y2 = min(frame_width - 1, x2), min(frame_height - 1, y2)
                     
                     color = (0, 0, 255) if self.alert_active else (0, 255, 0) # Red if alert, Green otherwise
                     cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                     
                     # Add track ID (optional)
                     track_id = track.track_id
                     cv2.putText(frame, f"ID:{track_id}", (x1 + 5, y1 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                 except Exception as e:
                      print(f"Error drawing track {getattr(track, 'track_id', 'N/A')}: {e}")
             else:
                  print(f"Warning: Track object {idx} missing 'to_tlbr' method.")


        # Draw Stats - Use white text with black outline for better visibility
        stats_color = (255, 255, 255)
        outline_color = (0, 0, 0)
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        # Position stats relative to frame size
        y_pos = 40
        line_height = 40
        
        cv2.putText(frame, f"People Count: {count}", (10, y_pos), font, 1, outline_color, 4, cv2.LINE_AA)
        cv2.putText(frame, f"People Count: {count}", (10, y_pos), font, 1, stats_color, 2, cv2.LINE_AA)
        y_pos += line_height

        cv2.putText(frame, f"FPS: {int(self.fps)}", (10, y_pos), font, 1, outline_color, 4, cv2.LINE_AA)
        cv2.putText(frame, f"FPS: {int(self.fps)}", (10, y_pos), font, 1, stats_color, 2, cv2.LINE_AA)
        y_pos += line_height

        cv2.putText(frame, f"Threshold: {threshold}", (10, y_pos), font, 1, outline_color, 4, cv2.LINE_AA)
        cv2.putText(frame, f"Threshold: {threshold}", (10, y_pos), font, 1, stats_color, 2, cv2.LINE_AA)


        # Draw alert overlay if active
        if self.alert_active:
            # Red semi-transparent overlay at the bottom
            overlay_height = 100
            overlay_y_start = frame_height - overlay_height
            
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, overlay_y_start), (frame_width, frame_height), (0, 0, 200), -1) # Solid red
            alpha = 0.6 # Transparency factor
            cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame) # Apply overlay

            # Alert text
            alert_msg = f"ALERT: Overcrowding! ({count}/{threshold})"
            text_size, _ = cv2.getTextSize(alert_msg, font, 1.2, 3)
            text_x = (frame_width - text_size[0]) // 2
            text_y = overlay_y_start + (overlay_height + text_size[1]) // 2
            
            cv2.putText(frame, alert_msg, (text_x, text_y), font, 1.2, outline_color, 6, cv2.LINE_AA)
            cv2.putText(frame, alert_msg, (text_x, text_y), font, 1.2, (255, 255, 0), 3, cv2.LINE_AA) # Yellow text


    def run(self, threshold=5):
        """Runs the detection loop using the initialized camera source."""
        if not self.cap or not self.cap.isOpened():
            print("Error: Camera not initialized or failed to open.")
            return
            
        self.threshold = threshold
        print(f"[INFO] Detection started. Alert threshold: {self.threshold}")
        
        while True: # Changed from cap.isOpened() to handle potential re-reads
            ret, frame = self.cap.read()
            if not ret:
                print("Warning: Failed to grab frame. End of video or camera error?")
                # Optional: Try to reopen camera or break
                time.sleep(0.5) # Wait a bit before retrying or breaking
                # Check if camera is still open
                if not self.cap.isOpened():
                     print("Camera is closed. Exiting run loop.")
                     break
                continue # Try reading next frame

            # Process the frame
            display_frame, current_count, alert_active = self.process_single_frame(frame)

            # Display the frame
            cv2.imshow("Overcrowding Detection", display_frame)

            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("Exit requested.")
                break
            elif key in [ord('+'), ord('=')]:
                self.threshold += 1
                print(f"Threshold increased to: {self.threshold}")
                # Immediately update alert status based on new threshold
                self._handle_alert(current_count, self.threshold) 
            elif key == ord('-') and self.threshold > 1:
                self.threshold -= 1
                print(f"Threshold decreased to: {self.threshold}")
                 # Immediately update alert status based on new threshold
                self._handle_alert(current_count, self.threshold)

        # Clean up
        print("Releasing resources.")
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        if self.sound_available:
             pygame.mixer.quit()
             pygame.quit()