import mediapipe as mp
import numpy as np
import time
import winsound
import cv2
from datetime import datetime

# Initialize Mediapipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Open the webcam
cap = cv2.VideoCapture(0)

# Drowsiness Detection Constants
EYE_AR_THRESHOLD = 0.25
TILT_THRESHOLD = 15
TIME_THRESHOLD = 2  # Must be drowsy for 2 seconds
BRIGHTNESS_THRESHOLD = 50  # If brightness is below this, apply enhancements
FACE_ABSENCE_THRESHOLD = 3  # Alert if no face detected for 3 seconds
start_eye_time = None
start_tilt_time = None
start_no_face_time = None

# UI Constants
BACKGROUND_COLOR = (25, 25, 25)  # Dark background
PANEL_COLOR = (45, 45, 45)  # Slightly lighter panels
TEXT_COLOR = (220, 220, 220)  # Light text
ACCENT_COLOR = (0, 120, 215)  # Blue accent
WARNING_COLOR = (0, 165, 255)  # Orange warning
ALERT_COLOR = (0, 0, 255)  # Red alert
SUCCESS_COLOR = (0, 255, 0)  # Green success
FONT = cv2.FONT_HERSHEY_SIMPLEX

# Initialize session stats
session_start = time.time()
alert_count = 0
last_frame_time = time.time()
fps_list = [30.0]  # Initialize with a default value to avoid division by zero


def get_brightness(frame):
    """Calculates the average brightness of the frame."""
    try:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return np.mean(gray)
    except:
        return 100  # Default value if conversion fails


def apply_low_light_enhancements(frame):
    """Enhances the frame in low-light conditions."""
    try:
        # Convert to YUV color space
        yuv = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)

        # Extract the Y channel (luminance)
        y = yuv[:, :, 0]

        # Apply CLAHE
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        y = clahe.apply(y)

        # Gamma correction
        gamma = 1.8
        look_up_table = np.array([((i / 255.0) ** (1.0 / gamma)) * 255 for i in np.arange(0, 256)]).astype("uint8")
        y = cv2.LUT(y, look_up_table)

        # Replace the Y channel and convert back to BGR
        yuv[:, :, 0] = y
        enhanced_frame = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)

        return enhanced_frame
    except:
        return frame  # Return original frame if enhancement fails


def draw_filled_rect(img, pt1, pt2, color):
    """Draw a filled rectangle with optional alpha blending"""
    cv2.rectangle(img, pt1, pt2, color, -1)


def draw_text_with_background(img, text, position, font, scale, text_color, bg_color, thickness=1, padding=5):
    """Draw text with a background rectangle"""
    (text_width, text_height), baseline = cv2.getTextSize(text, font, scale, thickness)
    x, y = position

    # Draw background rectangle
    draw_filled_rect(img,
                     (x - padding, y - text_height - padding),
                     (x + text_width + padding, y + padding),
                     bg_color)

    # Draw text
    cv2.putText(img, text, position, font, scale, text_color, thickness)


def create_ui(frame, metrics):
    """Create a simplified UI overlay on the frame"""
    height, width, _ = frame.shape
    ui = frame.copy()

    # Add semi-transparent overlay at the top for title
    header_height = 60
    draw_filled_rect(ui, (0, 0), (width, header_height), BACKGROUND_COLOR)

    # Add title
    cv2.putText(ui, "DRIVER DROWSINESS DETECTION SYSTEM", (20, 40),
                FONT, 0.8, TEXT_COLOR, 2)

    # Add time
    current_time = datetime.now().strftime("%H:%M:%S")
    time_text = f"Time: {current_time}"
    text_size = cv2.getTextSize(time_text, FONT, 0.6, 1)[0]
    cv2.putText(ui, time_text, (width - text_size[0] - 20, 40),
                FONT, 0.6, TEXT_COLOR, 1)

    # Add semi-transparent overlay on the right for metrics
    panel_width = 300
    draw_filled_rect(ui, (width - panel_width, header_height), (width, height), (0, 0, 0, 180))

    # Status indicators
    status_y = header_height + 40

    # Face detection status
    face_status = "FACE DETECTED" if metrics['face_detected'] else "NO FACE DETECTED"
    face_color = SUCCESS_COLOR if metrics['face_detected'] else ALERT_COLOR
    draw_text_with_background(ui, face_status, (width - panel_width + 20, status_y),
                              FONT, 0.6, face_color, PANEL_COLOR)

    # Eye status
    status_y += 40
    eye_status = "EYES OPEN" if metrics['eyes_open'] else "EYES CLOSED"
    eye_color = SUCCESS_COLOR if metrics['eyes_open'] else WARNING_COLOR
    draw_text_with_background(ui, eye_status, (width - panel_width + 20, status_y),
                              FONT, 0.6, eye_color, PANEL_COLOR)

    # Head tilt status
    status_y += 40
    tilt_status = "HEAD ALIGNED" if not metrics['head_tilted'] else "HEAD TILTED"
    tilt_color = SUCCESS_COLOR if not metrics['head_tilted'] else WARNING_COLOR
    draw_text_with_background(ui, tilt_status, (width - panel_width + 20, status_y),
                              FONT, 0.6, tilt_color, PANEL_COLOR)

    # Light status
    status_y += 40
    light_status = "GOOD LIGHTING" if metrics['brightness'] >= BRIGHTNESS_THRESHOLD else "LOW LIGHT"
    light_color = SUCCESS_COLOR if metrics['brightness'] >= BRIGHTNESS_THRESHOLD else WARNING_COLOR
    draw_text_with_background(ui, light_status, (width - panel_width + 20, status_y),
                              FONT, 0.6, light_color, PANEL_COLOR)

    # Alert status
    status_y += 60
    if metrics['alert_active']:
        draw_text_with_background(ui, metrics['alert_message'],
                                  (width - panel_width + 20, status_y),
                                  FONT, 0.7, ALERT_COLOR, PANEL_COLOR, 2)

    # Metrics
    metrics_y = status_y + 60

    # EAR value
    ear_text = f"Eye Aspect Ratio: {metrics['ear']:.2f}"
    draw_text_with_background(ui, ear_text, (width - panel_width + 20, metrics_y),
                              FONT, 0.6, TEXT_COLOR, PANEL_COLOR)

    # Head tilt value
    metrics_y += 40
    tilt_text = f"Head Tilt Angle: {metrics['tilt_angle']:.1f}°"
    draw_text_with_background(ui, tilt_text, (width - panel_width + 20, metrics_y),
                              FONT, 0.6, TEXT_COLOR, PANEL_COLOR)

    # Brightness value
    metrics_y += 40
    brightness_text = f"Brightness: {metrics['brightness']:.1f}"
    draw_text_with_background(ui, brightness_text, (width - panel_width + 20, metrics_y),
                              FONT, 0.6, TEXT_COLOR, PANEL_COLOR)

    # Session info
    metrics_y += 60
    session_duration = time.time() - session_start
    hours, remainder = divmod(int(session_duration), 3600)
    minutes, seconds = divmod(remainder, 60)
    session_text = f"Session: {hours:02}:{minutes:02}:{seconds:02}"
    draw_text_with_background(ui, session_text, (width - panel_width + 20, metrics_y),
                              FONT, 0.6, TEXT_COLOR, PANEL_COLOR)

    # FPS
    metrics_y += 40
    fps_text = f"FPS: {metrics['fps']:.1f}"
    draw_text_with_background(ui, fps_text, (width - panel_width + 20, metrics_y),
                              FONT, 0.6, TEXT_COLOR, PANEL_COLOR)

    # Alert count
    metrics_y += 40
    alert_text = f"Alerts: {metrics['alert_count']}"
    draw_text_with_background(ui, alert_text, (width - panel_width + 20, metrics_y),
                              FONT, 0.6, TEXT_COLOR, PANEL_COLOR)

    # Add semi-transparent overlay at the bottom for instructions
    footer_height = 40
    draw_filled_rect(ui, (0, height - footer_height), (width, height), BACKGROUND_COLOR)
    cv2.putText(ui, "Press 'Q' to quit", (20, height - 15),
                FONT, 0.6, TEXT_COLOR, 1)

    return ui


# Main loop
while True:
    try:
        # Calculate FPS
        current_time = time.time()
        dt = current_time - last_frame_time
        if dt > 0:  # Avoid division by zero
            fps = 1 / dt
            fps_list.append(fps)
            if len(fps_list) > 30:  # Average over last 30 frames
                fps_list.pop(0)
        last_frame_time = current_time
        avg_fps = sum(fps_list) / len(fps_list)

        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        frame = cv2.flip(frame, 1)  # Correct inverted feed

        # Initialize metrics dictionary with default values
        metrics = {
            'face_detected': False,
            'eyes_open': True,
            'head_tilted': False,
            'brightness': 100,
            'ear': 0.3,
            'tilt_angle': 0,
            'alert_active': False,
            'alert_message': "",
            'fps': avg_fps,
            'alert_count': alert_count
        }

        # Check brightness and apply enhancements if needed
        brightness = get_brightness(frame)
        metrics['brightness'] = brightness

        enhanced_frame = frame
        if brightness < BRIGHTNESS_THRESHOLD:
            # Apply enhanced low-light correction
            enhanced_frame = apply_low_light_enhancements(frame)

        # Process with MediaPipe
        frame_rgb = cv2.cvtColor(enhanced_frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(frame_rgb)

        # Check if face is detected
        if not results.multi_face_landmarks:
            # No face detected
            if start_no_face_time is None:
                start_no_face_time = time.time()
            elif time.time() - start_no_face_time >= FACE_ABSENCE_THRESHOLD:
                winsound.Beep(800, 500)  # Different tone for face absence alert
                metrics['alert_active'] = True
                metrics['alert_message'] = "ALERT: NO FACE DETECTED!"
                alert_count += 1
        else:
            # Face detected, reset the no-face timer
            if start_no_face_time is not None:
                start_no_face_time = None

            metrics['face_detected'] = True

            for face_landmarks in results.multi_face_landmarks:
                height, width, _ = frame.shape

                # Extract landmarks
                try:
                    landmarks = [(int(l.x * width), int(l.y * height)) for l in face_landmarks.landmark]

                    # Eye landmarks
                    LEFT_EYE = [362, 385, 387, 263, 373, 380]
                    RIGHT_EYE = [33, 160, 158, 133, 153, 144]
                    HEAD_TILT_POINTS = [10, 152]  # Forehead & Chin

                    # Draw landmarks for visualization
                    for idx in LEFT_EYE + RIGHT_EYE + HEAD_TILT_POINTS:
                        if 0 <= idx < len(landmarks):
                            cv2.circle(frame, landmarks[idx], 2, (0, 255, 0), -1)


                    def calculate_ear(eye_points, landmarks):
                        try:
                            A = np.linalg.norm(np.array(landmarks[eye_points[1]]) - np.array(landmarks[eye_points[5]]))
                            B = np.linalg.norm(np.array(landmarks[eye_points[2]]) - np.array(landmarks[eye_points[4]]))
                            C = np.linalg.norm(np.array(landmarks[eye_points[0]]) - np.array(landmarks[eye_points[3]]))
                            if C == 0:  # Avoid division by zero
                                return 0.3  # Default value
                            return (A + B) / (2.0 * C)
                        except:
                            return 0.3  # Default value if calculation fails


                    left_ear = calculate_ear(LEFT_EYE, landmarks)
                    right_ear = calculate_ear(RIGHT_EYE, landmarks)
                    avg_ear = (left_ear + right_ear) / 2.0
                    metrics['ear'] = avg_ear

                    # Calculate head tilt angle
                    try:
                        top_point = landmarks[HEAD_TILT_POINTS[0]]
                        bottom_point = landmarks[HEAD_TILT_POINTS[1]]
                        angle = np.degrees(np.arctan2(bottom_point[0] - top_point[0], bottom_point[1] - top_point[1]))
                        metrics['tilt_angle'] = angle
                    except:
                        angle = 0  # Default if calculation fails
                        metrics['tilt_angle'] = angle

                    # Eye Drowsiness Detection
                    if avg_ear < EYE_AR_THRESHOLD:
                        metrics['eyes_open'] = False
                        if start_eye_time is None:
                            start_eye_time = time.time()
                        elif time.time() - start_eye_time >= TIME_THRESHOLD:
                            winsound.Beep(1000, 1000)  # Beep if eyes are closed for too long
                            metrics['alert_active'] = True
                            metrics['alert_message'] = "ALERT: DROWSINESS DETECTED!"
                            alert_count += 1
                    else:
                        metrics['eyes_open'] = True
                        start_eye_time = None

                    # Head Tilt Detection
                    if abs(angle) > TILT_THRESHOLD:
                        metrics['head_tilted'] = True
                        if start_tilt_time is None:
                            start_tilt_time = time.time()
                        elif time.time() - start_tilt_time >= TIME_THRESHOLD:
                            winsound.Beep(1000, 1000)  # Beep if head is tilted for too long
                            metrics['alert_active'] = True
                            metrics['alert_message'] = "ALERT: HEAD TILT DETECTED!"
                            alert_count += 1
                    else:
                        metrics['head_tilted'] = False
                        start_tilt_time = None

                except Exception as e:
                    print(f"Error processing landmarks: {e}")
                    continue

        # Create UI overlay
        ui_frame = create_ui(frame, metrics)

        # Display the frame with UI
        cv2.imshow("Driver Drowsiness Detection System", ui_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    except Exception as e:
        print(f"Error in main loop: {e}")
        # If there's an error, wait a bit before continuing
        time.sleep(0.1)
        continue

# Clean up
cap.release()
cv2.destroyAllWindows()