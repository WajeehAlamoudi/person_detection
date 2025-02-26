import queue
import threading
import time
import torch.cuda
from ultralytics import YOLO
import cv2
import subprocess

detection_model = YOLO("models/model (2).pt")
class_names = detection_model.names
print(class_names)
device = "cuda" if torch.cuda.is_available() else "cpu"
Y_THRESHOLD = 100

sound_files = {0: "sounds/0.wav", 1: "sounds/1.wav"}
sound_queue = queue.Queue()
last_played = {}
COOLDOWN_TIME = 20
SOUND_DELAY = 4


def sound_player():
    """Continuously play sounds from the queue with cooldown management."""
    while True:
        sound_file = sound_queue.get()  # Get the next sound file
        if sound_file is None:
            break  # Stop thread if None is received

        # Check if cooldown time has passed
        current_time = time.time()
        if sound_file in last_played and (current_time - last_played[sound_file] < COOLDOWN_TIME):
            print(f"Skipped {sound_file}, still in cooldown.")
            continue  # Skip if cooldown is active

        # Play the sound (blocking call)
        subprocess.run(["start", sound_file], shell=True)  # Windows

        # Update last played time
        last_played[sound_file] = current_time

        # Ensure at least SOUND_DELAY seconds between different sounds
        time.sleep(SOUND_DELAY)


# Start sound player thread (runs in the background)
sound_thread = threading.Thread(target=sound_player, daemon=True)
sound_thread.start()


def person_detection(video_source):
    try:
        video = cv2.VideoCapture(video_source)
        # To stream video at its real FPS
        fps = int(video.get(cv2.CAP_PROP_FPS))
        delay = int(1000 / fps) if fps > 0 else 30

        if not video.isOpened():
            raise Exception(f"Error: Could not stream video {video_source}.")

    except Exception as e:
        print(str(e))
        exit()

    while video.isOpened():
        ret, frame = video.read()
        # Stop loop when the video ends
        if not ret:
            break

        # detect on video or live stream
        results = detection_model(frame, iou=0.5, half=(device == "cuda"))
        # draw threshold line max distance to produce sound 10 m
        frame = cv2.line(frame,
                         (0, Y_THRESHOLD),
                         (frame.shape[1],
                          Y_THRESHOLD),
                         (0, 255, 255),
                         2)
        for result in results:
            for box in result.boxes:

                class_id = int(box.cls[0])
                class_name = class_names.get(class_id, "Unknown")
                conf = round(float(box.conf[0]), 2)

                if class_name == "person" and conf >= 0.4:

                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    # draw on the detections
                    frame = cv2.putText(img=frame,
                                        text=f'Person {conf}',
                                        org=(x1, y1 - 10),
                                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                        fontScale=0.35,
                                        color=(255, 45, 76),
                                        thickness=1)

                    frame = cv2.rectangle(img=frame,
                                          pt1=(x1, y1),
                                          pt2=(x2, y2),
                                          color=(255, 45, 76)
                                          , thickness=1)

                    x, y = int((x1 + x2) / 2), int((y1 + y2) / 2)
                    # Check if the center is below the threshold line
                    if y >= Y_THRESHOLD:
                        # play the sound
                        play_sound_file = sound_files.get(video_source, "default.wav")
                        # Check cooldown before adding to queue
                        current_time = time.time()
                        if play_sound_file not in last_played or (
                                current_time - last_played[play_sound_file] > COOLDOWN_TIME):
                            sound_queue.put(play_sound_file)  # Add to queue
                            print(f"✅ Sound Queued: {play_sound_file}")
                        else:
                            print(f"⚠️ Skipped {play_sound_file} (cooldown active)")

        # optional: To display the video
        cv2.imshow(f"video source {video_source}", frame)
        if cv2.waitKey(delay) & 0xFF == ord('q'):  # Exit on 'q' press
            break

    # Release video and close window
    video.release()
    cv2.destroyAllWindows()
