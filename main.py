import cv2
import os
import random
from deepface import DeepFace
import pygame

# Initialize the mixer
pygame.mixer.init()

# Map emotion to folder name
emotion_to_folder = {
    'happy': 'music/happy',
    'sad': 'music/sad',
    'angry': 'music/angry',
    'neutral': 'music/neutral'
}

def play_music(emotion):
    folder = emotion_to_folder.get(emotion.lower(), 'music/neutral')
    if not os.path.exists(folder):
        print(f"No folder found for emotion: {emotion}")
        return

    songs = [f for f in os.listdir(folder) if f.endswith('.mp3')]
    if not songs:
        print(f"No songs in folder: {folder}")
        return

    song_path = os.path.join(folder, random.choice(songs))
    print(f"Playing: {song_path}")
    pygame.mixer.music.load(song_path)
    pygame.mixer.music.play()

def main():
    cap = cv2.VideoCapture(0)
    current_emotion = None
    print("Press 'q' to quit...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        try:
            result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
            emotion = result[0]['dominant_emotion']

            # Display the detected emotion on the screen
            cv2.putText(frame, f"Emotion: {emotion}", (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Play music only when emotion changes
            if emotion != current_emotion:
                current_emotion = emotion
                pygame.mixer.music.stop()
                play_music(emotion)

        except Exception as e:
            print("Detection error:", e)

        cv2.imshow("FaceMood", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    pygame.mixer.music.stop()

if __name__ == "__main__":
    main()
