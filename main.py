import cv2
import mediapipe as mp
import time

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)


def get_finger_states(landmarks):
    tips   = [8, 12, 16, 20]
    middle = [6, 10, 14, 18]

    fingers = {}
    names = ["index", "middle", "ring", "pinky"]

    for name, tip, mid in zip(names, tips, middle):
        fingers[name] = landmarks[tip].y < landmarks[mid].y

    thumb_tip  = landmarks[4]
    thumb_base = landmarks[2]
    fingers["thumb"] = abs(thumb_tip.x - thumb_base.x) > 0.06

    return fingers


def detect_letter(landmarks):
    f = get_finger_states(landmarks)

    all_curled   = not f["index"] and not f["middle"] and not f["ring"] and not f["pinky"]
    all_extended = f["index"] and f["middle"] and f["ring"] and f["pinky"]

    def dist(a, b):
        return ((a.x - b.x)**2 + (a.y - b.y)**2) ** 0.5

    thumb_tip = landmarks[4]
    index_tip = landmarks[8]
    mid_tip   = landmarks[12]

    wrist_y      = landmarks[0].y
    index_tip_y  = landmarks[8].y
    middle_tip_y = landmarks[12].y
    index_pip_y  = landmarks[6].y
    middle_pip_y = landmarks[10].y

    index_mcp_y = landmarks[5].y
    index_mcp_x = landmarks[5].x
    mid_mcp_y   = landmarks[9].y
    mid_mcp_x   = landmarks[9].x

    # --- K: index + middle UP, thumb UP between them, ring + pinky curled ---
    thumb_pointing_up = landmarks[4].y < landmarks[2].y
    thumb_between = (min(landmarks[5].x, landmarks[9].x) < landmarks[4].x <
                     max(landmarks[5].x, landmarks[9].x))

    thumb_pointing_out = abs(thumb_tip.x - landmarks[2].x) > 0.08
    if (f["index"]
        and not f["middle"] and not f["ring"] and not f["pinky"]
        and thumb_pointing_out
        and not dist(thumb_tip, mid_tip) < 0.08):   # excludes D
        return "L"

    if (f["index"] and f["middle"]
            and not f["ring"] and not f["pinky"]
            and thumb_pointing_up
            and thumb_between):
        return "K"

    # --- pinky only (base for I and J) ---
    pinky_only = (f["pinky"]
                  and not f["index"]
                  and not f["middle"]
                  and not f["ring"])

    # --- J: pinky up + thumb extended OUT ---
    if pinky_only and f["thumb"]:
        return "J"

    # --- I: pinky up + thumb tucked IN ---
    if pinky_only and not f["thumb"]:
        return "I"

    # --- H: index AND middle both horizontal, ring/pinky curled, thumb tucked ---
    index_horiz       = abs(landmarks[8].y - index_mcp_y) < 0.12
    index_out         = abs(landmarks[8].x - index_mcp_x) > 0.06
    middle_horiz      = abs(landmarks[12].y - mid_mcp_y) < 0.12
    middle_out        = abs(landmarks[12].x - mid_mcp_x) > 0.06
    ring_pinky_curled = not f["ring"] and not f["pinky"]

    if (index_horiz and index_out
            and middle_horiz and middle_out
            and ring_pinky_curled
            and not f["thumb"]):
        return "H"

    # --- G: index horizontal, thumb sideways, middle/ring/pinky curled ---
    index_horizontal   = abs(landmarks[8].y - index_mcp_y) < 0.12
    index_pointing_out = abs(landmarks[8].x - index_mcp_x) > 0.06
    thumb_sideways     = abs(thumb_tip.x - landmarks[0].x) > 0.08
    others_curled      = not f["middle"] and not f["ring"] and not f["pinky"]

    if index_horizontal and index_pointing_out and thumb_sideways and others_curled:
        return "G"

    # --- C: fingers curved open, thumb spread ---
    index_mid  = index_tip_y < wrist_y and index_tip_y > index_pip_y
    middle_mid = middle_tip_y < wrist_y and middle_tip_y > middle_pip_y

    if (not all_extended and not all_curled
            and f["thumb"]
            and index_mid
            and middle_mid):
        return "C"

    # --- D: index UP, middle/ring/pinky curled, thumb touches MIDDLE fingertip ---
    thumb_touches_middle = dist(thumb_tip, mid_tip) < 0.08
    if (f["index"]
            and not f["middle"] and not f["ring"] and not f["pinky"]
            and thumb_touches_middle):
        return "D"

    # --- F: index curled, middle/ring/pinky UP, thumb touches INDEX fingertip ---
    thumb_touches_index = dist(thumb_tip, index_tip) < 0.08
    if (not f["index"]
            and f["middle"] and f["ring"] and f["pinky"]
            and thumb_touches_index):
        return "F"

    # --- E: all 4 fingers curled, thumb tucked UNDER fingers ---
    thumb_tip_y  = landmarks[4].y
    thumb_tip_x  = landmarks[4].x
    index_base_x = landmarks[5].x

    thumb_tucked_under = (thumb_tip_y > index_pip_y and
                          abs(thumb_tip_x - index_base_x) < 0.12)

    if all_curled and thumb_tucked_under:
        return "E"

    # --- N: index + middle curled OVER thumb, ring + pinky curled, thumb tucked under index+middle ---
    index_curled  = not f["index"]
    middle_curled = not f["middle"]
    ring_curled   = not f["ring"]
    pinky_curled  = not f["pinky"]

    thumb_tip_x  = landmarks[4].x
    index_tip_x  = landmarks[8].x
    middle_tip_x = landmarks[12].x

    # Thumb is tucked between/under index and middle (horizontally between their tips)
    thumb_under_index_middle = (min(index_tip_x, middle_tip_x) - 0.05
                                 < thumb_tip_x <
                                 max(index_tip_x, middle_tip_x) + 0.05)

    # Thumb tip is lower than index and middle PIP joints (knuckles)
    thumb_below_knuckles = (landmarks[4].y > landmarks[7].y and
                            landmarks[4].y > landmarks[11].y)

    if (index_curled and middle_curled
            and ring_curled and pinky_curled
            and thumb_under_index_middle
            and thumb_below_knuckles):
        return "N"

    # --- A: fist, thumb on the side ---
    if all_curled and not f["thumb"]:
        return "A"

    # --- B: all 4 fingers up, thumb tucked in ---
    if all_extended and not f["thumb"]:
        return "B"

    return None


def main():
    with mp_hands.Hands(
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5,
    ) as hands:
        while True:
            ret, frame = cap.read()
            max_attempts = 5
            attempt = 0
            while not ret and attempt < max_attempts:
                attempt += 1
                time.sleep(0.1)
                ret, frame = cap.read()
            if not ret:
                print(f"Failed to read frame after {max_attempts} attempts.")
                break

            frame = cv2.flip(frame, 1)
            h, w, c = frame.shape
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)

            detected_letter = None

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_draw.draw_landmarks(
                        frame,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                    )

                    detected_letter = detect_letter(hand_landmarks.landmark)

                    fingertips = {
                        "thumb":  hand_landmarks.landmark[4],
                        "index":  hand_landmarks.landmark[8],
                        "middle": hand_landmarks.landmark[12],
                        "ring":   hand_landmarks.landmark[16],
                        "pinky":  hand_landmarks.landmark[20],
                    }
                    for name, landmark in fingertips.items():
                        x, y = int(landmark.x * w), int(landmark.y * h)
                        cv2.circle(frame, (x, y), 6, (0, 255, 0), -1)

            if detected_letter:
                cv2.putText(frame, detected_letter, (30, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 255, 0), 6, cv2.LINE_AA)
                cv2.putText(frame, f"Sign: {detected_letter}", (30, 160),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2, cv2.LINE_AA)
            else:
                cv2.putText(frame, "No sign detected", (30, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 100, 100), 2, cv2.LINE_AA)

            cv2.imshow("ASL Sign Detection - A to N", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()