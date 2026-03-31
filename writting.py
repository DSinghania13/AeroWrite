import numpy as np
import time
import cv2
import os


class WritingHandler:
    def __init__(self):
        self.canvas = None
        self.writing = False
        self.last_point = None

        self.smooth_x, self.smooth_y = None, None
        self.ALPHA = 0.2
        self.MAX_JUMP = 200
        self.MIN_MOVE = 4

        self.prev_mode = "NONE"
        self.last_fist_time = 0
        self.DOUBLE_FIST_GAP = 0.6

        self.pinch_start_time = None
        self.WRITE_HOLD_TIME = 0.25
        self.pinch_grace = 0

        self.fist_start_time = None
        self.ERASE_HOLD_TIME = 1.0
        self.eraser_mode = False

        self.text = ""
        self.submitted_text = ""
        self.clear_on_next_write = False
        self.submit_display_time = 0
        self.SUBMIT_SHOW_TIME = 2.0

        # 🔥 NEW: Personalized Template Dictionary
        self.templates = {}
        self.template_dir = "my_handwriting_templates"
        os.makedirs(self.template_dir, exist_ok=True)
        self.load_templates()

    def load_templates(self):
        """Loads your saved letters when the script starts."""
        for file in os.listdir(self.template_dir):
            if file.endswith(".png"):
                # This safely reads both old "A.png" and new "A_1.png" files
                char = file.split('_')[0].replace(".png", "")
                img = cv2.imread(os.path.join(self.template_dir, file), cv2.IMREAD_GRAYSCALE)

                # If this letter isn't in our dictionary yet, create an empty list for it
                if char not in self.templates:
                    self.templates[char] = []

                self.templates[char].append(img)

        if self.templates:
            total_variations = sum(len(variations) for variations in self.templates.values())
            print(f"Loaded {total_variations} saved variations across {len(self.templates)} letters!")

    def smooth(self, pt):
        if self.smooth_x is None:
            self.smooth_x, self.smooth_y = pt
        else:
            self.smooth_x = self.ALPHA * pt[0] + (1 - self.ALPHA) * self.smooth_x
            self.smooth_y = self.ALPHA * pt[1] + (1 - self.ALPHA) * self.smooth_y
        return int(self.smooth_x), int(self.smooth_y)

    def detect_mode(self, hand):
        def is_down(tip, mcp):
            return tip.y > mcp.y

        if all([is_down(hand[8], hand[5]), is_down(hand[12], hand[9]),
                is_down(hand[16], hand[13]), is_down(hand[20], hand[17])]):
            return "FIST"

        pinch = np.linalg.norm([hand[8].x - hand[4].x, hand[8].y - hand[4].y])
        if pinch < 0.07: return "PINCH"
        return "NONE"

    def init_canvas(self, frame):
        if self.canvas is None: self.canvas = np.zeros_like(frame)

    # 🔥 NEW: Extracts a perfectly cropped 64x64 picture of your letter
    def extract_feature(self):
        if self.canvas is None: return None
        gray = cv2.cvtColor(self.canvas, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
        coords = cv2.findNonZero(binary)
        if coords is None: return None

        x, y, w, h = cv2.boundingRect(coords)
        cropped = binary[y:y + h, x:x + w]
        return cv2.resize(cropped, (64, 64))  # Stretch to perfect square

    # 🔥 NEW: Saves your drawing to a folder
    def calibrate(self, char):
        feature = self.extract_feature()
        if feature is not None:
            char = char.upper()

            # Make sure the list exists
            if char not in self.templates:
                self.templates[char] = []

            # Figure out what number this variation is
            variation_num = len(self.templates[char]) + 1
            filename = f"{char}_{variation_num}.png"

            # Save it to memory and to your hard drive
            self.templates[char].append(feature)
            cv2.imwrite(os.path.join(self.template_dir, filename), feature)

            print(f"✅ Successfully learned variation #{variation_num} for: {char}")
            self.canvas = np.zeros_like(self.canvas)
            self.last_point = None

    def update(self, hand, frame):
        now = time.time()
        status_text = "IDLE"

        if now - self.submit_display_time < self.SUBMIT_SHOW_TIME:
            status_text = f"SUBMITTED: {self.submitted_text}"

        x, y = int(hand[8].x * frame.shape[1]), int(hand[8].y * frame.shape[0])
        pt = self.smooth((x, y))
        mode = self.detect_mode(hand)

        if mode == "PINCH":
            self.pinch_grace += 1
            if self.pinch_start_time is None:
                self.pinch_start_time = now
            elif now - self.pinch_start_time > self.WRITE_HOLD_TIME and not self.writing:
                if self.clear_on_next_write:
                    self.text = ""
                    self.clear_on_next_write = False
                self.writing = True
                self.eraser_mode = False
                self.last_point = None
                self.smooth_x, self.smooth_y = None, None
        else:
            if self.pinch_grace > 3: self.pinch_start_time = None
            self.pinch_grace = 0

        if mode == "FIST":
            if self.fist_start_time is None:
                self.fist_start_time = now
            elif now - self.fist_start_time > self.ERASE_HOLD_TIME:
                self.eraser_mode = True
                self.writing = False

            if self.prev_mode != "FIST":
                if now - self.last_fist_time < self.DOUBLE_FIST_GAP:
                    self.writing = False
                    self.eraser_mode = False

                    self.process_stroke()  # Run Image Matcher

                    self.submit_display_time = now
                    self.submitted_text = self.text
                    self.clear_on_next_write = True
                    status_text = f"SUBMITTED: {self.submitted_text}"
                    self.canvas = np.zeros_like(frame)
                    self.last_point = None
                else:
                    self.writing = False
                self.last_fist_time = now
        else:
            self.fist_start_time = None
            if mode != "PINCH": self.eraser_mode = False

        self.prev_mode = mode

        if self.eraser_mode:
            if status_text == "IDLE": status_text = "ERASER MODE"
            ex, ey = int(hand[0].x * frame.shape[1]), int(hand[0].y * frame.shape[0])
            cv2.circle(frame, (ex, ey), 40, (0, 0, 255), 2)
            cv2.circle(self.canvas, (ex, ey), 40, (0, 0, 0), -1)
            self.last_point = None

        elif self.writing:
            if status_text == "IDLE": status_text = "WRITING MODE"
            if mode == "PINCH":
                if self.last_point is not None:
                    dist = np.linalg.norm(np.array(pt) - np.array(self.last_point))
                    if dist < self.MIN_MOVE: return status_text
                    if dist > self.MAX_JUMP:
                        self.last_point = pt
                        return status_text
                    cv2.line(self.canvas, self.last_point, pt, (255, 0, 0), 12)
                self.last_point = pt
            else:
                self.last_point = None

        return status_text

    # 🔥 NEW: Compares your drawing to your saved snapshots
    def process_stroke(self):
        feature = self.extract_feature()
        if feature is None or not self.templates:
            print("Canvas empty, or no templates saved yet!")
            return

        best_char = None
        best_diff = float('inf')

        # Loop through every letter (A, B, C...)
        for char, variations in self.templates.items():
            # Loop through every saved picture of that letter
            for template in variations:
                # Compare pixel-by-pixel
                diff = np.sum(cv2.absdiff(feature, template))

                if diff < best_diff:
                    best_diff = diff
                    best_char = char

        if best_char:
            self.text += best_char
            print(f"Recognized: {best_char} (Difference score: {best_diff})")

        self.last_point = None
        self.smooth_x, self.smooth_y = None, None

    def get_canvas(self):
        return self.canvas

    def get_text(self):
        return self.text

    def clear_all(self):
        self.canvas = None
        self.text = ""
        self.submitted_text = ""
        self.clear_on_next_write = False
        self.last_point = None