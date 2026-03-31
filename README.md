# ✍️ AeroWrite

**AeroWrite** is a real-time, robust air-handwriting recognition system built using **MediaPipe Hand Landmarker** and **OpenCV**.

Unlike traditional math-based stroke trackers (like DTW) or heavy AI models, AeroWrite uses a **personalized visual calibration system** (a localized K-Nearest Neighbors approach). It captures your unique handwriting as a dataset of cropped, high-contrast images and uses pixel-difference matching to recognize multi-stroke letters and numbers with incredible accuracy and zero latency.

---

## 🚀 Features

- **✋ Real-Time Hand Tracking:** Uses MediaPipe to track 21 3D hand landmarks natively on your CPU.
- **✍️ Natural Writing Gestures:** Pinch to draw, open your hand to "lift the pen," and pump a double-fist to submit.
- **🧩 Multi-Stroke Support:** Easily handles complex letters like "B", "E", and "X" by waiting until you submit to process the final visual shape.
- **🧠 Personalized Calibration:** Train the system on _your_ exact handwriting. Save multiple variations of a single letter to dynamically improve accuracy.
- **⚡ Ultra-Lightweight:** Runs instantly using vectorized NumPy operations—no massive GPUs or heavy deep learning frameworks required.

---

## 📂 Project Structure

```
AeroWrite/
│
├── my_handwriting_templates/  # Auto-generated folder containing your saved letter variations (.png)
├── hand_landmarker.task       # MediaPipe hand tracking model (auto-downloads on first run)
├── main.py                    # Main execution loop, camera handling, and UI overlay
├── writting.py                # Core logic: gesture detection, smoothing, and image matching
└── README.md                  # Project documentation
```

---

## ⚙️ System Requirements

- **Operating System:** macOS, Windows, or Linux
- **Python Version:** Python **3.13** *(Fully tested and supported. Compatible with Python 3.8+)*.
- **Hardware:** A standard webcam.

---

## 🛠️ Installation & Setup

Follow these steps exactly to avoid environment or dependency errors.

### 1. Clone the repository:

```bash
git clone [https://github.com/yourusername/AeroWrite.git](https://github.com/yourusername/AeroWrite.git)
cd AeroWrite
```

### 2. Create a Virtual Environment (Recommended):

Isolating your project prevents version conflicts with other Python apps on your computer.

```bash
python3.12 -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
```

### 3. Install Dependencies:

To replicate the exact environment, install the required packages via the provided requirements file:
```bash
pip install -r requirements.txt
```

### 4. Run the Application:

```bash
python main.py
```

> Note: The very first time you run the app, it will securely download the 30MB MediaPipe hand_landmarker.task model automatically into your project folder.)

---

## 🎮 How to Use (Controls & Gestures)

AeroWrite relies on specific hand gestures and keyboard shortcuts to operate and learn. Ensure your hand is clearly visible in the webcam frame.

### 🖐️ Gesture Command Mapping

| Kinematic Gesture     | System Function / Action               |
|------------------|-------------------------------|
|**Index-Thumb Pinch** | **Stroke Generation:** Engages continuous input tracking (analogous to a stylus touching a surface). Captures and renders solid spatial trajectories on the active digital canvas.|
|**Open Palm (Extended)** | **Input Disengagement:** Suspends stroke tracking (analogous to lifting a stylus). Permits spatial repositioning without rendering ink, which is essential for the formation of multi-stroke characters (e.g., 'A', 'T').|
|**Sustained Closed Fist** | **Erasure Protocol:** Activates a localized deletion mode. Functions as a broad-area spatial brush to clear erroneous coordinate data from the canvas.|
|**Sequential Double Fist** | **Execution & Classification:** Triggers the recognition pipeline. Captures the finalized spatial geometry, executes the template-matching algorithm against the custom dataset, and outputs the classified character.|

### ⌨️ Keyboard Command Mapping

| Keystroke Input | System Function / Action |
| :--- | :--- |
| **`a` - `z`** | **Dataset Calibration & Feature Extraction:** Captures the current spatial matrix on the active canvas and serializes it as a ground-truth template for the corresponding alphanumeric character. |
| **`Shift` + `c`** | **Global Canvas Reinitialization:** Executes an immediate purge of all rendered coordinate data, restoring the digital workspace to a fundamental zero-state. |
| **`Shift` + `q`** | **System Termination:** Gracefully interrupts the main execution loop, releases all optical hardware (webcam) resources, and exits the application environment. |

---

## 🧠 Training AeroWrite (Crucial First Step)

Because this system uses personalized calibration, it needs to learn your handwriting before it can recognize it!

1. Run python `main.py`.
2. Pinch your fingers and draw a large **"A"** in the air.
3. Press the `a` key on your physical keyboard. The terminal will confirm: `✅ Successfully learned variation #1 for: A`. The screen will clear automatically.
4. Draw a slightly different, messier "A" and press a again to give it more data (`variation #2`).
5. Repeat this process for the entire alphabet and numbers 0-9.

> Pro Tip: The more variations you save (big, small, neat, sloppy), the more robust the recognition becomes! These images are permanently saved in the `my_handwriting_templates/` folder.

---

## 🔍 How It Works (The Pipeline)

1. **Hand Tracking:** MediaPipe identifies the index finger and thumb to calculate pinch distance.
2. **Writing Detection:** When a pinch is detected, OpenCV draws thick lines tracking the index finger tip. It utilizes an Exponential Moving Average (EMA) algorithm to stabilize the cursor and remove natural human hand tremors.
3. **Feature Extraction:** Upon submission, the system isolates the drawn ink, creates a tight bounding box to remove empty space, and normalizes the ink to a perfectly centered 64x64 high-contrast binary image.
4. **Template Matching:** The system calculates the absolute pixel difference (cv2.absdiff) between the new drawing and all saved variations in your custom dataset, returning the closest geometric match.

---

## 🚑 Troubleshooting & Common Errors

### 1. macOS Python SSL Certificate Error (`urllib.error.URLError`)

**The Problem:** When running `main.py` for the first time on a Mac, you might see an SSL Certificate verification error preventing the `hand_landmarker.task` model from downloading. This happens because Python on macOS does not come with pre-installed TLS/SSL certificates.

**The Fix:**

* Built-in bypass: AeroWrite includes `ssl._create_unverified_context` in `main.py` to automatically bypass this for you.
* System-wide fix: If it still fails, open your Mac Terminal and run the official Python certificate installer:

```bash
/Applications/Python\ 3.x/Install\ Certificates.command
```
> (Replace `3.x` with your installed version, e.g., `3.13`)

### 2. Dotted Lines or Dropped Strokes

**The Problem:** The drawn line breaks into dots if you move your hand too fast.

**The Fix:** This means your webcam's framerate is struggling or lighting is poor. Ensure you are in a well-lit room so the camera can track at a full 30 FPS.

### 3. Misclassifying Letters (e.g., confusing 'O' and 'D')

**The Problem:** The system outputs the wrong letter when you double-fist submit.

**The Fix:** You need more calibration data! Draw the letter that it missed, press the corresponding keyboard key to save that exact variation, and try again.

---

## 📌 Future Roadmap

* **Deep Learning Integration:** Replace the visual template matcher with a pre-trained CNN (like EMNIST) or an LSTM network for generalized recognition without user calibration.

* **Word-Level Recognition:** Implement spacing and timing detection to recognize full cursive words and sentences instead of single characters.

* **GUI Upgrade:** Transition from OpenCV's imshow to a modern UI wrapper using PyQt or Tkinter.

---

## 🤝 Contribution

We welcome contributions from the open-source and academic communities! If you would like to optimize the recognition pipeline, introduce new neural network architectures, or enhance the spatial tracking algorithms, please feel free to fork this repository, open an issue, or submit a pull request.

This system was originally engineered for educational and research purposes, exploring the intersections of Computer Vision, Human-Computer Interaction (HCI), and dynamic image classification.

---

## 📝 License:

This project is open-sourced and distributed under the **MIT License**. You are free to use, modify, and distribute this software for both academic and commercial purposes, provided that proper attribution is given to the original author.