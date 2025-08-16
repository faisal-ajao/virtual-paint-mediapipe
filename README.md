# 🎨 AI Virtual Paint with Mediapipe

This project is an **AI Virtual Paint App** built using **Python, OpenCV, and Mediapipe**.  
It allows users to draw in the air with hand gestures, change brush colors, and erase drawings — all using a webcam.

---

## 🚀 Features
- Real-time hand tracking with Mediapipe.
- Virtual brush with adjustable size.
- Eraser tool for corrections.
- Toolbar with multiple colors.
- Option to clear the canvas (press 'C').

---

## 📦 Installation

```bash
# Clone the repository
git clone https://github.com/AjaoFaisal/virtual-paint-mediapipe.git
cd virtual-paint-mediapipe

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\\Scripts\\activate    # Windows

# Install dependencies
pip install -r requirements.txt
```

---

## ▶️ Usage

Run the app:
```
python main.py
```

- Use your **index finger** to draw.  
- Use **index + middle fingers** to select colors/tools.  
- Move to the top toolbar to pick brush color or eraser.  
- Press **C** to clear the canvas.  
- Press **ESC** to exit.  

---

## 📊 Demo (YouTube)
[![Watch the demo](https://img.youtube.com/vi/LrVoDBY68iU/hqdefault.jpg)](https://youtu.be/LrVoDBY68iU?feature=shared)

---

## 📂 Project Structure
```
virtual-paint-mediapipe/
├── paint_templates/        # Toolbar images
├── HandTrackingModule.py   # Hand detection helper
├── main.py                 # Main application
├── requirements.txt
└── README.md
```

---

## 🧠 Tech Stack
- Python 3.11.5
- OpenCV
- Mediapipe
- NumPy

---

## 📜 License
This project is licensed under the MIT License – see the [LICENSE](LICENSE) file for details.
