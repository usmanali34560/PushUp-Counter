# AI Push-Up Counter — Python 3.13

A simple computer-vision Streamlit app that counts push-ups from an uploaded video using MediaPipe Pose + OpenCV.

## Features
- Upload MP4/MOV/AVI/MKV/WEBM video
- Live-style frame-by-frame processing in the Streamlit page
- Pose landmarks drawn over the person
- Push-up count displayed on screen
- UP / DOWN / MOVING / NO POSE states
- Elbow-angle display
- Adjustable pose visibility and push-up thresholds
- Python 3.13 friendly setup

## Recommended video
- Side view
- Full upper body and hips visible
- Good lighting
- Camera mostly fixed
- Avoid people overlapping the athlete

## Install

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Run

```powershell
streamlit run app.py
```

Then open the local Streamlit address shown in the terminal.

## Important
The counter uses elbow angle and pose landmarks. Different camera angles, clothing, occlusion, low resolution, and unusual push-up form can affect accuracy. This is an educational computer-vision project.
