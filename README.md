# Palm Health Detection System

AI-Powered Aerial Palm Plantation Health Monitoring System using YOLOv8 and Faster R-CNN.

## Features
- Real-time palm tree detection
- Health classification (Healthy/Unhealthy)
- Interactive dashboard with analytics
- Detection history tracking
- Automated validation with dual-model architecture

## Deployment

This app is deployed on Streamlit Community Cloud.

## Local Development

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the app:
```bash
streamlit run stm.py
```

## Models Required

Place the following model files in the specified paths:
- YOLO Detection: `runs/detect/YOLO_Detection/weights/best.pt`
- Faster R-CNN: `runs/detect/FasterRCNN_ResNet50_Optimized/weights/best.pt`

## Technology Stack
- YOLOv8 for palm detection
- Faster R-CNN for validation
- Streamlit for web interface
- SQLite for data storage
