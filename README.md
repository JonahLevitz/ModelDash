# 🚁 Medical Emergency Detection System

A comprehensive drone-based image recognition system designed to detect medical emergencies in real-time. The system runs on a Raspberry Pi and provides a professional web interface for monitoring and control.

## 🎯 Features

- **Real-time Medical Emergency Detection**: Detects people in need of medical attention
- **Professional Web Interface**: Modern UI with Tailwind CSS for real-time monitoring
- **Raspberry Pi Compatible**: Optimized for Raspberry Pi 4 with lightweight models
- **Automatic Logging**: Saves detection timestamps, images, and statistics
- **Configurable Detection**: Adjustable confidence thresholds and detection intervals

## 🏥 Detected Medical Emergencies

1. **Person Down** - Individuals lying on the ground
2. **Injured Person** - People with visible injuries
3. **Unconscious Person** - Unconscious individuals
4. **Medical Emergency** - General medical emergency situations
5. **Accident Victim** - People involved in accidents

## 🏗️ System Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Raspberry Pi  │    │   Web Interface │    │   Detection     │
│                 │    │                 │    │   System        │
│ ┌─────────────┐ │    │ ┌─────────────┐ │    │ ┌─────────────┐ │
│ │   Camera    │ │    │ │   Flask     │ │    │ │   YOLOv8    │ │
│ │   Module    │ │    │ │   Server    │ │    │ │   Model     │ │
│ └─────────────┘ │    │ └─────────────┘ │    │ └─────────────┘ │
│                 │    │                 │    │                 │
│ ┌─────────────┐ │    │ ┌─────────────┐ │    │ ┌─────────────┐ │
│ │   OpenCV    │ │    │ │  Socket.IO  │ │    │ │   Logging   │ │
│ │  Processing │ │    │ │  Real-time  │ │    │ │   System    │ │
│ └─────────────┘ │    │ └─────────────┘ │    │ └─────────────┘ │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 📋 Requirements

### Hardware
- Raspberry Pi 4 (4GB RAM recommended)
- Pi Camera Module or USB webcam
- MicroSD card (32GB+ recommended)

### Software
- Python 3.8+
- OpenCV
- PyTorch (CPU version for Pi)
- Ultralytics YOLOv8
- Flask + SocketIO

## 🚀 Installation

### 1. Clone the Repository
```bash
git clone <repository-url>
cd medical-emergency-detection
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Create Directory Structure
```bash
mkdir -p models detections logs
```

### 4. Train the Model (Optional)
If you have medical emergency training data:
```bash
python train_medical_emergency.py
```

Or use a pre-trained model by placing it in the `models/` directory as `medical_emergency.pt`.

## 🎮 Usage

### Starting the Web Interface
```bash
python web_interface.py
```

The web interface will be available at `http://localhost:5000`

### Running Detection Only
```bash
python detection_system.py
```

### Training Your Own Model
```bash
python train_medical_emergency.py
```

## 🌐 Web Interface Features

### Dashboard Components
- **Live Video Feed**: Real-time camera stream with detection overlays
- **Detection Statistics**: Real-time counts of different emergency types
- **Recent Detections**: Latest detection events with timestamps
- **Detection History**: Historical view of all detections
- **System Controls**: Start/stop detection with status indicators

### Real-time Features
- **Socket.IO Communication**: Real-time updates without page refresh
- **Emergency Alerts**: Modal popups for critical detections
- **FPS Monitoring**: Performance metrics display
- **Status Indicators**: System health and connection status

## 📊 Detection Logging

### CSV Log Format
```csv
timestamp,label,confidence,bbox,image_path
2024-01-15T10:30:45.123,person_down,0.85,"[100,200,300,400]","detections/2024-01-15T10-30-45-123_person_down_0.85.jpg"
```

### Saved Files
- **Detection Images**: Saved in `detections/` with timestamp and label
- **Log Files**: CSV format in `logs/detections.csv`
- **System Logs**: Application logs in `logs/detection_system.log`

## ⚙️ Configuration

### Detection Parameters
```python
# In detection_system.py
confidence_threshold = 0.5      # Minimum confidence for detections
detection_interval = 1.0        # Seconds between detections
```

### Model Configuration
```python
# Model paths and settings
model_path = "models/medical_emergency.pt"
device = "cpu"  # Use "cuda" for GPU
```

## 🔧 Customization

### Adding New Emergency Types
1. Update the `emergency_classes` dictionary in `detection_system.py`
2. Retrain the model with new classes
3. Update the web interface labels

### Modifying Detection Logic
- Edit `process_frame()` method in `MedicalEmergencyDetector`
- Adjust confidence thresholds and detection intervals
- Customize bounding box visualization

### Web Interface Customization
- Modify `templates/index.html` for UI changes
- Update `static/js/app.js` for frontend logic
- Customize CSS in the HTML template

## 📈 Performance Optimization

### Raspberry Pi Optimizations
- Use YOLOv8n (nano) model for faster inference
- Reduce input image resolution
- Increase detection interval for lower CPU usage
- Use USB 3.0 camera for better frame rates

### Model Optimization
```python
# Optimize model for inference
model = YOLO('medical_emergency.pt')
model.to('cpu')  # Ensure CPU usage
```

## 🛠️ Troubleshooting

### Common Issues

1. **Camera Not Found**
   ```bash
   # Check camera permissions
   sudo usermod -a -G video $USER
   # Reboot and try again
   ```

2. **Model Loading Errors**
   ```bash
   # Ensure model file exists
   ls models/medical_emergency.pt
   # Check file permissions
   chmod 644 models/medical_emergency.pt
   ```

3. **Web Interface Not Loading**
   ```bash
   # Check if port 5000 is available
   netstat -tulpn | grep 5000
   # Try different port
   socketio.run(app, port=5001)
   ```

### Performance Monitoring
- Monitor CPU usage: `htop`
- Check memory usage: `free -h`
- Monitor temperature: `vcgencmd measure_temp`

## 🔒 Security Considerations

- Change default Flask secret key
- Use HTTPS in production
- Implement authentication for web interface
- Secure camera access permissions
- Regular security updates

## 📝 Development

### Project Structure
```
medical-emergency-detection/
├── detection_system.py      # Main detection logic
├── web_interface.py         # Flask web server
├── train_medical_emergency.py  # Model training
├── requirements.txt         # Python dependencies
├── templates/              # HTML templates
│   └── index.html
├── static/                 # Static assets
│   ├── css/
│   └── js/
│       └── app.js
├── models/                 # Trained models
├── detections/             # Saved detection images
├── logs/                   # Log files
└── medical_emergency_dataset/  # Training data
```

### Contributing
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Support

For support and questions:
- Create an issue on GitHub
- Check the troubleshooting section
- Review the documentation

## 🚀 Future Enhancements

- [ ] Multi-camera support
- [ ] Cloud-based alerting
- [ ] Mobile app companion
- [ ] Advanced analytics dashboard
- [ ] Integration with emergency services
- [ ] Machine learning model improvements
- [ ] Edge computing optimizations

---

**Note**: This system is designed for educational and research purposes. For production use in emergency situations, ensure proper testing, validation, and compliance with local regulations. 