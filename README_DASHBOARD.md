# 🚁 Drone Emergency Detection Dashboard

A real-time web dashboard for monitoring drone emergency detections with analytics and multi-drone support.

## 🎯 Features

### 📊 Real-time Monitoring
- **Live detection feed** with instant alerts
- **Drone status tracking** (battery, location, connection)
- **Detection analytics** with charts and statistics
- **Emergency modal alerts** for high-confidence detections

### 🚨 Emergency Detection Types
- **Car Crash Detection** - Trained model with 40.9% mAP50 accuracy
- **Fire Detection** - Real-time fire detection
- **Person Down Detection** - Unconscious/fainted person detection
- **Medical Emergency** - Medical emergency situations

### 📈 Analytics Dashboard
- **Detection counts** by type and time
- **Drone performance** metrics
- **Geographic tracking** of detections
- **Confidence level** monitoring

## 🚀 Quick Start

### 1. Start the Dashboard Server
```bash
python3.11 drone_dashboard.py
```
The dashboard will be available at: **http://localhost:5000**

### 2. Test with Simulated Detections
```bash
python3.11 test_drone_detection.py
```
This simulates 3 drones making detections for 5 minutes.

### 3. Integrate with Real Detection System
Modify your `test_practical_emergency.py` to send detections to the dashboard:

```python
# Add this to your detection system
import requests

def send_to_dashboard(detection):
    requests.post("http://localhost:5000/api/add_detection", json={
        "drone_id": "drone-001",
        "type": detection['type'],
        "confidence": detection['confidence'],
        "reason": detection['reason']
    })
```

## 📱 Dashboard Interface

### Main Dashboard
- **Statistics Cards** - Total detections, active drones, recent activity
- **Real-time Feed** - Live detection alerts with images
- **Drone Status** - Battery levels, location, connection status
- **Analytics Chart** - Detection type distribution

### Emergency Alerts
- **High-confidence detections** (>70%) trigger emergency modals
- **Real-time notifications** via WebSocket
- **Detection images** saved and displayed
- **Location tracking** with GPS coordinates

## 🔧 API Endpoints

### Detection Management
- `POST /api/add_detection` - Add new detection
- `GET /api/detections` - Get recent detections
- `GET /api/stats` - Get detection statistics

### Drone Management
- `POST /api/register_drone` - Register new drone
- `POST /api/update_drone` - Update drone status
- `GET /api/drones` - Get all drone status

### WebSocket Events
- `new_detection` - Real-time detection alerts
- `drone_status_update` - Drone status changes
- `drone_registered` - New drone registration

## 🗄️ Database

The dashboard uses SQLite for data storage:
- **detections** table - All detection records
- **drones** table - Drone registration and status

## 🎨 Customization

### Adding New Detection Types
1. Update the detection model training
2. Add new type to dashboard analytics
3. Update color coding in CSS

### Multiple Drones
1. Register each drone with unique ID
2. Send detections with drone_id
3. Monitor individual drone performance

### Alert System
- Configure confidence thresholds
- Set up email/SMS notifications
- Customize emergency modal content

## 🔒 Security Features

- **Input validation** for all API endpoints
- **SQL injection protection** with parameterized queries
- **CORS configuration** for web security
- **Error handling** and logging

## 📊 Performance

- **Real-time updates** via WebSocket
- **Efficient database queries** with indexing
- **Image compression** for storage optimization
- **Responsive design** for mobile devices

## 🛠️ Development

### Prerequisites
```bash
pip install -r requirements_dashboard.txt
```

### File Structure
```
drone_dashboard.py          # Main dashboard server
templates/dashboard.html    # Dashboard UI
test_drone_detection.py    # Simulator for testing
static/detections/         # Detection images
drone_detections.db        # SQLite database
```

### Testing
1. Start dashboard server
2. Run simulator in separate terminal
3. Open dashboard in browser
4. Monitor real-time updates

## 🚀 Production Deployment

### Recommended Setup
- **Gunicorn** for WSGI server
- **Nginx** for reverse proxy
- **PostgreSQL** for production database
- **Redis** for WebSocket scaling

### Environment Variables
```bash
export FLASK_ENV=production
export SECRET_KEY=your-secret-key
export DATABASE_URL=postgresql://...
```

## 📞 Support

For issues or questions:
1. Check the logs in the terminal
2. Verify database connectivity
3. Test API endpoints directly
4. Review WebSocket connections

---

**Dashboard URL**: http://localhost:5000  
**API Base**: http://localhost:5000/api  
**WebSocket**: ws://localhost:5000/socket.io 