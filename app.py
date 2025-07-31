#!/usr/bin/env python3
"""
Production-ready Drone Emergency Detection Dashboard
Deployable version for online hosting
"""

import os
from drone_dashboard import app, socketio

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    socketio.run(app, host='0.0.0.0', port=port, debug=False) 