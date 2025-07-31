#!/usr/bin/env python3
"""
Test Drone Detection Simulator
Simulates drone detections for testing the dashboard
"""

import requests
import time
import random
import json
import datetime
from pathlib import Path

class DroneDetectionSimulator:
    """Simulates drone detections for dashboard testing"""
    
    def __init__(self):
        """Initialize the simulator"""
        self.base_url = "http://localhost:5000"
        self.drones = [
            {"id": "drone-001", "name": "Alpha Drone", "location": (40.7128, -74.0060)},
            {"id": "drone-002", "name": "Beta Drone", "location": (34.0522, -118.2437)},
            {"id": "drone-003", "name": "Gamma Drone", "location": (41.8781, -87.6298)}
        ]
        
        # Register drones
        for drone in self.drones:
            self.register_drone(drone)
        
        print("🚁 Drone Detection Simulator")
        print("=" * 40)
        print("📡 Simulating detections for dashboard testing")
        print("🎯 Dashboard URL: http://localhost:5000")
    
    def register_drone(self, drone):
        """Register a drone with the dashboard"""
        try:
            response = requests.post(f"{self.base_url}/api/register_drone", json={
                "drone_id": drone["id"],
                "name": drone["name"],
                "location": drone["location"]
            })
            if response.status_code == 200:
                print(f"✅ Registered drone: {drone['name']} ({drone['id']})")
            else:
                print(f"❌ Failed to register drone: {drone['name']}")
        except Exception as e:
            print(f"❌ Error registering drone: {e}")
    
    def update_drone_status(self, drone_id, status="active", battery_level=None):
        """Update drone status"""
        try:
            response = requests.post(f"{self.base_url}/api/update_drone", json={
                "drone_id": drone_id,
                "status": status,
                "battery_level": battery_level or random.randint(60, 100)
            })
            if response.status_code != 200:
                print(f"❌ Failed to update drone status: {drone_id}")
        except Exception as e:
            print(f"❌ Error updating drone status: {e}")
    
    def simulate_detection(self, drone_id, detection_type, confidence, reason=""):
        """Simulate a detection"""
        try:
            # Get drone location
            drone = next((d for d in self.drones if d["id"] == drone_id), None)
            location = drone["location"] if drone else None
            
            # Add some random offset to location
            if location:
                lat_offset = random.uniform(-0.001, 0.001)
                lng_offset = random.uniform(-0.001, 0.001)
                location = (location[0] + lat_offset, location[1] + lng_offset)
            
            response = requests.post(f"{self.base_url}/api/add_detection", json={
                "drone_id": drone_id,
                "type": detection_type,
                "confidence": confidence,
                "location": location,
                "reason": reason
            })
            
            if response.status_code == 200:
                print(f"🚨 Detection: {detection_type} by {drone_id} (confidence: {confidence:.2f})")
            else:
                print(f"❌ Failed to add detection: {detection_type}")
                
        except Exception as e:
            print(f"❌ Error simulating detection: {e}")
    
    def run_simulation(self, duration=300):  # 5 minutes
        """Run the detection simulation"""
        print(f"🎬 Starting simulation for {duration} seconds...")
        print("Press Ctrl+C to stop")
        
        detection_types = [
            ("car_crash", "Car crash detected on highway"),
            ("fire", "Fire detected in building"),
            ("person_down", "Person appears to be unconscious"),
            ("medical_emergency", "Medical emergency situation detected")
        ]
        
        start_time = time.time()
        
        try:
            while time.time() - start_time < duration:
                # Randomly select a drone
                drone = random.choice(self.drones)
                
                # Randomly select detection type
                detection_type, reason = random.choice(detection_types)
                
                # Random confidence based on detection type
                if detection_type == "car_crash":
                    confidence = random.uniform(0.4, 0.8)
                elif detection_type == "fire":
                    confidence = random.uniform(0.6, 0.9)
                elif detection_type == "person_down":
                    confidence = random.uniform(0.7, 0.95)
                else:
                    confidence = random.uniform(0.5, 0.8)
                
                # Simulate detection
                self.simulate_detection(drone["id"], detection_type, confidence, reason)
                
                # Update drone status occasionally
                if random.random() < 0.3:
                    battery_level = random.randint(60, 100)
                    self.update_drone_status(drone["id"], "active", battery_level)
                
                # Wait between 5-15 seconds
                wait_time = random.uniform(5, 15)
                time.sleep(wait_time)
                
        except KeyboardInterrupt:
            print("\n🛑 Simulation stopped by user")
        
        print("✅ Simulation completed")

def main():
    """Main function"""
    simulator = DroneDetectionSimulator()
    simulator.run_simulation()

if __name__ == "__main__":
    main() 