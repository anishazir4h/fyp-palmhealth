"""
Database module for Palm Health Detection System
Handles SQLite database operations for storing detection results
"""

import sqlite3
import json
from datetime import datetime
import os

class PalmDatabase:
    def __init__(self, db_path="palm_detection.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize database with required tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create detections table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS detections (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                image_name TEXT NOT NULL,
                image_path TEXT,
                total_palms INTEGER,
                healthy_palms INTEGER,
                unhealthy_palms INTEGER,
                health_rate REAL,
                avg_confidence REAL,
                detection_details TEXT,
                image_size TEXT
            )
        """)
        
        # Create individual_palms table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS individual_palms (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                detection_id INTEGER,
                palm_number INTEGER,
                status TEXT,
                confidence REAL,
                bbox TEXT,
                FOREIGN KEY (detection_id) REFERENCES detections(id)
            )
        """)
        
        conn.commit()
        conn.close()
    
    def save_detection(self, image_name, image_path, summary, image_size):
        """Save detection results to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Calculate health rate
        health_rate = (summary['healthy_palms'] / summary['total_palms'] * 100) if summary['total_palms'] > 0 else 0
        
        # Insert main detection record
        cursor.execute("""
            INSERT INTO detections 
            (image_name, image_path, total_palms, healthy_palms, unhealthy_palms, 
             health_rate, avg_confidence, detection_details, image_size)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            image_name,
            image_path,
            summary['total_palms'],
            summary['healthy_palms'],
            summary['unhealthy_palms'],
            health_rate,
            summary['avg_confidence'],
            json.dumps(summary['detections'], default=str),
            f"{image_size[0]}x{image_size[1]}"
        ))
        
        detection_id = cursor.lastrowid
        
        # Insert individual palm records
        for det in summary['detections']:
            cursor.execute("""
                INSERT INTO individual_palms 
                (detection_id, palm_number, status, confidence, bbox)
                VALUES (?, ?, ?, ?, ?)
            """, (
                detection_id,
                det['id'],
                det['status'],
                det['confidence'],
                json.dumps(det['bbox'].tolist() if hasattr(det['bbox'], 'tolist') else det['bbox'])
            ))
        
        conn.commit()
        conn.close()
        
        return detection_id
    
    def get_all_detections(self):
        """Get all detection records"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT id, timestamp, image_name, total_palms, healthy_palms, 
                   unhealthy_palms, health_rate, avg_confidence, image_path
            FROM detections
            ORDER BY timestamp DESC
        """)
        
        results = cursor.fetchall()
        conn.close()
        
        return results
    
    def get_detection_by_id(self, detection_id):
        """Get specific detection by ID"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT * FROM detections WHERE id = ?
        """, (detection_id,))
        
        result = cursor.fetchone()
        conn.close()
        
        return result
    
    def get_statistics(self):
        """Get overall statistics from all detections"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Total images
        cursor.execute("SELECT COUNT(*) FROM detections")
        total_images = cursor.fetchone()[0]
        
        # Total palms
        cursor.execute("SELECT SUM(total_palms) FROM detections")
        total_palms = cursor.fetchone()[0] or 0
        
        # Total healthy
        cursor.execute("SELECT SUM(healthy_palms) FROM detections")
        total_healthy = cursor.fetchone()[0] or 0
        
        # Total unhealthy
        cursor.execute("SELECT SUM(unhealthy_palms) FROM detections")
        total_unhealthy = cursor.fetchone()[0] or 0
        
        # Average health rate
        cursor.execute("SELECT AVG(health_rate) FROM detections")
        avg_health_rate = cursor.fetchone()[0] or 0
        
        # Average confidence
        cursor.execute("SELECT AVG(avg_confidence) FROM detections")
        avg_confidence = cursor.fetchone()[0] or 0
        
        # Get trend data (last 10 detections)
        cursor.execute("""
            SELECT timestamp, health_rate, total_palms 
            FROM detections 
            ORDER BY timestamp DESC 
            LIMIT 10
        """)
        trend_data = cursor.fetchall()
        
        conn.close()
        
        return {
            'total_images': total_images,
            'total_palms': total_palms,
            'total_healthy': total_healthy,
            'total_unhealthy': total_unhealthy,
            'avg_health_rate': avg_health_rate,
            'avg_confidence': avg_confidence,
            'trend_data': trend_data
        }
    
    def get_recent_detections(self, limit=5):
        """Get most recent detections"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT id, timestamp, image_name, total_palms, healthy_palms, 
                   unhealthy_palms, health_rate, image_path
            FROM detections
            ORDER BY timestamp DESC
            LIMIT ?
        """, (limit,))
        
        results = cursor.fetchall()
        conn.close()
        
        return results
    
    def delete_detection(self, detection_id):
        """Delete a detection and its associated palm records"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Delete individual palms first (foreign key constraint)
        cursor.execute("DELETE FROM individual_palms WHERE detection_id = ?", (detection_id,))
        
        # Delete main detection
        cursor.execute("DELETE FROM detections WHERE id = ?", (detection_id,))
        
        conn.commit()
        conn.close()
    
    def clear_all_data(self):
        """Clear all data from database (use with caution)"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("DELETE FROM individual_palms")
        cursor.execute("DELETE FROM detections")
        
        conn.commit()
        conn.close()
