"""
Database service for Cat Activities Monitor API.
"""

import logging
import asyncpg
import json
import hashlib
import os
from datetime import datetime, timedelta
import numpy as np

from models import CatDetectionWithActivity

logger = logging.getLogger(__name__)


class DatabaseService:
    """Service for managing PostgreSQL database operations."""
    
    def __init__(self, database_url: str = None):
        self.database_url = database_url or os.getenv(
            'DATABASE_URL', 
            'postgresql://db_user:db_password@localhost:5432/cats_monitor'
        )
        self.pool = None
    
    async def init_database(self):
        """Initialize PostgreSQL database with required tables."""
        if not self.pool:
            self.pool = await asyncpg.create_pool(self.database_url)
        
        async with self.pool.acquire() as conn:
        
            # Create feedback table
            await conn.execute('''
                CREATE TABLE IF NOT EXISTS feedback (
                    feedback_id TEXT PRIMARY KEY,
                    image_filename TEXT NOT NULL,
                    image_path TEXT NOT NULL,
                    original_detections JSONB NOT NULL,
                    user_annotations JSONB NOT NULL,
                    feedback_type TEXT NOT NULL,
                    notes TEXT,
                    timestamp TEXT NOT NULL,
                    user_id TEXT DEFAULT 'anonymous',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
        
            # Create cat_profiles table
            await conn.execute('''
                CREATE TABLE IF NOT EXISTS cat_profiles (
                    name TEXT PRIMARY KEY,
                    description TEXT,
                    color TEXT,
                    breed TEXT,
                    favorite_activities JSONB,
                    created_timestamp TEXT NOT NULL,
                    last_seen_timestamp TEXT,
                    total_detections INTEGER DEFAULT 0,
                    average_confidence REAL DEFAULT 0.0,
                    preferred_locations JSONB,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
        
            # Create detection_results table for storing YOLO detection data
            await conn.execute('''
                CREATE TABLE IF NOT EXISTS detection_results (
                    id SERIAL PRIMARY KEY,
                    source_name TEXT NOT NULL,
                    image_filename TEXT,
                    detected BOOLEAN NOT NULL,
                    count INTEGER NOT NULL,
                    confidence REAL NOT NULL,
                    detections JSONB NOT NULL,
                    activities JSONB,
                    total_animals INTEGER DEFAULT 0,
                    primary_activity TEXT,
                    image_array_hash TEXT,
                    timestamp TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(source_name, image_filename)
                )
            ''')
        
            # Create indexes for better performance
            await conn.execute('CREATE INDEX IF NOT EXISTS idx_feedback_timestamp ON feedback(timestamp)')
            await conn.execute('CREATE INDEX IF NOT EXISTS idx_feedback_type ON feedback(feedback_type)')
            await conn.execute('CREATE INDEX IF NOT EXISTS idx_cat_profiles_updated ON cat_profiles(updated_at)')
            await conn.execute('CREATE INDEX IF NOT EXISTS idx_detection_source ON detection_results(source_name)')
            await conn.execute('CREATE INDEX IF NOT EXISTS idx_detection_timestamp ON detection_results(timestamp)')
            
        logger.info(f"📊 Database initialized: {self.database_url}")
    
    async def get_db_connection(self):
        """Get a database connection from the pool."""
        if not self.pool:
            self.pool = await asyncpg.create_pool(self.database_url)
        return self.pool.acquire()
    
    # Feedback operations
    async def save_feedback(self, feedback_id: str, feedback_data: dict):
        """Save feedback to database."""
        async with self.pool.acquire() as conn:
            await conn.execute('''
                INSERT INTO feedback 
                (feedback_id, image_filename, image_path, original_detections, 
                 user_annotations, feedback_type, notes, timestamp, user_id)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                ON CONFLICT (feedback_id) DO UPDATE SET
                    image_filename = EXCLUDED.image_filename,
                    image_path = EXCLUDED.image_path,
                    original_detections = EXCLUDED.original_detections,
                    user_annotations = EXCLUDED.user_annotations,
                    feedback_type = EXCLUDED.feedback_type,
                    notes = EXCLUDED.notes,
                    timestamp = EXCLUDED.timestamp,
                    user_id = EXCLUDED.user_id
            ''', 
                feedback_id,
                feedback_data['image_filename'],
                feedback_data['image_path'],
                json.dumps(feedback_data['original_detections']),
                json.dumps(feedback_data['user_annotations']),
                feedback_data['feedback_type'],
                feedback_data.get('notes'),
                feedback_data['timestamp'],
                feedback_data.get('user_id', 'anonymous')
            )
    
    async def get_all_feedback(self):
        """Get all feedback from database."""
        async with self.pool.acquire() as conn:
            rows = await conn.fetch('SELECT * FROM feedback ORDER BY timestamp DESC')
            
            feedback_dict = {}
            for row in rows:
                feedback_dict[row['feedback_id']] = {
                    'image_filename': row['image_filename'],
                    'image_path': row['image_path'],
                    'original_detections': json.loads(row['original_detections']),
                    'user_annotations': json.loads(row['user_annotations']),
                    'feedback_type': row['feedback_type'],
                    'notes': row['notes'],
                    'timestamp': row['timestamp'],
                    'user_id': row['user_id']
                }
            
            return feedback_dict
    
    async def get_feedback_by_id(self, feedback_id: str):
        """Get specific feedback by ID."""
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow('SELECT * FROM feedback WHERE feedback_id = $1', feedback_id)
            
            if row:
                return {
                    'image_filename': row['image_filename'],
                    'image_path': row['image_path'],
                    'original_detections': json.loads(row['original_detections']),
                    'user_annotations': json.loads(row['user_annotations']),
                    'feedback_type': row['feedback_type'],
                    'notes': row['notes'],
                    'timestamp': row['timestamp'],
                    'user_id': row['user_id']
                }
            return None
    
    async def delete_feedback(self, feedback_id: str):
        """Delete feedback from database."""
        async with self.pool.acquire() as conn:
            result = await conn.execute('DELETE FROM feedback WHERE feedback_id = $1', feedback_id)
            # Extract row count from the result string (e.g., "DELETE 1" -> 1)
            deleted_count = int(result.split()[1]) if result.split()[1].isdigit() else 0
            return deleted_count > 0
    
    async def get_feedback_count(self):
        """Get total feedback count."""
        async with self.pool.acquire() as conn:
            count = await conn.fetchval('SELECT COUNT(*) FROM feedback')
            return count
    
    # Cat profile operations
    async def save_cat_profile(self, cat_name: str, profile_data: dict):
        """Save cat profile to database."""
        async with self.pool.acquire() as conn:
            await conn.execute('''
                INSERT INTO cat_profiles 
                (name, description, color, breed, favorite_activities, 
                 created_timestamp, last_seen_timestamp, total_detections, 
                 average_confidence, preferred_locations)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
                ON CONFLICT (name) DO UPDATE SET
                    description = EXCLUDED.description,
                    color = EXCLUDED.color,
                    breed = EXCLUDED.breed,
                    favorite_activities = EXCLUDED.favorite_activities,
                    last_seen_timestamp = EXCLUDED.last_seen_timestamp,
                    total_detections = EXCLUDED.total_detections,
                    average_confidence = EXCLUDED.average_confidence,
                    preferred_locations = EXCLUDED.preferred_locations,
                    updated_at = CURRENT_TIMESTAMP
            ''', 
                cat_name,
                profile_data.get('description'),
                profile_data.get('color'),
                profile_data.get('breed'),
                json.dumps(profile_data.get('favorite_activities', [])),
                profile_data['created_timestamp'],
                profile_data.get('last_seen_timestamp'),
                profile_data.get('total_detections', 0),
                profile_data.get('average_confidence', 0.0),
                json.dumps(profile_data.get('preferred_locations', []))
            )
    
    async def get_all_cat_profiles(self):
        """Get all cat profiles from database."""
        async with self.pool.acquire() as conn:
            rows = await conn.fetch('SELECT * FROM cat_profiles ORDER BY name')
            
            profiles = {}
            for row in rows:
                profiles[row['name']] = {
                    'name': row['name'],
                    'description': row['description'],
                    'color': row['color'],
                    'breed': row['breed'],
                    'favorite_activities': json.loads(row['favorite_activities'] or '[]'),
                    'created_timestamp': row['created_timestamp'],
                    'last_seen_timestamp': row['last_seen_timestamp'],
                    'total_detections': row['total_detections'],
                    'average_confidence': row['average_confidence'],
                    'preferred_locations': json.loads(row['preferred_locations'] or '[]')
                }
            
            return profiles
    
    async def get_cat_profile_by_name(self, cat_name: str):
        """Get specific cat profile by name."""
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow('SELECT * FROM cat_profiles WHERE name = $1', cat_name)
            
            if row:
                return {
                    'name': row['name'],
                    'description': row['description'],
                    'color': row['color'],
                    'breed': row['breed'],
                    'favorite_activities': json.loads(row['favorite_activities'] or '[]'),
                    'created_timestamp': row['created_timestamp'],
                    'last_seen_timestamp': row['last_seen_timestamp'],
                    'total_detections': row['total_detections'],
                    'average_confidence': row['average_confidence'],
                    'preferred_locations': json.loads(row['preferred_locations'] or '[]')
                }
            return None
    
    async def delete_cat_profile(self, cat_name: str):
        """Delete cat profile from database."""
        async with self.pool.acquire() as conn:
            result = await conn.execute('DELETE FROM cat_profiles WHERE name = $1', cat_name)
            # Extract row count from the result string (e.g., "DELETE 1" -> 1)
            deleted_count = int(result.split()[1]) if result.split()[1].isdigit() else 0
            return deleted_count > 0
    
    async def get_cat_profiles_count(self):
        """Get total cat profiles count."""
        async with self.pool.acquire() as conn:
            count = await conn.fetchval('SELECT COUNT(*) FROM cat_profiles')
            return count
    
    # Detection results operations
    async def save_detection_result(self, source_name: str, detection_result: CatDetectionWithActivity, image_array: np.ndarray = None, image_filename: str = None):
        """Save detection result to database. Never overwrites existing detections for the same image."""
        async with self.pool.acquire() as conn:
            # Create hash of image array for similarity comparison (optional)
            image_hash = None
            if image_array is not None:
                image_hash = hashlib.md5(image_array.tobytes()).hexdigest()
            
            # Convert detections to JSON
            detections_json = json.dumps([
                {
                    "class_id": d.class_id,
                    "class_name": d.class_name,
                    "confidence": d.confidence,
                    "bounding_box": d.bounding_box
                } for d in detection_result.detections
            ])
            
            # Convert activities to JSON
            activities_json = json.dumps([
                {
                    "activity": a.activity.value,
                    "confidence": a.confidence,
                    "reasoning": a.reasoning,
                    "bounding_box": a.bounding_box,
                    "duration_seconds": a.duration_seconds,
                    "cat_index": a.cat_index,
                    "detection_id": a.detection_id
                } for a in detection_result.activities
            ]) if detection_result.activities else None
            
            # Use INSERT ... ON CONFLICT DO NOTHING to never overwrite existing detections
            result = await conn.execute('''
                INSERT INTO detection_results 
                (source_name, image_filename, detected, count, confidence, detections, activities, 
                 total_animals, primary_activity, image_array_hash, timestamp)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
                ON CONFLICT (source_name, image_filename) DO NOTHING
            ''', 
                source_name,
                image_filename,
                detection_result.detected,
                detection_result.count,
                detection_result.confidence,
                detections_json,
                activities_json,
                detection_result.total_animals,
                detection_result.primary_activity.value if detection_result.primary_activity else None,
                image_hash,
                datetime.now().isoformat()
            )
            
            # Check if the insert was successful (not ignored due to duplicate)
            inserted_count = int(result.split()[1]) if result.split()[1].isdigit() else 0
            if inserted_count > 0:
                logger.debug(f"💾 Saved new detection result for {source_name} - {image_filename}")
            else:
                logger.debug(f"⏭️ Detection result already exists for {source_name} - {image_filename}, skipping")
    
    async def get_latest_detection_result(self, source_name: str):
        """Get the latest detection result for a source."""
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow('''
                SELECT * FROM detection_results 
                WHERE source_name = $1 
                ORDER BY created_at DESC 
                LIMIT 1
            ''', source_name)
            
            if row:
                return {
                    "detected": bool(row['detected']),
                    "count": row['count'],
                    "confidence": row['confidence'],
                    "detections": json.loads(row['detections']) if row['detections'] else [],
                    "activities": json.loads(row['activities']) if row['activities'] else [],
                    "total_animals": row['total_animals'],
                    "primary_activity": row['primary_activity'],
                    "timestamp": row['timestamp']
                }
            return None
    
    async def get_all_detection_results(self):
        """Get all detection results grouped by source."""
        async with self.pool.acquire() as conn:
            sources = await conn.fetch('''
                SELECT source_name, MAX(created_at) as latest_time
                FROM detection_results 
                GROUP BY source_name
            ''')
            
            results = {}
            
            for source_row in sources:
                source_name = source_row['source_name']
                
                # Get the latest detection for this source
                row = await conn.fetchrow('''
                    SELECT * FROM detection_results 
                    WHERE source_name = $1 
                    ORDER BY created_at DESC 
                    LIMIT 1
                ''', source_name)
                
                if row:
                    results[source_name] = {
                        "detected": bool(row['detected']),
                        "count": row['count'],
                        "confidence": row['confidence'],
                        "detections": json.loads(row['detections']) if row['detections'] else [],
                        "activities": json.loads(row['activities']) if row['activities'] else [],
                        "total_animals": row['total_animals'],
                        "primary_activity": row['primary_activity'],
                        "timestamp": row['timestamp']
                    }
            
            return results
    
    async def cleanup_old_detection_results(self, keep_days: int = 7):
        """Clean up old detection results, keeping only the specified number of days."""
        async with self.pool.acquire() as conn:
            cutoff_date = datetime.now() - timedelta(days=keep_days)
            
            result = await conn.execute('''
                DELETE FROM detection_results 
                WHERE created_at < $1
            ''', cutoff_date)
            
            # Extract row count from the result string (e.g., "DELETE 5" -> 5)
            deleted_count = int(result.split()[1]) if result.split()[1].isdigit() else 0
            
            if deleted_count > 0:
                logger.info(f"🧹 Cleaned up {deleted_count} old detection results older than {keep_days} days")
            
            return deleted_count
    
    async def get_recent_detection_results(self, limit_per_source: int = 10):
        """Get recent detection results for all sources to restore activity history."""
        async with self.pool.acquire() as conn:
            # Get all unique source names
            source_rows = await conn.fetch('SELECT DISTINCT source_name FROM detection_results')
            sources = [row['source_name'] for row in source_rows]
            
            results = {}
            
            for source_name in sources:
                # Get recent detections for this source
                rows = await conn.fetch('''
                    SELECT * FROM detection_results 
                    WHERE source_name = $1 
                    ORDER BY created_at DESC 
                    LIMIT $2
                ''', source_name, limit_per_source)
                
                source_results = []
                
                for row in rows:
                    source_results.append({
                        "detected": bool(row['detected']),
                        "count": row['count'],
                        "confidence": row['confidence'],
                        "detections": json.loads(row['detections']) if row['detections'] else [],
                        "activities": json.loads(row['activities']) if row['activities'] else [],
                        "total_animals": row['total_animals'],
                        "primary_activity": row['primary_activity'],
                        "timestamp": row['timestamp'],
                        "created_at": row['created_at']
                    })
                
                results[source_name] = source_results
            
            return results
    
    async def get_detection_result_by_image(self, image_filename: str):
        """Get detection result for a specific image filename."""
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow('''
                SELECT * FROM detection_results 
                WHERE image_filename = $1 
                ORDER BY created_at DESC 
                LIMIT 1
            ''', image_filename)
            
            if row:
                return {
                    "detected": bool(row['detected']),
                    "count": row['count'],
                    "confidence": row['confidence'],
                    "detections": json.loads(row['detections']) if row['detections'] else [],
                    "activities": json.loads(row['activities']) if row['activities'] else [],
                    "total_animals": row['total_animals'],
                    "primary_activity": row['primary_activity'],
                    "timestamp": row['timestamp'],
                    "source_name": row['source_name'],
                    "image_filename": row['image_filename']
                }
            return None 