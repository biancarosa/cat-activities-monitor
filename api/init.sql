-- Initialize Cat Activities Monitor Database
-- This script is run automatically when the PostgreSQL container starts

-- Create main database tables
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
);

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
);

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
);

-- Create indexes for better performance
CREATE INDEX IF NOT EXISTS idx_feedback_timestamp ON feedback(timestamp);
CREATE INDEX IF NOT EXISTS idx_feedback_type ON feedback(feedback_type);
CREATE INDEX IF NOT EXISTS idx_cat_profiles_updated ON cat_profiles(updated_at);
CREATE INDEX IF NOT EXISTS idx_detection_source ON detection_results(source_name);
CREATE INDEX IF NOT EXISTS idx_detection_timestamp ON detection_results(timestamp);