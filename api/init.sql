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

-- Create new tables for cat recognition feature extraction
CREATE TABLE IF NOT EXISTS cat_features (
    id SERIAL PRIMARY KEY,
    cat_name TEXT REFERENCES cat_profiles(name) ON DELETE CASCADE,
    feature_vector REAL[512] NOT NULL,  -- 512-dimensional feature vector
    image_path TEXT,
    extraction_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    quality_score REAL NOT NULL CHECK (quality_score >= 0.0 AND quality_score <= 1.0),
    pose_variant TEXT,
    detected_activity TEXT,
    activity_confidence REAL CHECK (activity_confidence IS NULL OR (activity_confidence >= 0.0 AND activity_confidence <= 1.0))
);

-- Create table for recognition model management
CREATE TABLE IF NOT EXISTS cat_recognition_models (
    id SERIAL PRIMARY KEY,
    model_name TEXT UNIQUE NOT NULL,
    model_path TEXT NOT NULL,
    feature_extractor TEXT NOT NULL DEFAULT 'resnet50',
    cats_included TEXT[] NOT NULL,
    created_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    accuracy_score REAL CHECK (accuracy_score IS NULL OR (accuracy_score >= 0.0 AND accuracy_score <= 1.0)),
    is_active BOOLEAN DEFAULT FALSE
);

-- Enhance existing detections table for recognition results
ALTER TABLE detection_results 
ADD COLUMN IF NOT EXISTS recognized_cat_names JSONB,
ADD COLUMN IF NOT EXISTS recognition_confidences JSONB,
ADD COLUMN IF NOT EXISTS is_manually_corrected BOOLEAN DEFAULT FALSE;

-- Create indexes for better performance
CREATE INDEX IF NOT EXISTS idx_feedback_timestamp ON feedback(timestamp);
CREATE INDEX IF NOT EXISTS idx_feedback_type ON feedback(feedback_type);
CREATE INDEX IF NOT EXISTS idx_cat_profiles_updated ON cat_profiles(updated_at);
CREATE INDEX IF NOT EXISTS idx_detection_source ON detection_results(source_name);
CREATE INDEX IF NOT EXISTS idx_detection_timestamp ON detection_results(timestamp);

-- Indexes for new recognition tables
CREATE INDEX IF NOT EXISTS idx_cat_features_cat_name ON cat_features(cat_name);
CREATE INDEX IF NOT EXISTS idx_cat_features_timestamp ON cat_features(extraction_timestamp);
CREATE INDEX IF NOT EXISTS idx_cat_features_quality ON cat_features(quality_score);
CREATE INDEX IF NOT EXISTS idx_cat_features_activity ON cat_features(detected_activity);
CREATE INDEX IF NOT EXISTS idx_recognition_models_active ON cat_recognition_models(is_active);
CREATE INDEX IF NOT EXISTS idx_recognition_models_created ON cat_recognition_models(created_timestamp);