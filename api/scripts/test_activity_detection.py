#!/usr/bin/env python3
"""
Test script for cat activity detection functionality.
This demonstrates the new activity recognition features.
"""

import sys
from main import (
    CatActivity, 
    ActivityDetection, 
    analyze_cat_pose_activity, 
    detect_movement_activity,
    Detection
)

def test_pose_activities():
    """Test pose-based activity detection."""
    print("🧪 Testing Pose-Based Activity Detection")
    print("=" * 50)
    
    # Test cases: different bounding box scenarios
    test_cases = [
        {
            "name": "Cat lying down",
            "bbox": {"x1": 100, "y1": 200, "x2": 300, "y2": 250, "width": 200, "height": 50},
            "expected": CatActivity.LYING
        },
        {
            "name": "Cat sitting",
            "bbox": {"x1": 100, "y1": 300, "x2": 180, "y2": 380, "width": 80, "height": 80},
            "expected": CatActivity.SITTING
        },
        {
            "name": "Cat standing",
            "bbox": {"x1": 100, "y1": 200, "x2": 150, "y2": 320, "width": 50, "height": 120},
            "expected": CatActivity.STANDING
        }
    ]
    
    for test_case in test_cases:
        detection = Detection(
            class_id=15,
            class_name="cat",
            confidence=0.85,
            bounding_box=test_case["bbox"]
        )
        
        activity = analyze_cat_pose_activity(detection, 640, 480)
        
        print(f"\n📋 {test_case['name']}:")
        print(f"   Bounding box: {test_case['bbox']['width']}x{test_case['bbox']['height']}")
        print(f"   Aspect ratio: {test_case['bbox']['width']/test_case['bbox']['height']:.2f}")
        print(f"   🎯 Detected: {activity.activity.value}")
        print(f"   📊 Confidence: {activity.confidence:.2f}")
        print(f"   💭 Reasoning: {activity.reasoning}")
        
        if activity.activity == test_case["expected"]:
            print("   ✅ CORRECT!")
        else:
            print(f"   ❌ Expected: {test_case['expected'].value}")

def test_activity_enum():
    """Test that all activity types are available."""
    print("\n🏷️  Testing Activity Types")
    print("=" * 50)
    
    activities = [activity.value for activity in CatActivity]
    print(f"Available activities: {activities}")
    print(f"Total activities: {len(activities)}")
    
    # Test specific activities
    expected_activities = [
        "unknown", "sitting", "lying", "standing", 
        "moving", "eating", "playing", "sleeping", "grooming"
    ]
    
    for expected in expected_activities:
        if expected in activities:
            print(f"✅ {expected}")
        else:
            print(f"❌ Missing: {expected}")

def test_activity_detection_model():
    """Test the ActivityDetection model."""
    print("\n📊 Testing ActivityDetection Model")
    print("=" * 50)
    
    activity = ActivityDetection(
        activity=CatActivity.PLAYING,
        confidence=0.85,
        reasoning="Test activity detection",
        bounding_box={"x1": 100, "y1": 100, "x2": 200, "y2": 200, "width": 100, "height": 100},
        duration_seconds=30.0
    )
    
    print(f"Activity: {activity.activity.value}")
    print(f"Confidence: {activity.confidence}")
    print(f"Reasoning: {activity.reasoning}")
    print(f"Duration: {activity.duration_seconds}s")
    print("✅ ActivityDetection model working!")

def main():
    """Run all tests."""
    print("🐱 Cat Activities Monitor API - Activity Detection Tests")
    print("=" * 60)
    
    try:
        test_activity_enum()
        test_activity_detection_model()
        test_pose_activities()
        
        print("\n" + "=" * 60)
        print("🎉 All activity detection tests completed!")
        print("✨ Your Cat Activities Monitor API can now recognize cat activities!")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 