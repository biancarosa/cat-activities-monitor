"""
Dashboard routes for cat activities and location monitoring.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from collections import defaultdict, Counter

from fastapi import APIRouter, Request, HTTPException
from sqlalchemy import select, func, desc, and_, distinct

from persistence.models import DetectionResult, CatProfile

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/dashboard")


@router.get(
    "/overview",
    summary="Dashboard Overview",
    description="Get overall dashboard statistics including total cats, recent activity, and location summary.",
    response_description="Dashboard overview statistics",
)
async def get_dashboard_overview(request: Request, hours: int = 24):
    """Get dashboard overview statistics for the specified time period."""
    try:
        database_service = request.app.state.database_service
        
        # Validate hours parameter
        if hours < 1 or hours > 168:  # Max 1 week
            raise HTTPException(status_code=400, detail="Hours must be between 1 and 168")
        
        cutoff_time = datetime.now() - timedelta(hours=hours)
        cutoff_time_str = cutoff_time.isoformat()
        
        async with database_service.get_session() as session:
            # Get total unique cats (named cats only)
            total_cats_stmt = select(func.count(distinct(CatProfile.cat_uuid)))
            total_cats_result = await session.execute(total_cats_stmt)
            total_named_cats = total_cats_result.scalar() or 0
            
            # Get recent detections within time period
            recent_detections_stmt = select(DetectionResult).where(
                DetectionResult.timestamp >= cutoff_time_str
            ).order_by(desc(DetectionResult.timestamp))
            recent_detections_result = await session.execute(recent_detections_stmt)
            recent_detections = recent_detections_result.scalars().all()
            
            # Process detection data
            total_detections = len(recent_detections)
            total_cats_detected = 0
            location_activity = defaultdict(int)
            named_cats_seen = set()
            recent_activities = []
            
            for detection in recent_detections:
                total_cats_detected += detection.cats_count or 0
                location_activity[detection.source_name] += detection.cats_count or 0
                
                # Parse detections JSON to get named cats
                if detection.detections:
                    import json
                    detections_data = (
                        detection.detections 
                        if isinstance(detection.detections, list) 
                        else json.loads(detection.detections)
                    )
                    
                    for det in detections_data:
                        if det.get('cat_name'):
                            named_cats_seen.add(det['cat_name'])
                        
                        # Add to recent activities
                        activity_item = {
                            "timestamp": detection.timestamp,
                            "location": detection.source_name,
                            "cat_name": det.get('cat_name'),
                            "activity": det.get('activity') or det.get('contextual_activity'),
                            "confidence": det.get('confidence', 0)
                        }
                        recent_activities.append(activity_item)
            
            # Sort locations by activity count
            top_locations = sorted(
                location_activity.items(), 
                key=lambda x: x[1], 
                reverse=True
            )[:5]
            
            # Sort recent activities by timestamp (most recent first)
            recent_activities.sort(key=lambda x: x['timestamp'], reverse=True)
            recent_activities = recent_activities[:10]  # Limit to 10 most recent
            
            return {
                "time_period_hours": hours,
                "total_named_cats": total_named_cats,
                "named_cats_seen_recently": len(named_cats_seen),
                "total_detections": total_detections,
                "total_cats_detected": total_cats_detected,
                "top_locations": [
                    {"location": loc, "activity_count": count} 
                    for loc, count in top_locations
                ],
                "recent_activities": recent_activities,
                "summary": {
                    "avg_cats_per_detection": (
                        round(total_cats_detected / total_detections, 2) 
                        if total_detections > 0 else 0
                    ),
                    "most_active_location": top_locations[0][0] if top_locations else None,
                    "named_cats_list": list(named_cats_seen),
                }
            }
            
    except Exception as e:
        logger.error(f"Error getting dashboard overview: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to get dashboard overview: {str(e)}"
        )


@router.get(
    "/cats",
    summary="Cat Activity Summary",
    description="Get per-cat activity summary with locations and recent behavior patterns.",
    response_description="Per-cat activity data",
)
async def get_cats_dashboard(request: Request, hours: int = 24):
    """Get per-cat activity summary for the specified time period."""
    try:
        database_service = request.app.state.database_service
        
        # Validate hours parameter
        if hours < 1 or hours > 168:  # Max 1 week
            raise HTTPException(status_code=400, detail="Hours must be between 1 and 168")
        
        cutoff_time = datetime.now() - timedelta(hours=hours)
        cutoff_time_str = cutoff_time.isoformat()
        
        async with database_service.get_session() as session:
            # Get all cat profiles
            cat_profiles_stmt = select(CatProfile)
            cat_profiles_result = await session.execute(cat_profiles_stmt)
            cat_profiles = cat_profiles_result.scalars().all()
            
            # Get recent detections within time period
            recent_detections_stmt = select(DetectionResult).where(
                DetectionResult.timestamp >= cutoff_time_str
            ).order_by(desc(DetectionResult.timestamp))
            recent_detections_result = await session.execute(recent_detections_stmt)
            recent_detections = recent_detections_result.scalars().all()
            
            # Process cat activity data
            cat_activities = {}
            
            # Initialize with cat profiles
            for profile in cat_profiles:
                cat_activities[profile.name] = {
                    "cat_name": profile.name,
                    "cat_uuid": profile.cat_uuid,
                    "description": profile.description,
                    "total_detections": 0,
                    "locations": defaultdict(int),
                    "activities": defaultdict(int),
                    "last_seen": None,
                    "last_location": None,
                    "confidence_scores": [],
                    "recent_timeline": []
                }
            
            # Process detection data
            import json
            for detection in recent_detections:
                if detection.detections:
                    detections_data = (
                        detection.detections 
                        if isinstance(detection.detections, list) 
                        else json.loads(detection.detections)
                    )
                    
                    for det in detections_data:
                        cat_name = det.get('cat_name')
                        if cat_name and cat_name in cat_activities:
                            cat_data = cat_activities[cat_name]
                            
                            # Update statistics
                            cat_data["total_detections"] += 1
                            cat_data["locations"][detection.source_name] += 1
                            cat_data["confidence_scores"].append(det.get('confidence', 0))
                            
                            # Update activity
                            activity = det.get('activity') or det.get('contextual_activity') or 'unknown'
                            cat_data["activities"][activity] += 1
                            
                            # Update last seen (timestamps are strings in ISO format)
                            if (cat_data["last_seen"] is None or 
                                detection.timestamp > cat_data["last_seen"]):
                                cat_data["last_seen"] = detection.timestamp
                                cat_data["last_location"] = detection.source_name
                            
                            # Add to timeline
                            cat_data["recent_timeline"].append({
                                "timestamp": detection.timestamp,
                                "location": detection.source_name,
                                "activity": activity,
                                "confidence": det.get('confidence', 0)
                            })
            
            # Format results
            cats_summary = []
            for cat_name, data in cat_activities.items():
                # Convert defaultdicts to regular dicts and sort
                locations_sorted = sorted(
                    data["locations"].items(), 
                    key=lambda x: x[1], 
                    reverse=True
                )
                activities_sorted = sorted(
                    data["activities"].items(), 
                    key=lambda x: x[1], 
                    reverse=True
                )
                
                # Sort timeline by timestamp (most recent first)
                data["recent_timeline"].sort(key=lambda x: x['timestamp'], reverse=True)
                data["recent_timeline"] = data["recent_timeline"][:5]  # Limit to 5 most recent
                
                cat_summary = {
                    "cat_name": cat_name,
                    "cat_uuid": data["cat_uuid"],
                    "description": data["description"],
                    "total_detections": data["total_detections"],
                    "favorite_locations": [
                        {"location": loc, "count": count} 
                        for loc, count in locations_sorted
                    ],
                    "common_activities": [
                        {"activity": act, "count": count} 
                        for act, count in activities_sorted
                    ],
                    "last_seen": data["last_seen"],
                    "last_location": data["last_location"],
                    "avg_confidence": (
                        round(sum(data["confidence_scores"]) / len(data["confidence_scores"]), 3)
                        if data["confidence_scores"] else 0
                    ),
                    "recent_timeline": data["recent_timeline"],
                    "is_active": data["total_detections"] > 0
                }
                cats_summary.append(cat_summary)
            
            # Sort by total detections (most active first)
            cats_summary.sort(key=lambda x: x["total_detections"], reverse=True)
            
            return {
                "time_period_hours": hours,
                "cats": cats_summary,
                "total_cats": len(cats_summary),
                "active_cats": len([cat for cat in cats_summary if cat["is_active"]])
            }
            
    except Exception as e:
        logger.error(f"Error getting cats dashboard: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to get cats dashboard: {str(e)}"
        )


@router.get(
    "/locations",
    summary="Location Activity Summary",
    description="Get per-location activity summary with cat presence and activity patterns.",
    response_description="Per-location activity data",
)
async def get_locations_dashboard(request: Request, hours: int = 24):
    """Get per-location activity summary for the specified time period."""
    try:
        database_service = request.app.state.database_service
        config_service = request.app.state.config_service
        
        # Validate hours parameter
        if hours < 1 or hours > 168:  # Max 1 week
            raise HTTPException(status_code=400, detail="Hours must be between 1 and 168")
        
        cutoff_time = datetime.now() - timedelta(hours=hours)
        cutoff_time_str = cutoff_time.isoformat()
        
        # Get configured cameras for location info
        config = config_service.config
        camera_info = {}
        if config and config.images:
            for camera in config.images:
                camera_info[camera.name] = {
                    "enabled": camera.enabled,
                    "interval_seconds": camera.interval_seconds,
                    "url": camera.url
                }
        
        async with database_service.get_session() as session:
            # Get recent detections within time period
            recent_detections_stmt = select(DetectionResult).where(
                DetectionResult.timestamp >= cutoff_time_str
            ).order_by(desc(DetectionResult.timestamp))
            recent_detections_result = await session.execute(recent_detections_stmt)
            recent_detections = recent_detections_result.scalars().all()
            
            # Process location data
            location_data = defaultdict(lambda: {
                "location": "",
                "total_detections": 0,
                "total_cats_detected": 0,
                "unique_cats": set(),
                "activities": defaultdict(int),
                "hourly_activity": defaultdict(int),
                "recent_timeline": [],
                "confidence_scores": [],
                "camera_config": None
            })
            
            # Process detection data
            import json
            for detection in recent_detections:
                location = detection.source_name
                location_data[location]["location"] = location
                location_data[location]["total_detections"] += 1
                location_data[location]["total_cats_detected"] += detection.cats_count or 0
                location_data[location]["camera_config"] = camera_info.get(location)
                
                # Hour for hourly activity pattern (parse timestamp string)
                timestamp_dt = datetime.fromisoformat(detection.timestamp)
                hour = timestamp_dt.hour
                location_data[location]["hourly_activity"][hour] += detection.cats_count or 0
                
                if detection.detections:
                    detections_data = (
                        detection.detections 
                        if isinstance(detection.detections, list) 
                        else json.loads(detection.detections)
                    )
                    
                    for det in detections_data:
                        # Track unique cats
                        cat_name = det.get('cat_name')
                        if cat_name:
                            location_data[location]["unique_cats"].add(cat_name)
                        
                        # Track activities
                        activity = det.get('activity') or det.get('contextual_activity') or 'unknown'
                        location_data[location]["activities"][activity] += 1
                        
                        # Track confidence scores
                        location_data[location]["confidence_scores"].append(det.get('confidence', 0))
                        
                        # Add to timeline
                        location_data[location]["recent_timeline"].append({
                            "timestamp": detection.timestamp,
                            "cat_name": cat_name,
                            "activity": activity,
                            "confidence": det.get('confidence', 0)
                        })
            
            # Format results
            locations_summary = []
            for location, data in location_data.items():
                # Convert sets and defaultdicts
                unique_cats_list = list(data["unique_cats"])
                activities_sorted = sorted(
                    data["activities"].items(), 
                    key=lambda x: x[1], 
                    reverse=True
                )
                
                # Create hourly activity array (24 hours)
                hourly_pattern = [data["hourly_activity"].get(hour, 0) for hour in range(24)]
                
                # Sort timeline by timestamp (most recent first)
                data["recent_timeline"].sort(key=lambda x: x['timestamp'], reverse=True)
                data["recent_timeline"] = data["recent_timeline"][:10]  # Limit to 10 most recent
                
                location_summary = {
                    "location": location,
                    "total_detections": data["total_detections"],
                    "total_cats_detected": data["total_cats_detected"],
                    "unique_cats_count": len(unique_cats_list),
                    "unique_cats": unique_cats_list,
                    "common_activities": [
                        {"activity": act, "count": count} 
                        for act, count in activities_sorted
                    ],
                    "hourly_activity_pattern": hourly_pattern,
                    "peak_hour": max(range(24), key=lambda h: data["hourly_activity"].get(h, 0)),
                    "avg_confidence": (
                        round(sum(data["confidence_scores"]) / len(data["confidence_scores"]), 3)
                        if data["confidence_scores"] else 0
                    ),
                    "avg_cats_per_detection": (
                        round(data["total_cats_detected"] / data["total_detections"], 2)
                        if data["total_detections"] > 0 else 0
                    ),
                    "recent_timeline": data["recent_timeline"],
                    "camera_config": data["camera_config"]
                }
                locations_summary.append(location_summary)
            
            # Sort by total activity (most active first)
            locations_summary.sort(key=lambda x: x["total_cats_detected"], reverse=True)
            
            return {
                "time_period_hours": hours,
                "locations": locations_summary,
                "total_locations": len(locations_summary)
            }
            
    except Exception as e:
        logger.error(f"Error getting locations dashboard: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to get locations dashboard: {str(e)}"
        )


@router.get(
    "/timeline",
    summary="Activity Timeline",
    description="Get activity timeline data for charts and graphs showing activity patterns over time.",
    response_description="Timeline activity data for visualization",
)
async def get_timeline_dashboard(request: Request, hours: int = 24, granularity: str = "hour"):
    """Get activity timeline data for the specified time period and granularity."""
    try:
        database_service = request.app.state.database_service
        
        # Validate parameters
        if hours < 1 or hours > 168:  # Max 1 week
            raise HTTPException(status_code=400, detail="Hours must be between 1 and 168")
        
        if granularity not in ["hour", "day", "15min"]:
            raise HTTPException(status_code=400, detail="Granularity must be 'hour', 'day', or '15min'")
        
        cutoff_time = datetime.now() - timedelta(hours=hours)
        cutoff_time_str = cutoff_time.isoformat()
        
        async with database_service.get_session() as session:
            # Get recent detections within time period
            recent_detections_stmt = select(DetectionResult).where(
                DetectionResult.timestamp >= cutoff_time_str
            ).order_by(DetectionResult.timestamp)  # Ascending for timeline
            recent_detections_result = await session.execute(recent_detections_stmt)
            recent_detections = recent_detections_result.scalars().all()
            
            # Create time buckets based on granularity
            timeline_data = defaultdict(lambda: {
                "timestamp": "",
                "total_detections": 0,
                "total_cats": 0,
                "locations": defaultdict(int),
                "activities": defaultdict(int),
                "named_cats": defaultdict(int),
                "cat_activities": defaultdict(lambda: defaultdict(int))  # cat_name -> activity -> count
            })
            
            # Determine bucket size
            if granularity == "15min":
                bucket_minutes = 15
            elif granularity == "hour":
                bucket_minutes = 60
            else:  # day
                bucket_minutes = 1440
            
            # Process detection data into time buckets
            import json
            for detection in recent_detections:
                # Calculate bucket timestamp (parse timestamp string)
                timestamp = datetime.fromisoformat(detection.timestamp)
                if granularity == "15min":
                    bucket_time = timestamp.replace(minute=(timestamp.minute // 15) * 15, second=0, microsecond=0)
                elif granularity == "hour":
                    bucket_time = timestamp.replace(minute=0, second=0, microsecond=0)
                else:  # day
                    bucket_time = timestamp.replace(hour=0, minute=0, second=0, microsecond=0)
                
                bucket_key = bucket_time.isoformat()
                
                # Update bucket data
                timeline_data[bucket_key]["timestamp"] = bucket_key
                timeline_data[bucket_key]["total_detections"] += 1
                timeline_data[bucket_key]["total_cats"] += detection.cats_count or 0
                timeline_data[bucket_key]["locations"][detection.source_name] += detection.cats_count or 0
                
                if detection.detections:
                    detections_data = (
                        detection.detections 
                        if isinstance(detection.detections, list) 
                        else json.loads(detection.detections)
                    )
                    
                    for det in detections_data:
                        # Track activities
                        activity = det.get('activity') or det.get('contextual_activity') or 'unknown'
                        timeline_data[bucket_key]["activities"][activity] += 1
                        
                        # Track named cats
                        cat_name = det.get('cat_name')
                        if cat_name:
                            timeline_data[bucket_key]["named_cats"][cat_name] += 1
                            # Track per-cat activities
                            timeline_data[bucket_key]["cat_activities"][cat_name][activity] += 1
            
            # Convert to sorted list
            timeline_list = []
            for bucket_key in sorted(timeline_data.keys()):
                data = timeline_data[bucket_key]
                
                # Convert defaultdicts to regular dicts
                locations_dict = dict(data["locations"])
                activities_dict = dict(data["activities"])
                named_cats_dict = dict(data["named_cats"])
                cat_activities_dict = {
                    cat_name: dict(activities) 
                    for cat_name, activities in data["cat_activities"].items()
                }
                
                timeline_item = {
                    "timestamp": bucket_key,
                    "total_detections": data["total_detections"],
                    "total_cats": data["total_cats"],
                    "locations": locations_dict,
                    "activities": activities_dict,
                    "named_cats": named_cats_dict,
                    "cat_activities": cat_activities_dict,
                    "unique_cats_count": len(named_cats_dict),
                    "most_active_location": max(locations_dict, key=locations_dict.get) if locations_dict else None,
                    "primary_activity": max(activities_dict, key=activities_dict.get) if activities_dict else None
                }
                timeline_list.append(timeline_item)
            
            return {
                "time_period_hours": hours,
                "granularity": granularity,
                "bucket_size_minutes": bucket_minutes,
                "timeline": timeline_list,
                "total_buckets": len(timeline_list),
                "summary": {
                    "total_detections": sum(item["total_detections"] for item in timeline_list),
                    "total_cats_detected": sum(item["total_cats"] for item in timeline_list),
                    "peak_activity_time": (
                        max(timeline_list, key=lambda x: x["total_cats"])["timestamp"] 
                        if timeline_list else None
                    )
                }
            }
            
    except Exception as e:
        logger.error(f"Error getting timeline dashboard: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to get timeline dashboard: {str(e)}"
        )