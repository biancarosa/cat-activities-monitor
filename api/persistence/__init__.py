"""
Persistence layer for cat activities monitor.

This package contains SQLAlchemy models and database-related utilities
for the cat activities monitoring application.
"""

from .models import Base, CatProfile, DetectionResult, Feedback

__all__ = ["Base", "CatProfile", "DetectionResult", "Feedback"]
