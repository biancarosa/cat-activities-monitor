"""
A FastAPI-based system for monitoring cats using YOLO object detection and activity analysis.
"""

import asyncio
import logging
from contextlib import asynccontextmanager
import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles


# Import services
from services import (
    ConfigService,
    DatabaseService,
    DetectionService,
    ImageService,
    TrainingService,
    ImageCleanupService,
)
from services.cat_identification_service import CatIdentificationService

# Import route modules
from routes import (
    main_routes,
    system_routes,
    camera_routes,
    detection_routes,
    activity_routes,
    feedback_routes,
    training_routes,
    cat_routes,
    maintenance_routes,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("api.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

# Global background tasks
background_fetch_task = None
background_cleanup_task = None
shutdown_event = asyncio.Event()

# If detection folder doesnt exist, create it
if not os.path.exists("detections"):
    os.makedirs("detections")


async def background_image_fetcher(app: FastAPI):
    """Background task to fetch images periodically."""
    logger.info("🔄 Starting background image fetcher")

    while not shutdown_event.is_set():
        try:
            config_service = app.state.config_service
            image_service = app.state.image_service

            config = config_service.config
            if config:
                enabled_images = config_service.get_enabled_images()

                if enabled_images:
                    logger.info(f"🔍 Fetching {len(enabled_images)} enabled images")

                    # Check for shutdown before starting fetch
                    if shutdown_event.is_set():
                        break

                    results = await image_service.fetch_all_images(
                        enabled_images,
                        config.global_.ml_model_config,
                        config.global_.max_concurrent_fetches,
                        config.global_.timeout_seconds,
                    )

                    successful = sum(1 for r in results if r["success"])
                    logger.info(
                        f"✅ Fetch completed: {successful}/{len(results)} successful"
                    )
                else:
                    logger.debug("No enabled images to fetch")

            # Wait for the default interval or until shutdown
            try:
                interval = config.global_.default_interval_seconds if config else 60
                await asyncio.wait_for(shutdown_event.wait(), timeout=interval)
                break  # If shutdown_event is set, exit the loop
            except asyncio.TimeoutError:
                continue  # Timeout reached, continue with next fetch cycle

        except Exception as e:
            logger.error(f"Error in background image fetcher: {e}")
            try:
                await asyncio.wait_for(shutdown_event.wait(), timeout=5)
                break
            except asyncio.TimeoutError:
                continue

    logger.info("🔄 Background image fetcher stopped")


async def background_image_cleanup(app: FastAPI):
    """Background task to clean up old images periodically."""
    logger.info("🧹 Starting background image cleanup service")

    # Wait a bit before starting cleanup to let the system initialize
    try:
        await asyncio.wait_for(shutdown_event.wait(), timeout=60)
        return  # If shutdown happened during initial wait, exit
    except asyncio.TimeoutError:
        pass  # Continue with cleanup

    while not shutdown_event.is_set():
        try:
            config_service = app.state.config_service
            image_cleanup_service = app.state.image_cleanup_service

            config = config_service.config
            if config:
                # Set the detection path from config
                image_cleanup_service.set_detection_path(
                    config.global_.ml_model_config.detection_image_path
                )

                # Run cleanup
                logger.info("🧹 Running scheduled image cleanup")
                cleanup_summary = await image_cleanup_service.cleanup_old_images()
                
                if cleanup_summary["images_deleted"] > 0 or cleanup_summary["errors"] > 0:
                    logger.info(
                        f"🧹 Cleanup completed: {cleanup_summary['images_deleted']} images deleted, "
                        f"{cleanup_summary['database_records_updated']} records updated, "
                        f"{cleanup_summary['errors']} errors"
                    )
                else:
                    logger.debug("🧹 Cleanup completed: No old images found")

            # Wait for configured interval before next cleanup, or until shutdown
            config = config_service.config
            cleanup_interval_seconds = (
                config.global_.image_cleanup.cleanup_interval_hours * 3600
                if config else 86400  # Default to 24 hours if no config
            )
            
            try:
                await asyncio.wait_for(shutdown_event.wait(), timeout=cleanup_interval_seconds)
                break  # If shutdown_event is set, exit the loop
            except asyncio.TimeoutError:
                continue  # Timeout reached, continue with next cleanup cycle

        except Exception as e:
            logger.error(f"Error in background image cleanup: {e}")
            try:
                # Wait 1 hour before retrying on error
                await asyncio.wait_for(shutdown_event.wait(), timeout=3600)
                break
            except asyncio.TimeoutError:
                continue

    logger.info("🧹 Background image cleanup service stopped")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle."""
    global background_fetch_task, background_cleanup_task

    logger.info("🚀 Starting Cat Activities Monitor API")

    try:
        # Initialize services in dependency order
        config_service = ConfigService()
        database_service = DatabaseService()
        detection_service = DetectionService(database_service)

        # Initialize async database
        await database_service.init_database()
        logger.info("💾 Database initialized")

        # Initialize cat identification service
        cat_identification_service = CatIdentificationService(database_service)

        # Try to load trained model for cat identification
        model_loaded = await cat_identification_service.load_trained_model()
        if model_loaded:
            logger.info("🧠 Cat identification model loaded successfully")
        else:
            logger.info(
                "ℹ️ No trained cat identification model found - using similarity matching only"
            )

        # TrainingService depends on DatabaseService
        training_service = TrainingService(database_service)

        # ImageService depends on DetectionService and DatabaseService
        image_service = ImageService(detection_service, database_service)
        
        # Load initial configuration
        config = config_service.load_config()
        logger.info(f"📋 Configuration loaded: {len(config.images)} image sources")
        
        # ImageCleanupService depends on DatabaseService and uses config values
        cleanup_config = config.global_.image_cleanup
        image_cleanup_service = ImageCleanupService(
            database_service, 
            retention_days=cleanup_config.retention_days
        )

        # Initialize ML pipeline
        await detection_service.initialize_ml_pipeline(config.global_.ml_model_config)
        logger.info(f"🤖 ML pipeline loaded: {config.global_.ml_model_config.model}")

        # Store services in app state for dependency injection
        app.state.config_service = config_service
        app.state.database_service = database_service
        app.state.detection_service = detection_service
        app.state.image_service = image_service
        app.state.training_service = training_service
        app.state.cat_identification_service = cat_identification_service
        app.state.image_cleanup_service = image_cleanup_service

        # Start background tasks
        background_fetch_task = asyncio.create_task(background_image_fetcher(app))
        
        # Only start cleanup task if enabled in config
        if cleanup_config.enabled:
            background_cleanup_task = asyncio.create_task(background_image_cleanup(app))
            logger.info(f"🧹 Image cleanup service enabled: {cleanup_config.retention_days} days retention, {cleanup_config.cleanup_interval_hours}h interval")
        else:
            background_cleanup_task = None
            logger.info("🧹 Image cleanup service disabled in configuration")

        # Remove signal handlers since uvicorn handles them
        # Let uvicorn handle signals and trigger lifespan shutdown

        logger.info("✅ Cat Activities Monitor API startup completed")

        yield

    except Exception as e:
        logger.error(f"❌ Error during startup: {e}")
        raise
    finally:
        # Cleanup
        logger.info("🔄 Shutting down Cat Activities Monitor API")

        # Signal background task to stop
        shutdown_event.set()

        # Wait for background tasks to complete with shorter timeout
        tasks_to_wait = []
        if background_fetch_task and not background_fetch_task.done():
            tasks_to_wait.append(background_fetch_task)
        if background_cleanup_task and not background_cleanup_task.done():
            tasks_to_wait.append(background_cleanup_task)
            
        if tasks_to_wait:
            try:
                await asyncio.wait_for(asyncio.gather(*tasks_to_wait, return_exceptions=True), timeout=5.0)
                logger.info("✅ Background tasks stopped gracefully")
            except asyncio.TimeoutError:
                logger.warning(
                    "⚠️ Background tasks did not stop gracefully, cancelling forcefully"
                )
                for task in tasks_to_wait:
                    if not task.done():
                        task.cancel()
                try:
                    await asyncio.wait_for(asyncio.gather(*tasks_to_wait, return_exceptions=True), timeout=2.0)
                except (asyncio.CancelledError, asyncio.TimeoutError):
                    logger.warning("⚠️ Background tasks cancelled")

        logger.info("👋 Cat Activities Monitor API shutdown completed")


app = FastAPI(
    title="Cat Activities Monitor API",
    description="""
    A comprehensive FastAPI-based system for monitoring cats using YOLO object detection and activity analysis.    """,
    version="2.0.0",
    license_info={
        "name": "MIT License",
        "url": "https://github.com/biancarosa/cat-activities-monitor/blob/main/LICENSE",
    },
    openapi_tags=[
        {
            "name": "main",
            "description": "Main application endpoints including root interface and API information.",
        },
        {
            "name": "system",
            "description": "System administration endpoints for status, configuration, health checks, and logs.",
        },
        {
            "name": "cameras",
            "description": "Camera and image source management endpoints for listing cameras and fetching images.",
        },
        {
            "name": "detections",
            "description": "Detection results and analysis endpoints for retrieving detection data and images.",
        },
        {
            "name": "activities",
            "description": "Activity analysis endpoints for tracking and summarizing cat activities across cameras.",
        },
        {
            "name": "feedback",
            "description": "Feedback system endpoints for submitting corrections and improvements to enhance model accuracy.",
        },
        {
            "name": "training",
            "description": "Model training and management endpoints for custom model creation and switching between models.",
        },
        {
            "name": "cats",
            "description": "Cat profile management endpoints for creating, updating, and tracking individual cats.",
        },
        {
            "name": "maintenance",
            "description": "System maintenance endpoints for image cleanup, database management, and system optimization.",
        },
    ],
    servers=[
        {"url": "http://localhost:8000", "description": "Development server"},
        {"url": "https://your-domain.com", "description": "Production server"},
    ],
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory="detections"), name="static")

app.include_router(main_routes.router, tags=["main"])
app.include_router(system_routes.router, tags=["system"])
app.include_router(camera_routes.router, tags=["cameras"])
app.include_router(detection_routes.router, tags=["detections"])
app.include_router(activity_routes.router, tags=["activities"])
app.include_router(feedback_routes.router, tags=["feedback"])
app.include_router(training_routes.router, tags=["training"])
app.include_router(cat_routes.router, tags=["cats"])
app.include_router(maintenance_routes.router, tags=["maintenance"])

if __name__ == "__main__":
    import uvicorn

    logger.info("🚀 Starting Cat Activities Monitor API")

    try:
        uvicorn.run(
            "main:app",
            host="0.0.0.0",
            port=8000,
            reload=True,  # Enable reload for development
            log_level="info",
            access_log=True,
        )
    except KeyboardInterrupt:
        logger.info("🛑 Server interrupted by user")
    except Exception as e:
        logger.error(f"❌ Server error: {e}")
    finally:
        logger.info("👋 Server shutdown complete")
