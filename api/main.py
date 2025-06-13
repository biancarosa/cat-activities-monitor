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
    TrainingService
)

# Import route modules
from routes import (
    main_routes,
    system_routes,
    camera_routes,
    detection_routes,
    activity_routes,
    feedback_routes,
    training_routes,
    cat_routes
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("api.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Global background task
background_task = None
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
                        config.global_.timeout_seconds
                    )
                    
                    successful = sum(1 for r in results if r['success'])
                    logger.info(f"✅ Fetch completed: {successful}/{len(results)} successful")
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


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle."""
    global background_task
    
    logger.info("🚀 Starting Cat Activities Monitor API")
    
    try:
        # Initialize services in dependency order
        config_service = ConfigService()
        database_service = DatabaseService()
        detection_service = DetectionService()
        
        # Initialize async database
        await database_service.init_database()
        logger.info("💾 Database initialized")
        
        # TrainingService depends on DatabaseService  
        training_service = TrainingService(database_service)
        
        # ImageService depends on DetectionService and DatabaseService
        image_service = ImageService(detection_service, database_service)
        
        # Load initial configuration
        config = config_service.load_config()
        logger.info(f"📋 Configuration loaded: {len(config.images)} image sources")
        
        # Initialize ML model
        ml_model = detection_service.initialize_ml_model(config.global_.ml_model_config)
        logger.info(f"🤖 ML model loaded: {config.global_.ml_model_config.model}")
        
        # Load activity history and previous detections from database
        await detection_service.load_activity_history_from_database(database_service)
        
        # Store services in app state for dependency injection
        app.state.config_service = config_service
        app.state.database_service = database_service
        app.state.detection_service = detection_service
        app.state.image_service = image_service
        app.state.training_service = training_service
        
        # Start background image fetching task
        background_task = asyncio.create_task(background_image_fetcher(app))
        
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
        
        # Wait for background task to complete with shorter timeout
        if background_task and not background_task.done():
            try:
                await asyncio.wait_for(background_task, timeout=5.0)
                logger.info("✅ Background task stopped gracefully")
            except asyncio.TimeoutError:
                logger.warning("⚠️ Background task did not stop gracefully, cancelling forcefully")
                background_task.cancel()
                try:
                    await asyncio.wait_for(background_task, timeout=2.0)
                except (asyncio.CancelledError, asyncio.TimeoutError):
                    logger.warning("⚠️ Background task cancelled")
        
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
            "description": "Main application endpoints including root interface and API information."
        },
        {
            "name": "system",
            "description": "System administration endpoints for status, configuration, health checks, and logs."
        },
        {
            "name": "cameras",
            "description": "Camera and image source management endpoints for listing cameras and fetching images."
        },
        {
            "name": "detections",
            "description": "Detection results and analysis endpoints for retrieving detection data and images."
        },
        {
            "name": "activities",
            "description": "Activity analysis endpoints for tracking and summarizing cat activities across cameras."
        },
        {
            "name": "feedback",
            "description": "Feedback system endpoints for submitting corrections and improvements to enhance model accuracy."
        },
        {
            "name": "training",
            "description": "Model training and management endpoints for custom model creation and switching between models."
        },
        {
            "name": "cats",
            "description": "Cat profile management endpoints for creating, updating, and tracking individual cats."
        }
    ],
    servers=[
        {
            "url": "http://localhost:8000",
            "description": "Development server"
        },
        {
            "url": "https://your-domain.com",
            "description": "Production server"
        }
    ],
    lifespan=lifespan
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
            access_log=True
        )
    except KeyboardInterrupt:
        logger.info("🛑 Server interrupted by user")
    except Exception as e:
        logger.error(f"❌ Server error: {e}")
    finally:
        logger.info("👋 Server shutdown complete") 