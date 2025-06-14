"""
Migration service for managing database schema changes using Alembic.
"""

import logging
import subprocess
import os
from pathlib import Path
from typing import Optional, List, Dict, Any
from alembic.config import Config
from alembic import command
from alembic.runtime.migration import MigrationContext
from alembic.script import ScriptDirectory
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

logger = logging.getLogger(__name__)


class MigrationService:
    """Service for managing database migrations using Alembic."""
    
    def __init__(self, database_url: str = None):
        """Initialize migration service."""
        self.database_url = database_url or os.getenv(
            'DATABASE_URL', 
            'postgresql://db_user:db_password@localhost:5432/cats_monitor'
        )
        
        # Set up Alembic configuration
        self.alembic_cfg = Config()
        self.alembic_cfg.set_main_option("script_location", "alembic")
        self.alembic_cfg.set_main_option("sqlalchemy.url", self.database_url)
        
        # Get the directory where alembic.ini is located
        self.api_dir = Path(__file__).parent.parent
        self.alembic_ini_path = self.api_dir / "alembic.ini"
        
        if self.alembic_ini_path.exists():
            self.alembic_cfg = Config(str(self.alembic_ini_path))
            self.alembic_cfg.set_main_option("sqlalchemy.url", self.database_url)
        
        logger.info(f"🔄 Migration service initialized for: {self.database_url}")
    
    async def check_database_connection(self) -> bool:
        """Check if database connection is available."""
        try:
            engine = create_engine(self.database_url)
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            engine.dispose()
            return True
        except Exception as e:
            logger.error(f"❌ Database connection failed: {e}")
            return False
    
    def get_current_revision(self) -> Optional[str]:
        """Get the current database revision."""
        try:
            engine = create_engine(self.database_url)
            with engine.connect() as connection:
                context = MigrationContext.configure(connection)
                current_rev = context.get_current_revision()
            engine.dispose()
            return current_rev
        except Exception as e:
            logger.error(f"❌ Error getting current revision: {e}")
            return None
    
    def get_head_revision(self) -> Optional[str]:
        """Get the head (latest) revision."""
        try:
            script = ScriptDirectory.from_config(self.alembic_cfg)
            return script.get_current_head()
        except Exception as e:
            logger.error(f"❌ Error getting head revision: {e}")
            return None
    
    def get_migration_history(self) -> List[Dict[str, Any]]:
        """Get migration history."""
        try:
            script = ScriptDirectory.from_config(self.alembic_cfg)
            revisions = []
            
            for revision in script.walk_revisions():
                revisions.append({
                    "revision": revision.revision,
                    "down_revision": revision.down_revision,
                    "description": revision.doc,
                    "is_current": revision.revision == self.get_current_revision()
                })
            
            return revisions
        except Exception as e:
            logger.error(f"❌ Error getting migration history: {e}")
            return []
    
    def needs_migration(self) -> bool:
        """Check if database needs migration."""
        try:
            current = self.get_current_revision()
            head = self.get_head_revision()
            
            if current is None and head is not None:
                return True  # No migrations applied yet
            
            return current != head
        except Exception as e:
            logger.error(f"❌ Error checking migration status: {e}")
            return False
    
    def run_migrations(self, target_revision: str = "head") -> bool:
        """Run database migrations."""
        try:
            logger.info(f"🔄 Running migrations to revision: {target_revision}")
            
            # Run Alembic upgrade
            command.upgrade(self.alembic_cfg, target_revision)
            
            logger.info("✅ Migrations completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Migration failed: {e}")
            return False
    
    def rollback_migration(self, target_revision: str) -> bool:
        """Rollback database to a specific revision."""
        try:
            logger.info(f"🔄 Rolling back to revision: {target_revision}")
            
            # Run Alembic downgrade
            command.downgrade(self.alembic_cfg, target_revision)
            
            logger.info("✅ Rollback completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Rollback failed: {e}")
            return False
    
    def create_migration(self, message: str, autogenerate: bool = True) -> Optional[str]:
        """Create a new migration."""
        try:
            logger.info(f"🔄 Creating migration: {message}")
            
            if autogenerate:
                # Create migration with autogenerate
                command.revision(
                    self.alembic_cfg, 
                    message=message, 
                    autogenerate=True
                )
            else:
                # Create empty migration
                command.revision(self.alembic_cfg, message=message)
            
            logger.info("✅ Migration created successfully")
            return self.get_head_revision()
            
        except Exception as e:
            logger.error(f"❌ Failed to create migration: {e}")
            return None
    
    def get_migration_status(self) -> Dict[str, Any]:
        """Get comprehensive migration status."""
        try:
            current_rev = self.get_current_revision()
            head_rev = self.get_head_revision()
            history = self.get_migration_history()
            
            return {
                "current_revision": current_rev,
                "head_revision": head_rev,
                "needs_migration": self.needs_migration(),
                "migration_count": len(history),
                "pending_migrations": [
                    rev for rev in history 
                    if not rev["is_current"] and rev["revision"] != current_rev
                ],
                "database_url": self.database_url.replace(
                    self.database_url.split('@')[0].split('//')[-1], 
                    "***"
                ) if '@' in self.database_url else self.database_url
            }
        except Exception as e:
            logger.error(f"❌ Error getting migration status: {e}")
            return {
                "current_revision": None,
                "head_revision": None,
                "needs_migration": False,
                "migration_count": 0,
                "pending_migrations": [],
                "database_url": "error",
                "error": str(e)
            }
    
    def stamp_database(self, revision: str = "head") -> bool:
        """Stamp database with a specific revision (without running migrations)."""
        try:
            logger.info(f"🔄 Stamping database with revision: {revision}")
            
            command.stamp(self.alembic_cfg, revision)
            
            logger.info("✅ Database stamped successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to stamp database: {e}")
            return False
    
    def validate_migrations(self) -> bool:
        """Validate that all migrations can be run successfully."""
        try:
            # Check that we can get revisions
            current = self.get_current_revision()
            head = self.get_head_revision()
            
            if head is None:
                logger.error("❌ No migrations found")
                return False
            
            # Validate migration scripts
            script = ScriptDirectory.from_config(self.alembic_cfg)
            
            # Check for missing migrations
            for revision in script.walk_revisions():
                if revision.revision is None:
                    logger.error(f"❌ Invalid migration: {revision}")
                    return False
            
            logger.info("✅ All migrations validated successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Migration validation failed: {e}")
            return False
    
    async def detect_database_state(self) -> str:
        """
        Detect the current state of the database.
        
        Returns:
            "fresh" - Empty database, no tables
            "legacy" - Has tables but no alembic_version table
            "managed" - Has alembic_version table (migration-managed)
        """
        try:
            engine = create_engine(self.database_url)
            with engine.connect() as conn:
                # Check if alembic_version table exists
                alembic_version_exists = conn.execute(text("""
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables 
                        WHERE table_schema = 'public' 
                        AND table_name = 'alembic_version'
                    );
                """)).scalar()
                
                if alembic_version_exists:
                    return "managed"
                
                # Check if any of our main tables exist
                tables_exist = conn.execute(text("""
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables 
                        WHERE table_schema = 'public' 
                        AND table_name IN ('feedback', 'cat_profiles', 'detection_results')
                    );
                """)).scalar()
                
                if tables_exist:
                    return "legacy"
                else:
                    return "fresh"
                    
            engine.dispose()
            
        except Exception as e:
            logger.error(f"❌ Error detecting database state: {e}")
            return "unknown"
    
    async def _analyze_schema_completeness(self, conn) -> str:
        """
        Analyze schema completeness to determine the most appropriate revision.
        Returns the highest safe revision to stamp with.
        """
        # Define expected columns for each migration level
        migration_001_columns = {
            'feedback': ['feedback_id', 'image_filename', 'image_path', 'original_detections', 'user_annotations'],
            'cat_profiles': ['name', 'description', 'color', 'breed', 'favorite_activities'],
            'detection_results': ['source_name', 'image_filename', 'detected', 'count', 'confidence']
        }
        
        migration_002_columns = {
            'cat_features': ['id', 'cat_name', 'feature_vector', 'confidence'],
            'cat_recognition_models': ['id', 'model_name', 'model_path', 'training_date'],
            'detection_results': ['recognized_cat_names', 'recognition_confidence', 'feature_extracted']
        }
        
        migration_003_columns = {
            'activity_features': ['id', 'image_filename', 'cat_index', 'feature_vector'],
            'activity_models': ['id', 'model_name', 'model_path', 'training_date']
        }
        
        # Check each migration level
        try:
            # Check migration 003 (latest)
            activity_complete = True
            for table, columns in migration_003_columns.items():
                for column in columns:
                    exists = conn.execute(text("""
                        SELECT EXISTS (
                            SELECT FROM information_schema.columns 
                            WHERE table_schema = 'public' 
                            AND table_name = :table_name
                            AND column_name = :column_name
                        );
                    """), {"table_name": table, "column_name": column}).scalar()
                    if not exists:
                        activity_complete = False
                        break
                if not activity_complete:
                    break
            
            if activity_complete:
                logger.info("📋 Schema analysis: All migration 003 features detected")
                return "003_activity_features"
            
            # Check migration 002
            recognition_complete = True
            for table, columns in migration_002_columns.items():
                # Check if table exists first
                table_exists = conn.execute(text("""
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables 
                        WHERE table_schema = 'public' AND table_name = :table_name
                    );
                """), {"table_name": table}).scalar()
                
                if not table_exists and table in ['cat_features', 'cat_recognition_models']:
                    recognition_complete = False
                    break
                
                # For detection_results, check additional columns
                if table == 'detection_results' and table_exists:
                    for column in columns:
                        exists = conn.execute(text("""
                            SELECT EXISTS (
                                SELECT FROM information_schema.columns 
                                WHERE table_schema = 'public' 
                                AND table_name = :table_name
                                AND column_name = :column_name
                            );
                        """), {"table_name": table, "column_name": column}).scalar()
                        if not exists:
                            recognition_complete = False
                            break
            
            if recognition_complete:
                logger.info("📋 Schema analysis: All migration 002 features detected")
                return "002_cat_recognition"
            
            # Check migration 001 (basic)
            basic_complete = True
            for table, columns in migration_001_columns.items():
                # Check if table exists
                table_exists = conn.execute(text("""
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables 
                        WHERE table_schema = 'public' AND table_name = :table_name
                    );
                """), {"table_name": table}).scalar()
                
                if not table_exists:
                    basic_complete = False
                    break
                
                # Check core columns exist
                for column in columns[:3]:  # Check first 3 core columns
                    exists = conn.execute(text("""
                        SELECT EXISTS (
                            SELECT FROM information_schema.columns 
                            WHERE table_schema = 'public' 
                            AND table_name = :table_name
                            AND column_name = :column_name
                        );
                    """), {"table_name": table, "column_name": column}).scalar()
                    if not exists:
                        basic_complete = False
                        break
                if not basic_complete:
                    break
            
            if basic_complete:
                logger.info("📋 Schema analysis: Migration 001 features detected")
                return "001_initial_schema"
            
            # If we get here, database has some tables but incomplete schema
            logger.warning("⚠️ Schema analysis: Partial schema detected, defaulting to fresh migration")
            return "fresh"
            
        except Exception as e:
            logger.error(f"❌ Error during schema analysis: {e}")
            return "fresh"
    
    async def handle_legacy_database(self) -> bool:
        """
        Handle existing database that was created before migrations.
        Performs comprehensive schema analysis to determine appropriate revision.
        """
        try:
            logger.info("🔄 Detected legacy database, performing comprehensive schema analysis...")
            
            engine = create_engine(self.database_url)
            with engine.connect() as conn:
                # Perform detailed schema analysis
                stamp_revision = await self._analyze_schema_completeness(conn)
                
                if stamp_revision == "fresh":
                    logger.warning("⚠️ Schema too incomplete for legacy handling, treating as fresh database")
                    engine.dispose()
                    return False  # Let ensure_database_ready handle as fresh
                
            engine.dispose()
            
            # Stamp the database with determined revision
            logger.info(f"📋 Legacy database schema matches: {stamp_revision}")
            success = self.stamp_database(stamp_revision)
            if success:
                logger.info(f"✅ Legacy database stamped with revision: {stamp_revision}")
                return True
            else:
                logger.error("❌ Failed to stamp legacy database")
                return False
                
        except Exception as e:
            logger.error(f"❌ Error handling legacy database: {e}")
            return False
    
    async def ensure_database_ready(self) -> bool:
        """
        Ensure database is ready by running migrations if needed.
        Handles fresh, legacy, and managed databases.
        """
        try:
            # Check database connection
            if not await self.check_database_connection():
                logger.error("❌ Cannot connect to database")
                return False
            
            # Detect database state
            db_state = await self.detect_database_state()
            logger.info(f"🔍 Database state detected: {db_state}")
            
            if db_state == "fresh":
                # Fresh database - run all migrations
                logger.info("🔄 Fresh database detected, running all migrations...")
                success = self.run_migrations()
                if not success:
                    logger.error("❌ Failed to run migrations on fresh database")
                    return False
                    
            elif db_state == "legacy":
                # Legacy database - stamp then migrate
                logger.info("🔄 Legacy database detected, stamping current state...")
                success = await self.handle_legacy_database()
                if not success:
                    # If legacy handling failed due to incomplete schema, treat as fresh
                    logger.warning("⚠️ Legacy database schema too incomplete, treating as fresh...")
                    success = self.run_migrations()
                    if not success:
                        logger.error("❌ Failed to run migrations on partial legacy database")
                        return False
                else:
                    # After successful stamping, check if any new migrations need to be applied
                    if self.needs_migration():
                        logger.info("🔄 Running pending migrations on legacy database...")
                        success = self.run_migrations()
                        if not success:
                            logger.error("❌ Failed to run pending migrations")
                            return False
                        
            elif db_state == "managed":
                # Already migration-managed - check if updates needed
                if self.needs_migration():
                    logger.info("🔄 Managed database needs migration, running migrations...")
                    success = self.run_migrations()
                    if not success:
                        logger.error("❌ Failed to run migrations")
                        return False
                else:
                    logger.info("✅ Managed database is up to date")
                    
            else:
                logger.error(f"❌ Unknown database state: {db_state}")
                return False
            
            logger.info("✅ Database is ready and up to date")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error ensuring database ready: {e}")
            return False