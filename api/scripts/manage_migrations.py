#!/usr/bin/env python3
"""
Database migration management script for Cat Activities Monitor.
"""

import sys
import os
import argparse
import asyncio
from pathlib import Path

# Add the api directory to the path so we can import modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from services.migration_service import MigrationService
from services.database_service import DatabaseService


async def main():
    """Main migration management function."""
    parser = argparse.ArgumentParser(description="Manage database migrations")
    
    # Subcommands
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Status command
    status_parser = subparsers.add_parser('status', help='Show migration status')
    
    # Upgrade command
    upgrade_parser = subparsers.add_parser('upgrade', help='Run migrations')
    upgrade_parser.add_argument('--revision', default='head', help='Target revision (default: head)')
    
    # Downgrade command
    downgrade_parser = subparsers.add_parser('downgrade', help='Rollback migrations')
    downgrade_parser.add_argument('revision', help='Target revision to rollback to')
    
    # History command
    history_parser = subparsers.add_parser('history', help='Show migration history')
    
    # Create command
    create_parser = subparsers.add_parser('create', help='Create new migration')
    create_parser.add_argument('message', help='Migration message')
    create_parser.add_argument('--autogenerate', action='store_true', help='Auto-generate migration from model changes')
    
    # Stamp command
    stamp_parser = subparsers.add_parser('stamp', help='Stamp database with revision')
    stamp_parser.add_argument('--revision', default='head', help='Revision to stamp (default: head)')
    
    # Validate command
    validate_parser = subparsers.add_parser('validate', help='Validate migrations')
    
    # Detect state command
    detect_parser = subparsers.add_parser('detect-state', help='Detect database state')
    
    # Handle legacy command
    legacy_parser = subparsers.add_parser('handle-legacy', help='Handle legacy database')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Initialize services
    database_url = os.getenv('DATABASE_URL', 'postgresql://db_user:db_password@localhost:5432/cats_monitor')
    migration_service = MigrationService(database_url)
    db_service = DatabaseService(database_url)
    
    try:
        if args.command == 'status':
            await show_status(migration_service)
        
        elif args.command == 'upgrade':
            await run_upgrade(migration_service, args.revision)
        
        elif args.command == 'downgrade':
            await run_downgrade(migration_service, args.revision)
        
        elif args.command == 'history':
            await show_history(migration_service)
        
        elif args.command == 'create':
            await create_migration(migration_service, args.message, args.autogenerate)
        
        elif args.command == 'stamp':
            await stamp_database(migration_service, args.revision)
        
        elif args.command == 'validate':
            await validate_migrations(migration_service)
        
        elif args.command == 'detect-state':
            await detect_database_state(migration_service)
        
        elif args.command == 'handle-legacy':
            await handle_legacy_database(migration_service)
        
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


async def show_status(migration_service: MigrationService):
    """Show migration status."""
    print("📊 Database Migration Status")
    print("=" * 40)
    
    # Check database connection
    connected = await migration_service.check_database_connection()
    print(f"Database Connection: {'✅ Connected' if connected else '❌ Failed'}")
    
    if not connected:
        print("Cannot check migration status without database connection.")
        return
    
    status = migration_service.get_migration_status()
    
    print(f"Current Revision: {status['current_revision'] or 'None'}")
    print(f"Head Revision: {status['head_revision'] or 'None'}")
    print(f"Needs Migration: {'✅ Yes' if status['needs_migration'] else '❌ No'}")
    print(f"Total Migrations: {status['migration_count']}")
    
    if status['pending_migrations']:
        print("\n🔄 Pending Migrations:")
        for migration in status['pending_migrations']:
            print(f"  - {migration['revision']}: {migration['description']}")
    else:
        print("\n✅ No pending migrations")


async def run_upgrade(migration_service: MigrationService, revision: str):
    """Run database upgrade."""
    print(f"🔄 Running migrations to revision: {revision}")
    
    # Check connection first
    connected = await migration_service.check_database_connection()
    if not connected:
        print("❌ Cannot connect to database")
        return
    
    success = migration_service.run_migrations(revision)
    if success:
        print("✅ Migrations completed successfully")
    else:
        print("❌ Migration failed")
        sys.exit(1)


async def run_downgrade(migration_service: MigrationService, revision: str):
    """Run database downgrade."""
    print(f"🔄 Rolling back to revision: {revision}")
    
    # Confirm the action
    confirm = input(f"Are you sure you want to rollback to {revision}? (y/N): ")
    if confirm.lower() != 'y':
        print("Rollback cancelled")
        return
    
    success = migration_service.rollback_migration(revision)
    if success:
        print("✅ Rollback completed successfully")
    else:
        print("❌ Rollback failed")
        sys.exit(1)


async def show_history(migration_service: MigrationService):
    """Show migration history."""
    print("📚 Migration History")
    print("=" * 50)
    
    history = migration_service.get_migration_history()
    
    if not history:
        print("No migrations found")
        return
    
    for migration in history:
        status_icon = "✅" if migration["is_current"] else "⚪"
        print(f"{status_icon} {migration['revision']}")
        print(f"   └─ {migration['description']}")
        if migration['down_revision']:
            print(f"   └─ Revises: {migration['down_revision']}")
        print()


async def create_migration(migration_service: MigrationService, message: str, autogenerate: bool):
    """Create new migration."""
    print(f"🔄 Creating migration: {message}")
    
    revision = migration_service.create_migration(message, autogenerate)
    if revision:
        print(f"✅ Migration created: {revision}")
    else:
        print("❌ Failed to create migration")
        sys.exit(1)


async def stamp_database(migration_service: MigrationService, revision: str):
    """Stamp database with revision."""
    print(f"🔄 Stamping database with revision: {revision}")
    
    success = migration_service.stamp_database(revision)
    if success:
        print("✅ Database stamped successfully")
    else:
        print("❌ Failed to stamp database")
        sys.exit(1)


async def validate_migrations(migration_service: MigrationService):
    """Validate migrations."""
    print("🔍 Validating migrations...")
    
    success = migration_service.validate_migrations()
    if success:
        print("✅ All migrations are valid")
    else:
        print("❌ Migration validation failed")
        sys.exit(1)


async def detect_database_state(migration_service: MigrationService):
    """Detect and display database state."""
    print("🔍 Detecting database state...")
    
    # Check database connection first
    connected = await migration_service.check_database_connection()
    if not connected:
        print("❌ Cannot connect to database")
        return
    
    state = await migration_service.detect_database_state()
    
    descriptions = {
        "fresh": "Empty database with no tables",
        "legacy": "Has tables but no migration tracking (pre-migration database)",
        "managed": "Migration-managed database with version tracking",
        "unknown": "Unable to determine database state"
    }
    
    print(f"📊 Database State: {state}")
    print(f"📝 Description: {descriptions.get(state, 'Unknown state')}")
    
    if state != "unknown":
        needs_migration = migration_service.needs_migration()
        print(f"🔄 Needs Migration: {'Yes' if needs_migration else 'No'}")
        
        if state == "managed":
            current = migration_service.get_current_revision()
            head = migration_service.get_head_revision()
            print(f"📌 Current Revision: {current or 'None'}")
            print(f"🎯 Head Revision: {head or 'None'}")


async def handle_legacy_database(migration_service: MigrationService):
    """Handle legacy database."""
    print("🔄 Handling legacy database...")
    
    # Check database connection first
    connected = await migration_service.check_database_connection()
    if not connected:
        print("❌ Cannot connect to database")
        return
    
    # Check current state
    state = await migration_service.detect_database_state()
    
    if state == "managed":
        print("✅ Database is already migration-managed, no action needed")
        return
    elif state == "fresh":
        print("✅ Database is fresh, run normal migrations instead")
        print("💡 Use: python scripts/manage_migrations.py upgrade")
        return
    elif state == "legacy":
        print("🔄 Legacy database detected, analyzing schema...")
        
        success = await migration_service.handle_legacy_database()
        
        if success:
            print("✅ Legacy database handled successfully")
            print("📌 Database stamped with appropriate revision")
            
            # Check if any additional migrations are needed
            if migration_service.needs_migration():
                print("🔄 Additional migrations available, run 'upgrade' to apply them")
            else:
                print("✅ Database is now fully up to date")
        else:
            print("❌ Failed to handle legacy database")
            sys.exit(1)
    else:
        print(f"❌ Unknown database state: {state}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())