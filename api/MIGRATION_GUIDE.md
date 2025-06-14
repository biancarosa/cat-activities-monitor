# Database Migration Guide

## Overview

The Cat Activities Monitor now uses Alembic for database schema management. This guide covers how to handle different database scenarios, especially existing databases created before the migration system.

## Database States

The system automatically detects three database states:

### 1. **Fresh Database** 🆕
- **Description**: Empty database with no tables
- **Action**: Runs all migrations from scratch
- **Log**: `🔄 Fresh database detected, running all migrations...`

### 2. **Legacy Database** 📚
- **Description**: Has existing tables but no migration tracking
- **Action**: Analyzes schema and stamps with appropriate revision
- **Log**: `🔄 Legacy database detected, stamping current state...`

### 3. **Managed Database** ✅
- **Description**: Migration-managed with `alembic_version` table
- **Action**: Runs pending migrations if needed
- **Log**: `✅ Managed database is up to date` or `🔄 Managed database needs migration...`

## Automatic Handling

### On App Startup

The system automatically handles all database states:

```python
# In main.py lifespan function
await database_service.init_database()
```

**Process:**
1. **Detect** database state (fresh/legacy/managed)
2. **Handle** accordingly:
   - Fresh → Run all migrations
   - Legacy → Stamp current state + run pending migrations
   - Managed → Run pending migrations if needed

### Legacy Database Detection

The system performs **comprehensive schema analysis** to determine the appropriate migration revision:

#### Schema Analysis Process
1. **Table Existence**: Checks if required tables exist
2. **Column Completeness**: Verifies all expected columns are present
3. **Schema Integrity**: Ensures tables have proper structure

#### Migration Levels Detected
- **Basic Schema (001)**: Core tables (`feedback`, `cat_profiles`, `detection_results`) with basic columns
- **Recognition Schema (002)**: + `cat_features`, `cat_recognition_models` tables + recognition columns in `detection_results`
- **Activity Schema (003)**: + `activity_features`, `activity_models` tables
- **Partial Schema**: Incomplete tables/columns → treated as fresh database

## Manual Management

### CLI Commands

If using docker, just use: `docker compose exec api bash` to get into the container and then run the commands below.

```bash
# Check migration status
python scripts/manage_migrations.py status

# Detect database state
python scripts/manage_migrations.py detect-state

# Handle legacy database manually
python scripts/manage_migrations.py handle-legacy

# Run migrations
python scripts/manage_migrations.py upgrade

# Show migration history
python scripts/manage_migrations.py history

# Create new migration
python scripts/manage_migrations.py create "Add new feature" --autogenerate

# Rollback migrations
python scripts/manage_migrations.py downgrade 002_cat_recognition

# Stamp database (advanced)
python scripts/manage_migrations.py stamp --revision 001_initial_schema

# Validate migrations
python scripts/manage_migrations.py validate
```

### Direct Alembic

```bash
# Check current revision
uv run alembic current

# Show migration history
uv run alembic history

# Stamp legacy database (manual)
uv run alembic stamp 001_initial_schema

# Run migrations
uv run alembic upgrade head
```

## Common Scenarios

### Scenario 1: Existing Production Database

**Situation**: You have a production database with tables created from `init.sql`

**What Happens:**
1. App detects `legacy` state
2. Analyzes existing tables
3. Stamps with appropriate revision (001, 002, or 003)
4. Runs any pending migrations
5. Database becomes `managed`

**Result**: ✅ No data loss, schema properly versioned

### Scenario 2: Fresh Installation

**Situation**: Brand new database, no tables

**What Happens:**
1. App detects `fresh` state
2. Runs all migrations (001 → 002 → 003)
3. Creates all tables and indexes
4. Database becomes `managed`

**Result**: ✅ Complete schema created with version tracking

### Scenario 3: Partially Updated Database

**Situation**: Database has some but not all new tables/columns

**What Happens:**
1. App detects `legacy` state
2. **Comprehensive schema analysis**:
   - Checks table existence
   - Verifies column completeness
   - Validates schema integrity
3. **Smart decision making**:
   - Complete schema → stamps with appropriate revision
   - Incomplete schema → treats as fresh database
4. Runs remaining migrations
5. Database becomes `managed`

**Result**: ✅ Schema completed safely, version tracking added

### Scenario 4: Corrupted or Incomplete Schema

**Situation**: Database has tables but missing critical columns

**What Happens:**
1. App detects `legacy` state
2. Schema analysis finds incomplete structure
3. **Safety mode**: Treats as fresh database
4. Recreates all tables and data (preserves existing data where possible)
5. Database becomes `managed`

**Result**: ✅ Clean schema rebuild, existing data preserved

## Migration Safety

### Backup Recommendations

```bash
# Before major updates, backup your database
pg_dump -h localhost -U db_user cats_monitor > backup_$(date +%Y%m%d_%H%M%S).sql
```

### Rollback Options

```bash
# Rollback to specific revision
python scripts/manage_migrations.py downgrade 002_cat_recognition
```

### Validation

```bash
# Validate migrations before applying
python scripts/manage_migrations.py validate
```

## Troubleshooting

### Problem: "Table already exists" Error

**Cause**: Database has tables but no migration tracking

**Solution**: System automatically handles this, but if manual intervention needed:

```bash
# Check database state
python scripts/manage_migrations.py detect-state

# If legacy, handle it
python scripts/manage_migrations.py handle-legacy
```

### Problem: Migration Fails Mid-Process

**Cause**: Database corruption or dependency issues

**Solution**:

```bash
# Check current state
python scripts/manage_migrations.py status

# Rollback to last known good state
python scripts/manage_migrations.py downgrade <previous_revision>

# Fix issues and retry
python scripts/manage_migrations.py upgrade
```

### Problem: Unsure of Database State

**Cause**: Complex migration history

**Solution**:

```bash
# Comprehensive status check
python scripts/manage_migrations.py status

# Database state detection
python scripts/manage_migrations.py detect-state

# Migration history
python scripts/manage_migrations.py history
```

## Best Practices

### For Development

1. **Always backup** before migrations
2. **Test migrations** on staging first
3. **Use CLI tools** for development
4. **Check status** before deploying

### For Production

1. **Monitor migration logs** during deployment
2. **Have rollback plan** ready
3. **Test on staging** with production data copy
4. **Use CLI tools** for migration management

### For Team Collaboration

1. **Never edit migration files** manually
2. **Create new migrations** for schema changes
3. **Coordinate migrations** across team
4. **Document migration reasoning**

## Environment Variables

```bash
# Database URL for migrations
DATABASE_URL=postgresql://user:password@host:port/database

# For Docker environments
POSTGRES_HOST=postgres
POSTGRES_DB=cats_monitor
POSTGRES_USER=db_user
POSTGRES_PASSWORD=db_password
POSTGRES_PORT=5432
```

## Migration Files

```
alembic/versions/
├── 001_initial_database_schema.py       # Basic tables
├── 002_add_cat_recognition_tables.py    # Recognition features
└── 003_add_enhanced_activity_features.py # Activity features
```

## Monitoring

### Log Messages to Watch

```bash
# Successful migration
✅ Database is ready and up to date

# Legacy handling
🔄 Legacy database detected, stamping current state...
✅ Legacy database stamped with revision: 001_initial_schema

# Migration running
🔄 Running migrations to revision: head
✅ Migrations completed successfully

# Errors
❌ Failed to run migrations
❌ Database connection failed
```

### Health Checks

```bash
# Check via CLI
python scripts/manage_migrations.py status

# Expected output for healthy system
📊 Database Migration Status
========================================
Database Connection: ✅ Connected
Current Revision: 003_activity_features
Head Revision: 003_activity_features
Needs Migration: ❌ No
Total Migrations: 3

✅ No pending migrations
```

This migration system ensures **zero-downtime** transitions from legacy databases to the new managed schema system! 🚀