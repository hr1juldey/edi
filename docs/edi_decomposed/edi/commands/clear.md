# Commands: Clear

[Back to Index](./index.md)

## Purpose

Data cleanup command - Contains the clear_command async function that deletes old session files, purges database records, with user confirmation required for full cleanup.

## Functions

- `async def clear_command(sessions=False, all=False)`: Cleans up data and session files

### Details

- Deletes old session files
- Purges database records
- User confirmation required for full cleanup (--all option)
- Safely removes EDI data

## Technology Stack

- AsyncIO for asynchronous operations
- File system operations
- Database operations

## See Docs

### AsyncIO Implementation Example

Async clear command for the EDI application:

```python
import asyncio
import aiofiles
import aiosqlite
import os
import shutil
from pathlib import Path
from typing import Dict, Any, List, Optional
import json
from datetime import datetime, timedelta
import tempfile

class EDICleanupManager:
    """Manages cleanup operations for the EDI application."""
    
    def __init__(self, edi_home: Optional[Path] = None):
        self.edi_home = edi_home or Path.home() / ".edi"
        self.cache_dir = self.edi_home / "cache"
        self.sessions_dir = self.edi_home / "sessions"
        self.logs_dir = self.edi_home / "logs"
        self.database_path = self.edi_home / "edi.db"
    
    async def clear_sessions(self, days_old: int = 7) -> Dict[str, Any]:
        """Clear session files older than specified number of days."""
        cutoff_date = datetime.now() - timedelta(days=days_old)
        deleted_count = 0
        total_size = 0
        
        if not self.sessions_dir.exists():
            return {"status": "success", "message": "Sessions directory does not exist", "deleted": 0, "size_freed_mb": 0}
        
        # List all session files
        session_files = list(self.sessions_dir.glob("*.json")) + list(self.sessions_dir.glob("*.session"))
        
        for file_path in session_files:
            # Get file modification time
            mod_time = datetime.fromtimestamp(file_path.stat().st_mtime)
            
            if mod_time < cutoff_date:
                try:
                    file_size = file_path.stat().st_size
                    file_path.unlink()
                    deleted_count += 1
                    total_size += file_size
                    print(f"Deleted session: {file_path}")
                except Exception as e:
                    print(f"Error deleting session {file_path}: {e}")
        
        size_freed_mb = total_size / (1024 * 1024)
        return {
            "status": "success",
            "message": f"Deleted {deleted_count} session files older than {days_old} days",
            "deleted": deleted_count,
            "size_freed_mb": round(size_freed_mb, 2)
        }
    
    async def clear_cache(self, days_old: int = 30) -> Dict[str, Any]:
        """Clear cache files older than specified number of days."""
        cutoff_date = datetime.now() - timedelta(days=days_old)
        deleted_count = 0
        total_size = 0
        
        if not self.cache_dir.exists():
            return {"status": "success", "message": "Cache directory does not exist", "deleted": 0, "size_freed_mb": 0}
        
        # Recursively find all cache files
        for file_path in self.cache_dir.rglob("*"):
            if file_path.is_file():
                mod_time = datetime.fromtimestamp(file_path.stat().st_mtime)
                
                if mod_time < cutoff_date:
                    try:
                        file_size = file_path.stat().st_size
                        file_path.unlink()
                        deleted_count += 1
                        total_size += file_size
                        print(f"Deleted cache: {file_path}")
                    except Exception as e:
                        print(f"Error deleting cache {file_path}: {e}")
        
        # Also remove empty directories
        for dir_path in sorted(self.cache_dir.rglob("*"), reverse=True):
            if dir_path.is_dir() and not any(dir_path.iterdir()):
                try:
                    dir_path.rmdir()
                    print(f"Removed empty directory: {dir_path}")
                except Exception as e:
                    print(f"Error removing directory {dir_path}: {e}")
        
        size_freed_mb = total_size / (1024 * 1024)
        return {
            "status": "success",
            "message": f"Deleted {deleted_count} cache files and empty directories",
            "deleted": deleted_count,
            "size_freed_mb": round(size_freed_mb, 2)
        }
    
    async def clear_logs(self, days_old: int = 14) -> Dict[str, Any]:
        """Clear log files older than specified number of days."""
        cutoff_date = datetime.now() - timedelta(days=days_old)
        deleted_count = 0
        total_size = 0
        
        if not self.logs_dir.exists():
            return {"status": "success", "message": "Logs directory does not exist", "deleted": 0, "size_freed_mb": 0}
        
        log_files = list(self.logs_dir.glob("*.log")) + list(self.logs_dir.glob("*.txt"))
        
        for file_path in log_files:
            mod_time = datetime.fromtimestamp(file_path.stat().st_mtime)
            
            if mod_time < cutoff_date:
                try:
                    file_size = file_path.stat().st_size
                    file_path.unlink()
                    deleted_count += 1
                    total_size += file_size
                    print(f"Deleted log: {file_path}")
                except Exception as e:
                    print(f"Error deleting log {file_path}: {e}")
        
        size_freed_mb = total_size / (1024 * 1024)
        return {
            "status": "success",
            "message": f"Deleted {deleted_count} log files older than {days_old} days",
            "deleted": deleted_count,
            "size_freed_mb": round(size_freed_mb, 2)
        }
    
    async def clear_database(self, confirm: bool = False) -> Dict[str, Any]:
        """Clear the database (only if confirmed)."""
        if not confirm:
            return {
                "status": "error",
                "message": "Database clear requires confirmation with --confirm flag"
            }
        
        if not self.database_path.exists():
            return {"status": "success", "message": "Database does not exist", "deleted": 0, "size_freed_mb": 0}
        
        try:
            # Get the database size before deletion
            file_size = self.database_path.stat().st_size
            
            # Remove the database file
            self.database_path.unlink()
            
            size_freed_mb = file_size / (1024 * 1024)
            return {
                "status": "success",
                "message": f"Database cleared, freed {round(size_freed_mb, 2)} MB",
                "deleted": 1,
                "size_freed_mb": round(size_freed_mb, 2)
            }
        except Exception as e:
            return {
                "status": "error",
                "message": f"Error clearing database: {str(e)}"
            }
    
    async def get_cleanup_summary(self) -> Dict[str, Any]:
        """Get a summary of what would be cleaned up without actually doing it."""
        summary = {}
        
        # Count session files
        session_count = 0
        session_size = 0
        if self.sessions_dir.exists():
            for file_path in list(self.sessions_dir.glob("*.json")) + list(self.sessions_dir.glob("*.session")):
                session_count += 1
                session_size += file_path.stat().st_size
        
        summary["sessions"] = {
            "count": session_count,
            "size_mb": round(session_size / (1024 * 1024), 2)
        }
        
        # Count cache files
        cache_count = 0
        cache_size = 0
        if self.cache_dir.exists():
            for file_path in self.cache_dir.rglob("*"):
                if file_path.is_file():
                    cache_count += 1
                    cache_size += file_path.stat().st_size
        
        summary["cache"] = {
            "count": cache_count,
            "size_mb": round(cache_size / (1024 * 1024), 2)
        }
        
        # Count log files
        log_count = 0
        log_size = 0
        if self.logs_dir.exists():
            log_files = list(self.logs_dir.glob("*.log")) + list(self.logs_dir.glob("*.txt"))
            for file_path in log_files:
                log_count += 1
                log_size += file_path.stat().st_size
        
        summary["logs"] = {
            "count": log_count,
            "size_mb": round(log_size / (1024 * 1024), 2)
        }
        
        # Database size
        if self.database_path.exists():
            db_size = self.database_path.stat().st_size
            summary["database"] = {
                "exists": True,
                "size_mb": round(db_size / (1024 * 1024), 2)
            }
        else:
            summary["database"] = {
                "exists": False,
                "size_mb": 0
            }
        
        return summary
    
    async def clear_all_data(self, confirm: bool = False) -> Dict[str, Any]:
        """Clear all EDI data (only if confirmed)."""
        if not confirm:
            return {
                "status": "error",
                "message": "Full data clear requires confirmation with --confirm flag"
            }
        
        print("WARNING: This will delete ALL EDI data including sessions, cache, logs, and database!")
        print("This cannot be undone. Please make sure you have backups if needed.")
        
        # Get total size before deletion
        summary = await self.get_cleanup_summary()
        total_size = (
            summary["sessions"]["size_mb"] + 
            summary["cache"]["size_mb"] + 
            summary["logs"]["size_mb"] + 
            summary["database"]["size_mb"]
        )
        
        print(f"Total data to be deleted: ~{round(total_size, 2)} MB")
        
        # Perform all cleanup operations
        results = await asyncio.gather(
            self.clear_sessions(),
            self.clear_cache(),
            self.clear_logs(),
            self.clear_database(confirm=True)
        )
        
        return {
            "status": "success",
            "message": "All EDI data cleared successfully",
            "details": results
        }

async def clear_command(sessions: bool = False, all: bool = False) -> Dict[str, Any]:
    """Cleans up data and session files."""
    print("EDI Clear Command")
    print("=" * 50)
    
    cleanup_manager = EDICleanupManager()
    
    # Get summary of data to be cleaned
    summary = await cleanup_manager.get_cleanup_summary()
    print("Current data summary:")
    print(f"  Sessions: {summary['sessions']['count']} files ({summary['sessions']['size_mb']} MB)")
    print(f"  Cache: {summary['cache']['count']} files ({summary['cache']['size_mb']} MB)")
    print(f"  Logs: {summary['logs']['count']} files ({summary['logs']['size_mb']} MB)")
    print(f"  Database: {'Exists' if summary['database']['exists'] else 'None'} ({summary['database']['size_mb']} MB)")
    
    if all:
        print("\nClearing ALL data...")
        result = await cleanup_manager.clear_all_data(confirm=True)
        print(f"Result: {result['message']}")
        
        # Print details of each operation
        for i, detail in enumerate(result.get('details', [])):
            operation_names = ["Sessions", "Cache", "Logs", "Database"]
            if i < len(operation_names):
                print(f"  {operation_names[i]}: {detail.get('message', 'Completed')}")
    elif sessions:
        print("\nClearing session files...")
        result = await cleanup_manager.clear_sessions()
        print(f"Result: {result['message']}")
    else:
        # Default behavior - clear cache and old logs
        print("\nClearing cache and old log files...")
        cache_result = await cleanup_manager.clear_cache()
        logs_result = await cleanup_manager.clear_logs()
        
        print(f"Cache result: {cache_result['message']}")
        print(f"Logs result: {logs_result['message']}")
    
    print("\nClear command completed.")
    return {"status": "success", "data_cleared": all or sessions}

# Example usage
if __name__ == "__main__":
    import argparse
    import sys
    
    parser = argparse.ArgumentParser(description='EDI Clear Command')
    parser.add_argument('--sessions', action='store_true', 
                       help='Clear session files')
    parser.add_argument('--all', action='store_true', 
                       help='Clear all data (requires confirmation)')
    parser.add_argument('--confirm', action='store_true',
                       help='Confirm destructive operations')
    
    args = parser.parse_args()
    
    # Run the clear command
    result = asyncio.run(
        clear_command(
            sessions=args.sessions,
            all=args.all
        )
    )
    
    print(f"Clear command result: {result}")
    sys.exit(0 if result.get("status") == "success" else 1)
```

### File System Operations Implementation Example

File system cleanup utilities for EDI:

```python
import asyncio
import aiofiles
from pathlib import Path
import shutil
import os
import tempfile
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import glob
import fnmatch

class FileSystemCleanup:
    """File system cleanup utilities for EDI."""
    
    @staticmethod
    async def safe_delete_file(file_path: Path, dry_run: bool = False) -> Dict[str, Any]:
        """Safely delete a file with verification."""
        result = {
            "path": str(file_path),
            "deleted": False,
            "size": 0,
            "error": None
        }
        
        if not file_path.exists():
            result["error"] = "File does not exist"
            return result
        
        try:
            # Verify it's actually a file
            if not file_path.is_file():
                result["error"] = "Path is not a file"
                return result
            
            # Get file size before deletion
            result["size"] = file_path.stat().st_size
            
            if not dry_run:
                # Perform the deletion
                file_path.unlink()
                result["deleted"] = True
            
            return result
        except PermissionError:
            result["error"] = "Permission denied"
            return result
        except Exception as e:
            result["error"] = str(e)
            return result
    
    @staticmethod
    async def safe_delete_directory(dir_path: Path, dry_run: bool = False) -> Dict[str, Any]:
        """Safely delete a directory and its contents."""
        result = {
            "path": str(dir_path),
            "deleted": False,
            "files_deleted": 0,
            "size_freed": 0,
            "error": None
        }
        
        if not dir_path.exists():
            result["error"] = "Directory does not exist"
            return result
        
        try:
            if not dir_path.is_dir():
                result["error"] = "Path is not a directory"
                return result
            
            # Count files and calculate size
            total_size = 0
            file_count = 0
            
            for item in dir_path.rglob("*"):
                if item.is_file():
                    total_size += item.stat().st_size
                    file_count += 1
            
            result["files_deleted"] = file_count
            result["size_freed"] = total_size
            
            if not dry_run:
                # Perform the deletion
                shutil.rmtree(dir_path)
                result["deleted"] = True
            
            return result
        except PermissionError:
            result["error"] = "Permission denied"
            return result
        except Exception as e:
            result["error"] = str(e)
            return result
    
    @staticmethod
    async def find_files_by_pattern(directory: Path, pattern: str) -> List[Path]:
        """Find files matching a given pattern."""
        if not directory.exists() or not directory.is_dir():
            return []
        
        # Use glob to find files matching the pattern
        matching_files = []
        for root, dirs, files in os.walk(directory):
            for file in files:
                if fnmatch.fnmatch(file, pattern):
                    matching_files.append(Path(root) / file)
        
        return matching_files
    
    @staticmethod
    async def find_files_older_than(directory: Path, days: int) -> List[Path]:
        """Find files older than specified number of days."""
        cutoff_date = datetime.now() - timedelta(days=days)
        old_files = []
        
        if not directory.exists() or not directory.is_dir():
            return old_files
        
        for file_path in directory.rglob("*"):
            if file_path.is_file():
                mod_time = datetime.fromtimestamp(file_path.stat().st_mtime)
                if mod_time < cutoff_date:
                    old_files.append(file_path)
        
        return old_files
    
    @staticmethod
    async def get_directory_size(directory: Path) -> int:
        """Get the total size of a directory in bytes."""
        total_size = 0
        
        if not directory.exists() or not directory.is_dir():
            return 0
        
        for file_path in directory.rglob("*"):
            if file_path.is_file():
                total_size += file_path.stat().st_size
        
        return total_size
    
    @staticmethod
    async def backup_directory(source: Path, backup_dir: Path, name: str = None) -> Dict[str, Any]:
        """Create a backup of a directory."""
        if not source.exists():
            return {"status": "error", "message": "Source directory does not exist"}
        
        if name is None:
            name = f"backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        backup_path = backup_dir / name
        
        try:
            # Ensure backup directory exists
            backup_dir.mkdir(parents=True, exist_ok=True)
            
            # Create backup
            shutil.copytree(source, backup_path)
            
            size = await FileSystemCleanup.get_directory_size(backup_path)
            
            return {
                "status": "success",
                "message": f"Backup created at {backup_path}",
                "backup_path": str(backup_path),
                "size_mb": round(size / (1024 * 1024), 2)
            }
        except Exception as e:
            return {
                "status": "error",
                "message": f"Failed to create backup: {str(e)}"
            }

class EDIDataCleaner:
    """High-level data cleaner for EDI."""
    
    def __init__(self, edi_home: Path):
        self.edi_home = edi_home
        self.fs_cleanup = FileSystemCleanup()
    
    async def cleanup_sessions(self, days_old: int = 7, dry_run: bool = False) -> Dict[str, Any]:
        """Clean up session files."""
        sessions_dir = self.edi_home / "sessions"
        
        if not sessions_dir.exists():
            return {"status": "success", "message": "Sessions directory does not exist", "deleted": 0, "size_freed_mb": 0}
        
        old_sessions = await self.fs_cleanup.find_files_older_than(sessions_dir, days_old)
        
        delete_tasks = []
        for session_file in old_sessions:
            delete_tasks.append(self.fs_cleanup.safe_delete_file(session_file, dry_run))
        
        results = await asyncio.gather(*delete_tasks)
        
        deleted_count = sum(1 for r in results if r["deleted"])
        size_freed = sum(r["size"] for r in results if r["deleted"])
        
        return {
            "status": "success",
            "message": f"Would delete {deleted_count} session files" if dry_run else f"Deleted {deleted_count} session files",
            "deleted": deleted_count,
            "size_freed_mb": round(size_freed / (1024 * 1024), 2),
            "dry_run": dry_run
        }
    
    async def cleanup_cache(self, days_old: int = 30, dry_run: bool = False) -> Dict[str, Any]:
        """Clean up cache files."""
        cache_dir = self.edi_home / "cache"
        
        if not cache_dir.exists():
            return {"status": "success", "message": "Cache directory does not exist", "deleted": 0, "size_freed_mb": 0}
        
        old_cache = await self.fs_cleanup.find_files_older_than(cache_dir, days_old)
        
        delete_tasks = []
        for cache_file in old_cache:
            delete_tasks.append(self.fs_cleanup.safe_delete_file(cache_file, dry_run))
        
        results = await asyncio.gather(*delete_tasks)
        
        deleted_count = sum(1 for r in results if r["deleted"])
        size_freed = sum(r["size"] for r in results if r["deleted"])
        
        # Remove empty directories after file deletion
        if not dry_run:
            for dir_path in sorted(cache_dir.rglob("*"), reverse=True):
                if dir_path.is_dir() and not any(dir_path.iterdir()):
                    try:
                        dir_path.rmdir()
                    except Exception:
                        pass  # Directory not empty or permission error
        
        return {
            "status": "success",
            "message": f"Would delete {deleted_count} cache files" if dry_run else f"Deleted {deleted_count} cache files",
            "deleted": deleted_count,
            "size_freed_mb": round(size_freed / (1024 * 1024), 2),
            "dry_run": dry_run
        }
    
    async def cleanup_logs(self, days_old: int = 14, dry_run: bool = False) -> Dict[str, Any]:
        """Clean up log files."""
        logs_dir = self.edi_home / "logs"
        
        if not logs_dir.exists():
            return {"status": "success", "message": "Logs directory does not exist", "deleted": 0, "size_freed_mb": 0}
        
        old_logs = await self.fs_cleanup.find_files_older_than(logs_dir, days_old)
        
        # Filter to only log files
        old_logs = [log for log in old_logs if log.suffix.lower() in ['.log', '.txt']]
        
        delete_tasks = []
        for log_file in old_logs:
            delete_tasks.append(self.fs_cleanup.safe_delete_file(log_file, dry_run))
        
        results = await asyncio.gather(*delete_tasks)
        
        deleted_count = sum(1 for r in results if r["deleted"])
        size_freed = sum(r["size"] for r in results if r["deleted"])
        
        return {
            "status": "success",
            "message": f"Would delete {deleted_count} log files" if dry_run else f"Deleted {deleted_count} log files",
            "deleted": deleted_count,
            "size_freed_mb": round(size_freed / (1024 * 1024), 2),
            "dry_run": dry_run
        }
    
    async def cleanup_temp(self, dry_run: bool = False) -> Dict[str, Any]:
        """Clean up temporary files."""
        temp_dir = self.edi_home / "temp"
        
        if not temp_dir.exists():
            return {"status": "success", "message": "Temp directory does not exist", "deleted": 0, "size_freed_mb": 0}
        
        # Clean all files in temp directory
        temp_files = [f for f in temp_dir.iterdir() if f.is_file()]
        
        delete_tasks = []
        for temp_file in temp_files:
            delete_tasks.append(self.fs_cleanup.safe_delete_file(temp_file, dry_run))
        
        results = await asyncio.gather(*delete_tasks)
        
        deleted_count = sum(1 for r in results if r["deleted"])
        size_freed = sum(r["size"] for r in results if r["deleted"])
        
        return {
            "status": "success",
            "message": f"Would delete {deleted_count} temp files" if dry_run else f"Deleted {deleted_count} temp files",
            "deleted": deleted_count,
            "size_freed_mb": round(size_freed / (1024 * 1024), 2),
            "dry_run": dry_run
        }

# Example usage
async def main():
    edi_home = Path.home() / ".edi-test-cleanup"
    
    # Create test directories and files
    edi_home.mkdir(exist_ok=True)
    (edi_home / "sessions").mkdir(exist_ok=True)
    (edi_home / "cache").mkdir(exist_ok=True)
    (edi_home / "logs").mkdir(exist_ok=True)
    (edi_home / "temp").mkdir(exist_ok=True)
    
    # Create some test files
    for i in range(5):
        # Create session files
        session_file = edi_home / "sessions" / f"session_{i}.json"
        session_file.write_text(f'{{"session_id": "{i}", "data": "test"}}')
        
        # Create cache files
        cache_file = edi_home / "cache" / f"cache_{i}.dat"
        cache_file.write_text(f"Cached data {i}")
        
        # Create log files
        log_file = edi_home / "logs" / f"log_{i}.log"
        log_file.write_text(f"Log entry {i} at {datetime.now()}")
        
        # Create temp files
        temp_file = edi_home / "temp" / f"temp_{i}.tmp"
        temp_file.write_text(f"Temp data {i}")
    
    print("Created test files for cleanup demonstration")
    
    # Initialize the data cleaner
    cleaner = EDIDataCleaner(edi_home)
    
    # Dry run cleanup
    print("\nDry run for sessions cleanup:")
    result = await cleaner.cleanup_sessions(days_old=0, dry_run=True)
    print(f"  {result['message']} - Size: {result['size_freed_mb']} MB")
    
    # Actual cleanup
    print("\nActual cleanup for sessions:")
    result = await cleaner.cleanup_sessions(days_old=0, dry_run=False)
    print(f"  {result['message']} - Size: {result['size_freed_mb']} MB")
    
    # Check cache cleanup
    result = await cleaner.cleanup_cache(days_old=0, dry_run=True)
    print(f"\nCache cleanup dry run: {result['message']} - Size: {result['size_freed_mb']} MB")
    
    # Check logs cleanup
    result = await cleaner.cleanup_logs(days_old=0, dry_run=True)
    print(f"\nLogs cleanup dry run: {result['message']} - Size: {result['size_freed_mb']} MB")
    
    # Check temp cleanup
    result = await cleaner.cleanup_temp(dry_run=True)
    print(f"\nTemp cleanup dry run: {result['message']} - Size: {result['size_freed_mb']} MB")
    
    print("\nFile system cleanup utilities example completed!")

if __name__ == "__main__":
    asyncio.run(main())
```

### Database Operations Implementation Example

Database cleanup for EDI:

```python
import asyncio
import aiosqlite
from pathlib import Path
from typing import Dict, Any, List, Optional
import json
from datetime import datetime, timedelta

class EDIDatabaseCleaner:
    """Database cleanup operations for EDI."""
    
    def __init__(self, db_path: Path):
        self.db_path = db_path
    
    async def connect(self):
        """Create a database connection."""
        return await aiosqlite.connect(self.db_path)
    
    async def get_table_info(self) -> List[Dict[str, Any]]:
        """Get information about all tables in the database."""
        async with self.connect() as db:
            cursor = await db.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = await cursor.fetchall()
            
            table_info = []
            for table in tables:
                table_name = table[0]
                
                # Get row count
                count_cursor = await db.execute(f"SELECT COUNT(*) FROM {table_name}")
                count = await count_cursor.fetchone()
                
                # Get column info
                col_cursor = await db.execute(f"PRAGMA table_info({table_name})")
                columns = await col_cursor.fetchall()
                
                table_info.append({
                    "name": table_name,
                    "rows": count[0] if count else 0,
                    "columns": len(columns),
                    "column_names": [col[1] for col in columns]
                })
            
            return table_info
    
    async def count_old_records(self, table: str, date_column: str, days_old: int) -> int:
        """Count records older than specified number of days."""
        cutoff_date = datetime.now() - timedelta(days=days_old)
        
        async with self.connect() as db:
            if date_column:
                cursor = await db.execute(
                    f"SELECT COUNT(*) FROM {table} WHERE {date_column} < ?",
                    (cutoff_date.isoformat(),)
                )
            else:
                # If no date column specified, return total count
                cursor = await db.execute(f"SELECT COUNT(*) FROM {table}")
            
            result = await cursor.fetchone()
            return result[0] if result else 0
    
    async def delete_old_records(self, table: str, date_column: str, days_old: int) -> Dict[str, Any]:
        """Delete records older than specified number of days."""
        cutoff_date = datetime.now() - timedelta(days=days_old)
        
        try:
            async with self.connect() as db:
                if date_column:
                    # Count records to be deleted first
                    count_cursor = await db.execute(
                        f"SELECT COUNT(*) FROM {table} WHERE {date_column} < ?",
                        (cutoff_date.isoformat(),)
                    )
                    count_result = await count_cursor.fetchone()
                    records_to_delete = count_result[0] if count_result else 0
                    
                    # Delete old records
                    await db.execute(
                        f"DELETE FROM {table} WHERE {date_column} < ?",
                        (cutoff_date.isoformat(),)
                    )
                else:
                    # Delete all records if no date column specified
                    count_cursor = await db.execute(f"SELECT COUNT(*) FROM {table}")
                    count_result = await count_cursor.fetchone()
                    records_to_delete = count_result[0] if count_result else 0
                    
                    await db.execute(f"DELETE FROM {table}")
                
                await db.commit()
                
                return {
                    "status": "success",
                    "deleted": records_to_delete,
                    "message": f"Deleted {records_to_delete} records from {table}"
                }
        except Exception as e:
            return {
                "status": "error",
                "message": f"Error deleting records from {table}: {str(e)}"
            }
    
    async def get_database_size(self) -> int:
        """Get the size of the database file in bytes."""
        if self.db_path.exists():
            return self.db_path.stat().st_size
        return 0
    
    async def vacuum_database(self) -> Dict[str, Any]:
        """Optimize the database by vacuuming."""
        try:
            async with self.connect() as db:
                await db.execute("VACUUM")
                await db.commit()
                
                return {
                    "status": "success",
                    "message": "Database vacuumed successfully"
                }
        except Exception as e:
            return {
                "status": "error",
                "message": f"Error vacuuming database: {str(e)}"
            }
    
    async def backup_database(self, backup_path: Path) -> Dict[str, Any]:
        """Create a backup of the database."""
        try:
            async with self.connect() as db:
                # Use SQLite's built-in backup command
                backup_db = await aiosqlite.connect(backup_path)
                await db.backup(backup_db)
                await backup_db.close()
                
                return {
                    "status": "success",
                    "message": f"Database backed up to {backup_path}",
                    "backup_path": str(backup_path)
                }
        except Exception as e:
            return {
                "status": "error",
                "message": f"Error backing up database: {str(e)}"
            }
    
    async def get_cleanup_summary(self) -> Dict[str, Any]:
        """Get a summary of what can be cleaned up in the database."""
        table_info = await self.get_table_info()
        
        summary = {}
        for table in table_info:
            # For EDI, assume sessions and logs are common tables with date columns
            date_column = None
            if 'session' in table['name'].lower():
                date_column = 'created_at' if 'created_at' in table['column_names'] else 'timestamp'
            elif 'log' in table['name'].lower():
                date_column = 'timestamp' if 'timestamp' in table['column_names'] else 'date'
            elif 'created_at' in table['column_names']:
                date_column = 'created_at'
            elif 'timestamp' in table['column_names']:
                date_column = 'timestamp'
            
            if date_column:
                old_count = await self.count_old_records(table['name'], date_column, 7)  # Count records older than 7 days
                summary[table['name']] = {
                    "total_rows": table['rows'],
                    "old_rows": old_count,
                    "date_column": date_column
                }
        
        return summary

class EDIDatabaseManager:
    """Manages the entire database for EDI."""
    
    def __init__(self, edi_home: Path):
        self.db_path = edi_home / "edi.db"
        self.db_cleaner = EDIDatabaseCleaner(self.db_path)
    
    async def create_tables(self):
        """Create necessary tables if they don't exist."""
        async with self.db_cleaner.connect() as db:
            # Sessions table
            await db.execute("""
                CREATE TABLE IF NOT EXISTS sessions (
                    id TEXT PRIMARY KEY,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    image_path TEXT NOT NULL,
                    naive_prompt TEXT NOT NULL,
                    status TEXT CHECK(status IN ('in_progress', 'completed', 'failed')),
                    final_alignment_score REAL
                )
            """)
            
            # Prompts table
            await db.execute("""
                CREATE TABLE IF NOT EXISTS prompts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT REFERENCES sessions(id),
                    iteration INT,
                    positive_prompt TEXT,
                    negative_prompt TEXT,
                    quality_score REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Entities table
            await db.execute("""
                CREATE TABLE IF NOT EXISTS entities (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT REFERENCES sessions(id),
                    entity_id TEXT,
                    label TEXT,
                    confidence REAL,
                    bbox_json TEXT,
                    mask_path TEXT,
                    color_hex TEXT,
                    area_percent REAL
                )
            """)
            
            # Validations table
            await db.execute("""
                CREATE TABLE IF NOT EXISTS validations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT REFERENCES sessions(id),
                    attempt_number INT,
                    alignment_score REAL,
                    preserved_count INT,
                    modified_count INT,
                    unintended_count INT,
                    user_feedback TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # User feedback table
            await db.execute("""
                CREATE TABLE IF NOT EXISTS user_feedback (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT REFERENCES sessions(id),
                    feedback_type TEXT CHECK(feedback_type IN ('accept', 'reject', 'partial')),
                    comments TEXT,
                    rating INT CHECK(rating BETWEEN 1 AND 5),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            await db.commit()
    
    async def cleanup_old_sessions(self, days_old: int = 7) -> Dict[str, Any]:
        """Clean up old session records."""
        # Clean up sessions table
        result = await self.db_cleaner.delete_old_records("sessions", "created_at", days_old)
        
        if result["status"] == "success":
            # Also clean up related records in other tables
            await self.db_cleaner.delete_old_records("prompts", "created_at", days_old)
            await self.db_cleaner.delete_old_records("validations", "created_at", days_old)
            await self.db_cleaner.delete_old_records("user_feedback", "created_at", days_old)
            
            # Note: entities are cleaned up by CASCADE when sessions are deleted
            # (would require foreign key constraints to be set up properly)
        
        return result
    
    async def cleanup_old_logs(self, days_old: int = 14) -> Dict[str, Any]:
        """Clean up old log records."""
        # Assuming there's a logs table (in a real implementation, this would be created)
        # For this example, we'll assume a logs table exists
        result = await self.db_cleaner.delete_old_records("validations", "created_at", days_old)
        return result
    
    async def get_database_stats(self) -> Dict[str, Any]:
        """Get statistics about the database."""
        table_info = await self.db_cleaner.get_table_info()
        db_size = await self.db_cleaner.get_database_size()
        cleanup_summary = await self.db_cleaner.get_cleanup_summary()
        
        return {
            "table_info": table_info,
            "database_size_mb": round(db_size / (1024 * 1024), 2),
            "cleanup_summary": cleanup_summary
        }
    
    async def reset_database(self, confirm: bool = False) -> Dict[str, Any]:
        """Reset the entire database (only if confirmed)."""
        if not confirm:
            return {
                "status": "error",
                "message": "Database reset requires confirmation"
            }
        
        try:
            # Close and delete the database file
            if self.db_path.exists():
                self.db_path.unlink()
            
            # Recreate tables
            await self.create_tables()
            
            return {
                "status": "success",
                "message": "Database reset successfully"
            }
        except Exception as e:
            return {
                "status": "error",
                "message": f"Error resetting database: {str(e)}"
            }

# Example usage
async def main():
    edi_home = Path.home() / ".edi-test-db"
    edi_home.mkdir(exist_ok=True)
    
    # Initialize the database manager
    db_manager = EDIDatabaseManager(edi_home)
    
    # Create tables
    await db_manager.create_tables()
    print("Created/verified database tables")
    
    # Insert some test data
    async with db_manager.db_cleaner.connect() as db:
        # Insert a test session
        await db.execute("""
            INSERT INTO sessions (id, image_path, naive_prompt, status, final_alignment_score)
            VALUES (?, ?, ?, ?, ?)
        """, ("test-session-1", "test.jpg", "make sky blue", "completed", 0.85))
        
        await db.execute("""
            INSERT INTO prompts (session_id, iteration, positive_prompt, negative_prompt, quality_score)
            VALUES (?, ?, ?, ?, ?)
        """, ("test-session-1", 0, "blue sky", "avoid clouds", 0.9))
        
        await db.commit()
    
    print("Inserted test data")
    
    # Get database stats
    stats = await db_manager.get_database_stats()
    print(f"Database stats: {stats['database_size_mb']} MB, {len(stats['table_info'])} tables")
    
    # Get cleanup summary
    summary = await db_manager.db_cleaner.get_cleanup_summary()
    print(f"Cleanup summary: {summary}")
    
    # Clean up old sessions (none should be old since they're recent)
    cleanup_result = await db_manager.cleanup_old_sessions(days_old=1)  # Only 1 day old
    print(f"Cleanup result: {cleanup_result}")
    
    # Vacuum the database
    vacuum_result = await db_manager.db_cleaner.vacuum_database()
    print(f"Vacuum result: {vacuum_result}")
    
    print("Database operations example completed!")

if __name__ == "__main__":
    asyncio.run(main())
```
