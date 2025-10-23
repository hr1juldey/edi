# Utils: Logging

[Back to Index](./index.md)

## Purpose
Logging setup - Contains functions for setting up logging with specific levels and file handlers.

## Functions
- `setup_logger(name, level)`: Sets up a logger with the given name and level
- Writes to ~/.edi/logs/edi.log with rotating file handler (10MB max, 5 backups)

### Details
- Provides consistent logging across the application
- Uses rotating file handlers to manage log size
- Configurable log levels

## Technology Stack

- Python logging module
- Rotating file handlers

## See Docs

### Python Implementation Example
Logging utilities implementation for the EDI application:

```python
import logging
import os
from pathlib import Path
from logging.handlers import RotatingFileHandler
from typing import Optional, Dict, Any
import sys
from datetime import datetime
import json

class LoggingError(Exception):
    """Custom exception for logging errors."""
    pass

class EDILogger:
    """
    EDI Logger with rotating file handlers and structured logging.
    """
    
    def __init__(self, 
                 name: str = "edi", 
                 level: str = "INFO",
                 log_dir: str = "~/.edi/logs",
                 max_file_size: int = 10 * 1024 * 1024,  # 10MB
                 backup_count: int = 5):
        self.name = name
        self.level = level
        self.log_dir = Path(os.path.expanduser(log_dir))
        self.max_file_size = max_file_size
        self.backup_count = backup_count
        self.logger = None
        self._setup_logger()
    
    def _setup_logger(self):
        """
        Sets up a logger with the given name and level.
        Writes to ~/.edi/logs/edi.log with rotating file handler (10MB max, 5 backups).
        """
        # Create log directory if it doesn't exist
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Create logger
        self.logger = logging.getLogger(self.name)
        self.logger.setLevel(getattr(logging, self.level.upper(), logging.INFO))
        
        # Clear existing handlers to avoid duplicates
        self.logger.handlers.clear()
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # Create console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
        
        # Create rotating file handler
        log_file = self.log_dir / "edi.log"
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=self.max_file_size,
            backupCount=self.backup_count
        )
        file_handler.setLevel(getattr(logging, self.level.upper(), logging.INFO))
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
        
        # Prevent propagation to avoid duplicate logs
        self.logger.propagate = False
    
    def debug(self, message: str, **kwargs):
        """
        Log a debug message.
        """
        if kwargs:
            message += f" | {json.dumps(kwargs)}"
        self.logger.debug(message)
    
    def info(self, message: str, **kwargs):
        """
        Log an info message.
        """
        if kwargs:
            message += f" | {json.dumps(kwargs)}"
        self.logger.info(message)
    
    def warning(self, message: str, **kwargs):
        """
        Log a warning message.
        """
        if kwargs:
            message += f" | {json.dumps(kwargs)}"
        self.logger.warning(message)
    
    def error(self, message: str, **kwargs):
        """
        Log an error message.
        """
        if kwargs:
            message += f" | {json.dumps(kwargs)}"
        self.logger.error(message)
    
    def critical(self, message: str, **kwargs):
        """
        Log a critical message.
        """
        if kwargs:
            message += f" | {json.dumps(kwargs)}"
        self.logger.critical(message)
    
    def log_session_event(self, 
                         session_id: str, 
                         event_type: str, 
                         message: str, 
                         **kwargs):
        """
        Log a session-related event with structured data.
        """
        structured_message = f"[SESSION:{session_id[:8]}] {event_type}: {message}"
        if kwargs:
            structured_message += f" | {json.dumps(kwargs)}"
        self.logger.info(structured_message)
    
    def log_model_interaction(self, 
                             model_name: str, 
                             interaction_type: str, 
                             prompt: str, 
                             response: str,
                             duration: Optional[float] = None,
                             **kwargs):
        """
        Log a model interaction with timing and content details.
        """
        structured_message = f"[MODEL:{model_name}] {interaction_type}: {prompt[:50]}..."
        if duration:
            structured_message += f" | Duration: {duration:.2f}s"
        if kwargs:
            structured_message += f" | {json.dumps(kwargs)}"
        self.logger.info(structured_message)
    
    def log_image_operation(self, 
                           image_path: str, 
                           operation: str, 
                           **kwargs):
        """
        Log an image operation with file details.
        """
        structured_message = f"[IMAGE:{Path(image_path).name}] {operation}"
        if kwargs:
            structured_message += f" | {json.dumps(kwargs)}"
        self.logger.info(structured_message)
    
    def log_user_action(self, 
                       action: str, 
                       **kwargs):
        """
        Log a user action with details.
        """
        structured_message = f"[USER] {action}"
        if kwargs:
            structured_message += f" | {json.dumps(kwargs)}"
        self.logger.info(structured_message)
    
    def get_log_file_path(self) -> Path:
        """
        Get the path to the current log file.
        """
        return self.log_dir / "edi.log"
    
    def get_recent_logs(self, lines: int = 100) -> list:
        """
        Get recent log entries.
        """
        log_file = self.get_log_file_path()
        if not log_file.exists():
            return []
        
        try:
            with open(log_file, 'r') as f:
                log_lines = f.readlines()
                return log_lines[-lines:] if len(log_lines) > lines else log_lines
        except Exception as e:
            self.logger.error(f"Failed to read recent logs: {str(e)}")
            return []
    
    def clear_logs(self) -> bool:
        """
        Clear all log files.
        """
        try:
            log_file = self.get_log_file_path()
            if log_file.exists():
                log_file.unlink()
            
            # Remove backup files
            for i in range(1, self.backup_count + 1):
                backup_file = self.log_dir / f"edi.log.{i}"
                if backup_file.exists():
                    backup_file.unlink()
            
            self.logger.info("Log files cleared")
            return True
        except Exception as e:
            self.logger.error(f"Failed to clear logs: {str(e)}")
            return False

# Global logger instance
_edi_logger: Optional[EDILogger] = None

def setup_logger(name: str = "edi", level: str = "INFO") -> EDILogger:
    """
    Sets up a logger with the given name and level.
    Writes to ~/.edi/logs/edi.log with rotating file handler (10MB max, 5 backups).
    
    Args:
        name: Logger name
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        
    Returns:
        EDILogger instance
    """
    global _edi_logger
    
    if _edi_logger is None:
        _edi_logger = EDILogger(name, level)
    
    return _edi_logger

def get_logger() -> EDILogger:
    """
    Get the global EDI logger instance.
    """
    global _edi_logger
    if _edi_logger is None:
        _edi_logger = setup_logger()
    return _edi_logger

# Convenience functions
def debug(message: str, **kwargs):
    """Log a debug message."""
    get_logger().debug(message, **kwargs)

def info(message: str, **kwargs):
    """Log an info message."""
    get_logger().info(message, **kwargs)

def warning(message: str, **kwargs):
    """Log a warning message."""
    get_logger().warning(message, **kwargs)

def error(message: str, **kwargs):
    """Log an error message."""
    get_logger().error(message, **kwargs)

def critical(message: str, **kwargs):
    """Log a critical message."""
    get_logger().critical(message, **kwargs)

def log_session_event(session_id: str, event_type: str, message: str, **kwargs):
    """Log a session-related event."""
    get_logger().log_session_event(session_id, event_type, message, **kwargs)

def log_model_interaction(model_name: str, interaction_type: str, prompt: str, response: str,
                        duration: Optional[float] = None, **kwargs):
    """Log a model interaction."""
    get_logger().log_model_interaction(model_name, interaction_type, prompt, response, duration, **kwargs)

def log_image_operation(image_path: str, operation: str, **kwargs):
    """Log an image operation."""
    get_logger().log_image_operation(image_path, operation, **kwargs)

def log_user_action(action: str, **kwargs):
    """Log a user action."""
    get_logger().log_user_action(action, **kwargs)

# Example usage
if __name__ == "__main__":
    # Setup logger
    logger = setup_logger("edi_example", "DEBUG")
    
    print("Logger setup completed")
    print(f"Log file: {logger.get_log_file_path()}")
    
    # Log different types of messages
    debug("This is a debug message", extra_data={"key": "value"})
    info("This is an info message", timestamp=datetime.now().isoformat())
    warning("This is a warning message", warning_code="W001")
    error("This is an error message", error_code="E001", details="Something went wrong")
    critical("This is a critical message", critical_code="C001", emergency=True)
    
    # Log structured events
    session_id = "session-12345678-90ab-cdef-1234-567890abcdef"
    log_session_event(session_id, "START", "Session started", image_path="/path/to/image.jpg")
    log_session_event(session_id, "PROCESSING", "Image processing completed", duration=120.5)
    log_session_event(session_id, "END", "Session ended", result="success")
    
    # Log model interactions
    log_model_interaction(
        model_name="qwen3:8b",
        interaction_type="GENERATE",
        prompt="make the sky more dramatic",
        response="dramatic sky with storm clouds",
        duration=2.3,
        temperature=0.7
    )
    
    # Log image operations
    log_image_operation("/path/to/image.jpg", "RESIZE", width=1024, height=768)
    log_image_operation("/path/to/image.jpg", "SAVE", format="JPEG", quality=85)
    
    # Log user actions
    log_user_action("EDIT_START", user_id="user-123", image_path="/path/to/image.jpg")
    log_user_action("EDIT_COMPLETE", user_id="user-123", result="success", duration=150.2)
    
    # Get recent logs
    recent_logs = logger.get_recent_logs(10)
    print(f"\nRecent logs ({len(recent_logs)} lines):")
    for log_line in recent_logs[-5:]:  # Show last 5 lines
        print(f"  {log_line.strip()}")
    
    print("\nLogging example completed!")
```

### Advanced Logging Implementation
Enhanced logging with structured data and performance monitoring:

```python
import logging
import os
import json
from pathlib import Path
from logging.handlers import RotatingFileHandler
from typing import Optional, Dict, Any, List
import sys
from datetime import datetime, timedelta
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum

class LogLevel(Enum):
    """Enumeration for log levels."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"

@dataclass
class LogConfig:
    """Configuration for logging system."""
    name: str = "edi"
    level: str = "INFO"
    log_dir: str = "~/.edi/logs"
    max_file_size: int = 10 * 1024 * 1024  # 10MB
    backup_count: int = 5
    enable_console: bool = True
    enable_file: bool = True
    structured_logging: bool = True
    performance_monitoring: bool = True

class StructuredLogRecord:
    """
    Structured log record for enhanced logging.
    """
    
    def __init__(self, 
                 level: str,
                 message: str,
                 timestamp: Optional[datetime] = None,
                 logger_name: str = "edi",
                 context: Optional[Dict[str, Any]] = None,
                 metadata: Optional[Dict[str, Any]] = None):
        self.level = level
        self.message = message
        self.timestamp = timestamp or datetime.now()
        self.logger_name = logger_name
        self.context = context or {}
        self.metadata = metadata or {}
        self.thread_id = threading.get_ident()
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert structured log record to dictionary.
        
        Returns:
            Dictionary representation of the log record
        """
        return {
            "timestamp": self.timestamp.isoformat(),
            "level": self.level,
            "logger_name": self.logger_name,
            "message": self.message,
            "context": self.context,
            "metadata": self.metadata,
            "thread_id": self.thread_id
        }
    
    def to_json(self) -> str:
        """
        Convert structured log record to JSON string.
        
        Returns:
            JSON string representation of the log record
        """
        return json.dumps(self.to_dict(), indent=2, default=str)

class PerformanceMonitor:
    """
    Performance monitoring for logging operations.
    """
    
    def __init__(self):
        self.metrics = {}
        self.lock = threading.Lock()
    
    def record_operation(self, operation: str, duration: float):
        """
        Record a performance metric.
        
        Args:
            operation: Name of the operation
            duration: Duration in seconds
        """
        with self.lock:
            if operation not in self.metrics:
                self.metrics[operation] = {
                    "count": 0,
                    "total_duration": 0.0,
                    "min_duration": float('inf'),
                    "max_duration": 0.0
                }
            
            self.metrics[operation]["count"] += 1
            self.metrics[operation]["total_duration"] += duration
            self.metrics[operation]["min_duration"] = min(self.metrics[operation]["min_duration"], duration)
            self.metrics[operation]["max_duration"] = max(self.metrics[operation]["max_duration"], duration)
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get performance metrics.
        
        Returns:
            Dictionary with performance metrics
        """
        with self.lock:
            metrics = {}
            for operation, data in self.metrics.items():
                avg_duration = data["total_duration"] / data["count"] if data["count"] > 0 else 0
                metrics[operation] = {
                    "count": data["count"],
                    "total_duration": data["total_duration"],
                    "avg_duration": avg_duration,
                    "min_duration": data["min_duration"],
                    "max_duration": data["max_duration"]
                }
            return metrics
    
    def reset_metrics(self):
        """
        Reset performance metrics.
        """
        with self.lock:
            self.metrics.clear()

class AdvancedLogger:
    """
    Advanced logger with structured logging and performance monitoring.
    """
    
    def __init__(self, config: LogConfig = None):
        self.config = config or LogConfig()
        self.logger = logging.getLogger(self.config.name)
        self.performance_monitor = PerformanceMonitor() if self.config.performance_monitoring else None
        self._setup_advanced_logger()
    
    def _setup_advanced_logger(self):
        """
        Sets up advanced logger with structured logging and performance monitoring.
        Writes to ~/.edi/logs/edi.log with rotating file handler (10MB max, 5 backups).
        """
        # Create log directory if it doesn't exist
        log_dir = Path(os.path.expanduser(self.config.log_dir))
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # Set logger level
        self.logger.setLevel(getattr(logging, self.config.level.upper(), logging.INFO))
        
        # Clear existing handlers to avoid duplicates
        self.logger.handlers.clear()
        
        # Create structured formatter
        if self.config.structured_logging:
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s - %(context)s - %(metadata)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
        else:
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
        
        # Create console handler if enabled
        if self.config.enable_console:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(getattr(logging, self.config.level.upper(), logging.INFO))
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)
        
        # Create rotating file handler if enabled
        if self.config.enable_file:
            log_file = log_dir / "edi_advanced.log"
            file_handler = RotatingFileHandler(
                log_file,
                maxBytes=self.config.max_file_size,
                backupCount=self.config.backup_count
            )
            file_handler.setLevel(getattr(logging, self.config.level.upper(), logging.INFO))
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
        
        # Prevent propagation to avoid duplicate logs
        self.logger.propagate = False
    
    @contextmanager
    def performance_monitoring(self, operation: str):
        """
        Context manager for performance monitoring.
        
        Args:
            operation: Name of the operation to monitor
        """
        if not self.performance_monitor:
            yield
            return
        
        start_time = datetime.now()
        try:
            yield
        finally:
            duration = (datetime.now() - start_time).total_seconds()
            self.performance_monitor.record_operation(operation, duration)
    
    def log_structured(self, 
                      level: str, 
                      message: str, 
                      context: Optional[Dict[str, Any]] = None,
                      metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Log a structured message with context and metadata.
        
        Args:
            level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            message: Log message
            context: Context information
            metadata: Additional metadata
        """
        with self.performance_monitoring("log_structured"):
            # Create structured log record
            record = StructuredLogRecord(
                level=level,
                message=message,
                context=context or {},
                metadata=metadata or {}
            )
            
            # Log with appropriate level
            log_level = getattr(logging, level.upper(), logging.INFO)
            
            if self.config.structured_logging:
                # Add context and metadata to log record
                extra = {
                    "context": json.dumps(context or {}, default=str),
                    "metadata": json.dumps(metadata or {}, default=str)
                }
                self.logger.log(log_level, message, extra=extra)
            else:
                # Log as regular message
                self.logger.log(log_level, message)
    
    def debug_structured(self, 
                        message: str, 
                        context: Optional[Dict[str, Any]] = None,
                        metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Log a structured debug message.
        
        Args:
            message: Debug message
            context: Context information
            metadata: Additional metadata
        """
        self.log_structured("DEBUG", message, context, metadata)
    
    def info_structured(self, 
                       message: str, 
                       context: Optional[Dict[str, Any]] = None,
                       metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Log a structured info message.
        
        Args:
            message: Info message
            context: Context information
            metadata: Additional metadata
        """
        self.log_structured("INFO", message, context, metadata)
    
    def warning_structured(self, 
                          message: str, 
                          context: Optional[Dict[str, Any]] = None,
                          metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Log a structured warning message.
        
        Args:
            message: Warning message
            context: Context information
            metadata: Additional metadata
        """
        self.log_structured("WARNING", message, context, metadata)
    
    def error_structured(self, 
                        message: str, 
                        context: Optional[Dict[str, Any]] = None,
                        metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Log a structured error message.
        
        Args:
            message: Error message
            context: Context information
            metadata: Additional metadata
        """
        self.log_structured("ERROR", message, context, metadata)
    
    def critical_structured(self, 
                          message: str, 
                          context: Optional[Dict[str, Any]] = None,
                          metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Log a structured critical message.
        
        Args:
            message: Critical message
            context: Context information
            metadata: Additional metadata
        """
        self.log_structured("CRITICAL", message, context, metadata)
    
    def log_session_event_structured(self, 
                                   session_id: str, 
                                   event_type: str, 
                                   message: str, 
                                   context: Optional[Dict[str, Any]] = None,
                                   metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Log a structured session-related event.
        
        Args:
            session_id: Session ID
            event_type: Type of event
            message: Event message
            context: Context information
            metadata: Additional metadata
        """
        event_context = {
            "session_id": session_id[:8],  # Abbreviate for readability
            "event_type": event_type,
            **(context or {})
        }
        
        self.info_structured(f"[SESSION] {message}", event_context, metadata)
    
    def log_model_interaction_structured(self, 
                                       model_name: str, 
                                       interaction_type: str, 
                                       prompt: str, 
                                       response: str,
                                       duration: Optional[float] = None,
                                       context: Optional[Dict[str, Any]] = None,
                                       metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Log a structured model interaction with timing and content details.
        
        Args:
            model_name: Name of the model
            interaction_type: Type of interaction
            prompt: Input prompt
            response: Model response
            duration: Duration in seconds
            context: Context information
            metadata: Additional metadata
        """
        interaction_context = {
            "model_name": model_name,
            "interaction_type": interaction_type,
            "prompt_length": len(prompt),
            "response_length": len(response),
            **(context or {})
        }
        
        if duration is not None:
            interaction_context["duration"] = duration
        
        self.info_structured(f"[MODEL] {prompt[:50]}...", interaction_context, metadata)
    
    def log_image_operation_structured(self, 
                                     image_path: str, 
                                     operation: str, 
                                     context: Optional[Dict[str, Any]] = None,
                                     metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Log a structured image operation with file details.
        
        Args:
            image_path: Path to image file
            operation: Operation performed
            context: Context information
            metadata: Additional metadata
        """
        operation_context = {
            "image_path": Path(image_path).name,
            "operation": operation,
            **(context or {})
        }
        
        self.info_structured(f"[IMAGE] {operation}", operation_context, metadata)
    
    def log_user_action_structured(self, 
                                  action: str, 
                                  context: Optional[Dict[str, Any]] = None,
                                  metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Log a structured user action with details.
        
        Args:
            action: User action performed
            context: Context information
            metadata: Additional metadata
        """
        action_context = {
            "action": action,
            **(context or {})
        }
        
        self.info_structured(f"[USER] {action}", action_context, metadata)
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """
        Get performance metrics from the logger.
        
        Returns:
            Dictionary with performance metrics
        """
        if self.performance_monitor:
            return self.performance_monitor.get_metrics()
        return {}
    
    def reset_performance_metrics(self):
        """
        Reset performance metrics.
        """
        if self.performance_monitor:
            self.performance_monitor.reset_metrics()
    
    def get_log_statistics(self) -> Dict[str, Any]:
        """
        Get log statistics.
        
        Returns:
            Dictionary with log statistics
        """
        log_file = Path(os.path.expanduser(self.config.log_dir)) / "edi_advanced.log"
        
        if not log_file.exists():
            return {"error": "Log file does not exist"}
        
        try:
            # Count log lines by level
            level_counts = {
                "DEBUG": 0,
                "INFO": 0,
                "WARNING": 0,
                "ERROR": 0,
                "CRITICAL": 0
            }
            
            with open(log_file, 'r') as f:
                for line in f:
                    for level in level_counts:
                        if f" - {level} - " in line:
                            level_counts[level] += 1
                            break
            
            # Get file size
            file_size = log_file.stat().st_size
            
            # Get log file age
            file_age = datetime.now() - datetime.fromtimestamp(log_file.stat().st_mtime)
            
            return {
                "file_size_bytes": file_size,
                "file_size_mb": round(file_size / (1024 * 1024), 2),
                "file_age_days": file_age.days,
                "level_breakdown": level_counts,
                "total_entries": sum(level_counts.values()),
                "created_at": datetime.fromtimestamp(log_file.stat().st_ctime).isoformat()
            }
            
        except Exception as e:
            return {"error": f"Failed to get log statistics: {str(e)}"}
    
    def search_logs(self, 
                   search_term: str,
                   level_filter: Optional[str] = None,
                   date_from: Optional[datetime] = None,
                   date_to: Optional[datetime] = None,
                   limit: int = 100) -> List[Dict[str, Any]]:
        """
        Search logs with filtering options.
        
        Args:
            search_term: Term to search for in log messages
            level_filter: Filter by log level
            date_from: Filter logs from this date
            date_to: Filter logs to this date
            limit: Maximum number of results to return
            
        Returns:
            List of matching log entries
        """
        log_file = Path(os.path.expanduser(self.config.log_dir)) / "edi_advanced.log"
        
        if not log_file.exists():
            return []
        
        try:
            matching_entries = []
            
            with open(log_file, 'r') as f:
                for line in f:
                    # Apply search term filter
                    if search_term.lower() not in line.lower():
                        continue
                    
                    # Apply level filter
                    if level_filter and f" - {level_filter.upper()} - " not in line:
                        continue
                    
                    # Parse timestamp from line (format: YYYY-MM-DD HH:MM:SS)
                    try:
                        timestamp_str = line.split(" - ")[0]
                        timestamp = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S")
                        
                        # Apply date filters
                        if date_from and timestamp < date_from:
                            continue
                        
                        if date_to and timestamp > date_to:
                            continue
                    except ValueError:
                        # If timestamp parsing fails, skip date filtering for this line
                        pass
                    
                    # Parse log entry
                    try:
                        parts = line.split(" - ")
                        if len(parts) >= 4:
                            entry = {
                                "timestamp": parts[0],
                                "logger": parts[1],
                                "level": parts[2],
                                "message": parts[3],
                                "context": parts[4] if len(parts) > 4 else "",
                                "metadata": parts[5].strip() if len(parts) > 5 else ""
                            }
                            matching_entries.append(entry)
                        else:
                            # Simple format
                            entry = {
                                "timestamp": parts[0] if parts else "",
                                "logger": parts[1] if len(parts) > 1 else "",
                                "level": parts[2] if len(parts) > 2 else "",
                                "message": " - ".join(parts[3:]) if len(parts) > 3 else ""
                            }
                            matching_entries.append(entry)
                    except Exception:
                        # If parsing fails, include the raw line
                        matching_entries.append({"raw_line": line.strip()})
                    
                    # Apply limit
                    if len(matching_entries) >= limit:
                        break
            
            return matching_entries
            
        except Exception as e:
            self.logger.error(f"Failed to search logs: {str(e)}")
            return []

# Global advanced logger instance
_advanced_edi_logger: Optional[AdvancedLogger] = None

def setup_advanced_logger(config: LogConfig = None) -> AdvancedLogger:
    """
    Sets up an advanced logger with structured logging and performance monitoring.
    Writes to ~/.edi/logs/edi_advanced.log with rotating file handler (10MB max, 5 backups).
    
    Args:
        config: LogConfig object with logger configuration
        
    Returns:
        AdvancedLogger instance
    """
    global _advanced_edi_logger
    
    if _advanced_edi_logger is None:
        _advanced_edi_logger = AdvancedLogger(config)
    
    return _advanced_edi_logger

def get_advanced_logger() -> AdvancedLogger:
    """
    Get the global advanced EDI logger instance.
    """
    global _advanced_edi_logger
    if _advanced_edi_logger is None:
        _advanced_edi_logger = setup_advanced_logger()
    return _advanced_edi_logger

# Convenience functions for structured logging
def debug_structured(message: str, context: Optional[Dict[str, Any]] = None, metadata: Optional[Dict[str, Any]] = None):
    """Log a structured debug message."""
    get_advanced_logger().debug_structured(message, context, metadata)

def info_structured(message: str, context: Optional[Dict[str, Any]] = None, metadata: Optional[Dict[str, Any]] = None):
    """Log a structured info message."""
    get_advanced_logger().info_structured(message, context, metadata)

def warning_structured(message: str, context: Optional[Dict[str, Any]] = None, metadata: Optional[Dict[str, Any]] = None):
    """Log a structured warning message."""
    get_advanced_logger().warning_structured(message, context, metadata)

def error_structured(message: str, context: Optional[Dict[str, Any]] = None, metadata: Optional[Dict[str, Any]] = None):
    """Log a structured error message."""
    get_advanced_logger().error_structured(message, context, metadata)

def critical_structured(message: str, context: Optional[Dict[str, Any]] = None, metadata: Optional[Dict[str, Any]] = None):
    """Log a structured critical message."""
    get_advanced_logger().critical_structured(message, context, metadata)

def log_session_event_structured(session_id: str, event_type: str, message: str, 
                              context: Optional[Dict[str, Any]] = None,
                              metadata: Optional[Dict[str, Any]] = None):
    """Log a structured session-related event."""
    get_advanced_logger().log_session_event_structured(session_id, event_type, message, context, metadata)

def log_model_interaction_structured(model_name: str, interaction_type: str, prompt: str, response: str,
                                   duration: Optional[float] = None,
                                   context: Optional[Dict[str, Any]] = None,
                                   metadata: Optional[Dict[str, Any]] = None):
    """Log a structured model interaction."""
    get_advanced_logger().log_model_interaction_structured(model_name, interaction_type, prompt, response, 
                                                         duration, context, metadata)

def log_image_operation_structured(image_path: str, operation: str,
                                 context: Optional[Dict[str, Any]] = None,
                                 metadata: Optional[Dict[str, Any]] = None):
    """Log a structured image operation."""
    get_advanced_logger().log_image_operation_structured(image_path, operation, context, metadata)

def log_user_action_structured(action: str,
                             context: Optional[Dict[str, Any]] = None,
                             metadata: Optional[Dict[str, Any]] = None):
    """Log a structured user action."""
    get_advanced_logger().log_user_action_structured(action, context, metadata)

# Example usage
if __name__ == "__main__":
    # Setup advanced logger
    config = LogConfig(
        name="edi_advanced_example",
        level="DEBUG",
        log_dir="~/.edi/test_logs",
        max_file_size=5 * 1024 * 1024,  # 5MB
        backup_count=3,
        enable_console=True,
        enable_file=True,
        structured_logging=True,
        performance_monitoring=True
    )
    
    logger = setup_advanced_logger(config)
    
    print("Advanced Logger initialized")
    print(f"Log file: {Path(os.path.expanduser(config.log_dir)) / 'edi_advanced.log'}")
    
    # Log structured messages
    debug_structured("Debug message with context", 
                    context={"debug_key": "debug_value"},
                    metadata={"source": "test_module", "version": "1.0"})
    
    info_structured("Info message with context",
                   context={"info_key": "info_value"},
                   metadata={"source": "test_module", "version": "1.0"})
    
    warning_structured("Warning message with context",
                      context={"warning_key": "warning_value"},
                      metadata={"source": "test_module", "priority": "medium"})
    
    error_structured("Error message with context",
                    context={"error_key": "error_value"},
                    metadata={"source": "test_module", "severity": "high"})
    
    critical_structured("Critical message with context",
                       context={"critical_key": "critical_value"},
                       metadata={"source": "test_module", "emergency": True})
    
    # Log structured events
    session_id = "session-12345678-90ab-cdef-1234-567890abcdef"
    log_session_event_structured(
        session_id, 
        "START", 
        "Session started with image processing",
        context={"image_path": "/path/to/image.jpg", "prompt": "make sky dramatic"},
        metadata={"processing_steps": 3, "estimated_time": "2m 30s"}
    )
    
    log_session_event_structured(
        session_id,
        "PROCESSING",
        "Image processing completed successfully",
        context={"duration": 120.5, "quality_score": 0.85},
        metadata={"model_used": "qwen3:8b", "processing_steps": 3}
    )
    
    log_session_event_structured(
        session_id,
        "END",
        "Session ended with user acceptance",
        context={"result": "success", "user_rating": 5},
        metadata={"final_alignment_score": 0.85, "processing_time": 150.2}
    )
    
    # Log model interactions
    log_model_interaction_structured(
        model_name="qwen3:8b",
        interaction_type="GENERATE",
        prompt="make the sky more dramatic with storm clouds",
        response="dramatic sky with storm clouds and lightning",
        duration=2.3,
        context={"temperature": 0.7, "max_tokens": 512},
        metadata={"model_confidence": 0.88, "quality_score": 0.92}
    )
    
    # Log image operations
    log_image_operation_structured(
        "/path/to/image.jpg",
        "RESIZE",
        context={"width": 1024, "height": 768, "aspect_ratio": "4:3"},
        metadata={"processing_time": 0.5, "quality": "high"}
    )
    
    log_image_operation_structured(
        "/path/to/image.jpg",
        "SAVE",
        context={"format": "JPEG", "quality": 85},
        metadata={"file_size_kb": 1250, "compression_ratio": 0.7}
    )
    
    # Log user actions
    log_user_action_structured(
        "EDIT_START",
        context={"user_id": "user-123", "image_path": "/path/to/image.jpg"},
        metadata={"session_type": "interactive", "interface": "TUI"}
    )
    
    log_user_action_structured(
        "EDIT_ACCEPT",
        context={"user_id": "user-123", "session_id": session_id[:8]},
        metadata={"quality_rating": 5, "processing_time": 150.2}
    )
    
    # Get performance metrics
    performance_metrics = logger.get_performance_metrics()
    print(f"\nPerformance metrics: {performance_metrics}")
    
    # Get log statistics
    log_stats = logger.get_log_statistics()
    print(f"Log statistics: {log_stats}")
    
    # Search logs
    search_results = logger.search_logs("session", level_filter="INFO", limit=10)
    print(f"Search results for 'session': {len(search_results)} entries")
    for entry in search_results[:3]:  # Show first 3
        print(f"  {entry['timestamp']} - {entry['level']} - {entry['message'][:50]}...")
    
    # Reset performance metrics
    logger.reset_performance_metrics()
    print("Performance metrics reset")
    
    print("\nAdvanced logging example completed!")
```