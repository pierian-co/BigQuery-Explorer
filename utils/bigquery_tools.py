from google.cloud import bigquery
from typing import List, Dict, Any, Optional, Callable
from datetime import datetime, timedelta
import pandas as pd
from google.cloud import resourcemanager_v3
from google.cloud.resourcemanager_v3 import ProjectsClient
import json
import os
from pathlib import Path
import time

class OperationStatus:
    """Class to track operation status and progress"""
    def __init__(self, operation_id: str, operation_type: str):
        self.operation_id = operation_id
        self.operation_type = operation_type
        self.start_time = datetime.now()
        self.end_time = None
        self.status = "running"  # running, completed, failed, interrupted
        self.progress = 0.0
        self.total_items = 0
        self.processed_items = 0
        self.current_item = None
        self.error = None
        self.checkpoint = None
        
    def to_dict(self) -> Dict[str, Any]:
        return {
            "operation_id": self.operation_id,
            "operation_type": self.operation_type,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "status": self.status,
            "progress": self.progress,
            "total_items": self.total_items,
            "processed_items": self.processed_items,
            "current_item": self.current_item,
            "error": str(self.error) if self.error else None,
            "checkpoint": self.checkpoint
        }

class BigQueryUtils:
    def __init__(self, project_id: str, state_dir: Optional[str] = None):
        self.client = bigquery.Client(project=project_id)
        self.project_id = project_id
        self.state_dir = state_dir or os.path.join(os.path.expanduser("~"), ".bqutils")
        Path(self.state_dir).mkdir(parents=True, exist_ok=True)
        
    def _get_operation_path(self, operation_id: str) -> str:
        """Get path to operation status file"""
        return os.path.join(self.state_dir, f"{operation_id}.json")
        
    def _save_operation_status(self, status: OperationStatus):
        """Save operation status to disk"""
        path = self._get_operation_path(status.operation_id)
        with open(path, 'w') as f:
            json.dump(status.to_dict(), f)
            
    def _load_operation_status(self, operation_id: str) -> Optional[OperationStatus]:
        """Load operation status from disk"""
        path = self._get_operation_path(operation_id)
        if os.path.exists(path):
            with open(path, 'r') as f:
                data = json.load(f)
                status = OperationStatus(data["operation_id"], data["operation_type"])
                status.__dict__.update(data)
                return status
        return None

    def _execute_with_retry(self, 
                          func: Callable, 
                          max_retries: int = 3, 
                          initial_delay: float = 1.0,
                          max_delay: float = 32.0,
                          timeout: Optional[float] = None) -> Any:
        """Execute a function with exponential backoff retry"""
        delay = initial_delay
        last_exception = None
        
        start_time = time.time()
        for attempt in range(max_retries):
            try:
                if timeout and time.time() - start_time > timeout:
                    raise TimeoutError("Operation timed out")
                    
                return func()
                
            except Exception as e:
                last_exception = e
                if attempt < max_retries - 1:
                    time.sleep(min(delay, max_delay))
                    delay *= 2
                    
        raise last_exception

    def execute_query(self, query: str) -> pd.DataFrame:
        """
        Execute a BigQuery SQL query and return results as a pandas DataFrame
        """
        try:
            return self.client.query(query).to_dataframe()
        except Exception as e:
            print(f"Query execution failed: {str(e)}")
            raise

    def get_table_schema(self, dataset_id: str, table_id: str) -> List[Dict[str, Any]]:
        """
        Get the schema of a specific table
        """
        table_ref = f"{self.project_id}.{dataset_id}.{table_id}"
        table = self.client.get_table(table_ref)
        return [{"name": field.name, "type": field.field_type} for field in table.schema]

    def list_datasets(self) -> List[str]:
        """
        List all datasets in the project
        """
        return [dataset.dataset_id for dataset in self.client.list_datasets()]

    def list_tables(self, dataset_id: str) -> List[str]:
        """
        List all tables in a specific dataset
        """
        dataset_ref = self.client.dataset(dataset_id)
        return [table.table_id for table in self.client.list_tables(dataset_ref)]

    def get_table_size(self, dataset_id: str, table_id: str) -> Dict[str, float]:
        """
        Get the size metrics of a specific table
        """
        table_ref = f"{self.project_id}.{dataset_id}.{table_id}"
        table = self.client.get_table(table_ref)
        return {
            "rows": table.num_rows,
            "size_mb": table.num_bytes / (1024 * 1024),
            "last_modified": table.modified.isoformat()
        } 

    def estimate_query_cost(self, query: str) -> Dict[str, Any]:
        """
        Estimate the cost and bytes processed for a query
        """
        job_config = bigquery.QueryJobConfig(dry_run=True)
        query_job = self.client.query(query, job_config=job_config)
        bytes_processed = query_job.total_bytes_processed
        estimated_cost_usd = (bytes_processed / 1_000_000_000) * 5  # $5 per TB processed
        
        return {
            "bytes_processed": bytes_processed,
            "estimated_cost_usd": estimated_cost_usd,
            "estimated_gb": bytes_processed / 1_000_000_000
        }

    def compare_schemas(self, dataset_id1: str, table_id1: str, 
                       dataset_id2: str, table_id2: str) -> Dict[str, List]:
        """
        Compare schemas of two tables and return differences
        """
        schema1 = self.get_table_schema(dataset_id1, table_id1)
        schema2 = self.get_table_schema(dataset_id2, table_id2)
        
        schema1_dict = {field['name']: field['type'] for field in schema1}
        schema2_dict = {field['name']: field['type'] for field in schema2}
        
        only_in_1 = [(name, type_) for name, type_ in schema1_dict.items() 
                     if name not in schema2_dict]
        only_in_2 = [(name, type_) for name, type_ in schema2_dict.items() 
                     if name not in schema1_dict]
        type_mismatch = [(name, schema1_dict[name], schema2_dict[name]) 
                        for name in set(schema1_dict) & set(schema2_dict)
                        if schema1_dict[name] != schema2_dict[name]]
        
        return {
            "only_in_first": only_in_1,
            "only_in_second": only_in_2,
            "type_mismatches": type_mismatch
        }

    def get_table_preview(self, dataset_id: str, table_id: str, 
                         limit: int = 5) -> pd.DataFrame:
        """
        Get a preview of table data with specified limit
        """
        query = f"""
        SELECT *
        FROM `{self.project_id}.{dataset_id}.{table_id}`
        LIMIT {limit}
        """
        return self.execute_query(query)

    def get_table_column_stats(self, dataset_id: str, table_id: str, 
                             column_name: str) -> Dict[str, Any]:
        """
        Get basic statistics for a specific column
        """
        query = f"""
        SELECT
            COUNT(*) as total_rows,
            COUNT(DISTINCT {column_name}) as unique_values,
            COUNT({column_name}) as non_null_count,
            CAST(COUNT(*) - COUNT({column_name}) AS INT64) as null_count
        FROM `{self.project_id}.{dataset_id}.{table_id}`
        """
        stats = self.execute_query(query).iloc[0].to_dict()
        
        # Calculate null percentage
        stats['null_percentage'] = (stats['null_count'] / stats['total_rows']) * 100
        
        return stats

    def get_table_partitioning_info(self, dataset_id: str, table_id: str) -> Dict[str, Any]:
        """
        Get partitioning information for a table
        """
        table_ref = f"{self.project_id}.{dataset_id}.{table_id}"
        table = self.client.get_table(table_ref)
        
        return {
            "is_partitioned": table.time_partitioning is not None,
            "partition_type": table.time_partitioning.type_ if table.time_partitioning else None,
            "partition_field": table.time_partitioning.field if table.time_partitioning else None,
            "expiration_ms": table.time_partitioning.expiration_ms if table.time_partitioning else None
        }

    def create_materialized_view(self, source_query: str, target_dataset: str, view_name: str, 
                               enable_refresh: bool = True, refresh_interval_minutes: int = 60) -> Dict[str, Any]:
        """
        Create a materialized view from a query
        """
        try:
            # Create the view using DDL since the MaterializedView class might not be available in all versions
            create_view_query = f"""
            CREATE MATERIALIZED VIEW `{self.project_id}.{target_dataset}.{view_name}`
            OPTIONS(
                enable_refresh = {str(enable_refresh).lower()},
                refresh_interval_minutes = {max(60, refresh_interval_minutes)}
            )
            AS {source_query}
            """
            
            self.execute_query(create_view_query)
            
            # Get the created view details
            view_details = self.get_materialized_view_details(target_dataset, view_name)
            return {
                "status": "success",
                "view_id": view_name,
                "dataset": target_dataset,
                "refresh_enabled": enable_refresh,
                "refresh_interval": refresh_interval_minutes,
                "last_refresh_time": view_details.get("last_refresh_time")
            }
        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }

    def list_materialized_views(self, dataset_id: str) -> List[Dict[str, Any]]:
        """
        List all materialized views in a dataset with their details
        """
        dataset_ref = self.client.dataset(dataset_id)
        views = []
        
        for table in self.client.list_tables(dataset_ref):
            table_detail = self.client.get_table(table)
            if hasattr(table_detail, 'table_type') and table_detail.table_type == "MATERIALIZED_VIEW":
                views.append({
                    "view_id": table.table_id,
                    "creation_time": table_detail.created.isoformat() if table_detail.created else None,
                    "last_modified": table_detail.modified.isoformat() if table_detail.modified else None,
                    "size_mb": table_detail.num_bytes / (1024 * 1024) if table_detail.num_bytes else 0,
                    "rows": table_detail.num_rows if table_detail.num_rows else 0
                })
        
        return views

    def get_materialized_view_details(self, dataset_id: str, view_id: str) -> Dict[str, Any]:
        """
        Get detailed information about a materialized view
        """
        view_ref = f"{self.project_id}.{dataset_id}.{view_id}"
        view = self.client.get_table(view_ref)
        
        if not hasattr(view, 'table_type') or view.table_type != "MATERIALIZED_VIEW":
            raise ValueError(f"{view_id} is not a materialized view")
        
        return {
            "view_id": view_id,
            "query": getattr(view, 'mview_query', None),
            "enable_refresh": getattr(view, 'mview_enable_refresh', None),
            "refresh_interval_minutes": getattr(view, 'refresh_interval_minutes', None),
            "last_refresh_time": view.last_refresh_time.isoformat() if hasattr(view, 'last_refresh_time') and view.last_refresh_time else None,
            "creation_time": view.created.isoformat() if view.created else None,
            "last_modified": view.modified.isoformat() if view.modified else None,
            "size_mb": view.num_bytes / (1024 * 1024) if view.num_bytes else 0,
            "rows": view.num_rows if view.num_rows else 0
        }

    def refresh_materialized_view(self, dataset_id: str, view_id: str) -> Dict[str, Any]:
        """
        Manually refresh a materialized view
        """
        try:
            query = f"""
            CALL BQ.REFRESH_MATERIALIZED_VIEW(
                '{self.project_id}.{dataset_id}.{view_id}'
            )
            """
            self.execute_query(query)
            
            # Get updated details
            view_details = self.get_materialized_view_details(dataset_id, view_id)
            return {
                "status": "success",
                "last_refresh_time": view_details["last_refresh_time"]
            }
        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }

    def get_slow_queries(self, days_back: int = 7, limit: int = 100, min_joins: int = 2) -> pd.DataFrame:
        """
        Get the slowest queries with multiple table joins
        
        Args:
            days_back: Number of days to look back
            limit: Maximum number of queries to return
            min_joins: Minimum number of JOIN operations to filter for
        """
        query = f"""
        WITH QueryStats AS (
            SELECT
                job_id,
                creation_time,
                user_email,
                total_bytes_processed,
                total_slot_ms,
                query,
                state,
                error_result,
                cache_hit,
                total_bytes_billed,
                TIMESTAMP_DIFF(end_time, start_time, MILLISECOND) as duration_ms,
                destination_table,
                -- Count number of JOIN keywords in the query
                (LENGTH(UPPER(query)) - LENGTH(REPLACE(UPPER(query), 'JOIN', ''))) / 4 as join_count
            FROM `{self.project_id}.region-us.INFORMATION_SCHEMA.JOBS`
            WHERE creation_time >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL {days_back} DAY)
                AND job_type = 'QUERY'
                AND state = 'DONE'
                -- Filter for queries containing JOIN
                AND UPPER(query) LIKE '%JOIN%'
        )
        SELECT
            job_id,
            creation_time,
            user_email,
            ROUND(total_bytes_processed / POW(1024, 3), 2) as gb_processed,
            ROUND(total_slot_ms / 1000, 2) as slot_seconds,
            ROUND(duration_ms / 1000, 2) as duration_seconds,
            ROUND(total_bytes_billed / POW(1024, 3), 2) as gb_billed,
            cache_hit,
            join_count as number_of_joins,
            -- Extract table names from query (simplified version)
            ARRAY(
                SELECT DISTINCT table_name 
                FROM UNNEST(REGEXP_EXTRACT_ALL(query, r'`?\[?([a-zA-Z0-9-_]+\.[a-zA-Z0-9-_]+\.[a-zA-Z0-9-_]+)\]?`?')) as table_name
            ) as tables_involved,
            query
        FROM QueryStats
        WHERE join_count >= {min_joins}
        ORDER BY duration_ms DESC
        LIMIT {limit}
        """
        return self.execute_query(query)

    def analyze_query_patterns(self, days_back: int = 7) -> Dict[str, pd.DataFrame]:
        """
        Analyze query patterns to identify optimization opportunities
        """
        base_query = f"""
        WITH QueryStats AS (
            SELECT
                DATE(creation_time) as query_date,
                user_email,
                total_bytes_processed,
                total_slot_ms,
                cache_hit,
                total_bytes_billed,
                TIMESTAMP_DIFF(end_time, start_time, MILLISECOND) as duration_ms,
                query
            FROM `{self.project_id}.region-us.INFORMATION_SCHEMA.JOBS`
            WHERE creation_time >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL {days_back} DAY)
                AND job_type = 'QUERY'
                AND state = 'DONE'
        )
        """
        
        # Daily query volume
        daily_volume = self.execute_query(f"""
            {base_query}
            SELECT
                query_date,
                COUNT(*) as query_count,
                ROUND(AVG(duration_ms) / 1000, 2) as avg_duration_seconds,
                ROUND(SUM(total_bytes_processed) / POW(1024, 4), 2) as total_tb_processed,
                ROUND(AVG(total_slot_ms) / 1000, 2) as avg_slot_seconds
            FROM QueryStats
            GROUP BY query_date
            ORDER BY query_date
        """)
        
        # Top users by resource usage
        top_users = self.execute_query(f"""
            {base_query}
            SELECT
                user_email,
                COUNT(*) as query_count,
                ROUND(AVG(duration_ms) / 1000, 2) as avg_duration_seconds,
                ROUND(SUM(total_bytes_processed) / POW(1024, 4), 2) as total_tb_processed,
                ROUND(AVG(total_slot_ms) / 1000, 2) as avg_slot_seconds
            FROM QueryStats
            GROUP BY user_email
            ORDER BY total_tb_processed DESC
            LIMIT 10
        """)
        
        # Cache hit analysis
        cache_analysis = self.execute_query(f"""
            {base_query}
            SELECT
                cache_hit,
                COUNT(*) as query_count,
                ROUND(AVG(duration_ms) / 1000, 2) as avg_duration_seconds,
                ROUND(SUM(total_bytes_processed) / POW(1024, 4), 2) as total_tb_processed
            FROM QueryStats
            GROUP BY cache_hit
        """)
        
        return {
            "daily_volume": daily_volume,
            "top_users": top_users,
            "cache_analysis": cache_analysis
        }

    def get_query_recommendations(self, query: str) -> Dict[str, Any]:
        """
        Analyze a query and provide optimization recommendations
        """
        try:
            # Dry run to get statistics
            job_config = bigquery.QueryJobConfig(dry_run=True)
            query_job = self.client.query(query, job_config=job_config)
            
            recommendations = []
            
            # Check for full table scans
            if "WHERE" not in query.upper():
                recommendations.append("Consider adding WHERE clauses to filter data")
            
            # Check for SELECT *
            if "SELECT *" in query.upper():
                recommendations.append("Specify needed columns instead of SELECT *")
            
            # Check for potential partitioning benefits
            if query_job.total_bytes_processed > 1_000_000_000:  # 1GB
                recommendations.append("Consider using partitioned tables for large queries")
            
            return {
                "bytes_processed": query_job.total_bytes_processed,
                "estimated_cost_usd": (query_job.total_bytes_processed / 1_000_000_000) * 5,
                "recommendations": recommendations
            }
        except Exception as e:
            return {
                "error": str(e),
                "recommendations": []
            }

    def analyze_join_patterns(self, days_back: int = 7) -> Dict[str, pd.DataFrame]:
        """
        Analyze patterns in queries with JOINs
        """
        base_query = f"""
        WITH QueryStats AS (
            SELECT
                query,
                total_bytes_processed,
                total_slot_ms,
                TIMESTAMP_DIFF(end_time, start_time, MILLISECOND) as duration_ms,
                (LENGTH(UPPER(query)) - LENGTH(REPLACE(UPPER(query), 'JOIN', ''))) / 4 as join_count,
                ARRAY(
                    SELECT DISTINCT table_name 
                    FROM UNNEST(REGEXP_EXTRACT_ALL(query, r'`?\[?([a-zA-Z0-9-_]+\.[a-zA-Z0-9-_]+\.[a-zA-Z0-9-_]+)\]?`?')) as table_name
                ) as tables_involved
            FROM `{self.project_id}.region-us.INFORMATION_SCHEMA.JOBS`
            WHERE creation_time >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL {days_back} DAY)
                AND job_type = 'QUERY'
                AND state = 'DONE'
                AND UPPER(query) LIKE '%JOIN%'
        )
        """
        
        # Join complexity analysis
        join_complexity = self.execute_query(f"""
            {base_query}
            SELECT
                join_count,
                COUNT(*) as query_count,
                ROUND(AVG(duration_ms) / 1000, 2) as avg_duration_seconds,
                ROUND(AVG(total_bytes_processed) / POW(1024, 3), 2) as avg_gb_processed
            FROM QueryStats
            GROUP BY join_count
            ORDER BY join_count
        """)
        
        # Most frequently joined tables
        common_table_pairs = self.execute_query(f"""
            {base_query},
            TablePairs AS (
                SELECT 
                    t1 as table1,
                    t2 as table2,
                    COUNT(*) as join_count
                FROM QueryStats,
                UNNEST(tables_involved) as t1,
                UNNEST(tables_involved) as t2
                WHERE t1 < t2  -- Avoid counting same pair twice
                GROUP BY t1, t2
                HAVING join_count > 1
            )
            SELECT *
            FROM TablePairs
            ORDER BY join_count DESC
            LIMIT 20
        """)
        
        return {
            "join_complexity": join_complexity,
            "common_table_pairs": common_table_pairs
        }

    def create_dataset_snapshot(self, 
                              source_dataset_id: str,
                              target_dataset_id: str,
                              expiration_days: Optional[int] = None,
                              operation_id: Optional[str] = None,
                              progress_callback: Optional[Callable[[OperationStatus], None]] = None) -> Dict[str, Any]:
        """
        Create a snapshot of an entire dataset with progress tracking and resumability
        """
        # Generate or load operation status
        operation_id = operation_id or f"snapshot_{source_dataset_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        status = self._load_operation_status(operation_id) or OperationStatus(operation_id, "dataset_snapshot")
        
        try:
            # If operation was interrupted, resume from checkpoint
            if status.status == "interrupted" and status.checkpoint:
                timestamp = status.checkpoint["timestamp"]
                processed_tables = set(status.checkpoint["processed_tables"])
            else:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                processed_tables = set()
                status.checkpoint = {"timestamp": timestamp, "processed_tables": []}
            
            # Ensure target dataset exists
            try:
                dataset = bigquery.Dataset(f"{self.project_id}.{target_dataset_id}")
                self._execute_with_retry(
                    lambda: self.client.create_dataset(dataset, exists_ok=True),
                    timeout=300  # 5 minute timeout for dataset creation
                )
            except Exception as e:
                status.status = "failed"
                status.error = f"Failed to create target dataset: {str(e)}"
                self._save_operation_status(status)
                return {"status": "error", "error": str(e)}
            
            # Get source tables
            source_tables = self.list_tables(source_dataset_id)
            remaining_tables = [t for t in source_tables if t not in processed_tables]
            
            status.total_items = len(source_tables)
            status.processed_items = len(processed_tables)
            status.progress = status.processed_items / status.total_items if status.total_items > 0 else 0
            
            if progress_callback:
                progress_callback(status)
            
            snapshot_results = []
            for table_id in remaining_tables:
                try:
                    status.current_item = table_id
                    
                    snapshot_table_id = f"{table_id}_snapshot_{timestamp}"
                    source_table_ref = f"{self.project_id}.{source_dataset_id}.{table_id}"
                    target_table_ref = f"{self.project_id}.{target_dataset_id}.{snapshot_table_id}"
                    
                    # Configure snapshot
                    copy_job_config = bigquery.CopyJobConfig()
                    
                    # Create snapshot with retry and timeout
                    def create_snapshot():
                        copy_job = self.client.copy_table(
                            source_table_ref,
                            target_table_ref,
                            job_config=copy_job_config
                        )
                        copy_job.result(timeout=3600)  # 1 hour timeout per table
                        
                        # Set expiration on the destination table after copy is complete
                        if expiration_days:
                            destination_table = self.client.get_table(target_table_ref)
                            expiration = datetime.now() + timedelta(days=expiration_days)
                            destination_table.expires = expiration
                            self.client.update_table(destination_table, ["expires"])
                        
                        return copy_job
                    
                    self._execute_with_retry(create_snapshot, max_retries=3)
                    
                    snapshot_results.append({
                        "source_table": table_id,
                        "snapshot_table": snapshot_table_id,
                        "status": "success"
                    })
                    
                    # Update checkpoint
                    processed_tables.add(table_id)
                    status.checkpoint["processed_tables"] = list(processed_tables)
                    status.processed_items = len(processed_tables)
                    status.progress = status.processed_items / status.total_items
                    self._save_operation_status(status)
                    
                    if progress_callback:
                        progress_callback(status)
                        
                except Exception as e:
                    snapshot_results.append({
                        "source_table": table_id,
                        "snapshot_table": snapshot_table_id,
                        "status": "error",
                        "error": str(e)
                    })
            
            status.status = "completed"
            status.end_time = datetime.now()
            self._save_operation_status(status)
            
            return {
                "status": "success",
                "operation_id": operation_id,
                "timestamp": timestamp,
                "source_dataset": source_dataset_id,
                "target_dataset": target_dataset_id,
                "snapshots": snapshot_results,
                "expiration_days": expiration_days
            }
            
        except Exception as e:
            status.status = "interrupted"
            status.error = str(e)
            self._save_operation_status(status)
            raise

    def list_dataset_snapshots(self, snapshot_dataset_id: str) -> List[Dict[str, Any]]:
        """
        List all snapshots in a dataset with their details
        """
        try:
            tables = self.list_tables(snapshot_dataset_id)
            snapshots = {}
            
            # Group tables by timestamp
            for table_id in tables:
                if "_snapshot_" in table_id:
                    # Extract original table name and timestamp
                    parts = table_id.split("_snapshot_")
                    if len(parts) == 2:
                        original_table, timestamp = parts
                        if timestamp not in snapshots:
                            snapshots[timestamp] = {
                                "timestamp": timestamp,
                                "tables": [],
                                "table_count": 0
                            }
                        snapshots[timestamp]["tables"].append({
                            "original_table": original_table,
                            "snapshot_table": table_id
                        })
                        snapshots[timestamp]["table_count"] += 1
            
            # Convert dictionary to list and sort by timestamp
            snapshot_list = list(snapshots.values())
            snapshot_list.sort(key=lambda x: x["timestamp"], reverse=True)
            
            return snapshot_list
            
        except Exception as e:
            print(f"Error listing snapshots: {str(e)}")
            return []

    def restore_dataset_snapshot(self, snapshot_dataset_id: str, 
                               target_dataset_id: str,
                               timestamp: str,
                               overwrite: bool = False) -> Dict[str, Any]:
        """
        Restore a dataset from a snapshot
        
        Args:
            snapshot_dataset_id: Dataset containing the snapshots
            target_dataset_id: Dataset to restore to
            timestamp: Timestamp of the snapshot to restore
            overwrite: Whether to overwrite existing tables
            
        Returns:
            Dict containing restoration details and status
        """
        try:
            # Ensure target dataset exists
            dataset = bigquery.Dataset(f"{self.project_id}.{target_dataset_id}")
            dataset = self.client.create_dataset(dataset, exists_ok=True)
            
            # Get all snapshot tables for the specified timestamp
            snapshot_tables = [
                table_id for table_id in self.list_tables(snapshot_dataset_id)
                if f"_snapshot_{timestamp}" in table_id
            ]
            
            restore_results = []
            for snapshot_table_id in snapshot_tables:
                # Extract original table name
                original_table = snapshot_table_id.split(f"_snapshot_{timestamp}")[0]
                
                source_table_ref = f"{self.project_id}.{snapshot_dataset_id}.{snapshot_table_id}"
                target_table_ref = f"{self.project_id}.{target_dataset_id}.{original_table}"
                
                # Configure copy job
                copy_job_config = bigquery.CopyJobConfig()
                if overwrite:
                    copy_job_config.write_disposition = bigquery.WriteDisposition.WRITE_TRUNCATE
                else:
                    copy_job_config.write_disposition = bigquery.WriteDisposition.WRITE_EMPTY
                
                # Restore table
                try:
                    copy_job = self.client.copy_table(
                        source_table_ref,
                        target_table_ref,
                        job_config=copy_job_config
                    )
                    copy_job.result()  # Wait for job to complete
                    
                    restore_results.append({
                        "table": original_table,
                        "status": "success"
                    })
                except Exception as e:
                    restore_results.append({
                        "table": original_table,
                        "status": "error",
                        "error": str(e)
                    })
            
            return {
                "status": "success",
                "timestamp": timestamp,
                "source_dataset": snapshot_dataset_id,
                "target_dataset": target_dataset_id,
                "restored_tables": restore_results
            }
            
        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }

    def list_available_projects(self) -> List[Dict[str, str]]:
        """
        List all available projects that the authenticated user has access to
        
        Returns:
            List of dicts containing project_id and display_name
        """
        try:
            client = ProjectsClient()
            projects = []
            
            # List all projects the user has access to
            # Use empty parent to list all accessible projects
            request = resourcemanager_v3.ListProjectsRequest(
                parent=""  # Empty parent lists all accessible projects
            )
            
            try:
                page_result = client.list_projects(request=request)
                
                for project in page_result:
                    if project.state == resourcemanager_v3.Project.State.ACTIVE:
                        projects.append({
                            "project_id": project.project_id,
                            "display_name": project.display_name or project.project_id
                        })
                
                return sorted(projects, key=lambda x: x['display_name'].lower())
                
            except Exception as e:
                # If listing fails, try alternative approach using search
                search_request = resourcemanager_v3.SearchProjectsRequest()
                search_result = client.search_projects(request=search_request)
                
                for project in search_result:
                    if project.state == resourcemanager_v3.Project.State.ACTIVE:
                        projects.append({
                            "project_id": project.project_id,
                            "display_name": project.display_name or project.project_id
                        })
                
                return sorted(projects, key=lambda x: x['display_name'].lower())
                
        except Exception as e:
            print(f"Error listing projects: {str(e)}")
            return []

    def delete_table(self, dataset_id: str, table_id: str) -> Dict[str, Any]:
        """Delete a specific table"""
        try:
            table_ref = f"{self.project_id}.{dataset_id}.{table_id}"
            self.client.delete_table(table_ref)
            return {
                "status": "success",
                "message": f"Table {table_id} deleted successfully"
            }
        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }

    def delete_dataset(self, 
                      dataset_id: str, 
                      delete_contents: bool = False,
                      operation_id: Optional[str] = None,
                      progress_callback: Optional[Callable[[OperationStatus], None]] = None) -> Dict[str, Any]:
        """Delete a dataset and optionally its contents"""
        try:
            dataset_ref = f"{self.project_id}.{dataset_id}"
            self.client.delete_dataset(
                dataset_ref,
                delete_contents=delete_contents,  # Pass through the delete_contents parameter
                not_found_ok=True
            )
            
            return {
                "status": "success",
                "message": f"Dataset {dataset_id} deleted successfully"
            }
            
        except Exception as e:
            return {"status": "error", "error": str(e)}

    def delete_snapshot(self, 
                       dataset_id: str, 
                       timestamp: str,
                       operation_id: Optional[str] = None,
                       progress_callback: Optional[Callable[[OperationStatus], None]] = None) -> Dict[str, Any]:
        """Delete all tables in a snapshot group with operation tracking"""
        operation_id = operation_id or f"delete_snapshot_{dataset_id}_{timestamp}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        status = self._load_operation_status(operation_id) or OperationStatus(operation_id, "snapshot_delete")
        
        try:
            # Get all snapshot tables for the specified timestamp
            tables = self.list_tables(dataset_id)
            snapshot_tables = [
                table_id for table_id in tables
                if f"_snapshot_{timestamp}" in table_id
            ]
            
            status.total_items = len(snapshot_tables)
            status.processed_items = 0
            self._save_operation_status(status)
            
            if progress_callback:
                progress_callback(status)
            
            results = []
            for table_id in snapshot_tables:
                try:
                    status.current_item = table_id
                    result = self.delete_table(dataset_id, table_id)
                    results.append({
                        "table": table_id,
                        "status": result["status"],
                        "message": result.get("message", result.get("error", "Unknown error"))
                    })
                    
                    status.processed_items += 1
                    status.progress = status.processed_items / status.total_items
                    self._save_operation_status(status)
                    
                    if progress_callback:
                        progress_callback(status)
                    
                except Exception as e:
                    results.append({
                        "table": table_id,
                        "status": "error",
                        "message": str(e)
                    })
            
            status.status = "completed"
            status.end_time = datetime.now()
            self._save_operation_status(status)
            
            return {
                "status": "success",
                "operation_id": operation_id,
                "timestamp": timestamp,
                "deleted_tables": results
            }
            
        except Exception as e:
            status.status = "failed"
            status.error = str(e)
            self._save_operation_status(status)
            return {
                "status": "error",
                "error": str(e)
            }