import sys
import os
# Add the project root directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import pandas as pd
from utils.bigquery_tools import BigQueryUtils
import time
from datetime import datetime

st.set_page_config(page_title="BigQuery Explorer", layout="wide")

# Initialize session state
if 'bq_utils' not in st.session_state:
    st.session_state.bq_utils = None

def init_bigquery():
    try:
        # Create temporary BigQueryUtils instance to list projects
        temp_utils = BigQueryUtils("")
        projects = temp_utils.list_available_projects()
        
        if not projects:
            st.warning("No projects found or unable to list projects. Enter project ID manually:")
            project_id = st.text_input("Enter your Google Cloud Project ID")
        else:
            # Create a dict for the selectbox
            project_options = {
                f"{p['display_name']} ({p['project_id']})": p['project_id'] 
                for p in projects
            }
            
            # Show dropdown of projects
            selected_option = st.selectbox(
                "Select Project",
                options=list(project_options.keys()),
                format_func=lambda x: x.split(" (")[0]  # Show only display name in dropdown
            )
            
            project_id = project_options[selected_option]
        
        if st.button("Connect"):
            try:
                st.session_state.bq_utils = BigQueryUtils(project_id)
                st.success("Successfully connected to BigQuery!")
            except Exception as e:
                st.error(f"Failed to connect: {str(e)}")
    except Exception as e:
        st.error(f"Error loading projects: {str(e)}")
        # Fallback to manual input
        project_id = st.text_input("Enter your Google Cloud Project ID")
        if st.button("Connect"):
            try:
                st.session_state.bq_utils = BigQueryUtils(project_id)
                st.success("Successfully connected to BigQuery!")
            except Exception as e:
                st.error(f"Failed to connect: {str(e)}")

def get_dataset_table_selection(key_prefix=""):
    """
    Helper function to create dataset and table selection in sidebar
    Returns tuple of (selected_dataset, selected_table)
    """
    with st.sidebar:
        st.subheader("Dataset & Table Selection")
        datasets = st.session_state.bq_utils.list_datasets()
        selected_dataset = st.selectbox(
            "Select Dataset", 
            datasets,
            key=f"{key_prefix}_dataset"
        )
        
        if selected_dataset:
            tables = st.session_state.bq_utils.list_tables(selected_dataset)
            selected_table = st.selectbox(
                "Select Table",
                tables,
                key=f"{key_prefix}_table"
            )
            return selected_dataset, selected_table
        
        return None, None

def switch_project():
    """
    Helper function to switch between Google Cloud projects with dropdown selection
    """
    try:
        # Get list of available projects
        projects = st.session_state.bq_utils.list_available_projects()
        
        if not projects:
            st.warning("No projects found or unable to list projects. Enter project ID manually:")
            new_project_id = st.text_input("Project ID")
        else:
            # Create a dict for the selectbox
            project_options = {
                f"{p['display_name']} ({p['project_id']})": p['project_id'] 
                for p in projects
            }
            
            # Show dropdown of projects
            selected_option = st.selectbox(
                "Select Project",
                options=list(project_options.keys()),
                format_func=lambda x: x.split(" (")[0]  # Show only display name in dropdown
            )
            
            new_project_id = project_options[selected_option]
            
        if st.button("Switch Project"):
            try:
                st.session_state.bq_utils = BigQueryUtils(new_project_id)
                st.success(f"Successfully switched to project: {new_project_id}")
                # Force a rerun to update all components
                st.rerun()
            except Exception as e:
                st.error(f"Failed to switch project: {str(e)}")
                
    except Exception as e:
        st.error(f"Error loading projects: {str(e)}")
        # Fallback to manual input
        new_project_id = st.text_input("Project ID")
        if st.button("Switch Project"):
            try:
                st.session_state.bq_utils = BigQueryUtils(new_project_id)
                st.success(f"Successfully switched to project: {new_project_id}")
                st.rerun()
            except Exception as e:
                st.error(f"Failed to switch project: {str(e)}")

def main():
    st.title("BigQuery Explorer")
    
    if not st.session_state.bq_utils:
        init_bigquery()
    else:
        # Add project switcher to sidebar
        with st.sidebar:
            st.subheader("Current Project")
            st.info(f"Project ID: {st.session_state.bq_utils.project_id}")
            
            # Add expander for project switching
            with st.expander("Switch Project"):
                switch_project()
        
        # Sidebar navigation
        page = st.sidebar.selectbox(
            "Select Feature",
            ["Dataset Explorer", "Schema Comparison", "Materialized Views", "Dataset Snapshots"]
        )
        
        if page == "Dataset Explorer":
            show_dataset_explorer()
        elif page == "Schema Comparison":
            show_schema_comparison()
        elif page == "Materialized Views":
            show_materialized_views()
        elif page == "Dataset Snapshots":
            show_dataset_snapshots()

def show_dataset_explorer():
    st.header("Datasets Explorer")
    
    try:
        datasets = st.session_state.bq_utils.list_datasets()
        
        # Format dataset options to show snapshot indicator
        dataset_options = []
        for dataset in datasets:
            is_snapshot = any("_snapshot_" in table for table in st.session_state.bq_utils.list_tables(dataset))
            label = f"{dataset} 📸" if is_snapshot else dataset
            dataset_options.append((dataset, label))
        
        col1, col2 = st.columns([3, 1])
        with col1:
            selected_dataset = st.selectbox(
                "Select Dataset", 
                options=[opt[0] for opt in dataset_options],
                format_func=lambda x: next(opt[1] for opt in dataset_options if opt[0] == x)
            )
        with col2:
            if selected_dataset:
                is_snapshot = any("_snapshot_" in table for table in st.session_state.bq_utils.list_tables(selected_dataset))
                delete_button_label = "Delete Snapshot Dataset" if is_snapshot else "Delete Dataset"
                delete_dataset_button = st.button(delete_button_label, key="delete_dataset")
                
                # Handle dataset deletion
                if delete_dataset_button:
                    st.session_state.confirm_delete_dataset = selected_dataset
                    st.session_state.show_dataset_delete_confirm = True
                
                if st.session_state.get('show_dataset_delete_confirm', False):
                    warning_text = (
                        f"Are you sure you want to delete the snapshot dataset '{selected_dataset}'?"
                        if is_snapshot else
                        f"Are you sure you want to delete dataset '{selected_dataset}'?"
                    )
                    st.warning(f"{warning_text} This cannot be undone!")
                    
                    # Add checkbox for deleting contents
                    delete_contents = st.checkbox(
                        "Delete all tables in this dataset",
                        value=True,
                        help="If checked, all tables in the dataset will be deleted. Otherwise, deletion will fail if the dataset contains tables."
                    )
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button("Confirm Delete", key="confirm_dataset_delete"):
                            try:
                                result = st.session_state.bq_utils.delete_dataset(
                                    selected_dataset,
                                    delete_contents=delete_contents
                                )
                                if result["status"] == "success":
                                    st.success(result["message"])
                                    # Clear the confirmation state and selection
                                    st.session_state.show_dataset_delete_confirm = False
                                    st.session_state.confirm_delete_dataset = None
                                    # Clear dataset selection
                                    if f"dataset" in st.session_state:
                                        del st.session_state["dataset"]
                                    st.rerun()
                                else:
                                    st.error(f"Failed to delete dataset: {result['error']}")
                            except Exception as e:
                                st.error(f"Error during deletion: {str(e)}")
                    with col2:
                        if st.button("Cancel", key="cancel_dataset_delete"):
                            st.session_state.show_dataset_delete_confirm = False
                            st.session_state.confirm_delete_dataset = None
                            st.rerun()
        
        if selected_dataset:
            tables = st.session_state.bq_utils.list_tables(selected_dataset)
            
            col1, col2 = st.columns([3, 1])
            with col1:
                selected_table = st.selectbox("Select Table", tables)
            with col2:
                if selected_table:
                    delete_table_button = st.button("Delete Table", key="delete_table")
                    
                    # Handle table deletion
                    if delete_table_button:
                        st.session_state.confirm_delete_table = selected_table
                        st.session_state.show_table_delete_confirm = True
                    
                    if st.session_state.get('show_table_delete_confirm', False):
                        st.warning(f"Are you sure you want to delete table '{selected_table}'? This cannot be undone!")
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            if st.button("Confirm Delete", key="confirm_table_delete"):
                                try:
                                    result = st.session_state.bq_utils.delete_table(
                                        selected_dataset,
                                        selected_table
                                    )
                                    if result["status"] == "success":
                                        st.success(result["message"])
                                        # Clear the confirmation state and selection
                                        st.session_state.show_table_delete_confirm = False
                                        st.session_state.confirm_delete_table = None
                                        # Clear table selection
                                        if f"table" in st.session_state:
                                            del st.session_state["table"]
                                        st.rerun()
                                    else:
                                        st.error(f"Failed to delete table: {result['error']}")
                                except Exception as e:
                                    st.error(f"Error during deletion: {str(e)}")
                        with col2:
                            if st.button("Cancel", key="cancel_table_delete"):
                                st.session_state.show_table_delete_confirm = False
                                st.session_state.confirm_delete_table = None
                                st.rerun()
            
            if selected_table:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Table Schema")
                    schema = st.session_state.bq_utils.get_table_schema(selected_dataset, selected_table)
                    st.table(pd.DataFrame(schema))
                
                with col2:
                    st.subheader("Table Metrics")
                    metrics = st.session_state.bq_utils.get_table_size(selected_dataset, selected_table)
                    st.write(f"Number of rows: {metrics['rows']:,}")
                    st.write(f"Size: {metrics['size_mb']:.2f} MB")
                    st.write(f"Last modified: {metrics['last_modified']}")
    
    except Exception as e:
        st.error(f"Error loading datasets: {str(e)}")

def show_query_editor():
    st.header("Query Editor")
    
    query = st.text_area("Enter your SQL query", height=200)
    
    col1, col2 = st.columns([1, 5])
    with col1:
        limit_rows = st.number_input("Limit rows", min_value=1, value=1000)
    
    if st.button("Run Query"):
        if query:
            try:
                # Add LIMIT clause if not present
                if "LIMIT" not in query.upper():
                    query = f"{query} LIMIT {limit_rows}"
                
                results = st.session_state.bq_utils.execute_query(query)
                st.dataframe(results)
                
                st.success(f"Query returned {len(results)} rows")
            except Exception as e:
                st.error(f"Query failed: {str(e)}")

def show_table_inspector():
    st.header("Table Inspector")
    
    try:
        dataset, table = get_dataset_table_selection("inspector")
        
        if dataset and table:
            preview_query = f"""
            SELECT *
            FROM `{st.session_state.bq_utils.project_id}.{dataset}.{table}`
            LIMIT 100
            """
            
            results = st.session_state.bq_utils.execute_query(preview_query)
            
            # Show table details
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Table Schema")
                schema = st.session_state.bq_utils.get_table_schema(dataset, table)
                st.table(pd.DataFrame(schema))
            
            with col2:
                st.subheader("Table Metrics")
                metrics = st.session_state.bq_utils.get_table_size(dataset, table)
                st.write(f"Number of rows: {metrics['rows']:,}")
                st.write(f"Size: {metrics['size_mb']:.2f} MB")
                st.write(f"Last modified: {metrics['last_modified']}")
            
            # Show preview data
            st.subheader("Data Preview")
            st.dataframe(results)
            
            # Download button
            st.download_button(
                label="Download Preview as CSV",
                data=results.to_csv(index=False).encode('utf-8'),
                file_name=f"{table}_preview.csv",
                mime="text/csv"
            )
    
    except Exception as e:
        st.error(f"Error in Table Inspector: {str(e)}")

def show_schema_comparison():
    st.header("Schema Comparison")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("First Table")
        datasets1 = st.session_state.bq_utils.list_datasets()
        dataset1 = st.selectbox("Select Dataset (1)", datasets1, key="dataset1")
        if dataset1:
            tables1 = st.session_state.bq_utils.list_tables(dataset1)
            table1 = st.selectbox("Select Table (1)", tables1, key="table1")
    
    with col2:
        st.subheader("Second Table")
        datasets2 = st.session_state.bq_utils.list_datasets()
        dataset2 = st.selectbox("Select Dataset (2)", datasets2, key="dataset2")
        if dataset2:
            tables2 = st.session_state.bq_utils.list_tables(dataset2)
            table2 = st.selectbox("Select Table (2)", tables2, key="table2")
    
    if st.button("Compare Schemas"):
        if dataset1 and table1 and dataset2 and table2:
            comparison = st.session_state.bq_utils.compare_schemas(dataset1, table1, dataset2, table2)
            
            if comparison["only_in_first"]:
                st.subheader("Fields only in first table")
                st.table(pd.DataFrame(comparison["only_in_first"], columns=["Field", "Type"]))
            
            if comparison["only_in_second"]:
                st.subheader("Fields only in second table")
                st.table(pd.DataFrame(comparison["only_in_second"], columns=["Field", "Type"]))
            
            if comparison["type_mismatches"]:
                st.subheader("Type mismatches")
                st.table(pd.DataFrame(comparison["type_mismatches"], 
                        columns=["Field", "Type (1)", "Type (2)"]))
            
            if not any(comparison.values()):
                st.success("Schemas are identical!")

def show_column_statistics():
    st.header("Column Statistics")
    
    dataset, table = get_dataset_table_selection("stats")
    
    if dataset and table:
        # Move column selection to sidebar
        with st.sidebar:
            st.subheader("Column Selection")
            schema = st.session_state.bq_utils.get_table_schema(dataset, table)
            columns = [field["name"] for field in schema]
            selected_column = st.selectbox("Select Column", columns)
        
        if selected_column:
            if st.button("Get Statistics"):
                stats = st.session_state.bq_utils.get_table_column_stats(
                    dataset, table, selected_column
                )
                
                # Display metrics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Rows", f"{stats['total_rows']:,}")
                with col2:
                    st.metric("Unique Values", f"{stats['unique_values']:,}")
                with col3:
                    st.metric("Null Percentage", f"{stats['null_percentage']:.2f}%")
                
                # Show distribution if numeric
                if stats.get('distribution'):
                    st.subheader("Value Distribution")
                    st.bar_chart(stats['distribution'])

def show_cost_analysis():
    st.header("Query Cost Analysis")
    
    query = st.text_area("Enter your SQL query", height=200)
    
    if st.button("Estimate Cost"):
        if query:
            try:
                cost_estimate = st.session_state.bq_utils.estimate_query_cost(query)
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Estimated Cost", f"${cost_estimate['estimated_cost_usd']:.4f}")
                with col2:
                    st.metric("GB Processed", f"{cost_estimate['estimated_gb']:.2f}")
                with col3:
                    st.metric("Bytes Processed", f"{cost_estimate['bytes_processed']:,}")
                
                st.info("Cost estimation is based on $5 per TB processed. Your actual cost may vary based on your pricing tier and other factors.")
            except Exception as e:
                st.error(f"Error estimating query cost: {str(e)}")

def show_materialized_views():
    st.header("Materialized Views Manager")
    
    tab1, tab2 = st.tabs(["Create View", "Manage Views"])
    
    with tab1:
        st.subheader("Create Materialized View")
        
        # Use sidebar for dataset selection
        dataset = st.sidebar.selectbox(
            "Select Target Dataset",
            st.session_state.bq_utils.list_datasets(),
            key="mv_create_dataset"
        )
        
        if dataset:
            # View name
            view_name = st.text_input("View Name")
            
            # Query input
            source_query = st.text_area("Source Query", height=200)
            
            # Refresh settings
            col1, col2 = st.columns(2)
            with col1:
                enable_refresh = st.checkbox("Enable Auto-Refresh", value=True)
            with col2:
                refresh_interval = st.number_input(
                    "Refresh Interval (minutes)", 
                    min_value=60, 
                    value=60, 
                    step=60,
                    help="Minimum refresh interval is 60 minutes"
                )
            
            if st.button("Create Materialized View"):
                if view_name and source_query:
                    result = st.session_state.bq_utils.create_materialized_view(
                        source_query=source_query,
                        target_dataset=dataset,
                        view_name=view_name,
                        enable_refresh=enable_refresh,
                        refresh_interval_minutes=refresh_interval
                    )
                    
                    if result["status"] == "success":
                        st.success(f"Materialized view {view_name} created successfully!")
                        st.json(result)
                    else:
                        st.error(f"Failed to create view: {result['error']}")
    
    with tab2:
        dataset = st.sidebar.selectbox(
            "Select Dataset",
            st.session_state.bq_utils.list_datasets(),
            key="mv_manage_dataset"
        )
        
        if dataset:
            views = st.session_state.bq_utils.list_materialized_views(dataset)
            
            if not views:
                st.info("No materialized views found in this dataset")
            else:
                view_df = pd.DataFrame(views)
                st.dataframe(view_df)
                
                selected_view = st.selectbox(
                    "Select View for Management",
                    [v["view_id"] for v in views]
                )
                
                if selected_view:
                    show_view_details(dataset, selected_view)

def show_query_analysis():
    st.header("Query Performance Analysis")
    
    tab1, tab2, tab3, tab4 = st.tabs(["Slow Queries", "Join Analysis", "Query Patterns", "Query Analyzer"])
    
    with tab1:
        st.subheader("Slowest Queries with JOINs")
        days = st.slider("Days to analyze", 1, 30, 7)
        limit = st.slider("Number of queries to show", 10, 1000, 100)
        min_joins = st.slider("Minimum number of JOINs", 1, 10, 2)
        
        if st.button("Analyze Slow Queries"):
            slow_queries = st.session_state.bq_utils.get_slow_queries(days, limit, min_joins)
            
            # Display summary metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Queries", len(slow_queries))
            with col2:
                st.metric("Avg Duration", f"{slow_queries['duration_seconds'].mean():.2f}s")
            with col3:
                st.metric("Avg Joins", f"{slow_queries['number_of_joins'].mean():.1f}")
            
            # Display detailed table
            st.dataframe(slow_queries)
            
            # Allow downloading results
            csv = slow_queries.to_csv(index=False).encode('utf-8')
            st.download_button(
                "Download Query Analysis",
                csv,
                "slow_queries.csv",
                "text/csv",
                key='download-csv'
            )
    
    with tab2:
        st.subheader("JOIN Pattern Analysis")
        analysis_days = st.slider("Analysis period (days)", 1, 30, 7, key="join_analysis_days")
        
        if st.button("Analyze Join Patterns"):
            patterns = st.session_state.bq_utils.analyze_join_patterns(analysis_days)
            
            # Show join complexity analysis
            st.subheader("Query Complexity vs Performance")
            complexity_df = patterns["join_complexity"]
            st.line_chart(complexity_df.set_index('join_count')[['avg_duration_seconds', 'avg_gb_processed']])
            
            # Show common table pairs
            st.subheader("Most Frequently Joined Tables")
            st.dataframe(patterns["common_table_pairs"])
    
    with tab3:
        st.subheader("Query Patterns")
        analysis_days = st.slider("Analysis period (days)", 1, 30, 7, key="pattern_days")
        
        if st.button("Analyze Patterns"):
            patterns = st.session_state.bq_utils.analyze_query_patterns(analysis_days)
            
            # Daily volume chart
            st.subheader("Daily Query Volume")
            st.line_chart(patterns["daily_volume"].set_index("query_date"))
            
            # Top users
            st.subheader("Top Users by Resource Usage")
            st.dataframe(patterns["top_users"])
            
            # Cache analysis
            st.subheader("Cache Hit Analysis")
            st.dataframe(patterns["cache_analysis"])
    
    with tab4:
        st.subheader("Query Analyzer")
        query = st.text_area("Enter a query to analyze", height=200)
        
        if st.button("Analyze Query"):
            if query:
                analysis = st.session_state.bq_utils.get_query_recommendations(query)
                
                if "error" not in analysis:
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Estimated GB", f"{analysis['bytes_processed'] / 1e9:.2f}")
                    with col2:
                        st.metric("Estimated Cost", f"${analysis['estimated_cost_usd']:.4f}")
                    
                    if analysis["recommendations"]:
                        st.subheader("Recommendations")
                        for rec in analysis["recommendations"]:
                            st.info(rec)
                    else:
                        st.success("No immediate optimization recommendations")
                else:
                    st.error(f"Analysis failed: {analysis['error']}")

def show_dataset_snapshots():
    st.header("Dataset Snapshots Manager")
    
    tab1, tab2, tab3 = st.tabs(["Create Snapshot", "Manage Snapshots", "Operation Status"])
    
    with tab1:
        st.subheader("Create Dataset Snapshot")
        
        # Source dataset selection
        source_dataset = st.selectbox(
            "Select Source Dataset",
            st.session_state.bq_utils.list_datasets(),
            key="snapshot_source_dataset"
        )
        
        # Target dataset input
        target_dataset = st.text_input(
            "Target Dataset ID",
            help="Enter the ID for the dataset where snapshots will be stored"
        )
        
        # Expiration settings
        enable_expiration = st.checkbox("Enable Snapshot Expiration", value=False)
        expiration_days = None
        if enable_expiration:
            expiration_days = st.number_input(
                "Expiration (days)",
                min_value=1,
                value=30
            )
        
        # Operation ID for resuming
        resume_operation = st.checkbox("Resume Previous Operation")
        operation_id = None
        if resume_operation:
            # List available operations
            operations = [f for f in os.listdir(st.session_state.bq_utils.state_dir) 
                        if f.startswith("snapshot_") and f.endswith(".json")]
            if operations:
                selected_op = st.selectbox(
                    "Select Operation to Resume",
                    operations,
                    format_func=lambda x: x.replace(".json", "")
                )
                operation_id = selected_op.replace(".json", "")
            else:
                st.info("No previous operations found")
                resume_operation = False
        
        if st.button("Create Snapshot"):
            if source_dataset and target_dataset:
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                def update_progress(status):
                    progress_bar.progress(status.progress)
                    status_text.text(
                        f"Processing {status.current_item or 'initializing'} "
                        f"({status.processed_items}/{status.total_items})"
                    )
                
                try:
                    with st.spinner("Creating dataset snapshot..."):
                        result = st.session_state.bq_utils.create_dataset_snapshot(
                            source_dataset_id=source_dataset,
                            target_dataset_id=target_dataset,
                            expiration_days=expiration_days,
                            operation_id=operation_id,
                            progress_callback=update_progress
                        )
                        
                        if result["status"] == "success":
                            st.success("Dataset snapshot created successfully!")
                            st.json(result)
                        else:
                            st.error(f"Failed to create snapshot: {result['error']}")
                            
                except Exception as e:
                    st.error(f"Operation interrupted: {str(e)}")
                    st.info("You can resume this operation later using the operation ID")
                    
    with tab2:
        st.subheader("Manage Snapshots")
        
        snapshot_dataset = st.selectbox(
            "Select Snapshot Dataset",
            st.session_state.bq_utils.list_datasets(),
            key="snapshot_view_dataset"
        )
        
        if snapshot_dataset:
            snapshots = st.session_state.bq_utils.list_dataset_snapshots(snapshot_dataset)
            
            if not snapshots:
                st.info("No snapshots found in this dataset")
            else:
                st.subheader("Available Snapshots")
                for snapshot in snapshots:
                    with st.expander(f"Snapshot {snapshot['timestamp']} ({snapshot['table_count']} tables)"):
                        col1, col2 = st.columns([3, 1])
                        with col1:
                            tables_df = pd.DataFrame(snapshot["tables"])
                            st.dataframe(tables_df)
                        with col2:
                            delete_snapshot_button = st.button("Delete Snapshot", key=f"delete_{snapshot['timestamp']}")
                            
                            # Handle snapshot deletion
                            if delete_snapshot_button:
                                st.session_state.confirm_delete_snapshot = snapshot['timestamp']
                                st.session_state.show_snapshot_delete_confirm = True
                            
                            if st.session_state.get('show_snapshot_delete_confirm', False) and st.session_state.confirm_delete_snapshot == snapshot['timestamp']:
                                st.warning(f"Are you sure you want to delete this snapshot? This cannot be undone!")
                                
                                col1, col2 = st.columns(2)
                                with col1:
                                    if st.button("Confirm Delete", key=f"confirm_delete_{snapshot['timestamp']}"):
                                        try:
                                            progress_bar = st.progress(0)
                                            status_text = st.empty()
                                            
                                            def update_progress(status):
                                                progress_bar.progress(status.progress)
                                                status_text.text(
                                                    f"Deleting {status.current_item or 'initializing'} "
                                                    f"({status.processed_items}/{status.total_items})"
                                                )
                                            
                                            result = st.session_state.bq_utils.delete_snapshot(
                                                snapshot_dataset,
                                                snapshot['timestamp'],
                                                progress_callback=update_progress
                                            )
                                            if result["status"] == "success":
                                                st.success("Snapshot deleted successfully!")
                                                st.json(result)
                                                # Clear the confirmation state and selection
                                                st.session_state.show_snapshot_delete_confirm = False
                                                st.session_state.confirm_delete_snapshot = None
                                                # Clear snapshot selection
                                                if "snapshot_view_dataset" in st.session_state:
                                                    del st.session_state["snapshot_view_dataset"]
                                                st.rerun()
                                            else:
                                                st.error(f"Failed to delete snapshot: {result['error']}")
                                        except Exception as e:
                                            st.error(f"Error during deletion: {str(e)}")
                                with col2:
                                    if st.button("Cancel", key=f"cancel_delete_{snapshot['timestamp']}"):
                                        st.session_state.show_snapshot_delete_confirm = False
                                        st.session_state.confirm_delete_snapshot = None
                                        st.rerun()
                
                # Restoration interface
                st.subheader("Restore Snapshot")
                
                selected_timestamp = st.selectbox(
                    "Select Snapshot Timestamp",
                    [s["timestamp"] for s in snapshots]
                )
                
                target_dataset = st.text_input(
                    "Restoration Target Dataset",
                    help="Enter the dataset ID where the snapshot will be restored"
                )
                
                overwrite = st.checkbox(
                    "Overwrite Existing Tables",
                    help="If checked, existing tables will be overwritten. Otherwise, restoration will fail if tables exist."
                )
                
                if st.button("Restore Snapshot"):
                    if target_dataset:
                        with st.spinner("Restoring snapshot..."):
                            result = st.session_state.bq_utils.restore_dataset_snapshot(
                                snapshot_dataset_id=snapshot_dataset,
                                target_dataset_id=target_dataset,
                                timestamp=selected_timestamp,
                                overwrite=overwrite
                            )
                            
                            if result["status"] == "success":
                                st.success("Snapshot restored successfully!")
                                st.json(result)
                            else:
                                st.error(f"Failed to restore snapshot: {result['error']}")
    
    with tab3:
        st.subheader("Operation Status")
        
        # List all operations
        operations = []
        for f in os.listdir(st.session_state.bq_utils.state_dir):
            if f.endswith(".json"):
                try:
                    status = st.session_state.bq_utils._load_operation_status(f.replace(".json", ""))
                    if status:
                        operations.append(status.to_dict())
                except Exception:
                    continue
        
        if operations:
            status_df = pd.DataFrame(operations)
            status_df["duration"] = pd.to_datetime(status_df["end_time"]) - pd.to_datetime(status_df["start_time"])
            status_df["duration"] = status_df["duration"].fillna(pd.Timedelta(seconds=0))
            
            st.dataframe(
                status_df[["operation_id", "operation_type", "status", "progress", 
                          "processed_items", "total_items", "duration"]]
            )
            
            # Allow clearing completed operations
            if st.button("Clear Completed Operations"):
                for f in os.listdir(st.session_state.bq_utils.state_dir):
                    if f.endswith(".json"):
                        try:
                            status = st.session_state.bq_utils._load_operation_status(f.replace(".json", ""))
                            if status and status.status == "completed":
                                os.remove(os.path.join(st.session_state.bq_utils.state_dir, f))
                        except Exception:
                            continue
                st.rerun()
        else:
            st.info("No operations found")

def show_view_details(dataset: str, view_id: str):
    """Display details of a materialized view"""
    try:
        details = st.session_state.bq_utils.get_materialized_view_details(dataset, view_id)
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Size", f"{details['size_mb']:.2f} MB")
            st.metric("Rows", f"{details['rows']:,}")
        with col2:
            st.metric("Last Refresh", details['last_refresh_time'] or "Never")
            st.metric("Refresh Interval", f"{details.get('refresh_interval_minutes', 'N/A')} min")
        
        if st.button("Refresh View"):
            result = st.session_state.bq_utils.refresh_materialized_view(dataset, view_id)
            if result["status"] == "success":
                st.success("View refreshed successfully!")
            else:
                st.error(f"Failed to refresh view: {result['error']}")
                
    except Exception as e:
        st.error(f"Error loading view details: {str(e)}")

if __name__ == "__main__":
    main() 