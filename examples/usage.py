from utils.bigquery_tools import BigQueryUtils

# Initialize the utility class
bq_utils = BigQueryUtils("your-project-id")

# List all datasets
datasets = bq_utils.list_datasets()
print("Available datasets:", datasets)

# Get schema for a specific table
schema = bq_utils.get_table_schema("dataset_id", "table_id")
print("Table schema:", schema)

# Execute a query
query = """
    SELECT column1, column2
    FROM `project.dataset.table`
    WHERE date >= '2024-01-01'
    LIMIT 1000
"""
results_df = bq_utils.execute_query(query)
print("Query results:", results_df.head())

# Get table metrics
metrics = bq_utils.get_table_size("dataset_id", "table_id")
print(f"Table size: {metrics['size_mb']:.2f} MB")
print(f"Number of rows: {metrics['rows']}") 