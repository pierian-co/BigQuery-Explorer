# BigQuery Explorer

A powerful Python utility and web interface for exploring, managing, and analyzing Google BigQuery resources.

## Features

### Dataset Management
- Browse and manage datasets and tables
- View table schemas and detailed metrics
- Create and manage dataset snapshots
- Delete datasets and tables with safety confirmations

### Query Tools
- Interactive SQL query editor with syntax highlighting
- Query cost estimation and optimization recommendations
- Performance analysis and execution metrics
- Query pattern analysis for identifying optimization opportunities

### Schema Analysis
- Compare schemas between tables
- View detailed column statistics
- Analyze table partitioning information
- Track schema changes over time

### Materialized Views
- Create and manage materialized views
- Configure auto-refresh settings
- Monitor view metrics and performance
- Manual refresh capabilities

### Dataset Snapshots
- Create point-in-time snapshots of entire datasets
- Manage snapshot lifecycle with expiration policies
- Restore snapshots to new datasets
- Track snapshot operations with progress monitoring

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/bigquery-explorer.git
cd bigquery-explorer
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up Google Cloud authentication (choose one method):

   Option A: Using gcloud CLI (recommended for development):
   ```bash
   # Install Google Cloud SDK if you haven't already
   # https://cloud.google.com/sdk/docs/install
   
   # Login with your Google account
   gcloud auth login
   
   # Set your project ID
   gcloud config set project YOUR_PROJECT_ID
   ```

   Option B: Using service account key:
   ```bash
   # Set environment variable to your service account key file
   export GOOGLE_APPLICATION_CREDENTIALS="path/to/your/credentials.json"
   ```

4. Run the setup script:
```bash
python setup.py
```

## Usage

### Web Interface

Launch the Streamlit web interface:
```bash
cd frontend
streamlit run frontend/app.py
```

### Python Library

```python
from utils.bigquery_tools import BigQueryUtils

# Initialize the utility class
bq_utils = BigQueryUtils("your-project-id")

# List datasets
datasets = bq_utils.list_datasets()
print("Available datasets:", datasets)

# Execute a query
query = """
    SELECT column1, column2
    FROM `project.dataset.table`
    WHERE date >= '2024-01-01'
    LIMIT 1000
"""
results_df = bq_utils.execute_query(query)
```

## Project Structure

```
bigquery_explorer/
├── utils/
│   ├── __init__.py
│   └── bigquery_tools.py
├── frontend/
│   └── app.py
├── requirements.txt
└── README.md
```

## Requirements

- Python 3.7+
- Google Cloud SDK
- Required Python packages:
  - streamlit>=1.30.0
  - google-cloud-bigquery>=3.11.4
  - pandas>=2.0.0
  - db-dtypes>=1.1.1
  - google-cloud-bigquery-storage>=2.24.0
  - google-cloud-resource-manager>=1.10.0

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Built with [Streamlit](https://streamlit.io/)
- Powered by [Google BigQuery](https://cloud.google.com/bigquery)

## Future Improvements

- [ ] Query Features
  - [ ] Query history and favorites
  - [ ] Query template library
  - [ ] Query cost estimation before execution
  - [ ] Query optimization suggestions
  - [ ] Export results in multiple formats (CSV, JSON, Parquet)

- [ ] Schema Management
  - [ ] Schema change tracking and versioning
  - [ ] Schema comparison between tables
  - [ ] Column-level data quality metrics
  - [ ] Auto-generated data dictionaries

- [ ] Security & Access Control
  - [ ] Role-based access control
  - [ ] Query access patterns monitoring
  - [ ] Cost monitoring and budgeting
  - [ ] Audit logging

- [ ] Performance Optimization
  - [ ] Cached query results
  - [ ] Materialized view recommendations
  - [ ] Partition and clustering analysis
  - [ ] Query performance monitoring

- [ ] User Interface
  - [ ] Dark mode support
  - [ ] Customizable dashboards
  - [ ] Query result visualizations
  - [ ] Mobile-responsive design

- [ ] Integration & Export
  - [ ] Scheduled query exports
  - [ ] Integration with dbt
  - [ ] Slack/Teams notifications
  - [ ] REST API endpoints

- [ ] Documentation
  - [ ] API documentation
  - [ ] User guides and tutorials
  - [ ] Query examples library
  - [ ] Best practices guide

Want to contribute? Pick an item from the list above and submit a pull request!
