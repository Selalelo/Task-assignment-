# ML-Powered Task Assignment System

This repository contains a machine learning system for optimizing task assignments to employees based on skills, experience, workload, and historical performance.

## Features

- **Skill-Based Matching**: Automatically matches employees to tasks based on required skills
- **Workload Balancing**: Considers current employee workload to prevent overallocation
- **ML-Powered Recommendations**: Uses Random Forest classifier trained on historical data
- **Priority Management**: Processes high-priority tasks first
- **Performance Optimization**: Considers employee success rate and experience

## Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/task-assignment-ml.git
cd task-assignment-ml
```

2. Create a virtual environment and activate it:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install the required packages:
```bash
pip install -r requirements.txt
```

## Usage

### Data Format

The system requires two CSV files:

1. `Employee.csv` with the following columns:
   - `Employee id`: Unique identifier for each employee
   - `Name`: Employee name
   - `Skills`: Comma-separated list of skills (e.g., "python,sql,javascript")
   - `Experience`: Experience level (numeric)
   - `Success rate`: Historical success rate percentage
   - `Workload`: Current workload (number of assigned tasks)

2. `Tasks.csv` with the following columns:
   - `Task id`: Unique identifier for each task
   - `Skills`: Comma-separated list of required skills
   - `Complexity`: Task complexity (low, medium, high)
   - `Priority level`: Task priority (low, medium, high, critical)

### Running the System

1. Place your data files in the `data/raw/` directory.

2. Run the main script:
```bash
python scripts/main.py
```

3. For specific task recommendations:
```python
from src.models.predict_model import recommend_employee_for_task

# Load your model and data
# ...

recommendations = recommend_employee_for_task(
    "T101", tasks, employees, model, feature_weights, feature_columns
)
print(recommendations)
```

## How It Works

1. **Data Preprocessing**: 
   - Loads employee and task data
   - Extracts and encodes skills
   - Encodes categorical features

2. **Feature Engineering**:
   - Creates skill matrices using one-hot encoding
   - Generates synthetic historical assignment data
   - Applies feature importance weighting

3. **Model Training**:
   - Trains a Random Forest classifier on historical data
   - Uses weighted features to emphasize skill matching

4. **Recommendation Engine**:
   - Scores potential employee-task matches
   - Combines ML predictions with a custom scoring algorithm
   - Considers workload for balanced assignments

5. **Task Assignment**:
   - Processes tasks in priority order
   - Assigns optimal employees while maintaining workload balance

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
