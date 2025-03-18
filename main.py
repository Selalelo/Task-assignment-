import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load datasets
employees = pd.read_csv("Employee.csv")
tasks = pd.read_csv("Tasks.csv")

# --------- FEATURE ENGINEERING ---------

# Process skills for employees (convert comma-separated skills to one-hot encoding)
skill_encoder = LabelEncoder()

# Extract all unique skills from both datasets
all_employee_skills = [skill for skills_list in employees["Skills"].str.split(',') for skill in skills_list]
all_task_skills = [skill for skills_list in tasks["Skills"].str.split(',') for skill in skills_list]
all_unique_skills = list(set(all_employee_skills + all_task_skills))

# Fit the encoder on all unique skills
skill_encoder.fit(all_unique_skills)

# Create skill matrices for employees
employee_skill_matrix = pd.DataFrame(0, index=employees.index, columns=skill_encoder.classes_)
for idx, skills in enumerate(employees["Skills"].str.split(',')):
    for skill in skills:
        employee_skill_matrix.loc[idx, skill] = 1

# Create skill matrices for tasks
task_skill_matrix = pd.DataFrame(0, index=tasks.index, columns=skill_encoder.classes_)
for idx, skills in enumerate(tasks["Skills"].str.split(',')):
    for skill in skills:
        task_skill_matrix.loc[idx, skill] = 1

# Encode complexity and priority if they exist
if "Complexity" in tasks.columns:
    complexity_encoder = LabelEncoder()
    tasks["ComplexityEncoded"] = complexity_encoder.fit_transform(tasks["Complexity"])

if "Priority level" in tasks.columns:
    priority_encoder = LabelEncoder()
    tasks["PriorityEncoded"] = priority_encoder.fit_transform(tasks["Priority level"])

# --------- CREATING TRAINING DATA ---------

# Create a dataset of historical task assignments
historical_assignments = []

# Generate synthetic historical data
np.random.seed(42)  # For reproducibility

# Create historical assignments with emphasis on skill matching
for _ in range(200):  # Increased sample size for better training
    task_id = np.random.choice(tasks["Task id"])
    task_row = tasks[tasks["Task id"] == task_id].iloc[0]
    
    # Find employees with matching skills
    task_skills = task_row["Skills"].split(',')
    skill_match_scores = []
    
    for idx, emp_row in employees.iterrows():
        emp_skills = emp_row["Skills"].split(',')
        emp_id = emp_row["Employee id"]
        
        # Calculate skill match percentage
        task_skills_set = set(task_skills)
        emp_skills_set = set(emp_skills)
        skill_match = len(task_skills_set.intersection(emp_skills_set)) / len(task_skills_set)
        
        # Skip if no skill match
        if skill_match == 0:
            continue
            
        # Penalize high workload
        workload_penalty = 1.0 if emp_row["Workload"] <= 3 else 0.7
        
        # Calculate weighted score (prioritizing skills, then success rate, then experience)
        # Use higher weights for skills (0.7), then success rate (0.2), then experience (0.1)
        weighted_score = (
            0.7 * skill_match + 
            0.2 * (emp_row["Success rate"] / 100) + 
            0.1 * (emp_row["Experience"] / 10)  # Normalize experience to 0-1 range
        ) * workload_penalty
        
        skill_match_scores.append((emp_id, weighted_score))
    
    # Sort by weighted score
    skill_match_scores.sort(key=lambda x: x[1], reverse=True)
    
    if not skill_match_scores:
        continue  # Skip if no matching employees
    
    # Select one of the top employees (with higher probability for top matches)
    selection_probs = [0.5, 0.3, 0.15, 0.05] + [0.0] * (len(skill_match_scores) - 4)
    selection_probs = selection_probs[:len(skill_match_scores)]
    # Normalize probabilities
    selection_probs = [p/sum(selection_probs) for p in selection_probs]
    
    selected_idx = np.random.choice(len(skill_match_scores), p=selection_probs)
    employee_id = skill_match_scores[selected_idx][0]
    employee_row = employees[employees["Employee id"] == employee_id].iloc[0]
    
    # Record the assignment with features
    task_skills_set = set(task_skills)
    emp_skills_set = set(employee_row["Skills"].split(','))
    skill_match_count = len(task_skills_set.intersection(emp_skills_set))
    skill_match_percentage = skill_match_count / len(task_skills_set) * 100
    
    assignment = {
        "Task id": task_id,
        "Employee id": employee_id,
        # Task features
        "Complexity": task_row.get("ComplexityEncoded", 0),
        "Priority": task_row.get("PriorityEncoded", 0),
        # Skill matching features - strongly emphasized
        "Skill_Match_Count": skill_match_count,
        "Skill_Match_Percentage": skill_match_percentage,
        # Employee features
        "Experience": employee_row["Experience"],
        "Success_Rate": employee_row["Success rate"],
        "Workload": employee_row["Workload"]
    }
    
    historical_assignments.append(assignment)

# Convert to DataFrame
historical_df = pd.DataFrame(historical_assignments)

# --------- TRAINING THE MODEL ---------

# Feature importance adjustments
# We'll use feature weights to emphasize skill match, success rate, and experience
feature_weights = {
    "Skill_Match_Percentage": 5.0,  # Highest weight for skill matching
    "Success_Rate": 3.0,           # Second highest for success rate
    "Experience": 2.0,             # Third highest for experience
    "Workload": 1.0,               # Lower weight for workload
    "Complexity": 1.0,             # Lower weight for complexity
    "Priority": 1.0,               # Lower weight for priority
    "Skill_Match_Count": 1.0       # Already captured in percentage
}

# Prepare features and target
X = historical_df.drop(["Task id", "Employee id"], axis=1)
y = historical_df["Employee id"]

# Apply feature weights by duplicating important columns
X_weighted = X.copy()
for feature, weight in feature_weights.items():
    if feature in X.columns:
        for i in range(int(weight) - 1):  # -1 because we already have one copy
            X_weighted[f"{feature}_{i}"] = X[feature]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_weighted, y, test_size=0.2, random_state=42)

# Train model with feature importance
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    class_weight='balanced'  # Balance classes for better prediction
)
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy*100:.2f}%")

# --------- USING THE MODEL FOR RECOMMENDATIONS ---------

def recommend_employee_for_task(task_id, top_n=3, current_workloads=None):
    """Recommend the best employees for a specific task with emphasis on skill matching and workload balancing"""
    if task_id not in tasks["Task id"].values:
        return "Task ID not found"
    
    task_row = tasks[tasks["Task id"] == task_id].iloc[0]
    task_skills = task_row["Skills"].split(',')
    
    # Use current workloads if provided, otherwise use the static workloads from the dataset
    if current_workloads is None:
        current_workloads = {emp_id: workload for emp_id, workload in 
                            zip(employees["Employee id"], employees["Workload"])}
    
    # Create candidate features for each employee
    candidates = []
    
    for idx, emp_row in employees.iterrows():
        emp_id = emp_row["Employee id"]
        emp_skills = emp_row["Skills"].split(',')
        
        # Calculate skill match
        task_skills_set = set(task_skills)
        emp_skills_set = set(emp_skills)
        skill_match_count = len(task_skills_set.intersection(emp_skills_set))
        
        # Skip employees with no skill match - strict requirement
        if skill_match_count == 0:
            continue
            
        skill_match_percentage = skill_match_count / len(task_skills_set) * 100
        
        # Get current workload
        current_workload = current_workloads.get(emp_id, emp_row["Workload"])
        
        # Create feature vector for this employee-task pair
        features = {
            "Employee id": emp_id,
            "Name": emp_row["Name"],
            "Complexity": task_row.get("ComplexityEncoded", 0),
            "Priority": task_row.get("PriorityEncoded", 0),
            "Skill_Match_Count": skill_match_count,
            "Skill_Match_Percentage": skill_match_percentage,
            "Experience": emp_row["Experience"],
            "Success_Rate": emp_row["Success rate"],
            "Workload": current_workload,  # Use updated workload
            # Add direct information for display
            "Skills": emp_row["Skills"],
            "Task_Skills": task_row["Skills"],
            "Current_Workload": current_workload  # For filtering
        }
        
        candidates.append(features)
    
    if not candidates:
        return "No eligible employees found for this task"
        
    # Convert to DataFrame
    candidates_df = pd.DataFrame(candidates)
    
    # First try with employees who have workload <= 3
    low_workload_candidates = candidates_df[candidates_df["Current_Workload"] <= 3]
    
    # If we have employees with workload <= 3, use only them
    if not low_workload_candidates.empty:
        candidates_df = low_workload_candidates
    else:
        # If all employees have high workload, proceed with all candidates
        print(f"Warning: All employees with matching skills for task {task_id} have workload > 3")
    
    # Create a custom score based on our priorities
    candidates_df["Custom_Score"] = (
        0.7 * candidates_df["Skill_Match_Percentage"] / 100 +  # 70% weight for skill match
        0.2 * candidates_df["Success_Rate"] / 100 +            # 20% weight for success rate
        0.1 * candidates_df["Experience"] / 10                 # 10% weight for experience
    )
    
    # Apply workload penalty for high workload employees
    candidates_df.loc[candidates_df["Current_Workload"] > 3, "Custom_Score"] *= 0.8
    
    # Also prepare features for the ML model
    X_candidates = candidates_df[X.columns].copy()
    
    # Apply the same feature weighting as in training
    X_candidates_weighted = X_candidates.copy()
    for feature, weight in feature_weights.items():
        if feature in X_candidates.columns:
            for i in range(int(weight) - 1):
                X_candidates_weighted[f"{feature}_{i}"] = X_candidates[feature]
    
    # Get model predictions
    try:
        # Use model prediction probabilities
        employee_probs = model.predict_proba(X_candidates_weighted)
        model_scores = np.max(employee_probs, axis=1)
        
        # Calculate final score: 70% custom score + 30% model score
        candidates_df["Final_Score"] = 0.7 * candidates_df["Custom_Score"] + 0.3 * model_scores
    except:
        # If model prediction fails, use only the custom score
        print("Warning: Model prediction failed, using custom score only")
        candidates_df["Final_Score"] = candidates_df["Custom_Score"]
    
    # Sort by final score
    candidates_df = candidates_df.sort_values("Final_Score", ascending=False)
    
    # Return top N candidates with scores
    return candidates_df[["Employee id", "Name", "Skills", "Skill_Match_Percentage", 
                         "Success_Rate", "Experience", "Current_Workload", 
                         "Final_Score"]].head(top_n)

# Function to assign optimal employees to all tasks
def assign_employees_to_all_tasks():
    """Assign the best employee to each task based on prioritized matching and workload balancing"""
    assignments = {}
    
    # Create a workload dictionary using Employee ID as keys
    current_workloads = {emp_id: workload for emp_id, workload in 
                        zip(employees["Employee id"], employees["Workload"])}
    
    # Sort tasks by priority (if available)
    if "PriorityEncoded" in tasks.columns:
        sorted_tasks = tasks.sort_values("PriorityEncoded", ascending=False)
    else:
        sorted_tasks = tasks
    
    for _, task in sorted_tasks.iterrows():
        task_id = task["Task id"]
        
        # Get recommendations with current workloads
        recommendations = recommend_employee_for_task(task_id, top_n=10, current_workloads=current_workloads)
        
        if isinstance(recommendations, str):
            assignments[task_id] = recommendations
            continue
        
        # Find the first available employee with workload <= 3
        assigned = False
        
        # First, try to find employees with workload <= 3
        low_workload_employees = recommendations[recommendations["Current_Workload"] <= 3]
        
        if not low_workload_employees.empty:
            # We have employees with low workload, use them
            for _, rec in low_workload_employees.iterrows():
                emp_id = rec["Employee id"]
                assignments[task_id] = {
                    "Employee id": emp_id,
                    "Name": rec["Name"],
                    "Skills": rec["Skills"],
                    "Match_Percentage": f"{rec['Skill_Match_Percentage']:.1f}%",
                    "Success_Rate": f"{rec['Success_Rate']}%",
                    "Experience": rec["Experience"],
                    "Workload": rec["Current_Workload"],
                    "Score": f"{rec['Final_Score']:.3f}"
                }
                # Update workload
                current_workloads[emp_id] += 1
                assigned = True
                break
        
        # If no low workload employee was assigned, use any available employee
        if not assigned:
            for _, rec in recommendations.iterrows():
                emp_id = rec["Employee id"]
                assignments[task_id] = {
                    "Employee id": emp_id,
                    "Name": rec["Name"],
                    "Skills": rec["Skills"],
                    "Match_Percentage": f"{rec['Skill_Match_Percentage']:.1f}%",
                    "Success_Rate": f"{rec['Success_Rate']}%",
                    "Experience": rec["Experience"],
                    "Workload": rec["Current_Workload"],
                    "Score": f"{rec['Final_Score']:.3f}"
                }
                # Update workload
                current_workloads[emp_id] += 1
                assigned = True
                break
                
        if not assigned:
            assignments[task_id] = "No available employees"
    
    return assignments

# Example usage:
# Get recommendations for a specific task
sample_task_id = tasks["Task id"].iloc[0]
recommendations = recommend_employee_for_task(sample_task_id)
print(f"\nTop recommendations for task {sample_task_id}:")
print(recommendations)

# Get assignments for all tasks
all_assignments = assign_employees_to_all_tasks()
print("\nAssignments for all tasks:")
for task_id, assignment in all_assignments.items():
    print(f"Task {task_id}: {assignment}")

# Print task details for reference
print("\nTask Details:")
for _, task in tasks.iterrows():
    print(f"Task {task['Task id']}: Skills = {task['Skills']}")