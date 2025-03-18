import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

employees = pd.read_csv("employees.csv")
tasks = pd.read_csv("tasks.csv")

label_encoder = LabelEncoder()
employees["Skills"] = employees["Skills"].apply(lambda x: x.split(','))
employees = employees.explode("Skills")
employees["Skills"] = label_encoder.fit_transform(employees["Skills"])

tasks["Skills"] = tasks["Skills"].apply(lambda x: x.split(','))
tasks = tasks.explode("Skills")
tasks["skills"] = label_encoder.fit_transform(tasks["Skills"])

merged_data = pd.merge(tasks, employees, left_on = "Skills", right_on = "Skills")

x = merged_data[["Skill","Experience","Workload","Success rate","Complexity","Priority level"]]
y = merged_data["Employee id"]

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.2, random_state = 42)
model = RandomForestClassifier(n_estimators = 100, random_state = 42)
model.fit(x_train,y_train)

accuracy = model.score(x_test,y_test)
print(f"Model Accuracy: {accuracy*100: .2f}%")

new_task = [[0,4,1,87,"high", "medium"]]
predicted_employee = model.predict(new_task)

print(f"Recommended Employee id: {predicted_employee[0]}")