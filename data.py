import pandas as pd
Employees = {
    "Employee id":["E001", "E002","E003"],
    "Name": ["Lolo", "Elton","Muntu"],
    "Skill":["python, sql", "html,css", "javascript"],
    "Experience":[3,4,2],
    "Success rate":[90,70,80]
}

Tasks = {
    "Task id":['T101',"T102",'T103'],
    "Skills": ["python,sql","html,sql","javascript"],
    "Complexity": ["high","medium",'high'],
    "Priority level": ["high","medium","high"]
}

df = pd.DataFrame(Employees)
dx = pd.DataFrame(Tasks)

df.to_csv("Employee.csv", index= False)
dx.to_csv("Tasks.csv", index= False)

print("csv file created successfully")