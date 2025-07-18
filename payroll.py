import pandas as pd

# Dummy sample
employees = [
    {"name": "John", "hours": 160, "rate": 100, "deductions": 5000}
]

for emp in employees:
    emp["net_pay"] = (emp["hours"] * emp["rate"]) - emp["deductions"]

df = pd.DataFrame(employees)
df.to_csv("payroll.csv", index=False)

print("✅ Payroll CSV exported.")
