# -----------------------------------------
# Fake HR tools (simulate real systems)
# -----------------------------------------

def get_leave_balance(employee_id: str):
    """
    Simulates fetching leave balance from HR system
    """

    # dummy data (in real life → DB / API)
    fake_db = {
        "E001": {"annual": 12, "sick": 5},
        "E002": {"annual": 8, "sick": 2}
    }

    if employee_id in fake_db:
        return fake_db[employee_id]

    return {"error": "Employee not found"}
