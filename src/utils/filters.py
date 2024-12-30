import pandas as pd

def update_filters(att_dict, selected_vertical, selected_department):
    """Dynamically update filters for Department and Function."""
    all_data = pd.concat([pd.DataFrame(data) for data in att_dict.values()])

    if selected_vertical != "All":
        all_data = all_data[all_data['Vertical'] == selected_vertical]

    departments = ["All"] + sorted(all_data['Department'].dropna().unique())

    if selected_department != "All":
        all_data = all_data[all_data['Department'] == selected_department]

    functions = ["All"] + sorted(all_data['Function'].dropna().unique())

    return departments, functions
