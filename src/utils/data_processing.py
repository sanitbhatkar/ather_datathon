import pickle
import pandas as pd
from datetime import datetime
from utils.common import parse_ot_to_hours

def load_pickle_to_dict(uploaded_file):
    """
    Load a pickle file into a dictionary.

    Args:
        uploaded_file (UploadedFile): Uploaded pickle file from Streamlit.

    Returns:
        dict: Loaded dictionary.
    """
    import pickle
    return pickle.load(uploaded_file)

def prepare_attendance_data_by_filters(att_dict, vertical, department, function):
    """
    Filter attendance data by Vertical, Department, and Function.

    Args:
        att_dict (dict): Dictionary containing attendance data.
        vertical (str): Selected vertical.
        department (str): Selected department.
        function (str): Selected function.

    Returns:
        pd.DataFrame: Filtered and aggregated attendance data.
    """
    attendance_data = []
    for date in att_dict:
        df = pd.DataFrame(att_dict[date])
        df['Date'] = pd.to_datetime(date.split('(')[0], format="%d-%b-%Y")
        attendance_data.append(df[['Vertical', 'Department', 'Function', 'Attendance', 'Date']])

    all_data = pd.concat(attendance_data)

    # Apply filters
    if vertical != "All":
        all_data = all_data[all_data['Vertical'] == vertical]
    if department != "All":
        all_data = all_data[all_data['Department'] == department]
    if function != "All":
        all_data = all_data[all_data['Function'] == function]

    if 'Date' in all_data:
        all_data['Month'] = all_data['Date'].dt.to_period('M').astype(str)
        grouped_data = all_data.groupby(['Month', 'Attendance']).size().reset_index(name='Count')
    else:
        grouped_data = pd.DataFrame()

    return grouped_data

def prepare_combined_data(att_dict, vertical, department):
    """
    Prepare combined metrics for Total Count, 'X', 'PP', and OT data.

    Args:
        att_dict (dict): Dictionary containing attendance data.
        vertical (str): Selected vertical.
        department (str): Selected department.

    Returns:
        tuple: DataFrames for Total Count, 'X', 'PP', and OT data.
    """
    combined_data = []
    for date in att_dict:
        df = pd.DataFrame(att_dict[date])
        df['Date'] = pd.to_datetime(date.split('(')[0], format="%d-%b-%Y")
        combined_data.append(df[['Vertical', 'Department', 'Attendance', 'OT', 'Date']])

    combined_df = pd.concat(combined_data)
    combined_df['Month'] = combined_df['Date'].dt.to_period('M').astype(str)

    # Filter data
    if vertical != "All":
        combined_df = combined_df[combined_df['Vertical'] == vertical]
    if department != "All":
        combined_df = combined_df[combined_df['Department'] == department]

    combined_df['OT_Hours'] = parse_ot_to_hours(combined_df['OT'])

    # Total Counts excluding 'X', 'NANA', 'NJ'
    total_counts = combined_df[~combined_df['Attendance'].isin(['X', 'NANA', 'NJ'])].groupby('Month').size().reset_index(name='Total_Count')

    # 'X' Attendance
    attendance_x = combined_df[combined_df['Attendance'] == 'X'].groupby('Month').size().reset_index(name='X_Count')

    # 'PP' Attendance
    attendance_pp = combined_df[combined_df['Attendance'] == 'PP'].groupby('Month').size().reset_index(name='PP_Count')

    # OT Data
    ot_data = combined_df.groupby('Month')['OT_Hours'].sum().reset_index()

    return total_counts, attendance_x, attendance_pp, ot_data

def analyze_vertical_department_ot_data(att_dict):
    """
    Analyze OT data grouped by Vertical and Department.

    Args:
        att_dict (dict): Dictionary where keys are dates and values are DataFrames.

    Returns:
        tuple: (overall_ot_summary, department_monthly_ot_summary)
            - overall_ot_summary: Total OT hours grouped by Vertical and Department.
            - department_monthly_ot_summary: Total OT hours grouped by Month, Vertical, and Department.
    """
    ot_data = []

    for date in att_dict:
        df = pd.DataFrame(att_dict[date])
        # Parse date and calculate OT hours
        df['Date'] = pd.to_datetime(date.split('(')[0], format="%d-%b-%Y")
        df['OT_Hours'] = parse_ot_to_hours(df['OT'])
        ot_data.append(df[['Vertical', 'Department', 'Date', 'OT_Hours']])

    combined_ot_df = pd.concat(ot_data)
    combined_ot_df['Month'] = combined_ot_df['Date'].dt.to_period('M').astype(str)

    # Overall OT Summary (Vertical and Department)
    overall_ot_summary = combined_ot_df.groupby(['Vertical', 'Department'])['OT_Hours'].sum().reset_index()

    # Monthly OT Summary (Vertical and Department)
    department_monthly_ot_summary = combined_ot_df.groupby(['Month', 'Vertical', 'Department'])['OT_Hours'].sum().reset_index()

    return overall_ot_summary, department_monthly_ot_summary
