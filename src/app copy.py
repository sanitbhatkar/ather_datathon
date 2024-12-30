import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Enable Streamlit's dynamic update
st.set_page_config(page_title="EDA Dashboard", layout="wide")
st.experimental_set_query_params(update="true")

# Helper Functions

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

def parse_ot_to_hours(ot_column):
    """
    Convert OT time in HH:MM format to hours as a float, with robust error handling.

    Args:
        ot_column (pd.Series): Series containing OT time in HH:MM format.

    Returns:
        pd.Series: Series with OT time converted to hours or NaN for invalid values.
    """
    def parse_time(value):
        try:
            # Check if value is a valid HH:MM format
            parts = value.split(':')
            if len(parts) == 2:
                hours, minutes = int(parts[0]), int(parts[1])
                return hours + minutes / 60
            return 0  # Default to 0 for invalid formats
        except Exception:
            return 0  # Default to 0 for unexpected errors

    return ot_column.apply(parse_time)

def analyze_vertical_department_ot_data(att_dict, vertical="All", department="All", function="All"):
    """
    Analyze OT data grouped by Vertical and Department with filtering.

    Args:
        att_dict (dict): Dictionary where keys are dates and values are DataFrames.
        vertical (str): Selected vertical filter. Default is "All".
        department (str): Selected department filter. Default is "All".
        function (str): Selected function filter. Default is "All".

    Returns:
        tuple: (overall_ot_summary, department_monthly_ot_summary)
    """
    ot_data = []

    for date in att_dict:
        df = pd.DataFrame(att_dict[date])
        parsed_date = datetime.strptime(date.split('(')[0], "%d-%b-%Y")
        df['Date'] = parsed_date
        df['OT_Hours'] = parse_ot_to_hours(df['OT'])
        ot_data.append(df[['Vertical', 'Department', 'Function', 'Date', 'OT_Hours']])

    combined_ot_df = pd.concat(ot_data)

    # Apply filters
    if vertical != "All":
        combined_ot_df = combined_ot_df[combined_ot_df['Vertical'] == vertical]
    if department != "All":
        combined_ot_df = combined_ot_df[combined_ot_df['Department'] == department]
    if function != "All":
        combined_ot_df = combined_ot_df[combined_ot_df['Function'] == function]

    if combined_ot_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    combined_ot_df['Month'] = combined_ot_df['Date'].dt.to_period('M').astype(str)

    overall_ot_summary = combined_ot_df.groupby(['Vertical', 'Department'])['OT_Hours'].sum().reset_index()
    department_monthly_ot_summary = combined_ot_df.groupby(['Month', 'Vertical', 'Department'])['OT_Hours'].sum().reset_index()

    return overall_ot_summary, department_monthly_ot_summary


def plot_vertical_department_ot_clean(overall_summary, monthly_summary, vertical, department="All", function="All"):
    """
    Create a 1x2 grid of plots for each Vertical showing overall and monthly OT data by Department.
    Filters can be applied for specific departments and functions.

    Args:
        overall_summary (pd.DataFrame): DataFrame with overall OT hours grouped by Vertical and Department.
        monthly_summary (pd.DataFrame): DataFrame with monthly OT hours grouped by Month, Vertical, and Department.
        vertical (str): Selected vertical to filter data.
        department (str): Selected department to filter data (default: "All").
        function (str): Selected function to filter data (default: "All").
    """
    overall_data = overall_summary[overall_summary['Vertical'] == vertical]
    monthly_data = monthly_summary[monthly_summary['Vertical'] == vertical]

    # Filter by department if specified
    if department != "All":
        overall_data = overall_data[overall_data['Department'] == department]
        monthly_data = monthly_data[monthly_data['Department'] == department]

    # Check if Function column exists and filter by function if specified
    if 'Function' in overall_data.columns and function != "All":
        overall_data = overall_data[overall_data['Function'] == function]
        monthly_data = monthly_data[monthly_data['Function'] == function]

    if overall_data.empty or monthly_data.empty:
        st.warning(f"No OT data found for the selected filters: Vertical={vertical}, Department={department}, Function={function}")
        return

    # Create a common color palette based on the departments
    departments = overall_data['Department'].unique()
    palette = sns.color_palette("tab10", len(departments))
    color_dict = {department: palette[i] for i, department in enumerate(departments)}

    # Create the 1x2 grid of plots
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), gridspec_kw={'width_ratios': [1, 1]})
    fig.suptitle(f"OT Hours for {vertical} - {department} - {function}", fontsize=16)

    # Overall Bar Chart
    sns.barplot(
        data=overall_data,
        x="Department",
        y="OT_Hours",
        ax=axes[0],
        palette=color_dict,
        errorbar=None,
    )
    axes[0].set_title("Overall OT Hours by Department")
    axes[0].set_xticks([])  # Remove x-ticks
    axes[0].set_ylabel("Total OT Hours")

    # Monthly Line Plot
    sns.lineplot(
        data=monthly_data,
        x="Month",
        y="OT_Hours",
        hue="Department",
        style="Department",
        ax=axes[1],
        palette=color_dict,
        markers=True
    )
    axes[1].set_title("Monthly OT Hours by Department")
    axes[1].set_xticklabels(monthly_data['Month'].unique(), rotation=45)
    axes[1].set_ylabel("Total OT Hours")
    axes[1].set_xlabel("Month")

    # Common Legend
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=5, title="Departments", bbox_to_anchor=(0.5, -0.05))

    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    st.pyplot(fig)



def update_filters(att_dict, selected_vertical, selected_department):
    """
    Dynamically update the department and function filters based on the selected vertical and department.

    Args:
        att_dict (dict): Dictionary where keys are dates and values are DataFrames.
        selected_vertical (str): Currently selected vertical filter.
        selected_department (str): Currently selected department filter.

    Returns:
        tuple: Updated lists of departments and functions.
    """
    all_data = pd.concat([pd.DataFrame(data) for data in att_dict.values()])

    if selected_vertical != "All":
        all_data = all_data[all_data['Vertical'] == selected_vertical]

    departments = ["All"] + sorted(all_data['Department'].dropna().unique())

    if selected_department != "All":
        all_data = all_data[all_data['Department'] == selected_department]

    functions = ["All"] + sorted(all_data['Function'].dropna().unique())

    return departments, functions



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

def plot_combined_metrics(total_counts, attendance_x, attendance_pp, ot_data, vertical, department):
    """
    Plot combined metrics (Total Count, X, PP, and OT) in a single chart.

    Args:
        total_counts (pd.DataFrame): Total attendance counts by month.
        attendance_x (pd.DataFrame): Attendance "X" counts by month.
        attendance_pp (pd.DataFrame): Attendance "PP" counts by month.
        ot_data (pd.DataFrame): OT hours by month.
        vertical (str): Vertical being analyzed.
        department (str): Department being analyzed.
    """
    if total_counts is None or attendance_x is None or attendance_pp is None or ot_data is None:
        st.warning("No data to plot.")
        return

    # Normalize data
    total_counts['Total_Normalized'] = total_counts['Total_Count'] / total_counts['Total_Count'].max()
    attendance_x['X_Normalized'] = attendance_x['X_Count'] / attendance_x['X_Count'].max()
    attendance_pp['PP_Normalized'] = attendance_pp['PP_Count'] / attendance_pp['PP_Count'].max()
    ot_data['OT_Normalized'] = ot_data['OT_Hours'] / ot_data['OT_Hours'].max()

    # Plot
    fig, ax1 = plt.subplots(figsize=(16, 8))

    ax1.plot(total_counts['Month'], total_counts['Total_Normalized'], label='Total Count (Normalized)', marker='o', color='blue')
    ax1.plot(attendance_x['Month'], attendance_x['X_Normalized'], label='X (Normalized)', marker='s', color='green')
    ax1.plot(attendance_pp['Month'], attendance_pp['PP_Normalized'], label='PP (Normalized)', marker='^', color='orange')
    ax1.set_ylabel("Normalized Count", fontsize=12, color='blue')
    ax1.set_xlabel("Month", fontsize=12)
    ax1.set_xticks(range(len(total_counts['Month'])))
    ax1.set_xticklabels(total_counts['Month'], rotation=45)
    ax1.legend(loc='upper left')

    ax2 = ax1.twinx()
    ax2.plot(ot_data['Month'], ot_data['OT_Normalized'], label='OT (Normalized)', marker='d', color='red')
    ax2.set_ylabel("OT (Normalized)", fontsize=12, color='red')
    ax2.legend(loc='upper right')

    plt.title(f"Combined Metrics for {vertical} - {department}", fontsize=16)
    plt.tight_layout()
    st.pyplot(fig)



def main():
    st.title("EDA Dashboard for Attendance and OT Analysis")

    # Load the data
    uploaded_file = st.file_uploader("Upload Cleaned Attendance Data (Pickle File):", type="pkl")
    if uploaded_file is not None:
        att_dict = load_pickle_to_dict(uploaded_file)
        st.success("Data successfully loaded.")

        # Sidebar filters
        vertical = st.sidebar.selectbox("Select Vertical:", options=["All"] + sorted(
            pd.concat([pd.DataFrame(data) for data in att_dict.values()])['Vertical'].dropna().unique()
        ))

        departments, functions = update_filters(att_dict, vertical, "All")
        department = st.sidebar.selectbox("Select Department:", options=departments)

        _, functions = update_filters(att_dict, vertical, department)
        function = st.sidebar.selectbox("Select Function:", options=functions)

        # Attendance Analysis
        st.header("Attendance Analysis")
        attendance_data = prepare_attendance_data_by_filters(att_dict, vertical, department, function)
        if not attendance_data.empty:
            pivoted_data = attendance_data.pivot(index='Month', columns='Attendance', values='Count').fillna(0)
            st.bar_chart(pivoted_data)

        # Combined Metrics Analysis
        st.header("Combined Metrics Analysis")
        total_counts, attendance_x, attendance_pp, ot_data = prepare_combined_data(att_dict, vertical, department)
        plot_combined_metrics(total_counts, attendance_x, attendance_pp, ot_data, vertical, department)

        # OT Data Analysis
        st.header("OT Data Analysis")
        overall_ot_summary, department_monthly_ot_summary = analyze_vertical_department_ot_data(att_dict)
        plot_vertical_department_ot_clean(overall_ot_summary, department_monthly_ot_summary, vertical, department, function)



if __name__ == "__main__":
    main()
