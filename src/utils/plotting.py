import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

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
