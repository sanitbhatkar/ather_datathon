import streamlit as st
from utils.plotting import plot_combined_metrics, plot_vertical_department_ot_clean
from utils.filters import update_filters
import pandas as pd
from utils.data_processing import (
    # load_pickle_to_dict,
    load_excel_to_dict,
    prepare_attendance_data_by_filters,
    prepare_combined_data,
    analyze_vertical_department_ot_data
)

def main():
    st.title("EDA Dashboard for Attendance and OT Analysis")

    # Load the Excel file
    uploaded_file = st.file_uploader("Upload Cleaned Attendance Data (Excel File):", type="xlsx")
    if uploaded_file is not None:
        att_dict = load_excel_to_dict(uploaded_file)
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
        if vertical != "All":
            plot_vertical_department_ot_clean(overall_ot_summary, department_monthly_ot_summary, vertical, department, function)

if __name__ == "__main__":
    main()


