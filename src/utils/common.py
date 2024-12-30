def parse_ot_to_hours(ot_column):
    """
    Convert OT time in HH:MM format to hours as a float, with robust error handling.

    Args:
        ot_column (pd.Series): Series containing OT time in HH:MM format.

    Returns:
        pd.Series: Series with OT time converted to hours or 0 for invalid values.
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
