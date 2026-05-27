# dags/common/gx_validator.py
import pandas as pd
import great_expectations as gx

def validate_source_csv(file_path: str):
    """
    Validates the source CSV using Great Expectations.
    Returns a lightweight summary safe for Airflow XCom.
    """
    df = pd.read_csv(file_path)
    ge_df = gx.from_pandas(df)

    ge_df.expect_column_to_exist("instant")
    ge_df.expect_column_to_exist("dteday")
    ge_df.expect_column_to_exist("cnt")
    ge_df.expect_column_to_exist("temp")
    ge_df.expect_column_values_to_be_between("cnt", min_value=0, max_value=1000)
    ge_df.expect_column_values_to_be_between("temp", min_value=0, max_value=1)
    ge_df.expect_column_values_to_not_be_null("instant")
    ge_df.expect_column_values_to_be_unique("instant")

    results = ge_df.validate()

    return {
        "success":                  bool(results["success"]),
        "evaluated_expectations":   len(results["results"]),
        "successful_expectations":  sum(1 for r in results["results"] if r["success"]),
        "failed_expectations":      sum(1 for r in results["results"] if not r["success"]),
    }