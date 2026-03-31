import great_expectations as gx
import pandas as pd
import os
import shutil


def validate_data(df):

    print("Running Great Expectations validation...")
    df = df.dropna(subset=["Temp"])


    # Create GX dataframe
    gx_df = gx.from_pandas(df)

    # Expectations
    gx_df.expect_column_to_exist("HR")

    gx_df.expect_column_values_to_not_be_null(
    "HR",
    mostly=0.9
    )

    gx_df.expect_column_values_to_be_between(
    "HR",
    min_value=20,
    max_value=220,
    mostly=0.9
    )

    gx_df.expect_column_values_to_not_be_null(
    "Temp",
    mostly=0.7
    )

    gx_df.expect_column_values_to_be_between(
    "Temp",
    min_value=30,
    max_value=45,
    mostly=0.9
    )

    # Run validation
    validation_result = gx_df.validate()

    print("Validation Result:")
    print(validation_result["statistics"])

    # Build HTML report
    generate_html_report(validation_result)

    return validation_result

def generate_html_report(validation_result):

    # FIX: define status here
    status = "Succeeded" if validation_result["success"] else "Failed"

    html_content = f"""
    <html>
    <head>
        <title>Great Expectations Validation Report</title>
        <style>
            body {{
                font-family: Arial;
                margin:40px;
                background:#f5f6fa;
            }}
            h1 {{
                color:#2f3640;
            }}
            .card {{
                background:white;
                padding:20px;
                margin-bottom:20px;
                border-radius:10px;
                box-shadow:0 2px 6px rgba(0,0,0,0.1);
            }}
            .success {{
                color:green;
                font-weight:bold;
            }}
            .failed {{
                color:red;
                font-weight:bold;
            }}
            table {{
                width:100%;
                border-collapse:collapse;
            }}
            th,td {{
                border:1px solid #ddd;
                padding:8px;
            }}
            th {{
                background:#273c75;
                color:white;
            }} 
        </style>
    </head>

    <body>

    <h1>Expectation Validation Result</h1>

    <div class="card">
    <h2>Overview</h2>
    <p>Status: <span class="success">{status}</span></p>
    </div>

    <div class="card">
    <h2>Statistics</h2>

    <table>
    <tr>
        <th>Evaluated Expectations</th>
        <th>Successful Expectations</th>
        <th>Unsuccessful Expectations</th>
        <th>Success Percent</th>
    </tr>

    <tr>
        <td>{validation_result['statistics']['evaluated_expectations']}</td>
        <td>{validation_result['statistics']['successful_expectations']}</td>
        <td>{validation_result['statistics']['unsuccessful_expectations']}</td>
        <td>{validation_result['statistics']['success_percent']}%</td>
    </tr>

    </table>
    </div>

    <div class="card">
    <h2>Expectation Details</h2>
    <table>
    <tr>
        <th>Expectation</th>
        <th>Status</th>
    </tr>
    """


    for result in validation_result["results"]:
        expectation = result["expectation_config"]["expectation_type"]
        status = "Succeeded" if result["success"] else "Failed"

        html_content += f"""
        <tr>
        <td>{expectation}</td>
        <td>{status}</td>
        </tr>
        """

    html_content += """
    </table>
    </div>

    </body>
    </html>
    """

    os.makedirs("reports", exist_ok=True)

    with open("reports/data_validation_report.html", "w") as f:
        f.write(html_content)

    print("HTML Validation Report Generated")
