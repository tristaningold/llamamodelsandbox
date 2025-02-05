import dash
import dash_table
from dash.dependencies import Input, Output
import pandas as pd
from flask import jsonify

# Load the processed output from the llm_output_processor.py script
processed_output = pd.read_json('processed_output.json')

# Create a Dash app
app = dash.Dash(__name__)

# Define the layout of the app
app.layout = dash_table.DataTable(
    id='data-table',
    columns=[{'name': i, 'id': i} for i in processed_output.columns],
    data=processed_output.to_dict('records')
)

# Add a REST API endpoint at /api/data
@app.server.route('/api/data', methods=['GET'])
def get_data():
    return jsonify(processed_output.to_dict('records'))

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
