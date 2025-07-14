import requests
import os
import pandas as pd
import io

def get_hierarchy(model_id: int, auth_header: str) -> pd.DataFrame:
    # Use the passed auth header for authentication
    if not auth_header:
        raise ValueError("Authentication header is required")
    
    header = {
        'Authorization': auth_header,
        'Content-Type': 'application/json'
    }
    
    response = requests.post(
        f'{os.environ.get("VENA_ENDPOINT")}/api/models/{model_id}/etl/query/hierarchies',
        headers=header,
        json={
            "destination": "ToCSV",
            "exportMemberIds": True,
            "queryString": None
        },
        stream=True
    )
    if response.status_code == 204 or response.status_code == 200:
        content = response.content
        return pd.read_csv(io.BytesIO(content))
    else:
        raise Exception("Failed to retrieve hierarchy CSV due to: " + response.text)
    
