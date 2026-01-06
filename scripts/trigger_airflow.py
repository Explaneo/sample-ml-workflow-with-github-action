import requests
import os
import sys
import json
from datetime import datetime, timezone

# ------------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------------
AIRFLOW_URL = os.getenv("AIRFLOW_URL").strip('/') 
USERNAME = os.getenv("AIRFLOW_USER")
PASSWORD = os.getenv("AIRFLOW_PASS")
DAG_ID = os.getenv("DAG_ID", "github_ec2_ml_training")
GIT_HASH = sys.argv[1] if len(sys.argv) > 1 else "main"

if not AIRFLOW_URL:
    print("❌ Error: AIRFLOW_URL env var is missing")
    sys.exit(1)

# Headers essentiels pour Ngrok et l'API
HEADERS = {
    "Content-Type": "application/json",
    "ngrok-skip-browser-warning": "true"  # INDISPENSABLE pour Ngrok
}

def trigger_dag():
    """
    Déclenche le DAG en utilisant l'API v2 (Standard Airflow 3)
    """
    # MODIFICATION ICI : v1 -> v2
    trigger_url = f"{AIRFLOW_URL}/api/v2/dags/{DAG_ID}/dagRuns"
    
    print(f"🚀 Triggering DAG: {DAG_ID} at {trigger_url}")
    
    # Payload avec Git Hash
    payload = {
        "conf": {"git_hash": GIT_HASH},
        "note": f"Triggered via GitHub Actions - SHA: {GIT_HASH[:7]}"
    }

    try:
        response = requests.post(
            trigger_url, 
            json=payload, 
            auth=(USERNAME, PASSWORD), 
            headers=HEADERS, 
            timeout=15
        )
        
        # Gestion des erreurs spécifique à la v2
        if response.status_code == 404:
            print("❌ Error 404: DAG not found. Verify DAG_ID is correct and DAG is unpaused.")
        elif response.status_code == 405:
            print("❌ Error 405: Method Not Allowed. Confirm Airflow 3 is using /api/v2/.")
        
        response.raise_for_status()
        
        data = response.json()
        # Dans l'API v2, le champ peut être 'dag_run_id' ou 'run_id'
        run_id = data.get('dag_run_id') or data.get('run_id')
        print(f"✅ Success! Run ID: {run_id}")
        
    except Exception as e:
        print(f"❌ Trigger Failed: {e}")
        if 'response' in locals():
            print(f"Response Body: {response.text}")
        sys.exit(1)

if __name__ == "__main__":
    trigger_dag()
