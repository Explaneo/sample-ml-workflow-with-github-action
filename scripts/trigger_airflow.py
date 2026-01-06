import requests
import os
import sys
import json
from datetime import datetime, timezone

# ------------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------------
# .strip('/') retire les slashs en trop à la fin de l'URL pour éviter les doubles slashs // dans l'endpoint
AIRFLOW_URL = os.getenv("AIRFLOW_URL").strip('/')  
USERNAME = os.getenv("AIRFLOW_USER")
PASSWORD = os.getenv("AIRFLOW_PASS")
DAG_ID = os.getenv("DAG_ID", "github_ec2_ml_training")
GIT_HASH = sys.argv[1] if len(sys.argv) > 1 else "main"

if not AIRFLOW_URL:
    print("❌ Error: AIRFLOW_URL env var is missing")
    sys.exit(1)

# Headers essentiels pour Ngrok et l'API v2
HEADERS = {
    "Content-Type": "application/json",
    "Accept": "application/json",
    "ngrok-skip-browser-warning": "true"  # Bypasse la page d'accueil Ngrok
}

def trigger_dag():
    """
    Déclenche le DAG en utilisant l'API v2 (Obligatoire pour Airflow 3)
    """
    # CORRECTION : Passage de v1 à v2
    trigger_url = f"{AIRFLOW_URL}/api/v2/dags/{DAG_ID}/dagRuns"
    
    print(f"🚀 Triggering DAG: {DAG_ID} at {trigger_url}")
    
    # Payload compatible Airflow 3
    payload = {
        "conf": {"git_hash": GIT_HASH},
        "note": f"Triggered via GitHub Actions - SHA: {GIT_HASH[:7]}"
    }

    try:
        # L'argument auth=(USERNAME, PASSWORD) gère le header Authorization: Basic
        response = requests.post(
            trigger_url, 
            json=payload, 
            auth=(USERNAME, PASSWORD), 
            headers=HEADERS, 
            timeout=15
        )
        
        # Gestion des codes retour spécifiques à Airflow 3
        if response.status_code == 404:
            print(f"❌ Error 404: DAG '{DAG_ID}' not found. Check if the DAG is active in Airflow.")
            sys.exit(1)
        elif response.status_code == 405:
            print("❌ Error 405: Method Not Allowed. Confirm you are hitting the /api/v2/ endpoint.")
            sys.exit(1)
        elif response.status_code == 401:
            print("❌ Error 401: Unauthorized. Check your AIRFLOW_USER and AIRFLOW_PASS.")
            sys.exit(1)
        
        # Soulève une exception pour les autres codes 4xx/5xx
        response.raise_for_status()
        
        data = response.json()
        # Note : Dans l'API v2, le champ peut être 'dag_run_id' ou simplement 'run_id'
        run_id = data.get('dag_run_id') or data.get('run_id')
        print(f"✅ Success! Run ID: {run_id}")
        
    except Exception as e:
        print(f"❌ Trigger Failed: {e}")
        if 'response' in locals():
            print(f"Response Body: {response.text}")
        sys.exit(1)

if __name__ == "__main__":
    trigger_dag()
