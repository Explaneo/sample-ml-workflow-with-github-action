import requests
import os
import sys
import base64
import time

# --- CONFIGURATION ---
URL_BASE = os.getenv("AIRFLOW_URL").strip().rstrip('/')
USER = os.getenv("AIRFLOW_USER")
PASS = os.getenv("AIRFLOW_PASS")
DAG_ID = "github_ec2_ml_training"
GIT_HASH = sys.argv[1] if len(sys.argv) > 1 else "main"

def trigger_dag():
    # URL V2 DIRECTE
    trigger_url = f"{URL_BASE}/api/v2/dags/{DAG_ID}/dagRuns"
    
    # Encodage Basic Auth (User:Pass)
    auth_str = base64.b64encode(f"{USER}:{PASS}".encode()).decode()

    headers = {
        "Authorization": f"Basic {auth_str}",
        "Content-Type": "application/json",
        "ngrok-skip-browser-warning": "true"
    }
    
    # ID de run unique pour éviter les doublons (409)
    run_id = f"github_{int(time.time())}"
    
    payload = {
        "dag_run_id": run_id,
        "conf": {"git_hash": GIT_HASH}
    }

    print(f"🚀 Appel direct de l'API v2 : {trigger_url}")

    try:
        response = requests.post(trigger_url, json=payload, headers=headers, timeout=30)
        
        if response.status_code in [200, 201]:
            print(f"✅ SUCCÈS ! DAG lancé avec l'ID : {run_id}")
        else:
            print(f"❌ ÉCHEC {response.status_code}")
            print(f"Détail : {response.text}")
            sys.exit(1)
            
    except Exception as e:
        print(f"💥 Erreur de connexion : {e}")
        sys.exit(1)

if __name__ == "__main__":
    trigger_dag()
