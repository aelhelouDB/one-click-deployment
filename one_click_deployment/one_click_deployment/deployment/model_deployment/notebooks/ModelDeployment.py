# Databricks notebook source
##################################################################################
# Model Deployment/Serving Notebook
#
# This notebook deploys a validated model from the model registry to a serving endpoint
# with comprehensive pre-deployment checks including endpoint validation and model verification.
#
# It runs as part of CD and by an automated model training job -> validation -> deployment job defined under ``one_click_deployment/resources/model-workflow-resource.yml``
#
# This notebook has the following parameters:
#
#  * env (required)                - String name of the current environment for model deployment (test, or prod).
#  * model_uri (required)          - URI of the model to deploy. Must be in the format "models:/<name>/<version-id>"
#  * endpoint_name (optional)      - Name of the serving endpoint. If not provided, defaults to model name.
#  * endpoint_config (optional)    - JSON config for endpoint creation (compute resources, etc.)
##################################################################################

# List of input args needed to run the notebook as a job.
# Provide them via DB widgets or notebook arguments.

# Name of the current environment
dbutils.widgets.dropdown("env", "None", ["None", "test", "prod"], "Environment Name")

# Optional endpoint name (defaults to model name if not provided)
dbutils.widgets.text("endpoint_name", "", "Serving Endpoint Name (optional)")

# Optional endpoint configuration
dbutils.widgets.text("endpoint_config", "{}", "Endpoint Configuration (JSON)")

# COMMAND ----------

import os
import sys
import json
import time
from mlflow.tracking import MlflowClient
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import ServedEntityInput, EndpointCoreConfigInput, TrafficConfig

notebook_path =  '/Workspace/' + os.path.dirname(dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get())
%cd $notebook_path
%cd ..
sys.path.append("../..")

# COMMAND ----------

# Initialize clients
mlflow_client = MlflowClient(registry_uri="databricks-uc")
w = WorkspaceClient()

# Get parameters
model_uri = dbutils.jobs.taskValues.get("Train", "model_uri", debugValue="")
model_name = dbutils.jobs.taskValues.get("Train", "model_name", debugValue="")
model_version = dbutils.jobs.taskValues.get("Train", "model_version", debugValue="")
env = dbutils.widgets.get("env")
endpoint_name = dbutils.widgets.get("endpoint_name")
endpoint_config = dbutils.widgets.get("endpoint_config")

# Validation
assert env != "None", "env notebook parameter must be specified"
assert model_uri != "", "model_uri notebook parameter must be specified"
assert model_name != "", "model_name notebook parameter must be specified"
assert model_version != "", "model_version notebook parameter must be specified"

# Set default endpoint name if not provided
if not endpoint_name:
    # Extract model name from full model path for endpoint naming
    model_base_name = model_name.split(".")[-1]  # Get the last part after the dots
    endpoint_name = f"{model_base_name}-{env}-endpoint"

print(f"Deploying model: {model_uri}")
print(f"Environment: {env}")
print(f"Endpoint name: {endpoint_name}")

# COMMAND ----------

# DBTITLE 1, Pre-deployment Checks
print("=" * 60)
print("RUNNING PRE-DEPLOYMENT CHECKS")
print("=" * 60)

# Check 1: Verify model exists in registry
print("\n1. Verifying model exists in registry...")
try:
    model_version_details = mlflow_client.get_model_version(model_name, model_version)
    print(f"   ✓ Model found: {model_name} version {model_version}")
    print(f"   ✓ Model stage: {model_version_details.current_stage}")
    print(f"   ✓ Model status: {model_version_details.status}")
except Exception as e:
    print(f"   ✗ Model not found: {str(e)}")
    raise Exception(f"Model {model_name} version {model_version} not found in registry")

# Check 2: Verify model has "challenger" alias (passed validation)
print("\n2. Verifying model has passed validation...")
try:
    model_aliases = mlflow_client.get_model_version_by_alias(model_name, "challenger")
    if model_aliases.version == model_version:
        print(f"   ✓ Model version {model_version} has 'challenger' alias")
    else:
        print(f"   ⚠ Warning: Model version {model_version} does not have 'challenger' alias")
        print(f"   Current challenger version: {model_aliases.version}")
except Exception as e:
    print(f"   ⚠ Warning: Could not verify 'challenger' alias: {str(e)}")

# Check 3: Test model loading
print("\n3. Testing model loading...")
try:
    import mlflow.transformers
    test_model = mlflow.transformers.load_model(model_uri)
    print("   ✓ Model loaded successfully")
    
    # Quick inference test
    test_result = test_model("Hello world", max_length=20, num_return_sequences=1, pad_token_id=50256)
    print("   ✓ Model inference test passed")
except Exception as e:
    print(f"   ✗ Model loading failed: {str(e)}")
    raise Exception(f"Model {model_uri} cannot be loaded")

# Check 4: Verify endpoint doesn't already exist or get existing endpoint info
print(f"\n4. Checking serving endpoint '{endpoint_name}'...")
try:
    existing_endpoint = w.serving_endpoints.get(endpoint_name)
    print(f"   ℹ Endpoint '{endpoint_name}' already exists")
    print(f"   Current state: {existing_endpoint.state}")
    endpoint_exists = True
except Exception:
    print(f"   ℹ Endpoint '{endpoint_name}' does not exist - will create new")
    endpoint_exists = False

print("\n" + "=" * 60)
print("PRE-DEPLOYMENT CHECKS COMPLETED")
print("=" * 60)

# COMMAND ----------

# DBTITLE 1, Deploy Model to Serving Endpoint
print("\n🚀 Starting model deployment...")

# Parse endpoint configuration
try:
    config = json.loads(endpoint_config) if endpoint_config != "{}" else {}
except json.JSONDecodeError:
    print("Warning: Invalid endpoint config JSON, using defaults")
    config = {}

# Set default configuration
default_config = {
    "served_entities": [{
        "entity_name": model_name,
        "entity_version": model_version,
        "workload_size": "Small",
        "scale_to_zero_enabled": True
    }],
    "traffic_config": {
        "routes": [{
            "served_model_name": f"{model_name}-{model_version}",
            "traffic_percentage": 100
        }]
    }
}

# Merge with user config
final_config = {**default_config, **config}

if endpoint_exists:
    print(f"Updating existing endpoint '{endpoint_name}'...")
    try:
        # Update the existing endpoint
        w.serving_endpoints.update_config(
            name=endpoint_name,
            served_entities=[
                ServedEntityInput(
                    entity_name=model_name,
                    entity_version=model_version,
                    workload_size=final_config["served_entities"][0].get("workload_size", "Small"),
                    scale_to_zero_enabled=final_config["served_entities"][0].get("scale_to_zero_enabled", True)
                )
            ],
            traffic_config=TrafficConfig(
                routes=[
                    {
                        "served_model_name": f"{model_name}-{model_version}",
                        "traffic_percentage": 100
                    }
                ]
            )
        )
        print(f"   ✓ Endpoint update initiated")
        
    except Exception as e:
        print(f"   ✗ Failed to update endpoint: {str(e)}")
        raise
else:
    print(f"Creating new endpoint '{endpoint_name}'...")
    try:
        # Create new endpoint
        w.serving_endpoints.create(
            name=endpoint_name,
            config=EndpointCoreConfigInput(
                served_entities=[
                    ServedEntityInput(
                        entity_name=model_name,
                        entity_version=model_version,
                        workload_size=final_config["served_entities"][0].get("workload_size", "Small"),
                        scale_to_zero_enabled=final_config["served_entities"][0].get("scale_to_zero_enabled", True)
                    )
                ],
                traffic_config=TrafficConfig(
                    routes=[
                        {
                            "served_model_name": f"{model_name}-{model_version}",
                            "traffic_percentage": 100
                        }
                    ]
                )
            )
        )
        print(f"   ✓ Endpoint creation initiated")
        
    except Exception as e:
        print(f"   ✗ Failed to create endpoint: {str(e)}")
        raise

# COMMAND ----------

# DBTITLE 1, Monitor Deployment Status
print("\n📊 Monitoring deployment status...")

max_wait_time = 1200  # 20 minutes
check_interval = 30   # 30 seconds
start_time = time.time()

while time.time() - start_time < max_wait_time:
    try:
        endpoint_status = w.serving_endpoints.get(endpoint_name)
        state = endpoint_status.state.value if endpoint_status.state else "UNKNOWN"
        
        print(f"   Current state: {state}")
        
        if state == "READY":
            print("   ✅ Endpoint is ready!")
            break
        elif state in ["FAILED", "CANCELLED"]:
            print(f"   ❌ Deployment failed with state: {state}")
            raise Exception(f"Endpoint deployment failed: {state}")
        
        print(f"   ⏳ Waiting... (elapsed: {int(time.time() - start_time)}s)")
        time.sleep(check_interval)
        
    except Exception as e:
        print(f"   ⚠ Error checking endpoint status: {str(e)}")
        time.sleep(check_interval)

if time.time() - start_time >= max_wait_time:
    print("   ⚠ Deployment monitoring timed out")
    print("   Check the endpoint status manually in the Databricks UI")
else:
    # Update model registry with serving endpoint info
    try:
        description = mlflow_client.get_model_version(model_name, model_version).description or ""
        if description:
            description += "\n\n---\n\n"
        description += f"Serving Endpoint: {endpoint_name}\nEnvironment: {env}\nDeployment Status: READY"
        mlflow_client.update_model_version(
            name=model_name, 
            version=model_version, 
            description=description
        )
        print("   ✓ Model registry updated with endpoint information")
    except Exception as e:
        print(f"   ⚠ Could not update model registry: {str(e)}")
        
    # Set model alias to "champion" if in prod environment
    if env == "prod":
        try:
            mlflow_client.set_registered_model_alias(model_name, "champion", model_version)
            print("   ✓ Model promoted to 'champion' alias in production")
        except Exception as e:
            print(f"   ⚠ Could not set 'champion' alias: {str(e)}")

# COMMAND ----------
print("\n" + "=" * 60)
print("🎉 MODEL DEPLOYMENT COMPLETED")
print("=" * 60)
print(f"Model: {model_uri}")
print(f"Endpoint: {endpoint_name}")
print(f"Environment: {env}")
print(f"Status: {'READY' if time.time() - start_time < max_wait_time else 'IN_PROGRESS'}")
print("=" * 60)