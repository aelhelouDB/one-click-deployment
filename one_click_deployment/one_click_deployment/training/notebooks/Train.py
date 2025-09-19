# Databricks notebook source
##################################################################################
# Model Training Notebook (Distilled RoBERTa Tone Detection)
#
# This notebook downloads a pre-trained distilled RoBERTa model from HuggingFace,
# wraps it in a PyFunc class for tone detection, saves model weights to Unity Catalog
# volume, and logs it to MLflow registry with locked dependencies using uv.
#
# Parameters:
# * env (required):                 - Environment the notebook is run in (test, or prod). Defaults to "test".
# * model_name (required)           - Three-level name (<catalog>.<schema>.<model_name>) to register the model in Unity Catalog.
# * experiment_name (required)      - MLflow experiment name for the training runs. Will be created if it doesn't exist.
# * uc_volume_path (required)       - Unity Catalog volume path to save model artifacts.
#  
##################################################################################

# COMMAND ----------

# MAGIC %load_ext autoreload
# MAGIC %autoreload 2

# COMMAND ----------

import os


notebook_path =  '/Workspace/' + os.path.dirname(dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get())

# COMMAND ----------

# MAGIC %pip install uv -r ../../requirements.txt

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1, Notebook arguments
# List of input args needed to run this notebook as a job.
# Provide them via DB widgets or notebook arguments.

# Notebook Environment
dbutils.widgets.dropdown("env", "test", ["test", "prod"], "Environment Name")
env = dbutils.widgets.get("env")

# MLflow experiment name.
dbutils.widgets.text(
    "experiment_name",
    f"/dev-one_click_deployment-experiment",
    label="MLflow experiment name",
)
# Unity Catalog registered model name to use for the trained model.
dbutils.widgets.text(
    "model_name", "amine_elhelou_staging.one_click_deployment.tone_detection_model", label="Full (Three-Level) Model Name"
)

# Unity Catalog volume path for model artifacts
dbutils.widgets.text(
    "uc_volume_path", "/Volumes/amine_elhelou_staging/one_click_deployment/model_artifacts", label="Unity Catalog Volume Path"
)

# COMMAND ----------

# DBTITLE 1,Define input and output variables
experiment_name = dbutils.widgets.get("experiment_name")
model_name = dbutils.widgets.get("model_name")
uc_volume_path = dbutils.widgets.get("uc_volume_path")

# COMMAND ----------

# DBTITLE 1, Set experiment
import mlflow
import os


# Set environment variable to lock model dependencies
os.environ["MLFLOW_LOCK_MODEL_DEPENDENCIES"] = "True"

mlflow.set_experiment(experiment_name)
mlflow.set_registry_uri('databricks-uc')

# COMMAND ----------

# DBTITLE 1, Download Distilled RoBERTa model from HuggingFace
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import mlflow.transformers
import torch
import os
import subprocess


model_name_hf = "j-hartmann/emotion-english-distilroberta-base"
print(f"Downloading {model_name_hf} from HuggingFace Hub...")

# Download model and tokenizer for tone/emotion detection
model = AutoModelForSequenceClassification.from_pretrained(model_name_hf)
tokenizer = AutoTokenizer.from_pretrained(model_name_hf)

print(f"Successfully downloaded {model_name_hf}")
print(f"Model labels: {model.config.id2label}")

# COMMAND ----------

# DBTITLE 1, Helper functions and PyFunc wrapper
from mlflow.tracking import MlflowClient
import mlflow.pyfunc
import pandas as pd
import numpy as np


def get_latest_model_version(model_name):
    latest_version = 1
    mlflow_client = MlflowClient()
    for mv in mlflow_client.search_model_versions(f"name='{model_name}'"):
        version_int = int(mv.version)
        if version_int > latest_version:
            latest_version = version_int
    return latest_version


class ToneDetectionModel(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        
        # Load model from artifacts
        model_path = os.path.join(context.artifacts["model"])
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model.eval()
    
    def predict(self, context, model_input):
        import torch
        import torch.nn.functional as F
        
        # Handle both string input and DataFrame input
        if isinstance(model_input, pd.DataFrame):
            texts = model_input.iloc[:, 0].tolist()  # Assume first column contains text
        elif isinstance(model_input, (list, np.ndarray)):
            texts = list(model_input)
        else:
            texts = [str(model_input)]
        
        results = []
        for text in texts:
            # Tokenize input
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
            
            # Get predictions
            with torch.no_grad():
                outputs = self.model(**inputs)
                probabilities = F.softmax(outputs.logits, dim=-1)
                predicted_class_id = outputs.logits.argmax().item()
                confidence = probabilities[0][predicted_class_id].item()
                
                predicted_label = self.model.config.id2label[predicted_class_id]
                
                results.append({
                    "text": text,
                    "predicted_tone": predicted_label,
                    "confidence": confidence,
                    "all_scores": {self.model.config.id2label[i]: prob.item() for i, prob in enumerate(probabilities[0])}
                })
        
        return results


# COMMAND ----------

# MAGIC %md
# MAGIC Save model to Unity Catalog volume and log as PyFunc model to MLflow.

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE VOLUME IF NOT EXISTS amine_elhelou_staging.one_click_deployment.model_artifacts

# COMMAND ----------

# DBTITLE 1, Save model to Unity Catalog Volume
# Create UC volume directory if it doesn't exist
dbutils.fs.mkdirs(uc_volume_path)
model_artifact_path = os.path.join(uc_volume_path, "model")

# Save model and tokenizer to UC volume
model.save_pretrained(model_artifact_path)
tokenizer.save_pretrained(model_artifact_path)
print(f"Model saved to Unity Catalog volume: {model_artifact_path}")

# COMMAND ----------

# DBTITLE 1, Lock requirements with uv
# Generate requirements.txt with uv
requirements_content = """torch>=2.0.0
transformers>=4.21.0
mlflow>=2.0.0
pandas>=1.5.0
numpy>=1.21.0
"""

requirements_path = os.path.join(uc_volume_path, "requirements.txt")
with open(requirements_path, "w") as f:
    f.write(requirements_content)

# Use uv to lock requirements
try:
    subprocess.run(["uv", "pip", "compile", requirements_path, "-o", os.path.join(uc_volume_path, "requirements.lock")], check=True)
    print("Successfully locked requirements with uv")
except subprocess.CalledProcessError:
    print("Warning: uv not available, using basic requirements.txt")

# COMMAND ----------

# DBTITLE 1, Log model with PyFunc wrapper
# Create sample input for the model signature
sample_input = pd.DataFrame({"text": ["I am feeling great today!", "This is terrible news"]})

# Create artifacts dictionary pointing to UC volume model
artifacts = {"model": model_artifact_path}

with mlflow.start_run(run_name="model-creation-baseline") as run:
    # Log the model with PyFunc wrapper
    mlflow.pyfunc.log_model(
        name="tone_detection_model",
        python_model=ToneDetectionModel(),
        artifacts=artifacts,
        input_example=sample_input,
        registered_model_name=model_name,
        pip_requirements=requirements_path
    )


# The returned model URI is needed by the model deployment notebook.
model_version = get_latest_model_version(model_name)
model_uri = f"models:/{model_name}/{model_version}"
dbutils.jobs.taskValues.set("model_uri", model_uri)
dbutils.jobs.taskValues.set("model_name", model_name)
dbutils.jobs.taskValues.set("model_version", model_version)
dbutils.jobs.taskValues.set("uc_volume_path", uc_volume_path)
dbutils.notebook.exit(model_uri)

# COMMAND ----------


