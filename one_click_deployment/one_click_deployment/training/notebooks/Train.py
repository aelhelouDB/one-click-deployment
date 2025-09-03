# Databricks notebook source
##################################################################################
# Model Training Notebook (HuggingFace GPT-2)
#
# This notebook downloads a pre-trained GPT-2 model from HuggingFace and logs it
# to MLflow registry in the dev/testing catalog. This serves as a template for
# model deployment/serving workflows.
#
# Parameters:
# * env (required):                 - Environment the notebook is run in (test, or prod). Defaults to "test".
# * model_name (required)           - Three-level name (<catalog>.<schema>.<model_name>) to register the model in Unity Catalog.
# * experiment_name (required)      - MLflow experiment name for the training runs. Will be created if it doesn't exist.
#  
##################################################################################

# COMMAND ----------

# MAGIC %load_ext autoreload
# MAGIC %autoreload 2

# COMMAND ----------

import os
notebook_path =  '/Workspace/' + os.path.dirname(dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get())
%cd $notebook_path

# COMMAND ----------

# MAGIC %pip install -r ../../requirements.txt

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
    "model_name", "dev.one_click_deployment.one_click_deployment-model", label="Full (Three-Level) Model Name"
)

# COMMAND ----------

# DBTITLE 1,Define input and output variables
experiment_name = dbutils.widgets.get("experiment_name")
model_name = dbutils.widgets.get("model_name")

# COMMAND ----------

# DBTITLE 1, Set experiment
import mlflow

mlflow.set_experiment(experiment_name)
mlflow.set_registry_uri('databricks-uc')

# COMMAND ----------

# DBTITLE 1, Download GPT-2 model from HuggingFace
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import mlflow.transformers

model_name_hf = "gpt2"
print(f"Downloading {model_name_hf} from HuggingFace Hub...")

# Download model and tokenizer
model = GPT2LMHeadModel.from_pretrained(model_name_hf)
tokenizer = GPT2Tokenizer.from_pretrained(model_name_hf)

print(f"Successfully downloaded {model_name_hf}")

# COMMAND ----------

# DBTITLE 1, Helper function
from mlflow.tracking import MlflowClient
import mlflow.pyfunc


def get_latest_model_version(model_name):
    latest_version = 1
    mlflow_client = MlflowClient()
    for mv in mlflow_client.search_model_versions(f"name='{model_name}'"):
        version_int = int(mv.version)
        if version_int > latest_version:
            latest_version = version_int
    return latest_version


# COMMAND ----------

# MAGIC %md
# MAGIC Log the downloaded GPT-2 model to MLflow with transformers flavor.

# COMMAND ----------

# DBTITLE 1, Log model and return output.
# Create a sample input for the model signature
sample_input = ["Hello, how are you?"]

# Log the model with transformers flavor
mlflow.transformers.log_model(
    transformers_model={"model": model, "tokenizer": tokenizer},
    artifact_path="gpt2_model",
    input_example=sample_input,
    registered_model_name=model_name
)

# The returned model URI is needed by the model deployment notebook.
model_version = get_latest_model_version(model_name)
model_uri = f"models:/{model_name}/{model_version}"
dbutils.jobs.taskValues.set("model_uri", model_uri)
dbutils.jobs.taskValues.set("model_name", model_name)
dbutils.jobs.taskValues.set("model_version", model_version)
dbutils.notebook.exit(model_uri)
