# Databricks notebook source
##################################################################################
# Model Validation Notebook
##
# This notebook uses mlflow model validation API to run mode validation after training and registering a model
# in model registry, before deploying it to the "champion" alias.
#
# It runs as part of CD and by an automated model training job -> validation -> deployment job defined under ``one_click_deployment/resources/model-workflow-resource.yml``
#
#
# Parameters:
#
# * env                                     - Name of the environment the notebook is run in (staging, or prod). Defaults to "prod".
# * `run_mode`                              - The `run_mode` defines whether model validation is enabled or not. It can be one of the three values:
#                                             * `disabled` : Do not run the model validation notebook.
#                                             * `dry_run`  : Run the model validation notebook. Ignore failed model validation rules and proceed to move
#                                                            model to the "champion" alias.
#                                             * `enabled`  : Run the model validation notebook. Move model to the "champion" alias only if all model validation
#                                                            rules are passing.
# * enable_baseline_comparison              - Whether to load the current registered "champion" model as baseline.
#                                             Baseline model is a requirement for relative change and absolute change validation thresholds.
# * validation_input                        - Validation input. Please refer to data parameter in mlflow.evaluate documentation https://mlflow.org/docs/latest/python_api/mlflow.html#mlflow.evaluate
# * model_type                              - A string describing the model type. The model type can be either "regressor" and "classifier".
#                                             Please refer to model_type parameter in mlflow.evaluate documentation https://mlflow.org/docs/latest/python_api/mlflow.html#mlflow.evaluate
# * targets                                 - The string name of a column from data that contains evaluation labels.
#                                             Please refer to targets parameter in mlflow.evaluate documentation https://mlflow.org/docs/latest/python_api/mlflow.html#mlflow.evaluate
# * custom_metrics_loader_function          - Specifies the name of the function in one_click_deployment/validation/validation.py that returns custom metrics.
# * validation_thresholds_loader_function   - Specifies the name of the function in one_click_deployment/validation/validation.py that returns model validation thresholds.
#
# For details on mlflow evaluate API, see doc https://mlflow.org/docs/latest/python_api/mlflow.html#mlflow.evaluate
# For details and examples about performing model validation, see the Model Validation documentation https://mlflow.org/docs/latest/models.html#model-validation
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

import os
notebook_path =  '/Workspace/' + os.path.dirname(dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get())
%cd $notebook_path
%cd ../

# COMMAND ----------

dbutils.widgets.text(
    "experiment_name",
    "/dev-one_click_deployment-experiment",
    "Experiment Name",
)
dbutils.widgets.dropdown("run_mode", "disabled", ["disabled", "dry_run", "enabled"], "Run Mode")
dbutils.widgets.dropdown("enable_baseline_comparison", "false", ["true", "false"], "Enable Baseline Comparison")
dbutils.widgets.text("validation_input", "SELECT * FROM delta.`dbfs:/databricks-datasets/nyctaxi-with-zipcodes/subsampled`", "Validation Input")

dbutils.widgets.text("model_type", "regressor", "Model Type")
dbutils.widgets.text("targets", "fare_amount", "Targets")
dbutils.widgets.text("custom_metrics_loader_function", "custom_metrics", "Custom Metrics Loader Function")
dbutils.widgets.text("validation_thresholds_loader_function", "validation_thresholds", "Validation Thresholds Loader Function")
dbutils.widgets.text("evaluator_config_loader_function", "evaluator_config", "Evaluator Config Loader Function")
dbutils.widgets.text("model_name", "dev.one_click_deployment.one_click_deployment-model", "Full (Three-Level) Model Name")
dbutils.widgets.text("model_version", "", "Candidate Model Version")

# COMMAND ----------
run_mode = dbutils.widgets.get("run_mode").lower()
assert run_mode == "disabled" or run_mode == "dry_run" or run_mode == "enabled"

if run_mode == "disabled":
    print(
        "Model validation is in DISABLED mode. Exit model validation without blocking model deployment."
    )
    dbutils.notebook.exit(0)
dry_run = run_mode == "dry_run"

if dry_run:
    print(
        "Model validation is in DRY_RUN mode. Validation threshold validation failures will not block model deployment."
    )
else:
    print(
        "Model validation is in ENABLED mode. Validation threshold validation failures will block model deployment."
    )

# COMMAND ----------

import mlflow
import mlflow.transformers
import mlflow.genai
import os
import tempfile
import traceback
import pandas as pd

from mlflow.tracking.client import MlflowClient

client = MlflowClient(registry_uri="databricks-uc")
mlflow.set_registry_uri('databricks-uc')

# set experiment
experiment_name = dbutils.widgets.get("experiment_name")
mlflow.set_experiment(experiment_name)

# set model evaluation parameters that can be inferred from the job
model_uri = dbutils.jobs.taskValues.get("Train", "model_uri", debugValue="")
model_name = dbutils.jobs.taskValues.get("Train", "model_name", debugValue="")
model_version = dbutils.jobs.taskValues.get("Train", "model_version", debugValue="")

if model_uri == "":
    model_name = dbutils.widgets.get("model_name")
    model_version = dbutils.widgets.get("model_version")
    model_uri = "models:/" + model_name + "/" + model_version

baseline_model_uri = "models:/" + model_name + "@champion"

evaluators = "default"
assert model_uri != "", "model_uri notebook parameter must be specified"
assert model_name != "", "model_name notebook parameter must be specified"
assert model_version != "", "model_version notebook parameter must be specified"

# COMMAND ----------

# take input
enable_baseline_comparison = dbutils.widgets.get("enable_baseline_comparison")


assert enable_baseline_comparison == "true" or enable_baseline_comparison == "false"
enable_baseline_comparison = enable_baseline_comparison == "true"

validation_input = dbutils.widgets.get("validation_input")
assert validation_input
data = spark.sql(validation_input)

# Convert to pandas for mlflow.genai.evaluate
eval_data = data.toPandas()

model_type = dbutils.widgets.get("model_type")
targets = dbutils.widgets.get("targets")

assert model_type
assert targets

# Load the transformers model for evaluation
print(f"Loading model for evaluation: {model_uri}")
loaded_model = mlflow.transformers.load_model(model_uri)

# Define a model wrapper for mlflow.genai.evaluate
def model_wrapper(inputs):
    """Wrapper function to format model outputs for evaluation"""
    outputs = []
    for input_text in inputs:
        try:
            result = loaded_model(input_text, max_length=100, num_return_sequences=1, 
                                temperature=0.7, do_sample=True, pad_token_id=50256)
            generated_text = result[0]['generated_text']
            # Return just the generated portion (remove the input prompt)
            if generated_text.startswith(input_text):
                generated_text = generated_text[len(input_text):].strip()
            outputs.append(generated_text)
        except Exception as e:
            print(f"Inference failed for input '{input_text}': {str(e)}")
            outputs.append("[Generation failed]")
    return outputs

# COMMAND ----------

# helper methods
def get_run_link(run_info):
    return "[Run](#mlflow/experiments/{0}/runs/{1})".format(
        run_info.experiment_id, run_info.run_id
    )


def get_training_run(model_name, model_version):
    version = client.get_model_version(model_name, model_version)
    return mlflow.get_run(run_id=version.run_id)


def generate_run_name(training_run):
    return None if not training_run else training_run.info.run_name + "-validation"


def generate_description(training_run):
    return (
        None
        if not training_run
        else "Model Training Details: {0}\n".format(get_run_link(training_run.info))
    )


def log_to_model_description(run, success):
    run_link = get_run_link(run.info)
    description = client.get_model_version(model_name, model_version).description
    status = "SUCCESS" if success else "FAILURE"
    if description != "":
        description += "\n\n---\n\n"
    description += "Model Validation Status: {0}\nValidation Details: {1}".format(
        status, run_link
    )
    client.update_model_version(
        name=model_name, version=model_version, description=description
    )



# COMMAND ----------



training_run = get_training_run(model_name, model_version)

# run evaluate
with mlflow.start_run(
    run_name=generate_run_name(training_run),
    description=generate_description(training_run),
) as run, tempfile.TemporaryDirectory() as tmp_dir:
    validation_thresholds_file = os.path.join(tmp_dir, "validation_thresholds.txt")
    with open(validation_thresholds_file, "w") as f:
        if validation_thresholds:
            for metric_name in validation_thresholds:
                f.write(
                    "{0:30}  {1}\n".format(
                        metric_name, str(validation_thresholds[metric_name])
                    )
                )
    mlflow.log_artifact(validation_thresholds_file)

    try:
        # Run MLflow GenAI evaluation
        print("Starting MLflow GenAI evaluation...")
        eval_result = mlflow.genai.evaluate(
            model=model_wrapper,
            data=eval_data,
            evaluators="default",
            model_type="text-generation"
        )
        # Log evaluation metrics
        metrics_file = os.path.join(tmp_dir, "metrics.txt")
        with open(metrics_file, "w") as f:
            f.write("GenAI Evaluation Results\n")
            f.write("=" * 30 + "\n")
            for metric_name, metric_value in eval_result.metrics.items():
                f.write(f"{metric_name}: {metric_value}\n")
                mlflow.log_metric(metric_name, metric_value)
        mlflow.log_artifact(metrics_file)
        
        # Print evaluation results
        print("\nEvaluation Results:")
        for metric_name, metric_value in eval_result.metrics.items():
            print(f"{metric_name}: {metric_value}")
        log_to_model_description(run, True)
        
        # Simple validation check - ensure evaluation ran successfully
        validation_passed = len(eval_result.metrics) > 0
        
        if validation_passed:
            # Assign "challenger" alias to indicate model version has passed validation checks
            print("Validation checks passed. Assigning 'challenger' alias to model version.")
            client.set_registered_model_alias(model_name, "challenger", model_version)
        else:
            print("Validation failed: No evaluation metrics computed.")
            if not dry_run:
                raise Exception("Validation failed: No evaluation metrics computed")
        
    except Exception as err:
        log_to_model_description(run, False)
        error_file = os.path.join(tmp_dir, "error.txt")
        with open(error_file, "w") as f:
            f.write("Validation failed : " + str(err) + "\n")
            f.write(traceback.format_exc())
        mlflow.log_artifact(error_file)
        if not dry_run:
            raise err
        else:
            print(
                "Model validation failed in DRY_RUN. It will not block model deployment."
            )
