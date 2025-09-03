# one_click_deployment
This project is a customized ML deployment template focused on **model serving** with HuggingFace Transformers. It provides a production-ready pipeline to download, validate, and deploy transformer models (using GPT-2 as an example) to Databricks serving endpoints.

The template has been streamlined to focus solely on model deployment/serving workflows, removing batch inference and monitoring components to create a lightweight, focused solution.

## Table of contents
* [Code structure](#code-structure): structure of this project.
* [Model Pipeline](#model-pipeline): overview of the train → validate → deploy pipeline.
* [Environment Setup](#environment-setup): test and production environment configuration.
* [Iterating on ML code](#iterating-on-ml-code): making and testing ML code changes.
* [Next steps](#next-steps)

## Model Pipeline

This template implements a **Train → Validate → Deploy** pipeline optimized for transformer model deployment:

1. **Training**: Downloads a pre-trained GPT-2 model from HuggingFace Hub and logs it to MLflow with transformers flavor
2. **Validation**: Loads the model, runs inference testing, and evaluates using `mlflow.genai.evaluate()`
3. **Deployment**: Comprehensive deployment to Databricks serving endpoints with pre-deployment checks

### Key Features:
- 🤖 **HuggingFace Integration**: Template for downloading and deploying transformer models
- 🔍 **Pre-deployment Checks**: Validates model existence, tests loading, and checks endpoint status
- 🎯 **GenAI Evaluation**: Uses MLflow's GenAI evaluation framework for model validation
- 🚀 **Serving Endpoints**: Automated deployment to Databricks Model Serving with monitoring
- 🏗️ **Two-Environment Setup**: Separate dev/testing and production catalogs

## Code structure
This project contains the following components:

| Component                  | Description                                                                                                                                                                                                                                                                                                                                             |
|----------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| ML Code                    | Transformer model deployment code with comprehensive validation and serving capabilities                                                                                                                                                                                                                                                                  |
| ML Resources as Code | Model deployment pipeline (training → validation → deployment) configured through [databricks CLI bundles](https://docs.databricks.com/dev-tools/cli/bundle-cli.html)                                                                                              |

contained in the following files:

```
one_click_deployment        <- Root directory
│
├── one_click_deployment       <- Contains python code, notebooks and ML resources for model deployment
│   │
│   ├── requirements.txt        <- Python dependencies for transformer models and MLflow
│   │
│   ├── databricks.yml          <- Bundle configuration with test/prod environments and catalogs
│   │
│   ├── training                <- Training: Downloads any model from HuggingFace and logs to MLflow
│   │   └── notebooks/
│   │       └── Train.py        <- Downloads and logs transformer model (i.e. GPT-2)
│   │
│   ├── validation              <- Model validation with GenAI evaluation
│   │   ├── notebooks/
│   │   │   └── ModelValidation.py  <- Loads model, runs inference, evaluates with mlflow.genai.evaluate()
│   │   └── validation.py       <- (Optional) Custom validation functions reference
│   │
│   ├── deployment              <- Model serving deployment
│   │   └── model_deployment/
│   │       ├── notebooks/
│   │       │   └── ModelDeployment.py <- Comprehensive serving endpoint deployment with checks
│   │       └── deploy.py       <- (Optional) Alternative deployment method
│   │
│   ├── tests                   <- Unit tests for the ML pipeline
│   │
│   └── resources               <- ML pipeline configuration
│       ├── model-workflow-resource.yml     <- Main train→validate→deploy workflow
│       └── ml-artifacts-resource.yml       <- Model and experiment definitions
```

## Environment Setup

The template uses a **two-environment, two-catalog architecture**:

### Environments:
- **Test Environment** (`test`): Development and testing with `dev_testing_catalog`
- **Production Environment** (`prod`): Production deployment with `production_catalog`

### Model Flow:
1. Models are initially logged to the **dev/testing catalog**
2. After validation, models get the **"challenger"** alias
3. Upon successful deployment to prod, models get the **"champion"** alias

### Workspace Configuration:
- Both environments can use the **same Databricks workspace**
- Separate **Unity Catalogs** provide isolation between dev/test and prod
- No staging workspace - simplified two-environment setup

## Iterating on ML code

### Deploy ML code and resources to test workspace using Bundles

```bash
# Deploy to test environment
databricks bundle deploy -t test

# Deploy to production environment  
databricks bundle deploy -t prod
```

Refer to [Local development and dev workspace](./resources/README.md#local-development-and-dev-workspace)
to use databricks CLI bundles to deploy ML code together with ML resource configs.

### Develop on Databricks using Databricks Repos

#### Prerequisites
You'll need:
* Access to run commands on a cluster running Databricks Runtime ML version 11.0 or above
* Unity Catalog access with permissions for the configured catalogs
* To set up [Databricks Repos](https://docs.databricks.com/repos/index.html): see instructions below

#### Configuring Databricks Repos
To use Repos, [set up git integration](https://docs.databricks.com/repos/repos-setup.html) in your workspace.

If the current project has already been pushed to a hosted Git repo, follow the
[UI workflow](https://docs.databricks.com/repos/git-operations-with-repos.html#add-a-repo-connected-to-a-remote-repo)
to clone it into your workspace and iterate.

### Testing the Pipeline

1. **Run Training**: Execute the `Train.py` notebook to download and log model (weights)
2. **Run Validation**: Execute `ModelValidation.py` to validate the model with GenAI evaluation _(optionnal)_
3. **Run Deployment**: Execute `ModelDeployment.py` to deploy to serving endpoint

The workflow can be triggered automatically on PR merge to main branch or run manually.

## Next Steps

### Setting up Production CI/CD

When ready for production deployment:

1. Configure Git integration for automatic triggering on PR merges
2. Set up proper Unity Catalog permissions for both environments
3. Configure serving endpoint quotas and resource limits
4. Set up monitoring and alerting for the deployed endpoints

### Customizing for Your Use Case

1. **Replace GPT-2** with your preferred transformer model in `Train.py`
2. **Update evaluation logic** in `ModelValidation.py` for your specific use case
3. **Modify serving configuration** in `ModelDeployment.py` for your performance requirements
4. **Configure catalogs and schemas** in `databricks.yml` to match your organization

### Advanced Features

- Add custom evaluation metrics in `validation.py`
- Implement A/B testing between model versions
- Add automatic model monitoring and drift detection
- Configure auto-scaling policies for serving endpoints

For more details on MLOps best practices, refer to the [Databricks MLOps Guide](https://docs.databricks.com/machine-learning/mlops/index.html).