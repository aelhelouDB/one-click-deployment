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

You can test the template in several ways depending on your needs:

#### 🧪 **Testing Options**

**1. Local Testing (Limited)**
For basic code validation and syntax checking:

```bash
# Navigate to the project directory
cd one_click_deployment/one_click_deployment

# Install dependencies
pip install -r requirements.txt

# Run unit tests
pytest tests/

# Validate YAML syntax
databricks bundle validate
```

> **Note:** Local testing is limited to syntax/import validation. MLflow Unity Catalog integration and Databricks-specific features require workspace testing.

**2. Databricks Workspace Testing (Recommended)**

**Prerequisites Setup:**
```bash
# 1. Configure workspace access
databricks configure --token --host https://e2-demo-field-eng.cloud.databricks.com

# 2. Ensure Unity Catalog access with these catalogs and schemas:
# - dev_testing_catalog.models (for test environment)
# - production_catalog.models (for prod environment)
```

**Option A: Bundle Deployment (Automated)**
```bash
# Deploy to test environment
databricks bundle deploy -t test

# Run the complete workflow
databricks bundle run model_training_job -t test
```

**Option B: Manual Notebook Execution**
1. Upload project to Databricks Repos (via Git integration) OR use Databricks VS Extension/DB Connect and run directly on Databricks Cluster via IDE
2. Execute notebooks individually:
   - **Training**: Run `training/notebooks/Train.py`
   - **Validation**: Run `validation/notebooks/ModelValidation.py` 
   - **Deployment**: Run `deployment/model_deployment/notebooks/ModelDeployment.py`

#### 🔍 **Step-by-Step Testing Guide**

**Step 1: Validate Configuration**
```bash
# Check bundle syntax
databricks bundle validate -t test

# Deploy to test environment
databricks bundle deploy -t test
```

**Step 2: Test Training Pipeline**
```bash
# Deploy and run training only
databricks bundle deploy -t test
databricks bundle run model_training_job -t test --job-key Train
```

**Step 3: Test Full Pipeline**
```bash
# Run complete workflow
databricks bundle run model_training_job -t test
```

**Step 4: Monitor Progress**
- Check **Databricks Jobs UI** for workflow progress
- Check **MLflow UI** for model registration and experiments  
- Check **Serving Endpoints** for deployment status

#### ✅ **Testing Checklist**

**Training Validation:**
- [ ] GPT-2 model downloads successfully from HuggingFace
- [ ] Model logs to MLflow with transformers flavor
- [ ] Model appears in Unity Catalog (`dev_testing_catalog.models`)
- [ ] Task values (model_uri, model_name, model_version) are set correctly

**Validation Testing:**
- [ ] Model loads from registry successfully
- [ ] Inference test passes with sample inputs
- [ ] GenAI evaluation runs and produces metrics
- [ ] "challenger" alias is assigned to model version
- [ ] Evaluation metrics are logged to MLflow

**Deployment Verification:**
- [ ] Pre-deployment checks pass (model exists, loads correctly)
- [ ] Serving endpoint creates/updates successfully
- [ ] Endpoint reaches "READY" state within timeout
- [ ] Model inference works via serving endpoint
- [ ] "champion" alias set (if running in prod environment)
- [ ] Model registry updated with endpoint information

#### 🐛 **Troubleshooting**

**Bundle Deployment Issues:**
```bash
# Check detailed error logs
databricks bundle deploy -t test --verbose

# Validate individual resource files
databricks bundle validate --verbose
```

**Model Registration Issues:**
- Check Unity Catalog permissions for `dev_testing_catalog` and `production_catalog`
- Ensure catalogs and schemas exist: `dev_testing_catalog.models`, `production_catalog.models`
- Verify MLflow registry URI is set to `databricks-uc`

**Serving Endpoint Issues:**
- Check endpoint logs in Databricks Serving UI
- Verify model version exists and is accessible in Unity Catalog
- Check compute resource availability and quotas
- Ensure proper permissions for endpoint creation

**Useful Debug Commands:**
```bash
# View bundle resources
databricks bundle schema

# Monitor job runs
databricks jobs list-runs --job-id <job-id>

# Check workspace files
databricks workspace list /Workspace/Users/<your-username>
```

#### 🧪 **Testing Different Scenarios**

**Test with Different Models:**
Update `training/notebooks/Train.py`:
```python
# Replace "gpt2" with other HuggingFace models
model_name_hf = "distilbert-base-uncased"  # or any transformer model
```

**Test Environment Promotion:**
```bash
# Test in development
databricks bundle deploy -t test
databricks bundle run model_training_job -t test

# Promote to production
databricks bundle deploy -t prod
databricks bundle run model_training_job -t prod
```

The workflow can be triggered automatically on PR merge to main branch or run manually using the commands above.

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