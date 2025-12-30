# Deployment Guide - Skin Lesion Classification MLOps Pipeline

This guide walks you through deploying the complete MLOps pipeline to AWS.

## Prerequisites

### Required Tools
- Python 3.8+
- AWS CLI configured with credentials
- Terraform 1.0+
- Docker (for local testing)
- Git

### AWS Account Requirements
- AWS Account with administrative access
- Sufficient service limits for:
  - SageMaker training instances (ml.p3.2xlarge)
  - SageMaker endpoint instances (ml.m5.xlarge)
  - S3 buckets
  - Lambda functions

### Estimated Costs
- **Development**: $50-100/month (with spot instances)
- **Production**: $200-500/month (depending on usage)

## Step 1: Clone and Setup Repository

```bash
# Clone the repository
git clone <your-repo-url>
cd skin-lesion-classification

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install -e .
```

## Step 2: Configure AWS Credentials

```bash
# Configure AWS CLI
aws configure

# Verify configuration
aws sts get-caller-identity
```

## Step 3: Update Configuration Files

### Update config/config.yaml

```yaml
aws:
  region: us-east-1  # Your preferred region
  account_id: "123456789012"  # Your AWS account ID

notifications:
  email: "your.email@example.com"  # Email for alerts
```

### Create Terraform variables file

```bash
cd infrastructure/terraform

# Create terraform.tfvars
cat > terraform.tfvars <<EOF
project_name = "skin-lesion-classification"
environment = "dev"
aws_region = "us-east-1"
alert_email = "your.email@example.com"
EOF
```

## Step 4: Deploy Infrastructure with Terraform

```bash
# Initialize Terraform
terraform init

# Review the plan
terraform plan

# Deploy infrastructure
terraform apply

# Save outputs
terraform output -json > ../../terraform-outputs.json

# Return to project root
cd ../..
```

**Expected Resources Created:**
- 4 S3 buckets (data, models, pipeline, logs)
- ECR repositories (training, inference)
- IAM roles and policies
- SageMaker model registry
- CloudWatch dashboards and alarms
- SNS topics for alerts
- Step Functions state machine
- Lambda functions
- EventBridge rules

## Step 5: Download Dataset

```bash
# Configure Kaggle API (if using ISIC 2019)
# 1. Create Kaggle account
# 2. Go to Account -> Create New API Token
# 3. Place kaggle.json in ~/.kaggle/

# Download ISIC 2019 dataset
python scripts/download_data.py --dataset isic2019 --output-dir data/raw --verify

# Or download HAM10000
python scripts/download_data.py --dataset ham10000 --output-dir data/raw --verify
```

## Step 6: Upload Data to S3

```bash
# Get bucket name from Terraform outputs
export DATA_BUCKET=$(cat terraform-outputs.json | jq -r '.data_bucket_name.value')

# Upload data
aws s3 sync data/raw s3://${DATA_BUCKET}/data/raw/

# Verify upload
aws s3 ls s3://${DATA_BUCKET}/data/raw/ --recursive | wc -l
```

Or use the helper script:

```bash
python scripts/upload_to_s3.py --bucket ${DATA_BUCKET} --input-dir data/raw --prefix data/raw
```

## Step 7: Build and Push Docker Images

```bash
# Get ECR repository URLs
export TRAINING_REPO=$(cat terraform-outputs.json | jq -r '.training_ecr_repository_url.value')
export INFERENCE_REPO=$(cat terraform-outputs.json | jq -r '.inference_ecr_repository_url.value')

# Login to ECR
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin ${TRAINING_REPO}

# Build training image
docker build -t ${TRAINING_REPO}:latest -f docker/Dockerfile.training .
docker push ${TRAINING_REPO}:latest

# Build inference image
docker build -t ${INFERENCE_REPO}:latest -f docker/Dockerfile.inference .
docker push ${INFERENCE_REPO}:latest
```

**Note:** You can use the provided Dockerfiles under `docker/`, or rely on SageMaker built-in containers initially.

## Step 8: Deploy SageMaker Pipeline

```bash
# Deploy the pipeline
python scripts/deploy_pipeline.py \
  --config config/config.yaml \
  --terraform-outputs terraform-outputs.json

# Verify pipeline was created
aws sagemaker list-pipelines
```

## Step 9: Start Pipeline Execution

### Option A: Via Script

```bash
# Start execution
python scripts/deploy_pipeline.py \
  --config config/config.yaml \
  --terraform-outputs terraform-outputs.json \
  --start

# Monitor execution
aws sagemaker list-pipeline-executions \
  --pipeline-name skin-lesion-pipeline
```

### Option B: Via AWS Console

1. Go to SageMaker Console
2. Navigate to Pipelines
3. Select `skin-lesion-pipeline`
4. Click "Create execution"
5. Use default parameters or customize
6. Click "Start"

### Option C: Via Python SDK

```python
import boto3

sagemaker_client = boto3.client('sagemaker')

response = sagemaker_client.start_pipeline_execution(
    PipelineName='skin-lesion-pipeline',
    PipelineParameters=[
        {
            'Name': 'TrainingInstanceType',
            'Value': 'ml.p3.2xlarge'
        },
        {
            'Name': 'Epochs',
            'Value': '50'
        }
    ]
)

print(f"Execution ARN: {response['PipelineExecutionArn']}")
```

## Step 10: Monitor Pipeline Execution

### CloudWatch Logs

```bash
# View training logs
aws logs tail /aws/sagemaker/TrainingJobs --follow

# View processing logs
aws logs tail /aws/sagemaker/ProcessingJobs --follow
```

### CloudWatch Dashboard

1. Go to CloudWatch Console
2. Navigate to Dashboards
3. Open `skin-lesion-classification-dev-dashboard`
4. Monitor metrics:
   - Training job status
   - Model performance
   - Resource utilization

### SageMaker Studio (if enabled)

1. Open SageMaker Studio
2. Navigate to Pipelines
3. View execution graph
4. Check step details and logs

## Step 11: Deploy Model to Endpoint (After Training)

Once the pipeline completes and registers a model:

```python
import boto3

sagemaker_client = boto3.client('sagemaker')

# Get latest approved model
response = sagemaker_client.list_model_packages(
    ModelPackageGroupName='skin-lesion-models',
    ModelApprovalStatus='Approved',
    SortBy='CreationTime',
    SortOrder='Descending'
)

model_package_arn = response['ModelPackageSummaryList'][0]['ModelPackageArn']

# Create endpoint configuration
endpoint_config_name = 'skin-lesion-endpoint-config'
sagemaker_client.create_endpoint_config(
    EndpointConfigName=endpoint_config_name,
    ProductionVariants=[{
        'VariantName': 'AllTraffic',
        'ModelPackageName': model_package_arn,
        'InstanceType': 'ml.m5.xlarge',
        'InitialInstanceCount': 1
    }]
)

# Create endpoint
endpoint_name = 'skin-lesion-endpoint'
sagemaker_client.create_endpoint(
    EndpointName=endpoint_name,
    EndpointConfigName=endpoint_config_name
)

print(f"Endpoint {endpoint_name} is being created...")
```

## Step 12: Test Inference

```python
import boto3
import json
from PIL import Image
import io

runtime = boto3.client('sagemaker-runtime')

# Load and prepare image
image = Image.open('test_image.jpg')
image_bytes = io.BytesIO()
image.save(image_bytes, format='JPEG')
image_bytes = image_bytes.getvalue()

# Invoke endpoint
response = runtime.invoke_endpoint(
    EndpointName='skin-lesion-endpoint',
    ContentType='application/x-image',
    Body=image_bytes
)

# Parse results
result = json.loads(response['Body'].read())
print(f"Prediction: {result}")
```

## Step 13: Enable Monitoring

### Enable Model Monitor

```python
from sagemaker.model_monitor import DefaultModelMonitor

monitor = DefaultModelMonitor(
    role='<sagemaker-role-arn>',
    instance_count=1,
    instance_type='ml.m5.xlarge',
    volume_size_in_gb=20,
    max_runtime_in_seconds=3600,
)

# Create baseline
monitor.suggest_baseline(
    baseline_dataset='s3://your-bucket/baseline-data/baseline.csv',
    dataset_format={'csv': {'header': True}},
    output_s3_uri='s3://your-bucket/baseline-results',
    wait=True
)

# Create monitoring schedule
monitor.create_monitoring_schedule(
    monitor_schedule_name='skin-lesion-monitoring',
    endpoint_input='skin-lesion-endpoint',
    output_s3_uri='s3://your-bucket/monitoring-results',
    statistics=monitor.baseline_statistics(),
    constraints=monitor.suggested_constraints(),
    schedule_cron_expression='cron(0 * * * ? *)',  # Hourly
)
```

## Step 14: Setup CI/CD (Optional)

### GitHub Actions

1. Add GitHub Secrets:
   - `AWS_ACCESS_KEY_ID`
   - `AWS_SECRET_ACCESS_KEY`

2. Push code to trigger workflow:

```bash
git add .
git commit -m "Initial deployment"
git push origin main
```

The CI/CD pipeline will:
- Run code quality checks
- Execute unit tests
- Validate Terraform
- Build Docker images
- Deploy infrastructure
- Deploy SageMaker pipeline

## Troubleshooting

### Common Issues

#### 1. Insufficient Service Limits

**Error:** `ResourceLimitExceeded`

**Solution:** Request service limit increase in AWS Console

#### 2. IAM Permission Errors

**Error:** `AccessDenied`

**Solution:** Ensure IAM role has required permissions:
- SageMakerFullAccess
- S3FullAccess (or scoped to specific buckets)
- ECRFullAccess
- CloudWatchFullAccess

#### 3. Spot Instance Interruption

**Error:** Training job fails with spot instance interruption

**Solution:** Enable checkpointing in training script or use on-demand instances

#### 4. Out of Memory During Training

**Error:** `OutOfMemoryError`

**Solution:**
- Reduce batch size
- Use larger instance type
- Enable gradient checkpointing

#### 5. Pipeline Execution Fails

**Solution:**
- Check CloudWatch logs for specific error
- Verify all S3 paths are accessible
- Ensure Docker images are pushed to ECR
- Validate data format

### Getting Help

- Check CloudWatch Logs
- Review SageMaker execution details
- Check Terraform state: `terraform show`
- AWS Support (if you have support plan)

## Cleanup

To avoid ongoing charges, delete resources when done:

```bash
# Delete SageMaker endpoint
aws sagemaker delete-endpoint --endpoint-name skin-lesion-endpoint

# Delete endpoint configuration
aws sagemaker delete-endpoint-config --endpoint-config-name skin-lesion-endpoint-config

# Delete infrastructure
cd infrastructure/terraform
terraform destroy

# Manually delete S3 buckets (Terraform may not delete non-empty buckets)
aws s3 rb s3://<bucket-name> --force
```

## Next Steps

1. **Experiment with Models**: Try different architectures in `config/config.yaml`
2. **Tune Hyperparameters**: Use SageMaker Automatic Model Tuning
3. **Enable Auto-Retraining**: Configure drift detection thresholds
4. **Setup A/B Testing**: Deploy multiple model versions
5. **Add Custom Metrics**: Extend monitoring with business metrics
6. **Implement Explainability**: Add SHAP/LIME analysis to pipeline
7. **Scale Inference**: Configure auto-scaling for endpoints
8. **Multi-Region Deployment**: Deploy to multiple regions for HA

## Additional Resources

- [AWS SageMaker Documentation](https://docs.aws.amazon.com/sagemaker/)
- [SageMaker Pipelines Guide](https://docs.aws.amazon.com/sagemaker/latest/dg/pipelines.html)
- [SageMaker Model Monitor](https://docs.aws.amazon.com/sagemaker/latest/dg/model-monitor.html)
- [Terraform AWS Provider](https://registry.terraform.io/providers/hashicorp/aws/latest/docs)

## Support

For issues or questions:
- Open GitHub issue
- Check documentation
- Review AWS SageMaker forums
