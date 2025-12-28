# Architecture Overview

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          Data Ingestion Layer                            │
├─────────────────────────────────────────────────────────────────────────┤
│  ISIC/Kaggle Datasets  →  S3 Raw Data Bucket  →  Data Validation        │
└─────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────┐
│                        SageMaker ML Pipeline                             │
├─────────────────────────────────────────────────────────────────────────┤
│  ┌──────────────┐   ┌──────────────┐   ┌──────────────┐                │
│  │ Preprocessing│ → │   Training   │ → │  Evaluation  │                │
│  │   Job        │   │     Job      │   │     Job      │                │
│  └──────────────┘   └──────────────┘   └──────────────┘                │
│         ↓                  ↓                   ↓                         │
│  ┌──────────────────────────────────────────────────┐                   │
│  │         Conditional Model Registration           │                   │
│  │         (if accuracy > threshold)                │                   │
│  └──────────────────────────────────────────────────┘                   │
└─────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────┐
│                      Model Registry & Deployment                         │
├─────────────────────────────────────────────────────────────────────────┤
│  Model Package → Manual/Auto Approval → Endpoint Deployment             │
│                                                                          │
│  ┌─────────────────┐        ┌──────────────────┐                       │
│  │ Model Registry  │   →    │  SageMaker       │                       │
│  │ (Versioned)     │        │  Endpoint        │                       │
│  └─────────────────┘        └──────────────────┘                       │
│                                     │                                    │
│                             Auto-scaling Group                           │
│                             (1-5 instances)                              │
└─────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────┐
│                        Monitoring & Observability                        │
├─────────────────────────────────────────────────────────────────────────┤
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐                  │
│  │ Data Quality │  │ Model Quality│  │  Drift       │                  │
│  │ Monitor      │  │ Monitor      │  │  Detection   │                  │
│  └──────────────┘  └──────────────┘  └──────────────┘                  │
│         ↓                 ↓                   ↓                          │
│  ┌────────────────────────────────────────────────────┐                 │
│  │           CloudWatch Metrics & Alarms              │                 │
│  └────────────────────────────────────────────────────┘                 │
└─────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────┐
│                    Automated Retraining Workflow                         │
├─────────────────────────────────────────────────────────────────────────┤
│  Triggers:                                                               │
│    • Scheduled (monthly)                                                 │
│    • Data drift detected                                                 │
│    • Model quality degradation                                           │
│    • Manual trigger                                                      │
│                                                                          │
│  ┌────────────────────────────────────────────────────────┐             │
│  │         Step Functions State Machine                   │             │
│  │  Check Data → Start Pipeline → Evaluate → Deploy       │             │
│  └────────────────────────────────────────────────────────┘             │
└─────────────────────────────────────────────────────────────────────────┘
```

## AWS Services Used

### Storage
- **Amazon S3**: Data lake for raw images, processed data, model artifacts, and logs
  - Raw data bucket
  - Processed data bucket
  - Model artifacts bucket
  - Logs bucket

### Compute
- **SageMaker Processing**: Data preprocessing and validation
- **SageMaker Training**: Model training with spot instances
- **SageMaker Endpoints**: Real-time inference
- **AWS Lambda**: Automation and orchestration tasks

### ML & AI
- **SageMaker Pipelines**: End-to-end ML workflow orchestration
- **SageMaker Model Registry**: Model versioning and lifecycle management
- **SageMaker Model Monitor**: Data drift and model quality monitoring
- **SageMaker Experiments**: Experiment tracking

### Container Registry
- **Amazon ECR**: Docker container images for training and inference

### Orchestration
- **AWS Step Functions**: Retraining workflow orchestration
- **Amazon EventBridge**: Scheduled and event-driven triggers

### Monitoring & Logging
- **Amazon CloudWatch**: Metrics, logs, dashboards, and alarms
- **AWS X-Ray**: Distributed tracing (optional)

### Security
- **AWS KMS**: Encryption key management
- **IAM**: Access control and permissions
- **AWS Secrets Manager**: Credentials management

### Notifications
- **Amazon SNS**: Alerts and notifications

### CI/CD
- **GitHub Actions**: CI/CD pipeline
- **AWS CodePipeline**: Alternative CI/CD (optional)

## Data Flow

### Training Pipeline Flow

1. **Data Ingestion**
   ```
   ISIC Dataset → S3 Raw Bucket → EventBridge Trigger
   ```

2. **Preprocessing**
   ```
   S3 Raw Data → SageMaker Processing Job →
   [Train/Val/Test Split, Resize, Normalize] →
   S3 Processed Bucket
   ```

3. **Training**
   ```
   S3 Processed Data → SageMaker Training Job →
   [ResNet/EfficientNet Training] →
   Model Artifacts → S3 Models Bucket
   ```

4. **Evaluation**
   ```
   Model Artifacts + Test Data →
   SageMaker Processing Job →
   [Metrics Calculation] →
   Evaluation Report → S3
   ```

5. **Model Registration**
   ```
   IF accuracy >= threshold:
     Register Model → Model Registry
     Status: PendingManualApproval
   ELSE:
     Skip registration
     Notify team
   ```

6. **Deployment**
   ```
   Approved Model → Create Endpoint Config →
   Create/Update Endpoint →
   Enable Auto-scaling →
   Enable Model Monitor
   ```

### Inference Flow

```
Client Request → API Gateway (optional) →
SageMaker Endpoint → Model Inference →
Response (JSON)
```

### Monitoring Flow

```
Inference Data → Model Monitor →
Drift Detection →
IF drift detected:
  Trigger EventBridge Rule →
  Start Step Functions →
  Initiate Retraining
```

## Infrastructure as Code

### Terraform Resources

```hcl
# S3 Buckets
- aws_s3_bucket.data
- aws_s3_bucket.models
- aws_s3_bucket.pipeline
- aws_s3_bucket.logs

# IAM Roles
- aws_iam_role.sagemaker_execution
- aws_iam_role.lambda_execution
- aws_iam_role.step_functions

# SageMaker
- aws_sagemaker_model_package_group
- aws_sagemaker_domain (dev only)
- aws_sagemaker_notebook_instance (dev only)

# ECR
- aws_ecr_repository.training
- aws_ecr_repository.inference

# Monitoring
- aws_cloudwatch_dashboard
- aws_cloudwatch_metric_alarm
- aws_sns_topic

# Automation
- aws_sfn_state_machine
- aws_lambda_function
- aws_cloudwatch_event_rule

# Security
- aws_kms_key
- aws_kms_alias
```

## Security Architecture

### Data Security
- Encryption at rest (S3, EBS) using KMS
- Encryption in transit (TLS)
- S3 bucket policies (block public access)
- VPC isolation (optional)

### Access Control
- IAM roles with least privilege
- SageMaker execution role
- Lambda execution role
- Step Functions execution role
- Resource-based policies

### Compliance
- HIPAA-eligible services
- PHI data handling
- Audit logging (CloudTrail)
- Data anonymization

## Cost Optimization Strategies

### Compute
- Spot instances for training (70% savings)
- Auto-scaling for endpoints
- Right-sizing instance types
- Serverless Lambda for automation

### Storage
- S3 Intelligent-Tiering
- Lifecycle policies (archive old data)
- Delete old model artifacts

### Monitoring
- Log retention policies (30 days)
- Metric filtering
- Reserved capacity (for production)

## Scalability

### Horizontal Scaling
- Multi-instance training
- Endpoint auto-scaling (1-5 instances)
- Distributed preprocessing

### Vertical Scaling
- Instance type flexibility
- GPU acceleration (P3 instances)
- Memory optimization

## High Availability

### Multi-AZ Deployment
- SageMaker endpoints across AZs
- S3 automatically replicated
- EventBridge and Step Functions managed services

### Disaster Recovery
- Model artifacts versioned in S3
- Infrastructure as Code (recreate in minutes)
- Multi-region deployment (optional)

## Performance Optimization

### Training
- Mixed precision training
- Gradient checkpointing
- Data caching
- Optimized data loading

### Inference
- Batch transform for bulk predictions
- Model compilation (SageMaker Neo)
- Caching frequently requested predictions
- Async inference for non-real-time use cases

## Monitoring Strategy

### Metrics to Track

**Model Performance**
- Accuracy, Precision, Recall, F1
- AUC-ROC
- Per-class metrics

**Operational Metrics**
- Endpoint latency (P50, P95, P99)
- Invocations per minute
- Error rate
- CPU/Memory utilization

**Data Quality**
- Input data distribution
- Missing values
- Feature drift

**Cost Metrics**
- Training cost per run
- Inference cost per 1000 requests
- Storage cost
- Total monthly cost

### Alerting Thresholds

- Endpoint errors > 10/hour → Critical
- Latency P95 > 1000ms → Warning
- Data drift detected → Warning
- Model accuracy drop > 5% → Critical
- Monthly cost > $500 → Warning

## Future Enhancements

1. **Multi-Model Endpoints**: Deploy multiple model versions
2. **A/B Testing**: Traffic splitting between models
3. **Explainability**: SHAP/LIME integration
4. **Federated Learning**: Multi-site training
5. **Edge Deployment**: IoT Greengrass for clinics
6. **Real-time Streaming**: Kinesis for online predictions
7. **Human-in-the-Loop**: Active learning pipeline
8. **Multi-Region**: Global deployment for low latency
