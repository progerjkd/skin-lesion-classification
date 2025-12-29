# Project Development History

## Initial Request

**Date**: December 2025

**Prompt**: "I am a devops professional and looking to break into mlops. I need to have a few ideas of projects with focus on aws, sagemaker, mlops pipelines, model retraining and monitoring for my github and portfolio. I am interested in healthcare projects with image diagnostics involving computer vision. Give me 10 project ideas, explain its challenges and datasets and aws architecture."

## Project Selection

Selected **Project #1**: Skin Lesion Classification with Automated Retraining Pipeline

## Key Development Steps

### 1. Project Setup
- Created Git repository
- Set up directory structure (22 directories, 42 files)
- Configured Python package structure

### 2. Infrastructure as Code
- Terraform files for AWS resources:
  - S3 buckets (data, models, logs, pipeline)
  - IAM roles (SageMaker, Lambda, Step Functions, EventBridge)
  - SageMaker resources (ECR, Model Registry, Notebooks)
  - Monitoring (CloudWatch alarms, dashboards, SNS)
  - Step Functions for automated retraining

### 3. ML Pipeline Implementation
- SageMaker Pipelines with:
  - Preprocessing step (train/val/test split, resize, normalize)
  - Training step (PyTorch, ResNet/EfficientNet/DenseNet support)
  - Evaluation step (accuracy, precision, recall, F1, AUC)
  - Conditional model registration (accuracy >= 85%)

### 4. Monitoring & Automation
- Data drift detection
- Model quality monitoring
- Automated retraining workflow
- EventBridge scheduling

### 5. CI/CD Pipeline
- GitHub Actions workflow:
  - Code quality checks
  - Unit tests
  - Terraform validation
  - Security scanning
  - Docker image builds
  - Automated deployment

### 6. Documentation & Diagrams
- README.md with professional badges
- ARCHITECTURE.md with detailed design decisions
- DEPLOYMENT_GUIDE.md with step-by-step instructions
- Generated 3 professional AWS architecture diagrams:
  - aws_architecture_diagram.png (complete system)
  - ml_pipeline_flow.png (ML pipeline details)
  - monitoring_workflow.png (monitoring & retraining)

## Key Technical Decisions

1. **AWS SageMaker** for managed ML infrastructure
2. **Terraform** for Infrastructure as Code
3. **PyTorch** for deep learning framework
4. **Spot Instances** for cost optimization (70% savings)
5. **S3 Intelligent-Tiering** for data storage
6. **KMS encryption** for security
7. **Step Functions** for retraining orchestration
8. **Model Registry** for version control

## Project Personalization

- **Developer**: Roger Vasconcelos
- **Email**: proger.mv@gmail.com
- **GitHub**: progerjkd
- **Repository**: https://github.com/progerjkd/skin-lesion-classification

## Next Steps

1. Push to GitHub
2. Set up AWS infrastructure with Terraform
3. Download ISIC dataset
4. Deploy ML pipeline
5. Optional: Add Streamlit UI, FastAPI service, blog post

## Project Status

✅ Complete and portfolio-ready
✅ All files committed to Git
✅ Professional diagrams generated
✅ Documentation complete
⏳ Ready to push to GitHub

## Cost Estimate

**Monthly Cost**: $200-500 (development), $500-1,500 (production)

See ARCHITECTURE.md for detailed cost breakdown.

## Datasets

- ISIC 2019: 25,331 dermoscopic images (8 categories)
- HAM10000: 10,015 dermatoscopic images
- BCN20000: 19,424 dermoscopic images

## License

MIT License
