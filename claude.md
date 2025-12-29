# Project Context for Claude

## Project Overview

**Name**: Skin Lesion Classification MLOps Pipeline
**Purpose**: Production-ready MLOps pipeline for detecting melanoma and other skin cancers using AWS SageMaker
**Owner**: Roger Vasconcelos (proger.mv@gmail.com)
**GitHub**: https://github.com/progerjkd/skin-lesion-classification
**Status**: Portfolio-ready, not yet deployed to AWS

## Project Type

This is an **MLOps demonstration project** showcasing:
- End-to-end ML pipeline automation
- AWS SageMaker best practices
- Infrastructure as Code with Terraform
- Healthcare computer vision application
- Transition from DevOps to MLOps

## Technology Stack

### Cloud & MLOps
- **AWS SageMaker**: Pipelines, Training Jobs, Endpoints, Model Monitor
- **AWS Step Functions**: Automated retraining orchestration
- **AWS EventBridge**: Scheduled and event-driven triggers
- **Amazon S3**: Data lake (4 buckets: raw data, processed, models, logs)
- **Amazon ECR**: Container registry for training/inference images
- **AWS Lambda**: Serverless automation functions
- **Amazon CloudWatch**: Monitoring, alarms, dashboards
- **AWS KMS**: Encryption key management
- **Terraform**: Infrastructure provisioning

### ML/AI
- **PyTorch 2.1+**: Deep learning framework
- **Transfer Learning**: ResNet50, EfficientNet, DenseNet121, MobileNetV2
- **Computer Vision**: Image classification (8 skin lesion categories)
- **Datasets**: ISIC 2019 (25,331 images), HAM10000, BCN20000

### DevOps
- **GitHub Actions**: CI/CD pipeline
- **Docker**: Containerization
- **Python 3.10+**: Development language
- **Git**: Version control

## Project Structure

```
.
├── config/              # Configuration files (YAML)
├── data/                # Data directories (gitignored, empty)
│   ├── raw/            # Raw ISIC dataset
│   ├── processed/      # Preprocessed images
│   └── features/       # Extracted features
├── docs/               # Documentation and diagram generation
├── images/             # Architecture diagrams (PNG)
├── infrastructure/
│   └── terraform/      # Complete AWS infrastructure as code
├── models/             # Saved models (gitignored, empty)
├── notebooks/          # Jupyter notebooks (placeholders)
├── scripts/            # Deployment and data scripts
├── src/
│   ├── pipeline/       # SageMaker pipeline definitions
│   ├── preprocessing/  # Data preprocessing
│   ├── training/       # Model training
│   ├── evaluation/     # Model evaluation
│   ├── deployment/     # Inference handlers
│   └── monitoring/     # Drift detection
└── tests/              # Unit tests
```

## Key Files to Know

### Configuration
- **config/config.yaml**: Main configuration for entire pipeline
- **requirements.txt**: Python dependencies
- **setup.py**: Package setup

### Infrastructure (Terraform)
- **infrastructure/terraform/main.tf**: Provider and backend config
- **infrastructure/terraform/s3.tf**: 4 S3 buckets with lifecycle policies
- **infrastructure/terraform/iam.tf**: IAM roles for all services
- **infrastructure/terraform/sagemaker.tf**: SageMaker resources
- **infrastructure/terraform/monitoring.tf**: CloudWatch alarms, SNS topics
- **infrastructure/terraform/step_functions.tf**: Retraining workflow

### ML Pipeline
- **src/pipeline/pipeline.py**: SageMaker Pipeline class with 4 steps
- **src/pipeline/steps.py**: Individual step definitions
- **src/preprocessing/preprocess.py**: Image preprocessing (resize, normalize, split)
- **src/training/train.py**: PyTorch training script with checkpointing
- **src/evaluation/evaluate.py**: Model evaluation metrics
- **src/monitoring/drift_detector.py**: Data/model quality monitoring

### Scripts
- **scripts/download_data.py**: Download ISIC dataset
- **scripts/deploy_pipeline.py**: Deploy SageMaker pipeline

### Documentation
- **README.md**: Main project documentation (updated with contact info)
- **ARCHITECTURE.md**: Detailed architecture decisions
- **DEPLOYMENT_GUIDE.md**: Step-by-step deployment instructions
- **PROJECT_HISTORY.md**: Development history and prompts used
- **docs/AWS_ARCHITECTURE.md**: Comprehensive AWS architecture reference
- **docs/generate_diagram.py**: Generate architecture diagrams

### Visual Assets
- **images/aws_architecture_diagram.png**: Complete system architecture
- **images/ml_pipeline_flow.png**: ML pipeline workflow
- **images/monitoring_workflow.png**: Monitoring & retraining workflow

## Current State

### What's Complete ✅
- [x] Complete project structure (43 files)
- [x] Terraform infrastructure code (9 files)
- [x] SageMaker pipeline implementation
- [x] Training, preprocessing, evaluation code
- [x] Monitoring and drift detection
- [x] CI/CD pipeline (GitHub Actions)
- [x] Professional architecture diagrams
- [x] Comprehensive documentation
- [x] Git repository initialized
- [x] All changes committed
- [x] Contact information personalized

### What's NOT Done ⏳
- [ ] AWS infrastructure not deployed (Terraform not applied)
- [ ] Data not downloaded (ISIC dataset)
- [ ] No actual model trained yet
- [ ] Not pushed to GitHub yet
- [ ] Lambda functions are placeholders
- [ ] Notebook files are empty
- [ ] No unit test implementations yet

## Important Context

### This is a Portfolio/Demo Project
- Designed to showcase MLOps skills to potential employers
- Not currently running in AWS (would cost $200-1500/month)
- Code is production-ready but untested in real AWS environment
- Focus is on architecture and MLOps practices, not model performance

### Design Decisions Made

1. **Spot Instances**: Training uses spot instances for 70% cost savings
2. **Conditional Registration**: Models only registered if accuracy >= 85%
3. **Auto-scaling**: Endpoints scale 1-5 instances based on traffic
4. **Encryption**: All data encrypted with KMS (at rest and in transit)
5. **Monitoring Schedule**: Data quality hourly, model quality every 6 hours
6. **Retraining Triggers**: Monthly schedule, drift detection, or manual
7. **S3 Lifecycle**: Intelligent-Tiering for cost optimization

### Cost Estimates
- **Development**: $200-500/month
- **Production**: $500-1,500/month
- **One-time setup**: $50-100

See ARCHITECTURE.md for detailed breakdown.

## Working with This Project

### If User Wants to Deploy to AWS
1. Ensure AWS CLI configured
2. Update config/config.yaml with actual AWS account details
3. Run: `cd infrastructure/terraform && terraform init && terraform apply`
4. Download ISIC dataset: `python scripts/download_data.py`
5. Deploy pipeline: `python scripts/deploy_pipeline.py`

### If User Wants to Push to GitHub
1. Create repo on GitHub: `progerjkd/skin-lesion-classification`
2. Add remote: `git remote add origin https://github.com/progerjkd/skin-lesion-classification.git`
3. Push: `git push -u origin main`

### If User Wants to Enhance
- Regenerate diagrams: `cd docs && python generate_diagram.py`
- Update architecture: Edit terraform files, then `terraform plan`
- Add new ML models: Update src/training/train.py model architectures

### If User Wants to Understand Architecture
- Start with: README.md → ARCHITECTURE.md → docs/AWS_ARCHITECTURE.md
- Visual: Look at images/*.png diagrams
- Code flow: src/pipeline/pipeline.py shows the orchestration

## Common Issues & Solutions

### Diagrams
- To regenerate: `cd docs && python generate_diagram.py`
- Requires: `pip install diagrams` and Graphviz installed
- Images saved to: `images/` folder

### Terraform
- Backend not configured (local state only)
- Variables need customization for real deployment
- Lambda function code is placeholder (needs implementation)

### Data
- data/ directories are empty (gitignored)
- ISIC dataset ~25GB, not included
- Download script provided but not run

## Development History

See **PROJECT_HISTORY.md** for:
- Original prompts and requirements
- Key decisions made during development
- Step-by-step development process
- What was built and why

## Next Steps Recommendations

**For Portfolio**: Push to GitHub, add demo video, write blog post
**For Learning**: Deploy to AWS, train actual model, monitor results
**For Job Applications**: Prepare to explain architecture decisions
**For Enhancement**: Add Streamlit UI, FastAPI service, A/B testing

## Questions to Ask User

Before making changes, clarify:
- Are they deploying to AWS or keeping it as portfolio project?
- Do they want to modify architecture or add features?
- Are they working on documentation or implementation?
- What's their budget if deploying to AWS?

## Tone & Approach

This user is:
- DevOps professional transitioning to MLOps
- Focused on portfolio and job applications
- Interested in AWS best practices
- Healthcare/medical imaging domain
- Wants production-ready, enterprise-grade code

Keep responses:
- Professional and technical
- Cost-conscious (mention AWS pricing)
- Focused on best practices
- Relevant to job interviews
