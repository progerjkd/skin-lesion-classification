"""
Generate AWS Architecture Diagram
This script creates a diagram that can be:
1. Exported to draw.io/Lucidchart XML format
2. Rendered as PNG/SVG
3. Exported to various formats

Install: pip install diagrams
"""

from diagrams import Diagram, Cluster, Edge
from diagrams.aws.storage import S3
from diagrams.aws.ml import Sagemaker, SagemakerModel, SagemakerNotebook
from diagrams.aws.compute import Lambda, EC2Instance
from diagrams.aws.integration import Eventbridge, StepFunctions, SNS
from diagrams.aws.management import Cloudwatch
from diagrams.aws.network import ALB, VPC
from diagrams.aws.security import IAM, KMS, SecretsManager
from diagrams.aws.devtools import Codebuild, Codepipeline
from diagrams.custom import Custom
from diagrams.programming.framework import React
from diagrams.programming.language import Python
from diagrams.onprem.container import Docker
from diagrams.onprem.vcs import Github
from diagrams.onprem.ci import GithubActions


def create_architecture_diagram():
    """Create the main AWS architecture diagram."""

    graph_attr = {
        "fontsize": "14",
        "bgcolor": "white",
        "pad": "0.5",
        "splines": "ortho",
    }

    with Diagram(
        "Skin Lesion Classification - MLOps Pipeline",
        filename="aws_architecture_diagram",
        direction="TB",
        graph_attr=graph_attr,
        show=False,
    ):

        # External Data Sources
        with Cluster("External Data Sources"):
            data_sources = [
                Python("ISIC Archive"),
                Python("Kaggle"),
                Python("HAM10000")
            ]

        # CI/CD Pipeline
        with Cluster("CI/CD Pipeline"):
            github = Github("GitHub")
            actions = GithubActions("GitHub Actions")
            docker_build = Docker("Build Images")

            github >> actions >> docker_build

        # Storage Layer
        with Cluster("Storage Layer"):
            s3_raw = S3("Raw Data\n~100GB")
            s3_processed = S3("Processed\n~50GB")
            s3_models = S3("Models\n~10GB")
            s3_logs = S3("Logs\n30-day retention")

        # Container Registry
        with Cluster("Container Registry"):
            ecr_training = Docker("Training Image\nPyTorch 2.1")
            ecr_inference = Docker("Inference Image\nFastAPI")

        # SageMaker ML Pipeline
        with Cluster("SageMaker ML Pipeline"):
            pipeline = Sagemaker("ML Pipeline\nOrchestrator")

            with Cluster("Processing"):
                preprocessing = Sagemaker("Preprocessing\nml.m5.xlarge")

            with Cluster("Training"):
                training = Sagemaker("Training Job\nml.p3.2xlarge\nSpot Instance")

            with Cluster("Evaluation"):
                evaluation = Sagemaker("Evaluation\nml.m5.xlarge")

            with Cluster("Model Registry"):
                registry = SagemakerModel("Model Registry\nVersioned")

        # Deployment
        with Cluster("Production Deployment"):
            with Cluster("Auto-Scaling Group"):
                alb = ALB("Load Balancer")
                endpoints = [
                    Sagemaker("Endpoint 1\nml.m5.xlarge"),
                    Sagemaker("Endpoint 2\nml.m5.xlarge"),
                    Sagemaker("Endpoint N\n(1-5 instances)")
                ]

        # Monitoring
        with Cluster("Monitoring & Observability"):
            model_monitor = Sagemaker("Model Monitor")
            cloudwatch = Cloudwatch("CloudWatch\nMetrics & Logs")
            dashboards = Cloudwatch("Dashboards")
            alarms = Cloudwatch("Alarms")

        # Automation
        with Cluster("Automation & Orchestration"):
            eventbridge = Eventbridge("EventBridge\nScheduled/Event")
            step_functions = StepFunctions("Step Functions\nRetraining Workflow")

            with Cluster("Lambda Functions"):
                lambda_funcs = [
                    Lambda("Check Data"),
                    Lambda("Evaluate"),
                    Lambda("Deploy")
                ]

        # Notifications
        with Cluster("Notifications"):
            sns = SNS("SNS Topics\nAlerts")

        # Security
        with Cluster("Security & Governance"):
            iam = IAM("IAM Roles\n& Policies")
            kms = KMS("KMS Keys\nEncryption")
            vpc = VPC("VPC\n(Optional)")

        # Data Flow
        data_sources >> s3_raw
        s3_raw >> preprocessing

        docker_build >> ecr_training
        docker_build >> ecr_inference

        ecr_training >> Edge(label="uses") >> training
        ecr_inference >> Edge(label="uses") >> alb

        # Pipeline flow
        pipeline >> preprocessing >> s3_processed
        s3_processed >> training >> s3_models
        s3_models >> evaluation
        evaluation >> Edge(label="if accuracy >= 85%") >> registry
        registry >> Edge(label="approved") >> alb
        alb >> endpoints

        # Monitoring flow
        endpoints >> Edge(label="logs") >> model_monitor
        model_monitor >> cloudwatch
        cloudwatch >> dashboards
        cloudwatch >> alarms

        # Retraining flow
        alarms >> Edge(label="drift detected") >> eventbridge
        eventbridge >> step_functions
        step_functions >> lambda_funcs[0]
        lambda_funcs >> step_functions
        step_functions >> Edge(label="trigger") >> pipeline

        # Notifications
        alarms >> sns
        step_functions >> sns

        # Security
        kms >> Edge(label="encrypts", style="dashed") >> s3_raw
        kms >> Edge(label="encrypts", style="dashed") >> s3_models
        iam >> Edge(label="secures", style="dashed") >> training
        iam >> Edge(label="secures", style="dashed") >> endpoints

        # Logs
        training >> Edge(label="logs") >> s3_logs
        endpoints >> Edge(label="logs") >> s3_logs
        lambda_funcs >> Edge(label="logs") >> s3_logs


def create_ml_pipeline_diagram():
    """Create detailed ML pipeline flow diagram."""

    with Diagram(
        "ML Pipeline Detailed Flow",
        filename="ml_pipeline_flow",
        direction="LR",
        show=False,
    ):

        # Input
        s3_input = S3("S3 Raw Data")

        # Preprocessing
        with Cluster("Preprocessing"):
            preprocess = Sagemaker("Split Data\n80/10/10")
            resize = Python("Resize\n224x224")
            normalize = Python("Normalize\nImageNet")

        # Training
        with Cluster("Training"):
            train = Sagemaker("PyTorch Training\nResNet50")
            checkpoint = S3("Checkpoints")

        # Evaluation
        with Cluster("Evaluation"):
            evaluate = Sagemaker("Calculate Metrics")
            metrics = Cloudwatch("Accuracy\nPrecision\nRecall\nAUC")

        # Registry
        registry = SagemakerModel("Model Registry\nv1.x")

        # Deployment
        endpoint = Sagemaker("SageMaker\nEndpoint")

        # Flow
        s3_input >> preprocess >> resize >> normalize
        normalize >> train >> checkpoint
        train >> evaluate >> metrics
        metrics >> Edge(label="≥ 85%") >> registry
        registry >> endpoint


def create_monitoring_diagram():
    """Create monitoring and retraining workflow diagram."""

    with Diagram(
        "Monitoring & Retraining Workflow",
        filename="monitoring_workflow",
        direction="TB",
        show=False,
    ):

        # Inference
        endpoint = Sagemaker("Production\nEndpoint")

        # Monitoring
        with Cluster("Model Monitor"):
            data_quality = Sagemaker("Data Quality\nHourly")
            model_quality = Sagemaker("Model Quality\n6-hourly")
            drift = Sagemaker("Drift Detection")

        # Observability
        cloudwatch = Cloudwatch("CloudWatch")
        alarms = Cloudwatch("Alarms")

        # Automation
        eventbridge = Eventbridge("EventBridge")

        with Cluster("Step Functions Workflow"):
            sf_start = StepFunctions("Start")
            check_data = Lambda("Check Data")
            start_pipeline = StepFunctions("Start Pipeline")
            wait_complete = StepFunctions("Wait")
            deploy = Lambda("Deploy Model")
            notify = SNS("Notify Team")

        # Flow
        endpoint >> data_quality
        endpoint >> model_quality
        data_quality >> drift
        model_quality >> drift

        drift >> cloudwatch >> alarms
        alarms >> eventbridge

        eventbridge >> sf_start
        sf_start >> check_data >> start_pipeline
        start_pipeline >> wait_complete >> deploy
        deploy >> notify


if __name__ == "__main__":
    print("Generating AWS Architecture Diagrams...")
    print("\nCreating main architecture diagram...")
    create_architecture_diagram()
    print("✓ Created: aws_architecture_diagram.png")

    print("\nCreating ML pipeline flow diagram...")
    create_ml_pipeline_diagram()
    print("✓ Created: ml_pipeline_flow.png")

    print("\nCreating monitoring workflow diagram...")
    create_monitoring_diagram()
    print("✓ Created: monitoring_workflow.png")

    print("\n" + "="*60)
    print("All diagrams generated successfully!")
    print("="*60)
    print("\nYou can:")
    print("1. View the PNG files directly")
    print("2. Import them into presentations")
    print("3. Convert to SVG for web use")
    print("4. Import into Lucidchart/draw.io")
    print("\nFor draw.io import:")
    print("- Open draw.io")
    print("- File → Import → Select PNG file")
    print("- The diagram will be imported with editable elements")
