# ECR Repository for custom containers
resource "aws_ecr_repository" "training" {
  name                 = "${local.resource_prefix}-training"
  image_tag_mutability = "MUTABLE"

  image_scanning_configuration {
    scan_on_push = true
  }

  encryption_configuration {
    encryption_type = var.enable_encryption ? "KMS" : "AES256"
    kms_key         = var.enable_encryption ? aws_kms_key.sagemaker[0].arn : null
  }

  tags = merge(
    local.common_tags,
    {
      Name    = "${local.resource_prefix}-training-ecr"
      Purpose = "Training container images"
    }
  )
}

resource "aws_ecr_repository" "inference" {
  name                 = "${local.resource_prefix}-inference"
  image_tag_mutability = "MUTABLE"

  image_scanning_configuration {
    scan_on_push = true
  }

  encryption_configuration {
    encryption_type = var.enable_encryption ? "KMS" : "AES256"
    kms_key         = var.enable_encryption ? aws_kms_key.sagemaker[0].arn : null
  }

  tags = merge(
    local.common_tags,
    {
      Name    = "${local.resource_prefix}-inference-ecr"
      Purpose = "Inference container images"
    }
  )
}

resource "aws_ecr_lifecycle_policy" "training" {
  repository = aws_ecr_repository.training.name

  policy = jsonencode({
    rules = [
      {
        rulePriority = 1
        description  = "Keep last 10 images"
        selection = {
          tagStatus     = "tagged"
          tagPrefixList = ["v"]
          countType     = "imageCountMoreThan"
          countNumber   = 10
        }
        action = {
          type = "expire"
        }
      },
      {
        rulePriority = 2
        description  = "Remove untagged images after 7 days"
        selection = {
          tagStatus   = "untagged"
          countType   = "sinceImagePushed"
          countUnit   = "days"
          countNumber = 7
        }
        action = {
          type = "expire"
        }
      }
    ]
  })
}

resource "aws_ecr_lifecycle_policy" "inference" {
  repository = aws_ecr_repository.inference.name

  policy = jsonencode({
    rules = [
      {
        rulePriority = 1
        description  = "Keep last 10 images"
        selection = {
          tagStatus     = "tagged"
          tagPrefixList = ["v"]
          countType     = "imageCountMoreThan"
          countNumber   = 10
        }
        action = {
          type = "expire"
        }
      },
      {
        rulePriority = 2
        description  = "Remove untagged images after 7 days"
        selection = {
          tagStatus   = "untagged"
          countType   = "sinceImagePushed"
          countUnit   = "days"
          countNumber = 7
        }
        action = {
          type = "expire"
        }
      }
    ]
  })
}

# SageMaker Model Package Group (Model Registry)
resource "aws_sagemaker_model_package_group" "skin_lesion" {
  model_package_group_name        = "${local.resource_prefix}-models"
  model_package_group_description = "Model registry for skin lesion classification models"

  tags = merge(
    local.common_tags,
    {
      Name = "${local.resource_prefix}-model-registry"
    }
  )
}

# SageMaker Feature Store (requires additional setup)
# Note: Feature groups are typically created programmatically

# SageMaker Domain for Studio (optional, for experimentation)
resource "aws_sagemaker_domain" "studio" {
  count       = var.environment == "dev" ? 1 : 0
  domain_name = "${local.resource_prefix}-studio"
  auth_mode   = "IAM"
  vpc_id      = var.enable_vpc ? var.vpc_id : null
  subnet_ids  = var.enable_vpc ? var.subnet_ids : null

  default_user_settings {
    execution_role = aws_iam_role.sagemaker_execution.arn

    sharing_settings {
      notebook_output_option = "Allowed"
    }
  }

  tags = merge(
    local.common_tags,
    {
      Name = "${local.resource_prefix}-studio-domain"
    }
  )
}

# SageMaker Notebook Instance (for development)
resource "aws_sagemaker_notebook_instance" "dev" {
  count                  = var.environment == "dev" ? 1 : 0
  name                   = "${local.resource_prefix}-notebook"
  role_arn               = aws_iam_role.sagemaker_execution.arn
  instance_type          = "ml.t3.medium"
  platform_identifier    = "notebook-al2-v2"
  volume_size            = 20
  direct_internet_access = "Enabled"

  lifecycle_config_name = aws_sagemaker_notebook_instance_lifecycle_configuration.dev[0].name

  tags = merge(
    local.common_tags,
    {
      Name = "${local.resource_prefix}-dev-notebook"
    }
  )
}

resource "aws_sagemaker_notebook_instance_lifecycle_configuration" "dev" {
  count = var.environment == "dev" ? 1 : 0
  name  = "${local.resource_prefix}-notebook-lc"

  on_start = base64encode(<<-EOF
    #!/bin/bash
    set -e

    # Install additional packages
    sudo -u ec2-user -i <<'USEREOF'
    source /home/ec2-user/anaconda3/bin/activate python3
    pip install --upgrade pip
    pip install sagemaker[all] boto3 pandas scikit-learn matplotlib seaborn
    source /home/ec2-user/anaconda3/bin/deactivate
    USEREOF

    echo "Notebook instance setup complete"
  EOF
  )
}

# CloudWatch Log Groups for SageMaker
resource "aws_cloudwatch_log_group" "sagemaker_training" {
  name              = "/aws/sagemaker/TrainingJobs"
  retention_in_days = 30

  kms_key_id = var.enable_encryption ? aws_kms_key.sagemaker[0].arn : null

  tags = merge(
    local.common_tags,
    {
      Name = "${local.resource_prefix}-training-logs"
    }
  )
}

resource "aws_cloudwatch_log_group" "sagemaker_endpoints" {
  name              = "/aws/sagemaker/Endpoints"
  retention_in_days = 30

  kms_key_id = var.enable_encryption ? aws_kms_key.sagemaker[0].arn : null

  tags = merge(
    local.common_tags,
    {
      Name = "${local.resource_prefix}-endpoint-logs"
    }
  )
}

resource "aws_cloudwatch_log_group" "sagemaker_processing" {
  name              = "/aws/sagemaker/ProcessingJobs"
  retention_in_days = 30

  kms_key_id = var.enable_encryption ? aws_kms_key.sagemaker[0].arn : null

  tags = merge(
    local.common_tags,
    {
      Name = "${local.resource_prefix}-processing-logs"
    }
  )
}
