# SageMaker Execution Role
resource "aws_iam_role" "sagemaker_execution" {
  name = "${local.resource_prefix}-sagemaker-execution-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Principal = {
          Service = "sagemaker.amazonaws.com"
        }
        Action = "sts:AssumeRole"
      }
    ]
  })

  tags = merge(
    local.common_tags,
    {
      Name = "${local.resource_prefix}-sagemaker-execution-role"
    }
  )
}

# SageMaker Execution Policy
resource "aws_iam_role_policy" "sagemaker_execution" {
  name = "${local.resource_prefix}-sagemaker-execution-policy"
  role = aws_iam_role.sagemaker_execution.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "s3:GetObject",
          "s3:PutObject",
          "s3:DeleteObject",
          "s3:ListBucket"
        ]
        Resource = [
          aws_s3_bucket.data.arn,
          "${aws_s3_bucket.data.arn}/*",
          aws_s3_bucket.models.arn,
          "${aws_s3_bucket.models.arn}/*",
          aws_s3_bucket.logs.arn,
          "${aws_s3_bucket.logs.arn}/*",
          aws_s3_bucket.pipeline.arn,
          "${aws_s3_bucket.pipeline.arn}/*"
        ]
      },
      {
        Effect = "Allow"
        Action = [
          "ecr:GetAuthorizationToken",
          "ecr:BatchCheckLayerAvailability",
          "ecr:GetDownloadUrlForLayer",
          "ecr:BatchGetImage"
        ]
        Resource = "*"
      },
      {
        Effect = "Allow"
        Action = [
          "cloudwatch:PutMetricData",
          "cloudwatch:GetMetricData",
          "cloudwatch:GetMetricStatistics",
          "cloudwatch:ListMetrics"
        ]
        Resource = "*"
      },
      {
        Effect = "Allow"
        Action = [
          "logs:CreateLogGroup",
          "logs:CreateLogStream",
          "logs:PutLogEvents",
          "logs:DescribeLogStreams"
        ]
        Resource = "arn:aws:logs:${local.region}:${local.account_id}:log-group:/aws/sagemaker/*"
      },
      {
        Effect = "Allow"
        Action = [
          "sagemaker:DescribeTrainingJob",
          "sagemaker:DescribeEndpoint",
          "sagemaker:DescribeEndpointConfig",
          "sagemaker:DescribeModel",
          "sagemaker:CreateModel",
          "sagemaker:CreateEndpointConfig",
          "sagemaker:CreateEndpoint",
          "sagemaker:UpdateEndpoint"
        ]
        Resource = "*"
      }
    ]
  })
}

# Attach AWS managed policies
resource "aws_iam_role_policy_attachment" "sagemaker_full_access" {
  role       = aws_iam_role.sagemaker_execution.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonSageMakerFullAccess"
}

# KMS permissions for SageMaker
resource "aws_iam_role_policy" "sagemaker_kms" {
  count = var.enable_encryption ? 1 : 0
  name  = "${local.resource_prefix}-sagemaker-kms-policy"
  role  = aws_iam_role.sagemaker_execution.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "kms:Decrypt",
          "kms:Encrypt",
          "kms:GenerateDataKey",
          "kms:DescribeKey"
        ]
        Resource = aws_kms_key.sagemaker[0].arn
      }
    ]
  })
}

# Lambda Execution Role for automation
resource "aws_iam_role" "lambda_execution" {
  name = "${local.resource_prefix}-lambda-execution-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Principal = {
          Service = "lambda.amazonaws.com"
        }
        Action = "sts:AssumeRole"
      }
    ]
  })

  tags = merge(
    local.common_tags,
    {
      Name = "${local.resource_prefix}-lambda-execution-role"
    }
  )
}

resource "aws_iam_role_policy_attachment" "lambda_basic" {
  role       = aws_iam_role.lambda_execution.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole"
}

# Lambda policy for SageMaker operations
resource "aws_iam_role_policy" "lambda_sagemaker" {
  name = "${local.resource_prefix}-lambda-sagemaker-policy"
  role = aws_iam_role.lambda_execution.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "sagemaker:StartPipelineExecution",
          "sagemaker:DescribePipeline",
          "sagemaker:ListPipelineExecutions",
          "sagemaker:StopPipelineExecution"
        ]
        Resource = "*"
      },
      {
        Effect = "Allow"
        Action = [
          "s3:GetObject",
          "s3:PutObject"
        ]
        Resource = [
          "${aws_s3_bucket.data.arn}/*",
          "${aws_s3_bucket.models.arn}/*"
        ]
      }
    ]
  })
}

# Step Functions Execution Role
resource "aws_iam_role" "step_functions" {
  name = "${local.resource_prefix}-step-functions-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Principal = {
          Service = "states.amazonaws.com"
        }
        Action = "sts:AssumeRole"
      }
    ]
  })

  tags = merge(
    local.common_tags,
    {
      Name = "${local.resource_prefix}-step-functions-role"
    }
  )
}

resource "aws_iam_role_policy" "step_functions" {
  name = "${local.resource_prefix}-step-functions-policy"
  role = aws_iam_role.step_functions.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "sagemaker:CreateTrainingJob",
          "sagemaker:DescribeTrainingJob",
          "sagemaker:StopTrainingJob",
          "sagemaker:CreateModel",
          "sagemaker:CreateEndpointConfig",
          "sagemaker:CreateEndpoint",
          "sagemaker:UpdateEndpoint",
          "sagemaker:DescribeEndpoint"
        ]
        Resource = "*"
      },
      {
        Effect = "Allow"
        Action = [
          "lambda:InvokeFunction"
        ]
        Resource = "*"
      },
      {
        Effect = "Allow"
        Action = [
          "sns:Publish"
        ]
        Resource = "*"
      },
      {
        Effect = "Allow"
        Action = [
          "iam:PassRole"
        ]
        Resource = aws_iam_role.sagemaker_execution.arn
      }
    ]
  })
}

# EventBridge Role for triggering retraining
resource "aws_iam_role" "eventbridge" {
  name = "${local.resource_prefix}-eventbridge-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Principal = {
          Service = "events.amazonaws.com"
        }
        Action = "sts:AssumeRole"
      }
    ]
  })

  tags = merge(
    local.common_tags,
    {
      Name = "${local.resource_prefix}-eventbridge-role"
    }
  )
}

resource "aws_iam_role_policy" "eventbridge" {
  name = "${local.resource_prefix}-eventbridge-policy"
  role = aws_iam_role.eventbridge.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "states:StartExecution"
        ]
        Resource = "*"
      }
    ]
  })
}
