# KMS Key for encryption
resource "aws_kms_key" "sagemaker" {
  count = var.enable_encryption ? 1 : 0

  description             = "KMS key for SageMaker encryption"
  deletion_window_in_days = 10
  enable_key_rotation     = true

  tags = merge(
    local.common_tags,
    {
      Name = "${local.resource_prefix}-sagemaker-kms-key"
    }
  )
}

resource "aws_kms_alias" "sagemaker" {
  count = var.enable_encryption ? 1 : 0

  name          = "alias/${local.resource_prefix}-sagemaker"
  target_key_id = aws_kms_key.sagemaker[0].key_id
}

# KMS Key Policy
resource "aws_kms_key_policy" "sagemaker" {
  count = var.enable_encryption ? 1 : 0

  key_id = aws_kms_key.sagemaker[0].id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid    = "Enable IAM User Permissions"
        Effect = "Allow"
        Principal = {
          AWS = "arn:aws:iam::${local.account_id}:root"
        }
        Action   = "kms:*"
        Resource = "*"
      },
      {
        Sid    = "Allow SageMaker to use the key"
        Effect = "Allow"
        Principal = {
          Service = "sagemaker.amazonaws.com"
        }
        Action = [
          "kms:Decrypt",
          "kms:Encrypt",
          "kms:GenerateDataKey",
          "kms:DescribeKey"
        ]
        Resource = "*"
      },
      {
        Sid    = "Allow CloudWatch Logs"
        Effect = "Allow"
        Principal = {
          Service = "logs.amazonaws.com"
        }
        Action = [
          "kms:Decrypt",
          "kms:Encrypt",
          "kms:GenerateDataKey"
        ]
        Resource = "*"
      }
    ]
  })
}
