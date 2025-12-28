# S3 Bucket for raw data
resource "aws_s3_bucket" "data" {
  bucket = "${var.s3_bucket_prefix}-data-${local.bucket_suffix}"

  tags = merge(
    local.common_tags,
    {
      Name        = "${local.resource_prefix}-data-bucket"
      Purpose     = "Raw and processed data storage"
      DataType    = "training-data"
    }
  )
}

resource "aws_s3_bucket_versioning" "data" {
  bucket = aws_s3_bucket.data.id

  versioning_configuration {
    status = var.enable_s3_versioning ? "Enabled" : "Suspended"
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "data" {
  bucket = aws_s3_bucket.data.id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm     = var.enable_encryption ? "aws:kms" : "AES256"
      kms_master_key_id = var.enable_encryption ? aws_kms_key.sagemaker[0].arn : null
    }
  }
}

resource "aws_s3_bucket_lifecycle_configuration" "data" {
  bucket = aws_s3_bucket.data.id

  rule {
    id     = "intelligent-tiering"
    status = "Enabled"

    transition {
      days          = 30
      storage_class = "INTELLIGENT_TIERING"
    }

    transition {
      days          = 90
      storage_class = "GLACIER_IR"
    }
  }

  rule {
    id     = "delete-old-versions"
    status = "Enabled"

    noncurrent_version_expiration {
      noncurrent_days = 90
    }
  }
}

resource "aws_s3_bucket_public_access_block" "data" {
  bucket = aws_s3_bucket.data.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

# S3 Bucket for model artifacts
resource "aws_s3_bucket" "models" {
  bucket = "${var.s3_bucket_prefix}-models-${local.bucket_suffix}"

  tags = merge(
    local.common_tags,
    {
      Name        = "${local.resource_prefix}-models-bucket"
      Purpose     = "Model artifacts and checkpoints"
      DataType    = "model-artifacts"
    }
  )
}

resource "aws_s3_bucket_versioning" "models" {
  bucket = aws_s3_bucket.models.id

  versioning_configuration {
    status = "Enabled"  # Always enable versioning for models
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "models" {
  bucket = aws_s3_bucket.models.id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm     = var.enable_encryption ? "aws:kms" : "AES256"
      kms_master_key_id = var.enable_encryption ? aws_kms_key.sagemaker[0].arn : null
    }
  }
}

resource "aws_s3_bucket_public_access_block" "models" {
  bucket = aws_s3_bucket.models.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

# S3 Bucket for logs and monitoring
resource "aws_s3_bucket" "logs" {
  bucket = "${var.s3_bucket_prefix}-logs-${local.bucket_suffix}"

  tags = merge(
    local.common_tags,
    {
      Name        = "${local.resource_prefix}-logs-bucket"
      Purpose     = "Training logs and monitoring data"
      DataType    = "logs"
    }
  )
}

resource "aws_s3_bucket_server_side_encryption_configuration" "logs" {
  bucket = aws_s3_bucket.logs.id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

resource "aws_s3_bucket_lifecycle_configuration" "logs" {
  bucket = aws_s3_bucket.logs.id

  rule {
    id     = "delete-old-logs"
    status = "Enabled"

    expiration {
      days = 90
    }
  }
}

resource "aws_s3_bucket_public_access_block" "logs" {
  bucket = aws_s3_bucket.logs.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

# S3 Bucket for SageMaker Pipeline artifacts
resource "aws_s3_bucket" "pipeline" {
  bucket = "${var.s3_bucket_prefix}-pipeline-${local.bucket_suffix}"

  tags = merge(
    local.common_tags,
    {
      Name        = "${local.resource_prefix}-pipeline-bucket"
      Purpose     = "SageMaker Pipeline artifacts"
      DataType    = "pipeline-artifacts"
    }
  )
}

resource "aws_s3_bucket_server_side_encryption_configuration" "pipeline" {
  bucket = aws_s3_bucket.pipeline.id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm     = var.enable_encryption ? "aws:kms" : "AES256"
      kms_master_key_id = var.enable_encryption ? aws_kms_key.sagemaker[0].arn : null
    }
  }
}

resource "aws_s3_bucket_public_access_block" "pipeline" {
  bucket = aws_s3_bucket.pipeline.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}
