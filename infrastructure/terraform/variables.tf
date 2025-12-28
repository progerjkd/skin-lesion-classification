variable "project_name" {
  description = "Name of the project"
  type        = string
  default     = "skin-lesion-classification"
}

variable "environment" {
  description = "Environment (dev, staging, prod)"
  type        = string
  default     = "dev"
}

variable "aws_region" {
  description = "AWS region"
  type        = string
  default     = "us-east-1"
}

variable "tags" {
  description = "Common tags for all resources"
  type        = map(string)
  default = {
    Project     = "skin-lesion-classification"
    ManagedBy   = "terraform"
    Environment = "dev"
  }
}

# S3 Configuration
variable "s3_bucket_prefix" {
  description = "Prefix for S3 bucket names"
  type        = string
  default     = "skin-lesion-ml"
}

variable "enable_s3_versioning" {
  description = "Enable S3 versioning"
  type        = bool
  default     = true
}

# SageMaker Configuration
variable "sagemaker_instance_type" {
  description = "SageMaker instance type for training"
  type        = string
  default     = "ml.p3.2xlarge"
}

variable "sagemaker_endpoint_instance_type" {
  description = "SageMaker instance type for inference"
  type        = string
  default     = "ml.m5.xlarge"
}

variable "enable_spot_instances" {
  description = "Enable spot instances for training"
  type        = bool
  default     = true
}

variable "max_runtime_in_seconds" {
  description = "Maximum runtime for training jobs"
  type        = number
  default     = 86400  # 24 hours
}

# Model Monitoring Configuration
variable "enable_model_monitoring" {
  description = "Enable SageMaker Model Monitor"
  type        = bool
  default     = true
}

variable "monitoring_schedule_cron" {
  description = "Cron expression for monitoring schedule"
  type        = string
  default     = "cron(0 * * * ? *)"  # Every hour
}

# Auto Scaling Configuration
variable "endpoint_min_capacity" {
  description = "Minimum number of instances for endpoint"
  type        = number
  default     = 1
}

variable "endpoint_max_capacity" {
  description = "Maximum number of instances for endpoint"
  type        = number
  default     = 5
}

variable "target_invocations_per_instance" {
  description = "Target invocations per instance for scaling"
  type        = number
  default     = 1000
}

# Retraining Configuration
variable "retraining_schedule" {
  description = "EventBridge schedule for automated retraining"
  type        = string
  default     = "rate(30 days)"
}

# KMS Configuration
variable "enable_encryption" {
  description = "Enable KMS encryption"
  type        = bool
  default     = true
}

# VPC Configuration (optional)
variable "enable_vpc" {
  description = "Enable VPC for SageMaker"
  type        = bool
  default     = false
}

variable "vpc_id" {
  description = "VPC ID (if enable_vpc is true)"
  type        = string
  default     = ""
}

variable "subnet_ids" {
  description = "Subnet IDs (if enable_vpc is true)"
  type        = list(string)
  default     = []
}

# Notification Configuration
variable "alert_email" {
  description = "Email address for CloudWatch alerts"
  type        = string
  default     = ""
}
