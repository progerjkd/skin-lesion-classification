# S3 Bucket Outputs
output "data_bucket_name" {
  description = "Name of the S3 bucket for data storage"
  value       = aws_s3_bucket.data.id
}

output "data_bucket_arn" {
  description = "ARN of the S3 bucket for data storage"
  value       = aws_s3_bucket.data.arn
}

output "models_bucket_name" {
  description = "Name of the S3 bucket for model artifacts"
  value       = aws_s3_bucket.models.id
}

output "models_bucket_arn" {
  description = "ARN of the S3 bucket for model artifacts"
  value       = aws_s3_bucket.models.arn
}

output "pipeline_bucket_name" {
  description = "Name of the S3 bucket for pipeline artifacts"
  value       = aws_s3_bucket.pipeline.id
}

output "logs_bucket_name" {
  description = "Name of the S3 bucket for logs"
  value       = aws_s3_bucket.logs.id
}

# IAM Role Outputs
output "sagemaker_execution_role_arn" {
  description = "ARN of the SageMaker execution role"
  value       = aws_iam_role.sagemaker_execution.arn
}

output "sagemaker_execution_role_name" {
  description = "Name of the SageMaker execution role"
  value       = aws_iam_role.sagemaker_execution.name
}

output "lambda_execution_role_arn" {
  description = "ARN of the Lambda execution role"
  value       = aws_iam_role.lambda_execution.arn
}

output "step_functions_role_arn" {
  description = "ARN of the Step Functions execution role"
  value       = aws_iam_role.step_functions.arn
}

# ECR Outputs
output "training_ecr_repository_url" {
  description = "URL of the training ECR repository"
  value       = aws_ecr_repository.training.repository_url
}

output "inference_ecr_repository_url" {
  description = "URL of the inference ECR repository"
  value       = aws_ecr_repository.inference.repository_url
}

# SageMaker Outputs
output "model_package_group_name" {
  description = "Name of the SageMaker model package group"
  value       = aws_sagemaker_model_package_group.skin_lesion.model_package_group_name
}

output "model_package_group_arn" {
  description = "ARN of the SageMaker model package group"
  value       = aws_sagemaker_model_package_group.skin_lesion.arn
}

output "sagemaker_domain_id" {
  description = "ID of the SageMaker Studio domain (if created)"
  value       = var.environment == "dev" ? aws_sagemaker_domain.studio[0].id : null
}

output "notebook_instance_name" {
  description = "Name of the SageMaker notebook instance (if created)"
  value       = var.environment == "dev" ? aws_sagemaker_notebook_instance.dev[0].name : null
}

# KMS Outputs
output "kms_key_id" {
  description = "ID of the KMS key for encryption"
  value       = var.enable_encryption ? aws_kms_key.sagemaker[0].id : null
}

output "kms_key_arn" {
  description = "ARN of the KMS key for encryption"
  value       = var.enable_encryption ? aws_kms_key.sagemaker[0].arn : null
}

# Monitoring Outputs
output "sns_alerts_topic_arn" {
  description = "ARN of the SNS topic for alerts"
  value       = aws_sns_topic.alerts.arn
}

output "cloudwatch_dashboard_name" {
  description = "Name of the CloudWatch dashboard"
  value       = aws_cloudwatch_dashboard.main.dashboard_name
}

# Step Functions Outputs
output "retraining_state_machine_arn" {
  description = "ARN of the retraining Step Functions state machine"
  value       = aws_sfn_state_machine.retraining_pipeline.arn
}

output "retraining_state_machine_name" {
  description = "Name of the retraining Step Functions state machine"
  value       = aws_sfn_state_machine.retraining_pipeline.name
}

# Lambda Function Outputs
output "check_data_lambda_arn" {
  description = "ARN of the check data Lambda function"
  value       = aws_lambda_function.check_data.arn
}

output "evaluate_model_lambda_arn" {
  description = "ARN of the evaluate model Lambda function"
  value       = aws_lambda_function.evaluate_model.arn
}

# EventBridge Outputs
output "scheduled_retraining_rule_name" {
  description = "Name of the scheduled retraining EventBridge rule"
  value       = aws_cloudwatch_event_rule.scheduled_retraining.name
}

# General Outputs
output "region" {
  description = "AWS region"
  value       = local.region
}

output "account_id" {
  description = "AWS account ID"
  value       = local.account_id
}

output "resource_prefix" {
  description = "Resource prefix used for naming"
  value       = local.resource_prefix
}

# Configuration for Python scripts
output "config_json" {
  description = "Configuration in JSON format for use in Python scripts"
  value = jsonencode({
    region                     = local.region
    account_id                 = local.account_id
    data_bucket                = aws_s3_bucket.data.id
    models_bucket              = aws_s3_bucket.models.id
    pipeline_bucket            = aws_s3_bucket.pipeline.id
    sagemaker_role_arn         = aws_iam_role.sagemaker_execution.arn
    model_package_group_name   = aws_sagemaker_model_package_group.skin_lesion.model_package_group_name
    training_ecr_repo          = aws_ecr_repository.training.repository_url
    inference_ecr_repo         = aws_ecr_repository.inference.repository_url
    kms_key_id                 = var.enable_encryption ? aws_kms_key.sagemaker[0].id : null
    retraining_state_machine   = aws_sfn_state_machine.retraining_pipeline.arn
  })
}
