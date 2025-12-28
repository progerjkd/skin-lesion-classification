# Step Functions State Machine for Retraining Pipeline
resource "aws_sfn_state_machine" "retraining_pipeline" {
  name     = "${local.resource_prefix}-retraining-pipeline"
  role_arn = aws_iam_role.step_functions.arn

  definition = jsonencode({
    Comment = "Automated ML retraining pipeline"
    StartAt = "CheckDataAvailability"
    States = {
      CheckDataAvailability = {
        Type     = "Task"
        Resource = "arn:aws:states:::lambda:invoke"
        Parameters = {
          FunctionName = aws_lambda_function.check_data.arn
          Payload = {
            bucket = aws_s3_bucket.data.id
          }
        }
        Next = "DataAvailable?"
        Catch = [
          {
            ErrorEquals = ["States.ALL"]
            Next        = "NotifyFailure"
          }
        ]
      }

      "DataAvailable?" = {
        Type = "Choice"
        Choices = [
          {
            Variable      = "$.Payload.dataAvailable"
            BooleanEquals = true
            Next          = "StartPipelineExecution"
          }
        ]
        Default = "NotifyInsufficientData"
      }

      StartPipelineExecution = {
        Type     = "Task"
        Resource = "arn:aws:states:::sagemaker:createPipelineExecution"
        Parameters = {
          PipelineName          = "${local.resource_prefix}-ml-pipeline"
          PipelineExecutionDisplayName = "auto-retraining-execution"
        }
        Next = "WaitForPipeline"
        Catch = [
          {
            ErrorEquals = ["States.ALL"]
            Next        = "NotifyFailure"
          }
        ]
      }

      WaitForPipeline = {
        Type    = "Wait"
        Seconds = 60
        Next    = "CheckPipelineStatus"
      }

      CheckPipelineStatus = {
        Type     = "Task"
        Resource = "arn:aws:states:::lambda:invoke"
        Parameters = {
          FunctionName = aws_lambda_function.check_pipeline_status.arn
          Payload = {
            pipelineExecutionArn = "$.PipelineExecutionArn"
          }
        }
        Next = "PipelineComplete?"
      }

      "PipelineComplete?" = {
        Type = "Choice"
        Choices = [
          {
            Variable     = "$.Payload.status"
            StringEquals = "Succeeded"
            Next         = "EvaluateModel"
          },
          {
            Variable     = "$.Payload.status"
            StringEquals = "Failed"
            Next         = "NotifyFailure"
          }
        ]
        Default = "WaitForPipeline"
      }

      EvaluateModel = {
        Type     = "Task"
        Resource = "arn:aws:states:::lambda:invoke"
        Parameters = {
          FunctionName = aws_lambda_function.evaluate_model.arn
          Payload = {
            modelPackageArn = "$.Payload.modelPackageArn"
          }
        }
        Next = "ModelMeetsThreshold?"
      }

      "ModelMeetsThreshold?" = {
        Type = "Choice"
        Choices = [
          {
            Variable      = "$.Payload.meetsThreshold"
            BooleanEquals = true
            Next          = "ApproveModel"
          }
        ]
        Default = "NotifyPoorPerformance"
      }

      ApproveModel = {
        Type     = "Task"
        Resource = "arn:aws:states:::lambda:invoke"
        Parameters = {
          FunctionName = aws_lambda_function.approve_model.arn
          Payload = {
            modelPackageArn = "$.Payload.modelPackageArn"
          }
        }
        Next = "DeployModel"
      }

      DeployModel = {
        Type     = "Task"
        Resource = "arn:aws:states:::lambda:invoke"
        Parameters = {
          FunctionName = aws_lambda_function.deploy_model.arn
          Payload = {
            modelPackageArn = "$.Payload.modelPackageArn"
            endpointName    = "${local.resource_prefix}-endpoint"
          }
        }
        Next = "NotifySuccess"
      }

      NotifySuccess = {
        Type     = "Task"
        Resource = "arn:aws:states:::sns:publish"
        Parameters = {
          TopicArn = aws_sns_topic.alerts.arn
          Subject  = "ML Retraining Successful"
          Message  = "The ML retraining pipeline completed successfully and a new model has been deployed."
        }
        End = true
      }

      NotifyFailure = {
        Type     = "Task"
        Resource = "arn:aws:states:::sns:publish"
        Parameters = {
          TopicArn = aws_sns_topic.alerts.arn
          Subject  = "ML Retraining Failed"
          Message  = "The ML retraining pipeline failed. Please check CloudWatch logs for details."
        }
        End = true
      }

      NotifyInsufficientData = {
        Type     = "Task"
        Resource = "arn:aws:states:::sns:publish"
        Parameters = {
          TopicArn = aws_sns_topic.alerts.arn
          Subject  = "ML Retraining Skipped - Insufficient Data"
          Message  = "Retraining was skipped due to insufficient new data."
        }
        End = true
      }

      NotifyPoorPerformance = {
        Type     = "Task"
        Resource = "arn:aws:states:::sns:publish"
        Parameters = {
          TopicArn = aws_sns_topic.alerts.arn
          Subject  = "ML Retraining - Model Performance Below Threshold"
          Message  = "The retrained model did not meet performance thresholds and was not deployed."
        }
        End = true
      }
    }
  })

  logging_configuration {
    log_destination        = "${aws_cloudwatch_log_group.step_functions.arn}:*"
    include_execution_data = true
    level                  = "ALL"
  }

  tags = merge(
    local.common_tags,
    {
      Name = "${local.resource_prefix}-retraining-state-machine"
    }
  )
}

resource "aws_cloudwatch_log_group" "step_functions" {
  name              = "/aws/vendedlogs/states/${local.resource_prefix}-retraining-pipeline"
  retention_in_days = 30

  tags = local.common_tags
}

# Lambda placeholders (these would be implemented separately)
resource "aws_lambda_function" "check_data" {
  filename      = "lambda_placeholder.zip"
  function_name = "${local.resource_prefix}-check-data"
  role          = aws_iam_role.lambda_execution.arn
  handler       = "index.handler"
  runtime       = "python3.11"
  timeout       = 60

  # Placeholder - actual implementation needed
  source_code_hash = filebase64sha256("lambda_placeholder.zip")

  environment {
    variables = {
      DATA_BUCKET = aws_s3_bucket.data.id
    }
  }

  tags = merge(
    local.common_tags,
    {
      Name = "${local.resource_prefix}-check-data-lambda"
    }
  )

  lifecycle {
    ignore_changes = [filename, source_code_hash]
  }
}

resource "aws_lambda_function" "check_pipeline_status" {
  filename      = "lambda_placeholder.zip"
  function_name = "${local.resource_prefix}-check-pipeline-status"
  role          = aws_iam_role.lambda_execution.arn
  handler       = "index.handler"
  runtime       = "python3.11"
  timeout       = 60

  source_code_hash = filebase64sha256("lambda_placeholder.zip")

  tags = merge(
    local.common_tags,
    {
      Name = "${local.resource_prefix}-check-pipeline-status-lambda"
    }
  )

  lifecycle {
    ignore_changes = [filename, source_code_hash]
  }
}

resource "aws_lambda_function" "evaluate_model" {
  filename      = "lambda_placeholder.zip"
  function_name = "${local.resource_prefix}-evaluate-model"
  role          = aws_iam_role.lambda_execution.arn
  handler       = "index.handler"
  runtime       = "python3.11"
  timeout       = 300

  source_code_hash = filebase64sha256("lambda_placeholder.zip")

  environment {
    variables = {
      MODEL_BUCKET = aws_s3_bucket.models.id
    }
  }

  tags = merge(
    local.common_tags,
    {
      Name = "${local.resource_prefix}-evaluate-model-lambda"
    }
  )

  lifecycle {
    ignore_changes = [filename, source_code_hash]
  }
}

resource "aws_lambda_function" "approve_model" {
  filename      = "lambda_placeholder.zip"
  function_name = "${local.resource_prefix}-approve-model"
  role          = aws_iam_role.lambda_execution.arn
  handler       = "index.handler"
  runtime       = "python3.11"
  timeout       = 60

  source_code_hash = filebase64sha256("lambda_placeholder.zip")

  tags = merge(
    local.common_tags,
    {
      Name = "${local.resource_prefix}-approve-model-lambda"
    }
  )

  lifecycle {
    ignore_changes = [filename, source_code_hash]
  }
}

resource "aws_lambda_function" "deploy_model" {
  filename      = "lambda_placeholder.zip"
  function_name = "${local.resource_prefix}-deploy-model"
  role          = aws_iam_role.lambda_execution.arn
  handler       = "index.handler"
  runtime       = "python3.11"
  timeout       = 600

  source_code_hash = filebase64sha256("lambda_placeholder.zip")

  environment {
    variables = {
      SAGEMAKER_ROLE = aws_iam_role.sagemaker_execution.arn
    }
  }

  tags = merge(
    local.common_tags,
    {
      Name = "${local.resource_prefix}-deploy-model-lambda"
    }
  )

  lifecycle {
    ignore_changes = [filename, source_code_hash]
  }
}

# Create placeholder Lambda zip
resource "null_resource" "create_lambda_placeholder" {
  provisioner "local-exec" {
    command = <<-EOT
      cat > /tmp/lambda_index.py << 'EOF'
import json

def handler(event, context):
    """Placeholder Lambda function"""
    return {
        'statusCode': 200,
        'body': json.dumps({'message': 'Placeholder - implement actual logic'})
    }
EOF
      cd /tmp && zip lambda_placeholder.zip lambda_index.py
      mv /tmp/lambda_placeholder.zip ${path.module}/lambda_placeholder.zip
    EOT
  }
}
