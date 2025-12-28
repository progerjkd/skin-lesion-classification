# SNS Topic for alerts
resource "aws_sns_topic" "alerts" {
  name              = "${local.resource_prefix}-alerts"
  display_name      = "Skin Lesion ML Pipeline Alerts"
  kms_master_key_id = var.enable_encryption ? aws_kms_key.sagemaker[0].id : null

  tags = merge(
    local.common_tags,
    {
      Name = "${local.resource_prefix}-alerts-topic"
    }
  )
}

resource "aws_sns_topic_subscription" "email_alerts" {
  count = var.alert_email != "" ? 1 : 0

  topic_arn = aws_sns_topic.alerts.arn
  protocol  = "email"
  endpoint  = var.alert_email
}

# CloudWatch Alarms for Model Performance
resource "aws_cloudwatch_metric_alarm" "endpoint_invocation_errors" {
  alarm_name          = "${local.resource_prefix}-endpoint-invocation-errors"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = 2
  metric_name         = "ModelInvocationErrors"
  namespace           = "AWS/SageMaker"
  period              = 300
  statistic           = "Sum"
  threshold           = 10
  alarm_description   = "This metric monitors SageMaker endpoint invocation errors"
  alarm_actions       = [aws_sns_topic.alerts.arn]
  treat_missing_data  = "notBreaching"

  dimensions = {
    EndpointName = "${local.resource_prefix}-endpoint"
  }

  tags = local.common_tags
}

resource "aws_cloudwatch_metric_alarm" "endpoint_high_latency" {
  alarm_name          = "${local.resource_prefix}-endpoint-high-latency"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = 2
  metric_name         = "ModelLatency"
  namespace           = "AWS/SageMaker"
  period              = 300
  statistic           = "Average"
  threshold           = 1000  # 1 second
  alarm_description   = "This metric monitors SageMaker endpoint latency"
  alarm_actions       = [aws_sns_topic.alerts.arn]
  treat_missing_data  = "notBreaching"

  dimensions = {
    EndpointName = "${local.resource_prefix}-endpoint"
  }

  tags = local.common_tags
}

resource "aws_cloudwatch_metric_alarm" "training_job_failures" {
  alarm_name          = "${local.resource_prefix}-training-job-failures"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = 1
  metric_name         = "TrainingJobsFailed"
  namespace           = "AWS/SageMaker"
  period              = 300
  statistic           = "Sum"
  threshold           = 0
  alarm_description   = "Alert when training jobs fail"
  alarm_actions       = [aws_sns_topic.alerts.arn]
  treat_missing_data  = "notBreaching"

  tags = local.common_tags
}

# CloudWatch Dashboard
resource "aws_cloudwatch_dashboard" "main" {
  dashboard_name = "${local.resource_prefix}-dashboard"

  dashboard_body = jsonencode({
    widgets = [
      {
        type = "metric"
        properties = {
          metrics = [
            ["AWS/SageMaker", "ModelInvocations", { stat = "Sum" }],
            [".", "ModelInvocationErrors", { stat = "Sum" }]
          ]
          period = 300
          stat   = "Sum"
          region = local.region
          title  = "Endpoint Invocations"
        }
      },
      {
        type = "metric"
        properties = {
          metrics = [
            ["AWS/SageMaker", "ModelLatency", { stat = "Average" }],
            ["...", { stat = "p95" }],
            ["...", { stat = "p99" }]
          ]
          period = 300
          stat   = "Average"
          region = local.region
          title  = "Endpoint Latency (ms)"
        }
      },
      {
        type = "metric"
        properties = {
          metrics = [
            ["AWS/SageMaker", "CPUUtilization", { stat = "Average" }],
            [".", "MemoryUtilization", { stat = "Average" }]
          ]
          period = 300
          stat   = "Average"
          region = local.region
          title  = "Resource Utilization"
        }
      },
      {
        type = "log"
        properties = {
          query   = <<-EOQ
            SOURCE '/aws/sagemaker/Endpoints'
            | fields @timestamp, @message
            | filter @message like /ERROR/
            | sort @timestamp desc
            | limit 20
          EOQ
          region  = local.region
          title   = "Recent Endpoint Errors"
        }
      }
    ]
  })
}

# EventBridge Rule for scheduled retraining
resource "aws_cloudwatch_event_rule" "scheduled_retraining" {
  name                = "${local.resource_prefix}-scheduled-retraining"
  description         = "Trigger ML pipeline retraining on schedule"
  schedule_expression = var.retraining_schedule

  tags = merge(
    local.common_tags,
    {
      Name = "${local.resource_prefix}-retraining-schedule"
    }
  )
}

resource "aws_cloudwatch_event_target" "retraining_sfn" {
  rule      = aws_cloudwatch_event_rule.scheduled_retraining.name
  target_id = "TriggerStepFunction"
  arn       = aws_sfn_state_machine.retraining_pipeline.arn
  role_arn  = aws_iam_role.eventbridge.arn
}

# EventBridge Rule for data drift detection
resource "aws_cloudwatch_event_rule" "data_drift_detected" {
  name        = "${local.resource_prefix}-data-drift-detected"
  description = "Trigger retraining when data drift is detected"

  event_pattern = jsonencode({
    source      = ["aws.sagemaker"]
    detail-type = ["SageMaker Model Monitor Alert"]
    detail = {
      MonitoringScheduleName = [{
        prefix = local.resource_prefix
      }]
      MonitoringAlertStatus = ["InAlert"]
    }
  })

  tags = merge(
    local.common_tags,
    {
      Name = "${local.resource_prefix}-drift-detection-rule"
    }
  )
}

resource "aws_cloudwatch_event_target" "drift_retraining_sfn" {
  rule      = aws_cloudwatch_event_rule.data_drift_detected.name
  target_id = "TriggerRetrainingOnDrift"
  arn       = aws_sfn_state_machine.retraining_pipeline.arn
  role_arn  = aws_iam_role.eventbridge.arn
}
