terraform {
  required_version = ">= 1.0"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }

  # Using local backend for demo/development
  # For production, uncomment and configure S3 backend
  # backend "s3" {
  #   bucket         = "your-terraform-state-bucket"
  #   key            = "skin-lesion-classification/terraform.tfstate"
  #   region         = "us-east-1"
  #   encrypt        = true
  #   dynamodb_table = "terraform-state-lock"
  # }
}

provider "aws" {
  region = var.aws_region

  default_tags {
    tags = var.tags
  }
}

# Data sources
data "aws_caller_identity" "current" {}
data "aws_region" "current" {}

# Random suffix for unique resource names
resource "random_string" "suffix" {
  length  = 8
  special = false
  upper   = false
}

locals {
  account_id = data.aws_caller_identity.current.account_id
  region     = data.aws_region.current.name

  # Resource naming
  resource_prefix = "${var.project_name}-${var.environment}"
  bucket_suffix   = random_string.suffix.result

  common_tags = merge(
    var.tags,
    {
      ProjectName = var.project_name
      Environment = var.environment
    }
  )
}
