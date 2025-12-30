# Terraform Issues and Fixes

This document tracks all Terraform configuration issues encountered during the project setup, their root causes, solutions implemented, and deployment options.

## Issues Identified and Fixed

### Issue 1: SageMaker Studio Domain VPC Configuration ❌ → ✅ Fixed

**Problem:**
- SageMaker Studio domain **requires** VPC configuration when created
- Cannot accept `null` values for `vpc_id` and `subnet_ids`
- Original code attempted conditional VPC setup that would fail when `enable_vpc = false`

**Original Code:**
```hcl
resource "aws_sagemaker_domain" "studio" {
  count       = var.environment == "dev" ? 1 : 0
  domain_name = "${local.resource_prefix}-studio"
  auth_mode   = "IAM"
  vpc_id      = var.enable_vpc ? var.vpc_id : null  # ❌ Cannot be null
  subnet_ids  = var.enable_vpc ? var.subnet_ids : null  # ❌ Cannot be null
  ...
}
```

**Error Message:**
```
Error: InvalidParameterValue: SageMaker Studio domain requires VPC configuration
```

**Solution Implemented:**
Made Studio domain creation conditional on **both** dev environment **and** VPC being enabled:

```hcl
resource "aws_sagemaker_domain" "studio" {
  count       = (var.environment == "dev" && var.enable_vpc) ? 1 : 0
  domain_name = "${local.resource_prefix}-studio"
  auth_mode   = "IAM"
  vpc_id      = var.vpc_id      # ✅ Now guaranteed to be non-null
  subnet_ids  = var.subnet_ids  # ✅ Now guaranteed to be non-null
  ...
}
```

**File Modified:** `infrastructure/terraform/sagemaker.tf` (lines 133-138)

**Impact:**
- Studio domain will only be created when VPC is explicitly enabled
- Prevents Terraform apply failures
- Allows simplified deployment without VPC for development/testing

---

### Issue 2: S3 Lifecycle Configuration Syntax ⚠️ → ✅ Fixed

**Problem:**
- Attempted to transition objects to `INTELLIGENT_TIERING` via lifecycle rule
- This approach is deprecated and causes warnings
- `INTELLIGENT_TIERING` should be enabled via separate resource, not lifecycle transition
- Missing `filter` blocks in lifecycle rules (required by newer AWS provider versions)

**Original Code:**
```hcl
resource "aws_s3_bucket_lifecycle_configuration" "data" {
  bucket = aws_s3_bucket.data.id

  rule {
    id     = "intelligent-tiering"
    status = "Enabled"

    transition {
      days          = 30
      storage_class = "INTELLIGENT_TIERING"  # ❌ Deprecated approach
    }

    transition {
      days          = 90
      storage_class = "GLACIER_IR"
    }
  }
  # ❌ Missing filter block
}
```

**Warning Messages:**
```
Warning: Invalid Attribute Combination
No attribute specified when one (and only one) of
[rule[0].filter,rule[0].prefix] is required
```

**Solution Implemented:**
1. Removed intelligent tiering transition (can be enabled manually in AWS Console if needed)
2. Added `filter` blocks to all lifecycle rules
3. Kept standard transitions (STANDARD → GLACIER_IR)

```hcl
resource "aws_s3_bucket_lifecycle_configuration" "data" {
  bucket = aws_s3_bucket.data.id

  rule {
    id     = "transition-to-glacier"
    status = "Enabled"

    filter {
      prefix = ""  # ✅ Filter block added
    }

    transition {
      days          = 90
      storage_class = "GLACIER_IR"
    }
  }

  rule {
    id     = "delete-old-versions"
    status = "Enabled"

    filter {
      prefix = ""  # ✅ Filter block added
    }

    noncurrent_version_expiration {
      noncurrent_days = 90
    }
  }
}
```

**Files Modified:**
- `infrastructure/terraform/s3.tf` (lines 34-70 for data bucket)
- `infrastructure/terraform/s3.tf` (lines 143-157 for logs bucket)

**Impact:**
- No more deprecation warnings
- Terraform validation passes cleanly
- Lifecycle rules work correctly
- Intelligent-Tiering can still be enabled manually if needed

---

### Issue 3: Lambda Placeholder Files ✅ Already Fixed

**Status:** Files exist and are properly configured

**Files:**
- `infrastructure/terraform/lambda_placeholder.py`
- `infrastructure/terraform/lambda_placeholder.zip`

**Note:** These are placeholder files for Lambda functions. Actual Lambda implementations should be added separately when implementing the retraining workflow.

---

## Verification

All issues have been resolved. Terraform validation passes:

```bash
cd infrastructure/terraform
terraform init -backend=false
terraform validate
# Output: Success! The configuration is valid.
```

## Deployment Options

Based on the fixes above, here are your deployment options:

### Option 1: Simplified Deployment (Recommended for Quick Start)

**Configuration:**
```hcl
# terraform.tfvars
environment = "dev"
enable_vpc = false
enable_encryption = true
```

**What Gets Created:**
- ✅ S3 buckets (4 buckets)
- ✅ IAM roles and policies
- ✅ ECR repositories
- ✅ SageMaker Model Registry
- ✅ CloudWatch dashboards and alarms
- ✅ Step Functions state machine
- ✅ Lambda functions (placeholders)
- ❌ SageMaker Studio domain (skipped - requires VPC)
- ❌ SageMaker Notebook instances (optional)

**Time:** 2-3 hours
**Cost:** ~$20-30/month (minimal usage)

### Option 2: Full Deployment with VPC

**Configuration:**
```hcl
# terraform.tfvars
environment = "dev"
enable_vpc = true
vpc_id = "vpc-xxxxxxxxx"  # Your existing VPC ID
subnet_ids = ["subnet-xxxxx", "subnet-yyyyy"]  # At least 2 subnets
enable_encryption = true
```

**What Gets Created:**
- ✅ Everything from Option 1
- ✅ SageMaker Studio domain (with VPC)
- ✅ SageMaker Notebook instances

**Prerequisites:**
- Existing VPC with at least 2 subnets in different AZs
- VPC endpoints for S3, SageMaker API, CloudWatch Logs (optional but recommended)

**Time:** 6-12 hours
**Cost:** ~$50-100/month

### Option 3: Production Deployment

**Configuration:**
```hcl
# terraform.tfvars
environment = "prod"
enable_vpc = true
enable_encryption = true
# ... production-specific settings
```

**Additional Considerations:**
- S3 backend for Terraform state
- Multi-AZ deployment
- Enhanced monitoring and alerting
- Cost optimization policies
- Security hardening

**Time:** 1-2 days
**Cost:** ~$200-500/month

## Decision Tree

```
Start
  │
  ├─ Do you need SageMaker Studio?
  │   │
  │   ├─ Yes → Option 2 (Full Deployment with VPC)
  │   │         Requires: VPC with subnets
  │   │
  │   └─ No → Option 1 (Simplified Deployment)
  │            No VPC required
  │
  └─ Is this for production?
      │
      ├─ Yes → Option 3 (Production Deployment)
      │         Requires: Full infrastructure setup
      │
      └─ No → Option 1 or 2 (based on Studio need)
```

## Common Issues and Solutions

### Issue: Terraform Plan Shows Errors

**Solution:**
1. Run `terraform init` to ensure providers are up to date
2. Run `terraform validate` to check syntax
3. Check that all required variables are set in `terraform.tfvars`
4. Verify AWS credentials are configured: `aws sts get-caller-identity`

### Issue: VPC Configuration Errors

**Solution:**
- If you don't need Studio, set `enable_vpc = false`
- If you need Studio, ensure VPC has at least 2 subnets in different AZs
- Verify VPC has internet gateway or NAT gateway for outbound access

### Issue: S3 Lifecycle Warnings

**Solution:**
- All lifecycle rules now include `filter` blocks
- If you see warnings, ensure you're using the latest version of the fixes

### Issue: Lambda Function Errors

**Solution:**
- Lambda placeholder files are provided but contain minimal code
- Implement actual Lambda functions when ready to use Step Functions workflow
- Functions can be updated after initial deployment

## Next Steps

1. **Choose Deployment Option** based on your needs
2. **Create `terraform.tfvars`** with your configuration
3. **Run `terraform plan`** to preview changes
4. **Run `terraform apply`** to deploy infrastructure
5. **Save Terraform outputs** for pipeline deployment:
   ```bash
   terraform output -json > ../../terraform-outputs.json
   ```

## Additional Resources

- [AWS SageMaker Studio Documentation](https://docs.aws.amazon.com/sagemaker/latest/dg/studio.html)
- [S3 Lifecycle Configuration](https://docs.aws.amazon.com/AmazonS3/latest/userguide/object-lifecycle-mgmt.html)
- [Terraform AWS Provider Documentation](https://registry.terraform.io/providers/hashicorp/aws/latest/docs)

## Summary

✅ **All Terraform issues resolved**
✅ **Configuration validates successfully**
✅ **Ready for deployment**

Choose the deployment option that best fits your needs and time constraints.
