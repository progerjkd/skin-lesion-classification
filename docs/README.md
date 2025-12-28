# Documentation

This directory contains comprehensive architecture documentation and diagrams for the Skin Lesion Classification MLOps Pipeline.

## Files

### 📄 AWS_ARCHITECTURE.md
Complete AWS architecture documentation in text format with:
- Detailed component descriptions
- ASCII diagrams for quick reference
- Service configurations
- Cost breakdown
- Security architecture
- Disaster recovery plans

**Use this for:**
- Understanding the complete system architecture
- Quick reference without visual tools
- Copy-paste into documentation
- Text-based reviews

### 🎨 generate_diagram.py
Python script to generate professional architecture diagrams using the `diagrams` library.

**Installation:**
```bash
pip install diagrams
```

**Usage:**
```bash
cd docs
python generate_diagram.py
```

**Generates:**
1. `aws_architecture_diagram.png` - Complete system architecture
2. `ml_pipeline_flow.png` - Detailed ML pipeline
3. `monitoring_workflow.png` - Monitoring and retraining workflow

**Output formats supported:**
- PNG (default)
- SVG (for web)
- PDF (for documents)
- DOT (for graphviz)

### 📊 How to Use the Diagrams

#### Option 1: View PNG Files Directly
After running the script, PNG files are created in the `docs/` directory:
```bash
open aws_architecture_diagram.png
open ml_pipeline_flow.png
open monitoring_workflow.png
```

#### Option 2: Import to draw.io
1. Go to https://app.diagrams.net/
2. Click **File → Import**
3. Select the PNG file
4. Edit and customize as needed
5. Export as PNG, SVG, PDF, or XML

#### Option 3: Import to Lucidchart
1. Go to https://www.lucidchart.com/
2. Create new document
3. Click **Import → Upload**
4. Select the PNG file
5. Use AWS icon library for additional customization

#### Option 4: Use in Presentations
The PNG files can be directly inserted into:
- PowerPoint presentations
- Google Slides
- Keynote
- Confluence pages
- Markdown documentation

### 🏗️ Architecture Diagram Contents

#### Main Architecture Diagram
Shows complete end-to-end MLOps pipeline:
- Data ingestion from external sources
- S3 storage layer (4 buckets)
- SageMaker ML Pipeline (preprocessing, training, evaluation)
- Model Registry and versioning
- Production endpoints with auto-scaling
- Monitoring and observability (CloudWatch, Model Monitor)
- Automated retraining workflow (Step Functions)
- Security layer (IAM, KMS, VPC)
- CI/CD pipeline (GitHub Actions, ECR)

#### ML Pipeline Flow Diagram
Detailed view of the ML pipeline:
- Data preprocessing steps
- Training configuration (instances, spot usage)
- Evaluation metrics calculation
- Conditional model registration
- Deployment to endpoints

#### Monitoring Workflow Diagram
Shows monitoring and automation:
- Real-time monitoring
- Drift detection
- Alert triggers
- Automated retraining workflow
- Notification system

### 📝 Customization Guide

#### Modify the Python Script

To customize colors:
```python
graph_attr = {
    "fontsize": "14",
    "bgcolor": "white",  # Change background
    "pad": "0.5",
}
```

To change diagram direction:
```python
with Diagram(..., direction="TB"):  # TB=Top-Bottom, LR=Left-Right
```

To add custom components:
```python
from diagrams.aws.ml import Sagemaker

my_component = Sagemaker("My Component\nDescription")
```

#### Export to Different Formats

Modify the script to export SVG:
```python
with Diagram(..., outformat="svg"):
```

Or PDF:
```python
with Diagram(..., outformat="pdf"):
```

### 🎨 AWS Icon Library

The diagrams use official AWS icons from:
- https://aws.amazon.com/architecture/icons/

For manual diagram creation, download the icon set and use in:
- Lucidchart (built-in AWS library)
- draw.io (AWS shape library)
- PowerPoint (import PNG icons)

### 📐 Architecture Principles

The diagrams follow these principles:

1. **Layered Architecture**:
   - Data Layer → Processing Layer → Deployment Layer → Monitoring Layer

2. **Color Coding**:
   - Blue: ML Pipeline components
   - Green: Production endpoints
   - Orange: Monitoring services
   - Purple: Infrastructure/automation
   - Red: Security components

3. **Flow Direction**:
   - Data flows top-to-bottom
   - Feedback loops clearly marked
   - Dotted lines for configuration/security

4. **Grouping**:
   - Related services in clusters
   - Clear service boundaries
   - VPC boundaries when applicable

### 🔄 Updating Diagrams

When you update the architecture:

1. **Update AWS_ARCHITECTURE.md**
   - Modify the text-based diagrams
   - Update component descriptions
   - Add new services

2. **Update generate_diagram.py**
   - Add new components
   - Update connections
   - Regenerate diagrams

3. **Regenerate Images**
   ```bash
   python generate_diagram.py
   ```

4. **Commit Changes**
   ```bash
   git add docs/
   git commit -m "Update architecture diagrams"
   ```

### 📚 Additional Resources

#### For Lucidchart Users
- Import templates: https://www.lucidchart.com/pages/templates/aws-architecture-diagram
- AWS shapes: Built-in library in Lucidchart
- Tutorial: https://www.lucidchart.com/pages/tutorials/aws-architecture-diagram

#### For draw.io Users
- AWS icons: File → Open Library from → AWS19
- Templates: https://www.drawio.com/blog/aws-diagrams
- Tutorial: https://drawio-app.com/blog/draw-aws-diagrams/

#### For Automation
- Terraform graph: `terraform graph | dot -Tpng > graph.png`
- AWS CloudFormation Designer: Visual infrastructure editor
- AWS Application Composer: Serverless architecture designer

### 💡 Tips for Portfolio

1. **Include in README**: Link to diagram images in main README.md
2. **Create Slides**: Use diagrams in presentation slides
3. **Blog Posts**: Embed diagrams in technical blog posts
4. **Documentation**: Include in project documentation
5. **Interviews**: Use to explain architecture decisions

### 🎯 Diagram Checklist

When creating/updating diagrams:

- [ ] All major AWS services included
- [ ] Data flow direction clear
- [ ] Security components highlighted
- [ ] Cost optimization features noted
- [ ] Monitoring and alerting shown
- [ ] Auto-scaling capabilities indicated
- [ ] Disaster recovery paths marked
- [ ] CI/CD integration displayed
- [ ] Legend/key provided
- [ ] High-quality resolution (300+ DPI for print)

---

## Quick Start

Generate all diagrams with one command:

```bash
# Install dependencies
pip install diagrams

# Generate diagrams
cd docs
python generate_diagram.py

# View results
open aws_architecture_diagram.png
```

For best results:
- Use diagrams in presentations (PNG format)
- Use in documentation (SVG format for web)
- Import to Lucidchart/draw.io for customization
- Keep text-based version for reference

---

**Pro Tip**: Keep both visual diagrams AND text-based documentation. Visual for presentations, text for version control and quick reference.
