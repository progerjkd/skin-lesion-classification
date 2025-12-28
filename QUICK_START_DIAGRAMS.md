# Quick Start - Generate AWS Architecture Diagrams

## 🎨 Generate Professional Diagrams in 3 Steps

### Step 1: Install Dependencies

```bash
pip install diagrams
```

### Step 2: Generate Diagrams

```bash
cd docs
python generate_diagram.py
```

### Step 3: View Results

Three PNG files will be created:

1. **aws_architecture_diagram.png** - Complete MLOps system
2. **ml_pipeline_flow.png** - Detailed ML pipeline
3. **monitoring_workflow.png** - Monitoring & retraining

```bash
# On macOS
open *.png

# On Linux
xdg-open *.png

# On Windows
start *.png
```

## 📊 What You Get

### Main Architecture Diagram
- Complete end-to-end MLOps pipeline
- All AWS services with official icons
- Data flow arrows and connections
- Color-coded components
- Professional layout

### ML Pipeline Flow
- Preprocessing → Training → Evaluation → Registry
- Instance types and configurations
- Conditional model registration
- Deployment flow

### Monitoring Workflow
- Real-time monitoring setup
- Drift detection
- Automated retraining workflow
- Alert and notification system

## 🎯 Import to Lucidchart

1. Go to https://www.lucidchart.com/
2. Create new document (or open existing)
3. Click **Import** or **File → Import**
4. Select **Upload** and choose the PNG file
5. The diagram will appear - you can now:
   - Edit text and labels
   - Rearrange components
   - Add additional AWS icons from library
   - Export to PDF/PNG/SVG

## 🎯 Import to draw.io

1. Go to https://app.diagrams.net/
2. Click **File → Import From → Device**
3. Select the PNG file
4. The diagram will be imported
5. Customize:
   - File → Open Library from → AWS19 (for AWS icons)
   - Edit, resize, rearrange components
   - Export: File → Export as → PNG/SVG/PDF

## 🎨 Customization Options

### Change Output Format

Edit `generate_diagram.py`:

```python
# For SVG (web-friendly)
with Diagram(..., outformat="svg"):

# For PDF (documents)
with Diagram(..., outformat="pdf"):

# For DOT (graphviz)
with Diagram(..., outformat="dot"):
```

### Change Layout Direction

```python
# Left to Right
with Diagram(..., direction="LR"):

# Top to Bottom (default)
with Diagram(..., direction="TB"):
```

### Change Colors

```python
graph_attr = {
    "bgcolor": "white",    # Background color
    "fontsize": "14",      # Font size
}
```

## 📁 Alternative: Use Text-Based Documentation

If you prefer text-based diagrams for Lucidchart:

1. Open [docs/AWS_ARCHITECTURE.md](docs/AWS_ARCHITECTURE.md)
2. Copy the ASCII diagrams
3. Use as reference to manually create in Lucidchart
4. Benefit: More control over exact layout

## 💡 Pro Tips

### For Presentations
- Generate PNG at high DPI for print quality
- Use transparent background for slides
- Export individual components for focus slides

### For Documentation
- Generate SVG for web documentation
- Smaller file sizes
- Scales perfectly at any resolution

### For Portfolio
- Generate all three diagrams
- Include in GitHub README
- Add to project presentations
- Use in blog posts

## 🔧 Troubleshooting

### Error: "diagrams module not found"
```bash
pip install diagrams
# or
pip install -r requirements.txt
```

### Error: Graphviz not installed
```bash
# macOS
brew install graphviz

# Ubuntu/Debian
sudo apt-get install graphviz

# Windows
# Download from: https://graphviz.org/download/
```

### Images not generating
- Check that you're in the `docs/` directory
- Ensure you have write permissions
- Try running with `python3` instead of `python`

## 📚 Additional Resources

### Official AWS Icons
- Download: https://aws.amazon.com/architecture/icons/
- Use in: PowerPoint, Visio, Lucidchart, draw.io

### Diagram Templates
- Lucidchart AWS Templates: https://www.lucidchart.com/pages/templates/aws-architecture-diagram
- draw.io AWS Templates: https://www.drawio.com/blog/aws-diagrams

### Documentation
- Python diagrams library: https://diagrams.mingrammer.com/
- AWS Architecture Best Practices: https://aws.amazon.com/architecture/

## 🎬 Complete Example

```bash
# Full workflow
cd /path/to/skin-lesion-classification

# Install dependencies
pip install diagrams

# Generate diagrams
cd docs
python generate_diagram.py

# View results
open aws_architecture_diagram.png
open ml_pipeline_flow.png
open monitoring_workflow.png

# Copy to presentation folder
cp *.png ../presentations/

# Generate SVG for web
# (edit generate_diagram.py to use outformat="svg")
python generate_diagram.py

# Now you have both PNG and SVG versions!
```

## ✅ Success Checklist

- [ ] Python diagrams library installed
- [ ] Graphviz installed
- [ ] Generated all three diagrams
- [ ] Verified PNG files created
- [ ] Diagrams look professional
- [ ] Ready to import to Lucidchart/draw.io
- [ ] Ready to use in presentations

---

**Need help?** Check [docs/README.md](docs/README.md) for detailed documentation or [docs/AWS_ARCHITECTURE.md](docs/AWS_ARCHITECTURE.md) for text-based architecture reference.
