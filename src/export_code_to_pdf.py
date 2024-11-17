from pygments import highlight
from pygments.lexers import PythonLexer
from pygments.formatters import HtmlFormatter
from xhtml2pdf import pisa


def export_code_to_pdf(input_files, output_pdf):
    # Start HTML content for the PDF
    html_content = """
    <html>
    <head>
        <style>
        body { font-family: monospace; }
        pre { font-size: 8pt; line-height: 1.2; }
        </style>
    </head>
    <body>
    <h1>Python Files</h1>
    """

    # Highlight each file and convert to HTML
    for input_file in input_files:
        html_content += f"<h2>File: {input_file}</h2>"
        with open(input_file, "r") as file:
            code = file.read()
            formatter = HtmlFormatter(style="default", full=True, noclasses=True)
            highlighted_code = highlight(code, PythonLexer(), formatter)
            html_content += f"<pre>{highlighted_code}</pre><hr>"
    html_content += "</body></html>"
    # Export HTML content to PDF
    with open(output_pdf, "wb") as pdf_file:
        pisa.CreatePDF(html_content, dest=pdf_file)

# Python files to export to PDF
python_files = [
    "default_config.py",
    "loaders.py",
    "convolutionalNetwork.py",
    "train_model.py",
    "result_handler.py",
    "plot_scores.py",
    "predict.py",
    "plot_individual_feature_maps.py",
    "plot_transformations.py",
]
output_pdf = "code.pdf"
export_code_to_pdf(python_files, output_pdf)