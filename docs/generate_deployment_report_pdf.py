"""One-off script: renders a docs/*.md file as a PDF.

Uses markdown -> styled HTML -> Playwright's page.pdf(), rather than adding
a new PDF-rendering dependency (reportlab/weasyprint/fpdf) — both `markdown`
and `playwright` are already installed for other parts of this project.

Run: python docs/generate_deployment_report_pdf.py [source.md]
Defaults to agmarknet_scraper_vm_deployment.md if no argument is given.
"""

import sys
from pathlib import Path

import markdown
from playwright.sync_api import sync_playwright

_DEFAULT_SRC = "agmarknet_scraper_vm_deployment.md"
SRC_PATH = Path(__file__).parent / (sys.argv[1] if len(sys.argv) > 1 else _DEFAULT_SRC)
OUT_PATH = SRC_PATH.with_suffix(".pdf")

ACCENT = "#1F5C3D"  # dark green, agri-themed — matches generate_market_price_docx.py

HTML_TEMPLATE = """<!doctype html>
<html>
<head>
<meta charset="utf-8">
<style>
  body {{
    font-family: Calibri, "Segoe UI", Arial, sans-serif;
    color: #222;
    font-size: 10.5pt;
    line-height: 1.5;
    max-width: 100%;
  }}
  h1 {{
    color: {accent};
    font-size: 20pt;
    border-bottom: 2px solid {accent};
    padding-bottom: 6px;
    margin-top: 0;
  }}
  h2 {{
    color: {accent};
    font-size: 14pt;
    margin-top: 24px;
    border-bottom: 1px solid #ccc;
    padding-bottom: 3px;
  }}
  h3 {{
    color: #333;
    font-size: 12pt;
    margin-top: 16px;
  }}
  code {{
    font-family: Consolas, "Courier New", monospace;
    background: #f2f2f2;
    padding: 1px 4px;
    border-radius: 3px;
    font-size: 9.5pt;
  }}
  pre {{
    background: #f2f2f2;
    padding: 10px 12px;
    border-radius: 4px;
    border-left: 3px solid {accent};
    overflow-x: auto;
    font-size: 9pt;
  }}
  pre code {{
    background: none;
    padding: 0;
  }}
  ul, ol {{
    padding-left: 22px;
  }}
  li {{
    margin-bottom: 4px;
  }}
  strong {{
    color: #111;
  }}
  hr {{
    border: none;
    border-top: 1px solid #ccc;
    margin: 20px 0;
  }}
</style>
</head>
<body>
{content}
</body>
</html>
"""


def main() -> None:
    md_text = SRC_PATH.read_text(encoding="utf-8")
    html_body = markdown.markdown(md_text, extensions=["fenced_code", "tables"])
    full_html = HTML_TEMPLATE.format(accent=ACCENT, content=html_body)

    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        page.set_content(full_html, wait_until="load")
        page.pdf(
            path=str(OUT_PATH),
            format="A4",
            margin={"top": "20mm", "bottom": "18mm", "left": "18mm", "right": "18mm"},
            print_background=True,
        )
        browser.close()

    print(f"Wrote {OUT_PATH} ({OUT_PATH.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
