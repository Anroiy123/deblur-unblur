"""
Extract formatting from the reference docx and create a scientific report
for the deblur-unblur project.
"""
import os
import sys

# First, install dependencies
os.system(f"{sys.executable} -m pip install python-docx --break-system-packages --quiet 2>&1")

from docx import Document
from docx.shared import Pt, Inches, Cm, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.style import WD_STYLE_TYPE
from docx.oxml.ns import qn
import copy

UPLOAD_DIR = r"C:\Users\hunga\AppData\Local\Claude-3p\local-agent-mode-sessions\a09e4deb-699d-458f-905b-4257bea32003\00000000-0000-4000-8000-000000000001\local_90fdf037-5671-4dcb-b5f9-4a0947d2456d\uploads"
REF_FILE = os.path.join(UPLOAD_DIR, "TieuLuan_KhaiPhaDuLieu.docx")
OUTPUT = r"C:\Users\hunga\OneDrive\Desktop\project\deblur-unblur\BaoCao_DeBlur.docx"

# Read reference document to understand formatting
print("Reading reference document...")
ref_doc = Document(REF_FILE)

# Print style details
print("\n=== PARAGRAPH STYLES ===")
for style in ref_doc.styles:
    if style.type == WD_STYLE_TYPE.PARAGRAPH and style.name and 'Heading' not in style.name and 'Normal' not in style.name and 'TOC' not in style.name:
        try:
            font = style.font
            pf = style.paragraph_format
            print(f"  Style: {style.name}")
            print(f"    Font: {font.name}, Size: {font.size}, Bold: {font.bold}, Italic: {font.italic}")
            print(f"    Alignment: {pf.alignment}")
            print(f"    Space before: {pf.space_before}, after: {pf.space_after}")
        except:
            pass

print("\n=== DOCUMENT SECTIONS ===")
for i, section in enumerate(ref_doc.sections):
    print(f"  Section {i}: page_width={section.page_width}, page_height={section.page_height}")
    print(f"    margins: left={section.left_margin}, right={section.right_margin}, top={section.top_margin}, bottom={section.bottom_margin}")

# Print first few paragraphs to understand structure
print("\n=== FIRST 20 PARAGRAPHS ===")
for i, para in enumerate(ref_doc.paragraphs[:30]):
    style = para.style.name if para.style else "None"
    text = para.text[:120] if para.text else "[empty]"
    font_info = ""
    if para.runs:
        run = para.runs[0]
        font_info = f"font={run.font.name}, size={run.font.size}, bold={run.font.bold}"
    alignment = para.alignment
    print(f"  [{i}] style='{style}' align={alignment} {font_info}: '{text}'")

# Check header/footer
print("\n=== SECTIONS HEADERS/FOOTERS ===")
for i, section in enumerate(ref_doc.sections):
    print(f"  Section {i}:")
    if section.header:
        for j, para in enumerate(section.header.paragraphs[:5]):
            print(f"    Header[{j}]: '{para.text[:100]}'")
    if section.footer:
        for j, para in enumerate(section.footer.paragraphs[:5]):
            print(f"    Footer[{j}]: '{para.text[:100]}'")

# Check for images
print("\n=== IMAGES IN DOCUMENT ===")
for rel_id, rel in ref_doc.part.rels.items():
    if "image" in rel.reltype:
        print(f"  Image: {rel.target_ref}")

# Check tables
print("\n=== TABLES ===")
for i, table in enumerate(ref_doc.tables[:5]):
    print(f"  Table {i}: {len(table.rows)} rows x {len(table.columns)} cols")
    for j, row in enumerate(table.rows[:3]):
        cells = [cell.text[:40] for cell in row.cells]
        print(f"    Row {j}: {cells}")

# Now read full text to understand structure
print("\n=== FULL DOCUMENT TEXT (first 2000 chars) ===")
full_text = ""
for para in ref_doc.paragraphs:
    full_text += para.text + "\n"
print(full_text[:2000])

print("\n=== ANALYSIS COMPLETE ===")
