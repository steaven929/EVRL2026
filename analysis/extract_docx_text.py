from pathlib import Path
import zipfile
import xml.etree.ElementTree as ET


docx_path = Path(r"F:\Claude code\EVRL260207\article\SAEV-MARLc1.docx")
out_path = Path(r"F:\Claude code\EVRL260207\analysis\SAEV-MARLc1_extracted.txt")

ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}

with zipfile.ZipFile(docx_path, "r") as zf:
    xml_data = zf.read("word/document.xml")

root = ET.fromstring(xml_data)
lines = []

for p in root.findall(".//w:body/w:p", ns):
    texts = []
    for t in p.findall(".//w:t", ns):
        if t.text:
            texts.append(t.text)
    line = "".join(texts).strip()
    if line:
        lines.append(line)

out_path.write_text("\n".join(lines), encoding="utf-8")
print(out_path)
print(f"lines={len(lines)}")
