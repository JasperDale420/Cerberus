import pdfplumber


def extract_pdf_preview(pdf_path, out_path, num_pages=50):
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        total_pages = len(pdf.pages)
        pages_to_extract = min(num_pages, total_pages)
        for i in range(pages_to_extract):
            page = pdf.pages[i]
            extracted = page.extract_text()
            if extracted:
                text += f"\n--- Page {i + 1} ---\n"
                text += extracted

    with open(out_path, "w") as f:
        f.write(text)

    print(f"Extracted {pages_to_extract} pages to {out_path}")


if __name__ == "__main__":
    extract_pdf_preview("151 trading strategies.pdf", "artifacts/151_strategies_preview.txt")
