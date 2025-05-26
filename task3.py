from PyPDF2 import PdfReader, PdfWriter
from io import BytesIO


# Load the original PDF
input_path = "C:/Users/gillk/OneDrive/Documents/Gurleen/Resume.pdf"
reader = PdfReader(input_path)
writer = PdfWriter()

# Function to extract original text content (for context) and then create a new version with updated sections
# Since we can't directly edit PDFs in place here, we'll prepare new content for a rewritten version

# Add all pages except the last if it's blank (assume last page may be blank based on user comment)
num_pages = len(reader.pages)
for i in range(num_pages):
    if i == num_pages - 1:
        text = reader.pages[i].extract_text().strip()
        if text:  # Only include last page if it has text
            writer.add_page(reader.pages[i])
                       
    else:
        writer.add_page(reader.pages[i])

# Save the modified PDF without the blank last page
output_path = "C:/Users/gillk/OneDrive/Documents/Gurleen/Resume_no_blank.pdf"
with open(output_path, "wb") as f:
    writer.write(f)

output_path


