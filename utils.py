import os
import re
import pdftotext

def convert_pdf_to_txt(directory):
    for filename in os.listdir(directory):
        if filename.endswith(".pdf"):
            txt_filename = f"{os.path.splitext(filename)[0]}.txt"
            txt_path = os.path.join(directory,"txts", txt_filename)
            # Check if the txt file already exists
            if not os.path.exists(txt_path):
                pdf_path = os.path.join(directory, filename)
                with open(pdf_path, "rb") as f:
                    pdf = pdftotext.PDF(f)
                text = "\n\n".join(pdf)
                
                # Remove References or Bibliography parts of the texts
                # Make sure to check if the word, 'References' or 'Bibliography'
                # appears after at least 60% of the texts. 
                if 'References' in text[int(0.6*len(text)):]:
                    text = text[:text.rfind('References')]
                elif 'Bibliography' in text[int(0.6*len(text)):]:
                    text = text[:text.rfind('Bibliography')]
                    
                with open(txt_path, "w") as text_file:
                    text_file.write(text)

dir_path = "/Users/minseokbae/ANL/gpt3_finetune/pdfs"
convert_pdf_to_txt(dir_path)