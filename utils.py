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
                # appears after at least 70% of the texts. 
                text_lower = text.lower()

                # Extract the last 30% of the text
                last_30_percent = text_lower[int(0.7*len(text)):]

                # Find the index of 'references' and 'acknowledgements' within the last 30% of the text
                references_index = last_30_percent.find('references')
                acknowledgements_index = last_30_percent.find('acknowledgements')

                # Initialize the index at which to start removing text as -1 (not found)
                remove_text_index = -1

                # If 'references' is found within the last 30% of the text, update remove_text_index
                if references_index != -1:
                    remove_text_index = references_index + int(0.7*len(text))

                # If 'acknowledgements' is found within the last 30% of the text and appears before 'references', update remove_text_index
                if acknowledgements_index != -1 and (acknowledgements_index < references_index or references_index == -1):
                    remove_text_index = acknowledgements_index + int(0.7*len(text))

                # If either term was found within the last 30% of the text, remove everything from that point onwards
                if remove_text_index != -1:
                    text = text[:remove_text_index]

                with open(txt_path, "w") as text_file:
                    text_file.write(text)

dir_path = "/Users/minseokbae/ANL/gpt3_finetune/pdfs"
convert_pdf_to_txt(dir_path)