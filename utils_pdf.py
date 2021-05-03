from fpdf import FPDF, TitleStyle
from datetime import date
from datetime import datetime as dt

author = "T. J. Lux"
title_style_0 = TitleStyle("Times", "B", 16)
title_style_1 = TitleStyle("Times", "B", 14)
title_style_2 = TitleStyle("Times", "B", 14)
title_style_3 = TitleStyle("Times", "B", 14)
date_format = "%d.%m.%Y"

class PDF(FPDF):
    def header(self):
        # Logo
        self.image('data/logo.png', 8, 8, 15)
        # helvetica bold 15
        self.set_font('Times', '', 10)
        # Move to the right
        self.cell(20)
        # Title
        self.cell(30, 4, self.title)
        # Line break
        self.ln(20)

    # Page footer
    def footer(self):
        # Position at 1.5 cm from bottom
        self.set_y(-15)
        # helvetica italic 8
        self.set_font('Times', 'I', 8)
        # Page number
        self.cell(0, 10, 'Page ' + str(self.page_no()) + '/{nb}', 0, 0, 'C')

def init_pdf(pdf: PDF):
    pdf.set_creation_date()
    pdf.set_author(author)
    pdf.set_creator(author)
    pdf.set_auto_page_break(auto = True, margin=2)
    pdf.set_title("Material Report, Endoscopy")
    pdf.set_font('Times', '', 16)

def add_summary(pdf, summary_dict, _type:str):
    pdf.add_page()
    pdf.start_section(f"Summary {_type}", level = 0)

    if _type == "dgvs":
        for key, value in summary_dict.items():
            pdf.start_section(key, level = 1)
            pdf.ln(20)
            pdf.cell(w = 100, h = 20, txt = f"Anzahl:\t\t{value['n_performed']}")
            pdf.ln(20)
            pdf.cell(w = 100, h = 20, txt = f"Mittlere Kosten:\t{value['mean_cost']}")
            pdf.ln(20)
            pdf.cell(w = 100, h = 20, txt = f"Mittlere Dauer:\t{value['mean_duration']}")

    elif _type == "intervention_type":
        pass
    else:
        print("_type must be 'dgvs' or 'intervention_type'")