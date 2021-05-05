from fpdf import FPDF, TitleStyle
from datetime import date
from datetime import datetime as dt
from typing import List, Optional
import plotly.express as px
import os
import pathlib
import pandas as pd

# author = "T. J. Lux"
# title_style_0 = TitleStyle("Times", "B", 16)
# title_style_1 = TitleStyle("Times", "B", 14)
# title_style_2 = TitleStyle("Times", "B", 14)
# title_style_3 = TitleStyle("Times", "B", 14)
# date_format = "%d.%m.%Y"

path_lookup_dgvs_keys = "data/DGVS-Leistungskatalog-Leistungsgruppen-fuer-OPS-Version-2021.csv"
dgvs_lookup = pd.read_csv(path_lookup_dgvs_keys, sep = ";", encoding = "latin1")

def make_weekly_plot(endo_mat, group_col:str, y_col: str, y_type: str, w: int = 800, h: int = 400, filter_for_values: Optional[List]=None, prefix: Optional[str] = None):
    df = endo_mat.mat
    if filter_for_values:
        df = df[df[group_col].isin(filter_for_values)]
    plot_folder = pathlib.Path("plots")
    if not plot_folder.exists():
        os.mkdir(plot_folder)
    df = df.loc[:, ["date", group_col, y_col]]
    df["date"] = pd.to_datetime(df["date"])
    df = df.groupby([group_col, pd.Grouper(key='date', freq='W-MON')])
    df = df.describe()
    df = df.loc[:, (y_col, y_type)]
    df = df.reset_index()
    name_value_col = f"{y_col}_{y_type}"
    df.columns = [group_col, "date", name_value_col]
    fig = px.line(df, x="date", y=name_value_col, color=group_col)
    plot_name = f"weekly_{name_value_col}_by_{group_col}.png"
    if prefix:
        plot_name = prefix + "_" + plot_name
    plot_path = plot_folder.joinpath(plot_name)
    fig.write_image(plot_path.as_posix(), width = w, height = h)
    return plot_path

class PDF(FPDF):
    def __init__(self, em, plot_kwargs):
        super().__init__()
        self.plot_paths = []
        self.author = "T. J. Lux"
        self.title_style_0 = TitleStyle("Times", "B", 16)
        self.title_style_1 = TitleStyle("Times", "B", 14)
        self.title_style_2 = TitleStyle("Times", "B", 14)
        self.title_style_3 = TitleStyle("Times", "B", 14)
        self.date_format = "%d.%m.%Y"
        self.line_space = 8
        self.tab_space = 30
        self.font = "Times"
        self.base_font_size = 12
        self.full_line_width = 100
        self.dgvs_lookup = {row["Code"]: row["Leistungsgruppe"] for index, row in dgvs_lookup.iterrows()}
        
        for kwargs in plot_kwargs:
            self.plot_paths.append(make_weekly_plot(em, **kwargs))
        
    def header(self):
        # Logo
        self.image('data/logo.png', 8, 8, 15)
        # helvetica bold 15
        self.set_font(self.font, '', 10)
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
        self.set_font(self.font, 'I', 8)
        # Page number
        self.cell(0, 10, 'Page ' + str(self.page_no()) + '/{nb}', 0, 0, 'C')

    def new_line(self, spacing: int = None):
        if spacing == None:
            spacing = self.line_space
            self.ln(spacing)

    def indent(self, spacing:int=None):
        if spacing == None:
            spacing = self.tab_space
            self.cell(w = spacing)

    def standard_cell(self, txt:str, w: int = None):
        if w == None:
            w = self.full_line_width
        self.cell(w = w, txt = txt)
        
    def add_image(self, image_path, w: Optional[int] = None, h:Optional[int]=None):
        self.image(image_path, w = w, h = h)#, x = None, y = None, w = w, h = h)
        self.new_line()

    def init_pdf(self, title: str, start_date: date = date(2021, 1,1), end_date: date = dt.now().date()):
        self.set_creation_date()
        self.set_author(self.author)
        self.set_creator(self.author)
        self.set_auto_page_break(auto = True, margin = 20)
        self.set_section_title_styles(
        level0 = self.title_style_0,
        level1 = self.title_style_1,
        level2 = self.title_style_2,
        level3 = self.title_style_3
        )
        self.set_title(title)
        self.add_page()
        self.set_font(self.font, "", self.base_font_size)
        self.new_line()
        self.new_line()
        self.start_section("Report")
        self.new_line()
        self.standard_cell("Universitätsklinikum Würzburg")
        self.new_line()
        self.standard_cell("Medizinische Klinik II")
        self.new_line()
        self.standard_cell("Gastroenterologische Endoskopie")
        self.new_line()
        self.standard_cell(f"Von: {start_date.strftime(self.date_format)}")
        self.new_line()
        self.standard_cell(f"Bis: {end_date.strftime(self.date_format)}")
        self.add_page()
        
    def add_plots(self, w:Optional[int]=180, h:Optional[int]=0):
        for image_path in self.plot_paths:
            self.add_image(image_path, w, h)
        self.add_page()
        
    def add_summary(self, summary_dict, _type:str, w: int = None, cell_height: int = 0, spacing: int = None):
        if w == None:
            w = self.full_line_width
        if spacing == None:
            spacing = self.line_space
        
        self.start_section(f"{_type}", level = 1)
        self.new_line(spacing)
        
        for key, value in summary_dict.items():
            self.new_line(spacing)
            if _type == "Leistungen":
                try:
                    self.start_section(f"{self.dgvs_lookup[key]} ({key})", level = 2)
                except:
                    self.start_section(f"({key})", level = 2)
            else: 
                self.start_section(f"{key}", level = 2)
            self.new_line()
            self.cell(w = w, h = cell_height, txt = f"Anzahl:               {value['n_performed']}")
            self.new_line()
            self.cell(w = w, h = cell_height, txt = f"Mittlere Kosten:  {value['mean_cost']} Euro")
            self.new_line()
            self.cell(w = w, h = cell_height, txt = f"Mittlere Dauer:   {value['mean_duration']} min")
            self.new_line()
            self.cell(w = w, h = cell_height, txt = "Häufigste Materialien")
            self.new_line()
            for material in value["top_most_used_mat_ids"]:
                self.indent()
                self.cell(w = w-self.tab_space, h = cell_height, txt = f"{material[3]}: Anzahl {material[1]}, Gesamtkosten: {material[2]} Euro")
                self.new_line()
            self.cell(w = w, h = cell_height, txt = "Kumulativ Teuerste Materialien")
            self.new_line()
            for material in value["top_highest_cost"]:
                self.indent()
                self.cell(w = w-self.tab_space, h = cell_height, txt = f"{material[3]}: Anzahl {material[1]}, Gesamtkosten: {material[2]} Euro")
                self.new_line()
                

    def add_material_info(self, mat_info, spacing:int = 6):
        self.add_page()
        self.start_section("Materialliste", level = 0)
        self.ln(spacing)
        for key, value in mat_info.items():
            self.new_line()
            self.set_font(self.font, 'B', self.base_font_size)
            self.standard_cell(txt = value["name"])
            self.set_font(self.font, '', self.base_font_size)
            self.new_line()
            self.indent()
            self.cell(w = self.full_line_width - self.tab_space, txt = f'ID: {int(key)}')
            self.new_line()
            self.indent()
            self.cell(w = self.full_line_width - self.tab_space,txt = f"Preis: {round(value['price'],2)} Euro")
            self.new_line()
            self.indent()
            self.cell(w = self.full_line_width - self.tab_space,txt = f"Hersteller: {value['manufacturer']}")