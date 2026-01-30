import os
import pandas as pd
import re
import unicodedata


class AlloyFinder:
    def __init__(self):
        self.articles_dir = '/home/shared/llm-extraction-pipeline/articles_all'
        self.found = []
        self.not_found = []
        self.matches = []

    def normalize(self, text: str):
        text = unicodedata.normalize("NFKC", text)
        DASHES = {
            '\u2010', # hyphen
            '\u2011', # non-breaking hyphen
            '\u2012', # figure dash
            '\u2013', # en dash
            '\u2014', # em dash
            '\u2015', # horizontal bar
            '\u2212', # minus sign
            '-',      # ASCII hyphen-minus
        }

        for dash in DASHES:
            text = re.sub(dash, "-", text)
        
        text = text.replace("\u00A0", " ")

        text = re.sub(r'\s+', ' ', text)

        return text.strip()
    
    def contains_whole_string(self, query: str, paragraph: str) -> bool:

        pattern = re.escape(query)
        return re.search(pattern, paragraph) is not None


    def searchInParagraphs(self, acronym: str, doi:str, caption: str, footer: str, table_path: str) -> None:

        if not (os.path.isdir(os.path.join(self.articles_dir, doi))):
            return

        article_path = os.path.join(self.articles_dir,doi, 'Processed_Text_Data.csv')

        if not (os.path.isfile(article_path)):
            return
        
        
        article = pd.read_csv(article_path)
        article.dropna(subset=['Paragraph'], inplace=True)


        acronym = self.normalize(acronym)
        no_of_matches = 0
        match_found = False
        for idx, row in article.iterrows():
            para = row['Paragraph']
            para = self.normalize(para)
            if self.contains_whole_string(acronym, para):
                if no_of_matches == 0:
                    self.found.append([doi, para, acronym, caption, footer, table_path])
                    match_found = True
                no_of_matches += 1
        if not match_found:
            self.not_found.append([doi, acronym, caption, footer, table_path])       

        self.matches.append([acronym, no_of_matches, doi]) 

        return