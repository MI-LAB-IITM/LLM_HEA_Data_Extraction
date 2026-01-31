# ---------------------------------------------------------------------------------------------------------
# The table extraction code has been adapted from the following repository: https://github.com/olivettigroup/table_extractor
# The license for the same is provided below.
# ---------------------------------------------------------------------------------------------------------
# MIT License
#
# Copyright (c) 2018
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# ---------------------------------------------------------------------------------------------------------

from __future__ import annotations
from bs4 import BeautifulSoup
from html import unescape
from scipy import stats
import unidecode, traceback, sys

class TableExtractor:
    """Lightly cleaned wrapper keeping original behavior."""

    def __init__(self):
        self.doi: str = ""

    def get_caption(self, table, format: str):
        if format == "xml":
            if "10.1016" in self.doi:
                caption = table.find("caption")
                caption, ref = self._search_for_reference(caption, format)
                caption = unidecode.unidecode(unescape(caption.text)).strip() if caption else ""
                return caption, ref
            elif "10.1021" in self.doi:
                caption = table.find("title") or table.parent.find("caption")
                caption, ref = self._search_for_reference(caption, format)
                caption = unidecode.unidecode(unescape(caption.text)).strip() if caption else ""
                return caption, ref
        else:
            raise NotImplementedError
        return "", []

    def get_footer(self, table, format: str):
        footer_dict: dict[str, str] = {}
        if format == "xml":
            if "10.1016" in self.doi:
                footer = table.find_all("table-footnote")
                if footer:
                    for f in footer:
                        sup = f.find("label")
                        key = sup.text if sup is not None else "NA"
                        if sup is not None and hasattr(f, "label"):
                            f.label.decompose()
                        footer_dict[key.strip()] = unidecode.unidecode(unescape(f.text)).strip()
                else:
                    footer = table.find("legend")
                    if not footer: return None
                    for a in footer.find_all("simple-para"):
                        sup = a.find("sup")
                        key = sup.text if sup is not None else "NA"
                        if sup is not None and hasattr(a, "sup"):
                            a.sup.decompose()
                        footer_dict[key.strip()] = unidecode.unidecode(unescape(a.text)).strip()
            elif "10.1021" in self.doi:
                up = table.parent
                footer = up.find("table-wrap-foot")
                if not footer:
                    return None
                dts, dds = footer.find_all("label"), footer.find_all("p")
                if len(dts) != len(dds):
                    # fallback parsing
                    ts = footer.find_all("sup")
                    dts = [t for t in ts if t.text != ""]
                    if len(dds) == 1 and len(dts) > 1:
                        para = dds[0]
                        cont = para.contents
                        c = []
                        for co in cont:
                            c.append(co.text if hasattr(co, "text") else co)
                        ind = [i for i, x in enumerate(c) if x == ""]
                        dts, dds = [], []
                        if ind:
                            curr = ind[0]
                            for i in ind[1:]:
                                dts.append(c[curr - 1])
                                dds.append("".join(c[(curr + 1):(i - 1)]))
                                curr = i
                            dts.append(c[curr - 1])
                            dds.append("".join(c[(curr + 1):]))
                        for d, t in zip(dds, dts):
                            footer_dict[str(t).strip()] = unidecode.unidecode(unescape(str(d))).strip().replace("\n", " ")
                    else:
                        return None
                else:
                    for d, t in zip(dds, dts):
                        footer_dict[t.text.strip()] = unidecode.unidecode(unescape(d.text)).strip().replace("\n", " ")
        else:
            raise NotImplementedError
        return footer_dict

    def get_xml_tables(self, xml_path: str):
        all_tables, all_captions, all_footers, all_ids, all_labels = [], [], [], [], []
        file_content = open(xml_path, "r", encoding="utf-8").read()
        soup = BeautifulSoup(file_content, "xml")
        tables = soup.find_all("table")
        for table in tables:
            try:
                caption = None
                footer = None
                table_id = table.get("id")
                label_node = table.find("ce:label")
                table_label = label_node.get_text(strip=True) if label_node else ""

                try:
                    caption = self.get_caption(table, format="xml")[0]
                except Exception as e:
                    print(e, "Problem in caption")
                try:
                    footer = self.get_footer(table, format="xml")
                except Exception as e:
                    print(e, "problem in footer")

                # build big matrix with span handling
                tab = [[None] * 400 for _ in range(400)]
                rows = table.find_all("row")
                for i, row in enumerate(rows):
                    counter = 0
                    for ent in row:
                        if not hasattr(ent, "attrs"):  # skip NavigableString
                            continue
                        curr_col = int(ent.get("colname", "0").replace("col", "")) if ent.get("colname") else 0
                        beg = int(ent.get("namest", "0").replace("col", "")) if ent.get("namest") else 0
                        end = int(ent.get("nameend", "0").replace("col", "")) if ent.get("nameend") else 0
                        more_row = int(ent.get("morerows", "0").replace("col", "")) if ent.get("morerows") else 0

                        ent = self._search_for_reference(ent, "xml")[0]
                        text = unidecode.unidecode(unescape(ent.get_text())).strip().replace("\n", " ")

                        if beg and end and more_row:
                            for j in range(beg, end + 1):
                                for k in range(more_row + 1):
                                    tab[i + k][j - 1] = text
                        elif beg and end:
                            for j in range(beg, end + 1):
                                tab[i][j - 1] = text
                        elif more_row:
                            for j in range(more_row + 1):
                                tab[i + j][counter] = text
                        elif curr_col:
                            tab[i][curr_col - 1] = text
                        else:
                            # first empty slot from counter
                            c2 = counter
                            while tab[i][c2] is not None:
                                c2 += 1
                            tab[i][c2] = text
                        counter = counter + 1 + max(0, end - beg)

                # trim None columns/rows
                tab = [list(filter(lambda x: x is not None, row)) for row in tab]
                tab = [row for row in tab if row]

                if not tab:
                    continue

                lens = [len(t) for t in tab]
                size = int(stats.mode(lens, keepdims=True)[0][0]) if lens else 0
                if size:
                    for t in tab:
                        if len(t) != size:
                            t.extend([""] * (size - len(t)))

                all_tables.append(tab)
                all_captions.append(caption)
                all_footers.append(footer)
                all_ids.append(table_id)
                all_labels.append(table_label)
            except Exception:
                print("Failed to extract XML table")
                tb = sys.exc_info()[-1]
                print(traceback.extract_tb(tb, limit=1)[-1][1])
        return all_tables, all_captions, all_footers, all_ids, all_labels

    def _search_for_reference(self, soup, format: str):
        if format != "xml":
            raise NotImplementedError
        tags = []
        if not soup:
            return soup, tags
        ref = soup.find_all("xref")
        if len(ref) == 0:
            if soup.name == "caption":
                return soup, tags
            ref = soup.find_all("sup")
            for r in ref:
                for t in r.text.split(","):
                    if len(t) == 1 and t.isalpha():
                        tags.append(t)
                        if hasattr(soup, "sup"):
                            soup.sup.decompose()
            return soup, tags
        else:
            for _ in ref:
                if len(_.text) < 4 and hasattr(soup, "xref"):
                    tag = soup.xref.extract()
                    tags.append(tag.text)
            return soup, tags
