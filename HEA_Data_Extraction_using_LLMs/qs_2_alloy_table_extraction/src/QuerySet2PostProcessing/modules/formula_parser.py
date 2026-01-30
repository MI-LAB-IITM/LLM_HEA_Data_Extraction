import re
from typing import Union, List, Dict
import math

def parse_formula(formulas: Union[List[str], str]) -> list[dict[str,float]]:
    ELEMENT_SYMBOLS = {
        'H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne',
        'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar', 'K', 'Ca',
        'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn',
        'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr', 'Rb', 'Sr', 'Y', 'Zr',
        'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn',
        'Sb', 'Te', 'I', 'Xe', 'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd',
        'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb',
        'Lu', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg',
        'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn', 'Fr', 'Ra', 'Ac', 'Th',
        'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm',
        'Md', 'No', 'Lr', 'Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Ds',
        'Rg', 'Cn', 'Nh', 'Fl', 'Mc', 'Lv', 'Ts', 'Og'
    }

    sorted_symbols = sorted(ELEMENT_SYMBOLS, key=len, reverse=True)
    element_pattern = r'(\([^)]+\)|' + '|'.join(sorted_symbols) + r')(\d+(?:\.\d+)|\.?\d+)?'
    pattern = re.compile(element_pattern)

    results = []
    
    if isinstance(formulas, list):
        to_parse = formulas
    elif isinstance(formulas, str):
        to_parse = [formulas]
    else:
        raise TypeError("Input must be a sting or a list of strings.")
    
    for formula in to_parse:
        matches = pattern.findall(formula)
        matched_str = ''.join(elem+num for elem, num in matches)
        if matched_str != formula:
            results.append({})
            continue
        comp_dict = {}
        for elem, num in matches:
            value = float(num) if num else 1.0
            if elem not in comp_dict:
                comp_dict[elem] = value
            else:
                comp_dict = {}
                break
        results.append(comp_dict)

    return results