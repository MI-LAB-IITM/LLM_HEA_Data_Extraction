import re
from collections import defaultdict
from typing import Union, Dict
from .formula_parser import parse_formula


class ChemicalCompositionValidator:
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
    
    def _validate_numeric_values(
        self,
        composition: Dict[str, Union[str, int, float]]
    ) -> bool:
        """
        Validate that all values in the composition can be converted to float.
        """
        for key, value in composition.items():
            try:
                value = float(value)
                if value > 100:
                    return False
            except (ValueError, TypeError):
                return False
        return True
    
    def _expand_compound_formula(
        self,
        composition: Dict[str, Union[str, int, float]]
    ) -> Dict[str, float]:
        """
        Expand compound formulas into their constituent elements.

        Example:
            {'Cr0.01Zn0.99': 2, 'Na': 1} -> {'Cr': 0.02, 'Zn': 1.98, 'Na': 1.0}
        """
        expanded_composition = dict()
        
        for formula, quantity in composition.items():
            # Remove parentheses and convert quantity to float
            clean_formula = formula.strip("()")
            outer_multiplier = float(quantity)

            
            # Find all element-number pairs in the formula
            matches = parse_formula(clean_formula)[0]
            if not matches:
                return {}       
            for element, value in matches.items():
                # Default to 1 if no number is specified
                if element in expanded_composition:
                    return {}
                element_count = float(value) if value else 1.0
                expanded_composition[element] = element_count * outer_multiplier
                    
        return expanded_composition
    
    def _all_elements_valid(self, elements: Dict[str, float]) -> bool:
        return all(element in self.ELEMENT_SYMBOLS for element in elements.keys())
    
    def _valid_composition(
        self,
        composition: Dict[str, Union[str, int, float]]
    ) -> bool:

        if not composition:
            return False
            
        # First check: If all values can be converted to float and are within 100
        if not self._validate_numeric_values(composition):
            return False
        
        # Second check: If all keys are valid element symbols, return True immediately
        if self._all_elements_valid(composition):
            return True

        
        # Third check: Try to expand compound formulas
        try:
            expanded_composition = self._expand_compound_formula(composition)
            
            # Check if expansion resulted in any elements
            if not expanded_composition:
                return False
            
            # Validate that all expanded elements are valid
            return self._all_elements_valid(expanded_composition)
            
        except Exception:
            # If any error occurs during expansion, composition is invalid
            return False


def valid_composition(comp: Dict[str, Union[str, int, float]]) -> bool:
    validator = ChemicalCompositionValidator()
    return validator._valid_composition(comp)
