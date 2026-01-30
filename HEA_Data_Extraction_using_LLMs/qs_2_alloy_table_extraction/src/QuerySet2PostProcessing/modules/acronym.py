from sortedcontainers import SortedDict
from collections import defaultdict
import re
from modules.validator import valid_composition
import math

class Acronym:
    def __init__(self, acronym, composition, unit):
        self.acronym = acronym
        self.original_composition = composition
        self.unit = unit
        self.standardized_composition = self.standardize()
        self.serialized_composition = self.serialize()

    def standardize(self):
        if ";" in self.original_composition:
            tmp = self.original_composition.split(";")
            comp = SortedDict()
            for item in tmp:
                try:
                    ele, val = [i.strip().replace('"','') for i in item.split(":")]
                    if "±" in val:
                        val = val.split('±')[0].strip()
                    elif "+-" in val:
                        val = val.split("+-")[0].strip()
                    
                    val = float(val)
                    if math.isnan(val):
                        val = 0.0
                    comp[ele] = val
                except ValueError:
                    return {"101": "MalformedItemError"}

            if valid_composition(comp):
                return comp
            else:
                return {"102" : "InvalidElementKeyError"}
        else:
            return {"103": "InvalidFormError"}

    def serialize(self):
        s = ""
        for key, val in self.standardized_composition.items():
            s += key+str(val)

        return s
        

        
        

