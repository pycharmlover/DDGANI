from typing import FrozenSet, List, Dict, Iterable

__version__ = "0.0.1"
__author__ = 'Tullio Pizzuti'
__credits__ = 'DAIS Lab (Computer Science Department, University of Salerno)'

class Attribute(str):
    threshold: float
    funct:str
    match_operator:str
    name:str

    def __new__(cls, name:str, threshold: float = 0.0, funct:str = None, match_operator:str = None, thr_precision:int=6):
        return super().__new__(cls, name)

    def __init__(self, name:str, threshold: float = 0.0, funct:str = None, match_operator:str = None, thr_precision:int=6):
        self.name = name.strip()
        self.threshold = round(threshold, thr_precision)
        self.funct = funct
        self.match_operator = match_operator

    def __eq__(self, other):
        if isinstance(other, Attribute):
            return super().__eq__(other) and self.threshold == other.threshold
        return False

    def __hash__(self):
        return hash((super().__hash__(), self.threshold))

    def __str__(self):
        return f"{super().__str__()}@{self.threshold}"
    def __repr__(self):
        return self.__str__()


class AttributeSet(FrozenSet[Attribute]):
    def __new__(cls, attributes: Iterable[Attribute]|Attribute=None):
        if attributes is None:
            attributes = {}
        elif isinstance(attributes, Attribute):
            attributes={attributes}
        return super(AttributeSet, cls).__new__(cls, attributes)

    def __init__(self, attributes: Iterable[Attribute] | Attribute = None):
        super().__init__()

    def __str__(self):
        return f"[{', '.join(map(lambda a: str(a), self))}]"
    def __repr__(self):
        return self.__str__()

    def remove_attributes(self, attributes: str|Iterable[str]):
        if isinstance(attributes, str):
            attributes = [attributes]
        attr_to_remove=set()
        for a in self:
            if a.name in attributes:
                attr_to_remove.add(a)
        return AttributeSet(self.difference(attr_to_remove))

    def attribute_names(self) -> set[str]: return set(map(lambda a: a.name, self))

    def get_attribute(self, name:str):
        return AttributeSet(list(self.remove_attributes(
            self.attribute_names().difference({name})
        )))


class Dependencies(Dict[AttributeSet, AttributeSet]):
    extent:float
    cov_measure:str
    def __new__(cls, extent:float = 0.0, cov_measure:str = None):
        return super().__new__(cls)

    def __init__(self, extent: float = 0.0, cov_measure: str = None):
        super().__init__()
        self.extent = extent
        self.cov_measure = cov_measure


    def __setitem__(self, key: AttributeSet, value: AttributeSet|Attribute):
        if not super().__contains__(key):
            super().__setitem__(key, AttributeSet())
        new_attrs = AttributeSet(list(super().__getitem__(key)) + list((AttributeSet(value) if isinstance(value, Attribute) else list(value))))
        super().__setitem__(key, new_attrs)

    def __str__(self):
        s='\n'.join(map(lambda a: f'{str(a)} -> {str(self[a])}', self))
        return f"[{s}]"

    def __repr__(self):
        return self.__str__()

    def get_unique_attributes_set(self):
        toret=set()
        for k,v in self.items():
            toret.update(k)
            toret.update(v)
        return toret

    def hassubset_on_lhs(self, attributes:AttributeSet|Attribute):
        if isinstance(attributes, Attribute):
            attributes=AttributeSet(attributes)
        for key in self.keys():
            if key.issubset(attributes):
                return True
        return False

    def subset_on_lhs(self, attributes:AttributeSet|Attribute):
        if isinstance(attributes, Attribute):
            attributes=AttributeSet(attributes)
        to_ret = Dependencies(self.extent, self.cov_measure)
        for key in self.keys():
            if key.issubset(attributes):
                to_ret[key] = self[key]
        return to_ret

    def hassuperset_on_lhs(self, attributes:AttributeSet|Attribute):
        if isinstance(attributes, Attribute):
            attributes=AttributeSet(attributes)
        for key in self.keys():
            if key.issuperset(attributes):
                return True
        return False

    def superset_on_lhs(self, attributes:AttributeSet|Attribute):
        if isinstance(attributes, Attribute):
            attributes=AttributeSet(attributes)
        to_ret = Dependencies(self.extent, self.cov_measure)
        for key in self.keys():
            if key.issuperset(attributes):
                to_ret[key] = self[key]
        return to_ret

    def hassubset_on_rhs(self, attributes:AttributeSet|Attribute):
        if isinstance(attributes, Attribute):
            attributes=AttributeSet(attributes)
        for key in self.keys():
            if self[key].issubset(attributes):
                return True
        return False

    def subset_on_rhs(self, attributes:AttributeSet|Attribute):
        if isinstance(attributes, Attribute):
            attributes=AttributeSet(attributes)
        to_ret = Dependencies(self.extent, self.cov_measure)
        for key in self.keys():
            if self[key].issubset(attributes):
                to_ret[key] = self[key]
        return to_ret

    def hassuperset_on_rhs(self, attributes:AttributeSet|Attribute):
        if isinstance(attributes, Attribute):
            attributes=AttributeSet(attributes)
        for key in self.keys():
            if self[key].issuperset(attributes):
                return True
        return False

    def superset_on_rhs(self, attributes:AttributeSet|Attribute):
        if isinstance(attributes, Attribute):
            attributes=AttributeSet(attributes)
        to_ret = Dependencies(self.extent, self.cov_measure)
        for key in self.keys():
            if self[key].issuperset(attributes):
                to_ret[key] = self[key]
        return to_ret

    def superset_on_lhs_by_name(self, attributes:str|Iterable[str]):
        if isinstance(attributes, str):
            attributes = {attributes}
        if isinstance(attributes, Iterable):
            attributes = set(attributes)
        to_ret = Dependencies(self.extent, self.cov_measure)
        for lhs, rhs in self.items():
            lhs_name=set(map(lambda a: a.name, lhs))
            if lhs_name.issuperset(attributes):
                to_ret[lhs]=rhs
        return to_ret

    def subset_on_lhs_by_name(self, attributes:str|Iterable[str]):
        if isinstance(attributes, str):
            attributes = {attributes}
        if isinstance(attributes, Iterable):
            attributes = set(attributes)
        to_ret = Dependencies(self.extent, self.cov_measure)
        for lhs, rhs in self.items():
            lhs_name=set(map(lambda a: a.name, lhs))
            if lhs_name.issubset(attributes):
                to_ret[lhs]=rhs
        return to_ret

    def superset_on_rhs_by_name(self, attributes: str | Iterable[str]):
        if isinstance(attributes, str):
            attributes= {attributes}
        if isinstance(attributes, Iterable):
            attributes = set(attributes)
        to_ret = Dependencies(self.extent, self.cov_measure)
        for lhs, rhs in self.items():
            rhs_name = set(map(lambda a: a.name, rhs))
            if rhs_name.issuperset(attributes):
                to_ret[lhs] = rhs
        return to_ret

    def subset_on_rhs_by_name(self, attributes: str | Iterable[str]):
        if isinstance(attributes, str):
            attributes = {attributes}
        if isinstance(attributes, Iterable):
            attributes = set(attributes)
        to_ret = Dependencies(self.extent, self.cov_measure)
        for lhs, rhs in self.items():
            rhs_name = set(map(lambda a: a.name, rhs))
            if rhs_name.issubset(attributes):
                to_ret[lhs] = rhs
        return to_ret

    def subset_leq_on_rhs(self, attributes:Attribute|AttributeSet):
        contained=Dependencies(self.extent, self.cov_measure)
        if isinstance(attributes, Attribute):
            attributes=AttributeSet(attributes)
        for lhs, rhs in self.superset_on_rhs_by_name(attributes.attribute_names()).items():
            insert=True
            rhs_to_add=AttributeSet()
            for attr_to_check in attributes:
                rhs_to_check = rhs.get_attribute(attr_to_check.name)
                for a in rhs_to_check:
                    if not a.threshold<=attr_to_check.threshold and len(rhs_to_check)==1:
                        insert=False
                        break
                    elif not a.threshold<=attr_to_check.threshold and len(rhs_to_check)>1:
                        pass
                    else:
                        rhs_to_add|=AttributeSet(a)
            if insert:
                contained[lhs]=rhs_to_add
        return contained


    def subset_geq_on_rhs(self, attributes:Attribute|AttributeSet):
        contained=Dependencies(self.extent, self.cov_measure)
        if isinstance(attributes, Attribute):
            attributes=AttributeSet(attributes)
        for lhs, rhs in self.superset_on_rhs_by_name(attributes.attribute_names()).items():
            insert=True
            rhs_to_add=AttributeSet()
            for attr_to_check in attributes:
                rhs_to_check = rhs.get_attribute(attr_to_check.name)
                for a in rhs_to_check:
                    if not a.threshold>=attr_to_check.threshold and len(rhs_to_check)==1:
                        insert=False
                        break
                    elif not a.threshold>=attr_to_check.threshold and len(rhs_to_check)>1:
                        pass
                    else:
                        rhs_to_add|=AttributeSet(a)
            if insert:
                contained[lhs]=rhs_to_add
        return contained


    def subset_leq_on_lhs(self, attributes:Attribute|AttributeSet):
        contained=Dependencies(self.extent, self.cov_measure)
        if isinstance(attributes, Attribute):
            attributes=AttributeSet(attributes)
        for lhs, rhs in self.subset_on_lhs_by_name(attributes.attribute_names()).items():
            insert=True
            for a in lhs:
                for _a in attributes.get_attribute(a.name):
                    if not a.threshold<=_a.threshold:
                        insert=False
                        break
            if insert:
                contained[lhs]=rhs
        return contained

    def subset_geq_on_lhs(self, attributes:Attribute|AttributeSet):
        contained=Dependencies(self.extent, self.cov_measure)
        if isinstance(attributes, Attribute):
            attributes=AttributeSet(attributes)
        for lhs, rhs in self.subset_on_lhs_by_name(attributes.attribute_names()).items():
            insert=True
            for a in lhs:
                for _a in attributes.get_attribute(a.name):
                    if not a.threshold>=_a.threshold:
                        insert=False
                        break
            if insert:
                contained[lhs]=rhs
        return contained



    def apply_dominance(self):
        filtered=Dependencies(self.extent, self.cov_measure)
        for lhs, rhs in self.items():
            for a in rhs:
                contained_a=self.subset_leq_on_rhs(a)
                contained_b=contained_a.subset_geq_on_lhs(lhs)
                if contained_b.count_dependencies()==1:
                    filtered[lhs]=a
        return filtered

    def count_dependencies(self):
        return sum(map(lambda rhs: len(rhs),self.values()))


class DependenciesFormatter:
    def __init__(self, lhs_rhs_separator:str="->", comparison_threshold_separator:str="@", attributes_separator:str=",",
                 to_remove=None):
        if to_remove is None:
            to_remove = ["KEY", "DEP", "(", ")","[", "]"]
        self.lhs_rhs_separator = lhs_rhs_separator
        self.comparison_threshold_separator = comparison_threshold_separator
        self.attributes_separator = attributes_separator
        self.to_remove = to_remove

    def cast_dependency(self, dependency_string:str):
        dependency_string_copy = dependency_string
        for s in self.to_remove:
            dependency_string_copy=dependency_string_copy.replace(s,"")
        dependency_string_copy=dependency_string_copy.strip()
        lhs, rhs = dependency_string_copy.split(self.lhs_rhs_separator)
        rhs = self.__cast_attribute_set__(rhs.strip())
        lhs = self.__cast_attribute_set__(lhs.strip())
        return lhs,rhs


    def __cast_attribute_set__(self, attribute_set_string:str):
        if attribute_set_string=="":
            return AttributeSet()
        attribute_set=[]
        attrs=attribute_set_string.split(self.attributes_separator)
        for attr in attrs:
            thr="0.0"
            attr_name=attr
            if (self.comparison_threshold_separator is None or self.comparison_threshold_separator!="") and self.comparison_threshold_separator in attr:
                attr_name, thr=attr.split(self.comparison_threshold_separator)

            attribute_set.append(Attribute(attr_name.strip(), float(thr)))
        return AttributeSet(attribute_set)

class DependenciesLoader:
    def __init__(self, lhs_rhs_separator:str="->", comparison_threshold_separator:str="@", attributes_separator:str=",",
                 to_remove=None, extent:float = 0.0, cov_measure:str = None ):

        self.dependencies_formatter=DependenciesFormatter(lhs_rhs_separator, comparison_threshold_separator, attributes_separator, to_remove)
        self.dependencies = Dependencies(extent=0.0, cov_measure=None)

    def add_dependency(self, dependency_string:str):
        lhs, rhs= self.dependencies_formatter.cast_dependency(dependency_string)
        self.dependencies[lhs]=rhs



# if __name__ == "__main__":
#     dep_list = ["CustomerName@0.0 -> Region@0.0, Country@0.0",
#            "Country@1.0, OrderYear@1.0, ModelSeries@1.0, OrderMonth@1.0 -> DeliveryYear@3.0",
#            "OrderYear@0.0, ModelSeries@0.0, DeliveryYear@0.0, Country@0.0 -> Engine@0.0",
#            "OrderYear@0.0, ModelSeries@0.0, CustomerName@3.0, DeliveryYear@0.0 -> Engine@0.0",
#            "ModelSeries@1.0, OrderMonth@0.0, DeliveryYear@0.0, Country@0.0 -> Engine@0.0",
#            "ModelSeries@2.0, Country@0.0, DeliveryYear@0.0, OrderYear@0.0, OrderMonth@0.0 -> Engine@0.0",
#            "OrderYear@1.0 -> OrderMonth@3.0",
#            "Region@2.0 -> OrderMonth@3.0",
#            "Country@0.0 -> Region@0.0",
#            "Country@1.0 -> Region@2.0"]
#     dp = DependenciesLoader()
#     for s in dep_list:
#         dp.add_dependency(s)
#     print(dp.dependencies)
#     print("-"*50)
#     print(dp.dependencies.apply_dominance())

