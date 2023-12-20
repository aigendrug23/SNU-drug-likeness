class TDC:
    def __init__(self, name, task):
        self.name = name
        self.task = task

    @classmethod
    def create_instance(cls, name, task):
        return cls(name, task)

    @classmethod
    def get_ordered_list(cls, tdc_list):
        # Sort the list of TDC instances
        sorted_tdc_list = sorted(tdc_list, key=lambda tdc: tdc.name)

        return sorted_tdc_list

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name

    def isRegression(self):
        return self.task == "regression"


# Create instances using the factory method
TDC.BBB = TDC.create_instance("BBB", "classification")
TDC.CYP3A4 = TDC.create_instance("CYP3A4", "classification")
TDC.Solubility = TDC.create_instance("Solubility", "regression")
TDC.Clearance = TDC.create_instance("Clearance", "regression")

TDC.allList = TDC.get_ordered_list(
    [TDC.BBB, TDC.CYP3A4, TDC.Solubility, TDC.Clearance]
)  # Order matters
