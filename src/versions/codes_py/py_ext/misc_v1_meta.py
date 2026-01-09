from prettytable import PrettyTable


def tabularPrintingConstructing(s, field_names, ifNeedToAddTs1):

    x = PrettyTable()
    if ifNeedToAddTs1:
        x.field_names = ["ts1"] + field_names
    else:
        x.field_names = field_names
    for k in s.keys():
        if ifNeedToAddTs1:
            l = [k]
        else:
            l = []
        for kc in field_names:
            l.append(s[k][kc])
        x.add_row(l)
    return x
