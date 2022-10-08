
def wrapper_loop(d):
    #print(d)
    ci = d["column_index"]
    oi = d["operator_index"]
    co = d["condition"]
    ret = []
    for i in range(len(ci)):
        ret.append([ci[i], oi[i], co[i]])
    #print(ret)
    return ret

class Query:

    agg_ops = ['', 'MAX', 'MIN', 'COUNT', 'SUM', 'AVG']
    cond_ops = ['=', '>', '<', 'OP']
    syms = ['SELECT', 'WHERE', 'AND', 'COL', 'TABLE', 'CAPTION', 'PAGE', 'SECTION', 'OP', 'COND', 'QUESTION', 'AGG', 'AGGOPS', 'CONDOPS']

    def __init__(self, sel_index, agg_index, columns,  conditions=tuple(),  ordered=False):
        self.sel_index = sel_index
        self.agg_index = agg_index
        self.conditions = (conditions)
        self.ordered = ordered
        self.columns = columns


    @classmethod
    def from_dict(cls, d, columns,  ordered=False):
        return cls(sel_index=d['sel'], agg_index=d['agg'], columns = columns, conditions=d['conds'], ordered=ordered)



    def __repr__(self):
        if(self.agg_index == 0):
            rep = 'SELECT {sel} FROM table'.format(
                sel=self.columns[self.sel_index],
            )
        else:
            rep = 'SELECT {agg} {sel} FROM table'.format(
                agg=self.agg_ops[self.agg_index],
                sel=self.columns[self.sel_index],
            )
        if len(wrapper_loop(self.conditions))!=0:
            rep +=  ' WHERE ' + ' AND '.join(['{} {} {}'.format(self.columns[i], self.cond_ops[o], v) for i, o, v in wrapper_loop(self.conditions)])

        return rep