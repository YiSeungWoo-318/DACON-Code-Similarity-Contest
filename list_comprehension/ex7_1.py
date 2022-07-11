m = {"col":1, "ind":2}
q = {k:v for k,v in m.items()if type(v) == int}
print(q)