e1 = [i for i in range(10)]
print(e1)
x = []
for i in range(10):
      x.append(i)
print(x)
#################

e2 = [i*i for i in range(10)]
print(e2)
x2 = []
for i in range(10):
      x2.append(i*i)
print(x2)
###########################

e3 = [i for i in range(10, -1, -1)]
print(e3)
x3 = []
for i in range(10, -1, -1):
      x3.append(i)
print(x3)
#######################
f1 = [i for i in [i for i in range(20)] if i %2 == 1]
print(f1)
f2 = []
for i in range(20):
      if i % 2 == 1:
            f2.append(i)
print(f2)
######################
#different
k = [[i for i in range(10)],[i for i in range(10,20)], [i for i in range(20, 30)]]
g = [[i for i in item] for item in k]
print(g)
t = []
for r in range(3):
      if r==0:
            j = []
            for i in range(10):
                  j.append(i)
            t.append(j)
      elif r==1:
            j = []
            for k in range(10, 20):
                  j.append(k)
            t.append(j)
      else:
            z = []
            for zz in range(20, 30):
                  z.append(zz)
            t.append(z)
u = []
for i in t:
      for j in i:
            u.append(j)
print(u)
##################################################
#same
k = [[i for i in range(10)],[i for i in range(10,20)], [i for i in range(20, 30)]]
g = [[i for i in item] for item in k]
print(g)
t = []
for r in range(3):
      if r==0:
            j = []
            for i in range(10):
                  j.append(i)
            t.append(j)
      elif r==1:
            j = []
            for k in range(10, 20):
                  j.append(k)
            t.append(j)
      else:
            z = []
            for zz in range(20, 30):
                  z.append(zz)
            t.append(z)
u = []
for i in t:
      u.append(i)
print(u)
##############################################
m = {"col":1, "ind":2}
q = {k:v for k,v in m.items()if type(v) == int}
##################################
m = {"col":1, "ind":2}
z = {}
for k, v in m.items():
      if type(v) == int:
            z[v] = k
print(z)
############################