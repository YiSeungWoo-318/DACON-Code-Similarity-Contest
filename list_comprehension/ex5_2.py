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