def F1(n):
    if(n<=0):
             return n
    p = [0,1]
    p[0]=0
    p[1]=1
    for i in range(2,n+1):
        p.append(p[i-1]+p[i-2])
    return p[n]
def F2(n):
    if(n<=1):
        return n
    return F2(n-1)+F2(n-2)

print(F1(30))