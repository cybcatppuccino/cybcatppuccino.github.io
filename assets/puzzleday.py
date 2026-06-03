import numpy as np

p1 = np.asarray([[2,2,0],[0,2,0],[0,2,2]], dtype=np.int8)
p2 = np.asarray([[3,3,3,3],[3,0,0,0]], dtype=np.int8)
p3 = np.asarray([[4,4,4],[4,0,0],[4,0,0]], dtype=np.int8)
p4 = np.asarray([[5,5,5],[5,5,0]], dtype=np.int8)
p5 = np.asarray([[6,6,6],[6,0,6]], dtype=np.int8)
p6 = np.asarray([[7,7,7,7],[0,7,0,0]], dtype=np.int8)
p7 = np.asarray([[8,8,8],[8,8,8]], dtype=np.int8)
p8 = np.asarray([[9,9,9,0],[0,0,9,9]], dtype=np.int8)
BOUND = 5 # smallest component size
MAXLEN = 4 # Largest piece is in a 4*4

table = np.asarray([[0,0,0,0,0,0,1]]*2 + [[0,0,0,0,0,0,0]]*4 + [[0,0,0,1,1,1,1]], dtype=np.int8)
pieces = [p1,p8,p3,p6,p2,p5,p4,p7]

def copypiece(inpce, rot, lr, ud):
    if not(rot):
        outpce = np.arange(inpce.size, dtype=np.int8).reshape(inpce.shape[0], inpce.shape[1])
        for a in range(len(inpce)):
            for b in range(len(inpce[1])):
                outpce[a][b] = inpce[len(inpce) - 1 - a if lr else a][len(inpce[0]) - 1 - b if ud else b]
    else:
        outpce = np.arange(inpce.size, dtype=np.int8).reshape(inpce.shape[1], inpce.shape[0])
        for a in range(len(inpce)):
            for b in range(len(inpce[1])):
                outpce[b][a] = inpce[len(inpce) - 1 - a if lr else a][len(inpce[0]) - 1 - b if ud else b]
    return outpce

def gen(month, day):
    table0 = np.copy(table)
    table0[(month-1) // 6][(month-1) % 6] = 1
    table0[2 + (day-1) // 7][(day-1) % 7] = 1
    return table0

def notin(inpce1, inlst):
    for inpce2 in inlst:
        if inpce1.shape == inpce2.shape:
            if np.all(inpce1 == inpce2):
                return False
    return True

def all1piece(inpce):
    outlst = list()
    for rot in range(2):
        for lr in range(2):
            for ud in range(2):
                newpce = copypiece(inpce, rot, lr, ud)
                if notin(newpce, outlst):
                    outlst.append(newpce)
    return outlst

def allpieces(inpcs):
    outlst = list()
    for pcs in inpcs:
        outlst.append(all1piece(pcs))
    return outlst

alist = allpieces(pieces)

alistnew = np.arange(len(alist) * 9 * MAXLEN * MAXLEN, dtype=np.int8).reshape(len(alist), 9, MAXLEN, MAXLEN)
alistlen = np.arange(len(alist) * 9 * 2).reshape(len(alist), 9, 2)
alistlen.fill(-1)
for pcen in range(len(alist)):
    for pced in range(len(alist[pcen])):
        xx = len(alist[pcen][pced])
        yy = len(alist[pcen][pced][0])
        for aa in range(xx):
            for bb in range(yy):
                alistnew[pcen][pced][aa][bb] = alist[pcen][pced][aa][bb]
        alistlen[pcen][pced][0] = xx
        alistlen[pcen][pced][1] = yy

print("alistdone! The first time solving a table would be slow! ")

from numba import jit

@jit
def addonepce1a(inpce, newtable, x, y):
    for a in range(len(inpce)):
        for b in range(len(inpce[0])):
            if newtable[x+a][y+b] == 0:
                newtable[x+a][y+b] = inpce[a][b]
    return newtable

@jit
def addonepce1abool(inpce, newtable, x, y):
    for a in range(len(inpce)):
        for b in range(len(inpce[0])):
            if newtable[x+a][y+b] != 0 and inpce[a][b] != 0:
                return False
    return True

@jit
def tableto1(intable):
    return np.minimum(intable.copy(), 1)

@jit
def component(intable, x, y, arg):
    if intable[x][y] != 0:
        return 0
    intable[x][y] = -1
    out = 1
    if x > 0 and arg != 1:
        out += component(intable, x-1, y, 2)
    if x < len(intable) - 1 and arg != 2:
        out += component(intable, x+1, y, 1)
    if y > 0 and arg != 3:
        out += component(intable, x, y-1, 4)
    if y < len(intable[0]) - 1 and arg != 4:
        out += component(intable, x, y+1, 3)
    return out

@jit
def fastkill(newtbl):
    for a in range(len(newtbl)):
        for b in range(len(newtbl[0])):
            if newtbl[a][b] == 0:
                if component(newtbl, a, b, 0) < BOUND:
                    return False
    return True

# start addonepce3

@jit
def addonepce1b(inpce, newtable, x, y):
    for a in range(len(inpce)):
        for b in range(len(inpce[0])):
            if newtable[x+a][y+b] != 1:
                newtable[x+a][y+b] = max(inpce[a][b], newtable[x+a][y+b])
            elif inpce[a][b] != 0:
                return newtable
    return newtable

@jit
def addonepce1bbool(inpce, newtable, x, y):
    for a in range(len(inpce)):
        for b in range(len(inpce[0])):
            if newtable[x+a][y+b] == 1 and inpce[a][b] != 0:
                return False
    return True

@jit
def addonepce3(inpce, outtbl):
    for x in range(len(outtbl) - len(inpce) + 1):
        for y in range(len(outtbl[0]) - len(inpce[0]) + 1):
            if addonepce1bbool(inpce, outtbl, x, y):
                addonepce1b(inpce, outtbl, x, y)

# end addonepce3

@jit
def addonepce2jit(inpce, intable):
    outlst = np.arange(0, dtype=np.int8).reshape(0, intable.shape[0], intable.shape[1])
    for x in range(len(intable) - len(inpce) + 1):
        for y in range(len(intable[0]) - len(inpce[0]) + 1):
            if addonepce1abool(inpce, intable, x, y):
                newtbl = addonepce1a(inpce, np.copy(intable), x, y)
                if fastkill(np.copy(newtbl)):
                    outlst = np.concatenate((newtbl.reshape(1, intable.shape[0], intable.shape[1]), outlst))
    return outlst

@jit
def addpce(intable, pcenum, use3=True):
    if use3:
        intable1 = tableto1(np.copy(intable))
        intable1ok = False
    for pnum in range(pcenum, len(alistnew)):
        plist = np.arange(0, dtype=np.int8).reshape(0, intable.shape[0], intable.shape[1])
        for pdir in range(9):
            if  alistlen[pnum][pdir][0] == -1:
                break
            pce = alistnew[pnum][pdir][:alistlen[pnum][pdir][0], :alistlen[pnum][pdir][1],]
            plist = np.concatenate((plist, addonepce2jit(pce, intable)))
            if use3 and not intable1ok:
                addonepce3(pce, intable1)
                intable1ok = np.all(intable1 != 0)
        if len(plist) == 0:
            return np.arange(0, dtype=np.int8).reshape(0, intable.shape[0], intable.shape[1])
        if pnum == pcenum:
            outlst = plist.copy()
    if use3 and not(intable1ok):
        return np.arange(0, dtype=np.int8).reshape(0, intable.shape[0], intable.shape[1])
    return outlst

def solve(intable, use3=True, prn=False):
    if prn:
        print("Start Solving!")
    slist = intable.copy().reshape(1, intable.shape[0], intable.shape[1])
    for pcenum in range(len(pieces)):
        slistnew = np.arange(0, dtype=np.int8).reshape(0, intable.shape[0], intable.shape[1])
        for num in range(len(slist)):
            tbl = slist[num]
            slistnew = np.concatenate((slistnew, addpce(tbl, pcenum, use3)))
            if num % 2000 == 1000 and prn:
                print("pcenum=", pcenum, ", num=", num, ", all=", len(slist), ", slistlen=", len(slistnew))
                if len(slistnew) > 0:
                    for a in range(len(intable)):
                        print(slistnew[-1][a])
        slist = slistnew
        if prn:
            print(pcenum, len(slist))
    return slist

def solvedate(monate, tag, use3=True, prn=False):
    gentable = gen(monate, tag)
    rst = solve(gentable, use3, prn)
    if prn:
        print(len(rst))
    if len(rst) > 0 and prn:
        print(rst[0])
    return rst

def year():
    lst = []
    for monate in range(1, 13):
        for tag in range(1, 32):
            lst.append((monate, tag, len(solvedate(monate, tag))))
            print((monate, tag, len(solvedate(monate, tag))))
    return lst

if __name__ == "__main__":
    rst = solvedate(1, 1, True, True)
    yr = year()