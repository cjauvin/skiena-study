from random import *

def quicksort(arr):
    if len(arr) <= 1: return arr
    pivot = arr[0]
    left, right = [], []
    for i in arr:
        if i < pivot: left.append(i)
        elif i > pivot: right.append(i)
    return quicksort(left) + [pivot] + quicksort(right)

def merge(a, b):
    na, nb = len(a), len(b)
    i, j = 0, 0
    c = []
    for _ in range(na + nb):
        if i >= na:
            c.append(b[j])
            j += 1
        elif j >= nb:
            c.append(a[i])
            i += 1
        elif a[i] <= b[j]:
            c.append(a[i])
            i += 1
        else:
            c.append(b[j])
            j += 1
    return c

def mergesort(arr):
    n = len(arr)
    if n <= 1: return arr
    return merge(mergesort(arr[0:n/2]), mergesort(arr[n/2:n]))
    
a = range(100000)
shuffle(a)

#print merge([1,4,5,9], [2,3,6,10])

assert quicksort(a) == range(100000)
assert mergesort(a) == range(100000)
    
