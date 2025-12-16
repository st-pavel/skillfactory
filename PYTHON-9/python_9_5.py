import numpy as np

a = np.int8(25)
print(np.iinfo(np.int8))
print(np.iinfo(np.int16))
print(np.iinfo(np.int32))
print(np.iinfo(np.int64))


#print(*sorted(map(str, set(np.sctypeDict.values()))), sep='\n')
try:
    print(np.uint8(-456))
except Exception:
    print(-456%128)

zeros_3d = np.zeros((5,4,3), dtype=np.float32)
print(zeros_3d)


arr = np.array([345234, 876362.12, 0, -1000, 99999999])

arr = np.float64(arr)
print(arr)
print(arr.dtype)

arr, step = np.linspace(-6, 21, 60, endpoint=False, retstep=True)
print(step)

#np.finfo(np.float128)