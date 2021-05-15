import numpy as np

QN = np.diag([1.0, 0.0, 0.0])

# anw_p = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
# anw = np.array(anw_p[3::3])

xr = np.ones(12)
ub = np.array([2, 3, 4])
lb = np.array([6, 7, 8])

xr[3::3] = (lb + ub) / 2

print(xr)
print(xr[:3])
print(xr[3:])

print()
print(xr[-3:])

# Test 2
print(f"Test 2")
v_ref = 2
delta_s = 3

print(1 / v_ref * delta_s)

# Test 3
print("Test 3")
a = np.kron(np.ones(3), [5, 6])
print(a)