"""
rank, pi_family = null(Q)

print(pi_family.shape)
pi_family = pi_family[:,0]
print(pi_family.shape)
print(pi_family.sum())

plt.figure("pi_family")
x = np.arange(pi_family.shape[0])
plt.plot(x, pi_family)
plt.show()

pi = pi_family / np.transpose(pi_family)
pi = pi * np.ones(pi.shape[0])

print(pi.shape)
print(pi.sum())

plt.figure("stationary distrib")
x = np.arange(pi.shape[0])
plt.plot(x, pi)
plt.show()
"""
 
"""
pt = w_matrix.transpose()

eigenvalues, eigenvectors = np.linalg.eig(pt)

index_eigenvalue_1 = np.where(np.isclose(eigenvalues, 1))[0][0]

steady_state_distribution = np.real(eigenvectors[:, index_eigenvalue_1])

print("sum stationnary distrib", steady_state_distribution.sum())

steady_state_distribution /= np.sum(steady_state_distribution)


matrix = np.zeros((Zr+1, Zp+1))
for i in range(numStates):
    ir, ip = rec_V(i)
    matrix[ir, ip] = steady_state_distribution[i]
    
plt.matshow(matrix)

plt.figure("stationary distrib")
x = np.arange(numStates)
plt.plot(x, steady_state_distribution)
plt.show()
"""
"""
Q = pt - np.eye(pt.shape[0])

Q[-1, :] = 1

steady_state_distribution = np.linalg.lstsq(Q, np.zeros(pt.shape[0]), rcond=None)[0]

print("sum stationnary distrib", steady_state_distribution.sum())

steady_state_distribution /= np.sum(steady_state_distribution)

matrix = np.zeros((Zr+1, Zp+1))
for i in range(numStates):
    ir, ip = rec_V(i)
    matrix[ir, ip] = steady_state_distribution[i]
    
plt.matshow(matrix)
"""

"""

valp, vecp = np.linalg.eig(w_matrix)
print("valp:", valp)
print("vecp:", vecp)
index_eigenvalue_1 = np.where(np.isclose(valp, 1))[0][0]
print(index_eigenvalue_1)
eigenvector_1 = vecp[:,index_eigenvalue_1]
print(eigenvector_1)
print(eigenvector_1.sum())


matrix = np.zeros((Zr+1, Zp+1))
for i in range(numStates):
    ir, ip = rec_V(i)
    matrix[ir, ip] = np.abs(eigenvector_1[i])
    
plt.matshow(matrix)


for i, val in enumerate(valp):
    #if np.real(val) == 1.0 and np.imag(val) == 0.0:
    if np.abs(val) == 1.0:
        result = vecp[i]
        print("find a result !")

        
end = time.time()

print("elapsed time =", end-start, "seconds")

"""


"""
mi = np.arange(matrix.shape[0])
mj = np.arange(matrix.shape[1])
color = matrix/matrix.max()
color = 1 - color
plt.scatter(x=mj, y=mi, c=color, alpha=1, s=2)
plt.show()

fig, ax = plt.subplots()
for i in range(m.shape[0]):
    for j in range(m.shape[1]):
        color = (m[i][j])/(m.max())
        ax.scatter(j, i, c=[[1-color,1-color,1-color]], alpha=1, s=2)  # j and i are reversed to match matrix indexing
ax.set_aspect('equal')
plt.show()
"""