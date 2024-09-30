import sys
import math
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime


def redirect_output(output_file):
    sys.stdout = open(output_file, 'w')

def mse(np_trans, recon_data):
    return np.mean(np.sum((np_trans - recon_data) ** 2, axis=1))

def pca(np_trans):
    n_comps=None
    X_centered = np_trans - np.mean(np_trans, axis=0)
    cov_matrix = np.cov(X_centered, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
    idx = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    if n_comps is None:
        n_comps = np_trans.shape[1]
    return eigenvectors[:, :n_comps], eigenvalues

def power_iteration(cov_numpy, max_iterations=1000, threshold=1e-6):
    d = cov_numpy.shape[0]
    x = np.random.rand(d)
    x = x / np.linalg.norm(x)

    for _ in range(max_iterations):
        x_prev = x
        x = np.dot(cov_numpy, x)
        m = np.argmax(np.abs(x))
        x = x / x[m]

        if np.linalg.norm(x - x_prev) < threshold:
            break

    # Normalize the eigenvector
    dom_eigvec_man = x / np.linalg.norm(x)

    # Calculate the eigenvalue
    dom_eigval_man = np.dot(np.dot(cov_numpy, dom_eigvec_man), dom_eigvec_man)

    return dom_eigval_man, dom_eigvec_man
def imported_file(file_name):
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.expanduser(f"~\\Desktop\\assign1-iumaugerra.txt")
        redirect_output(output_file)
        open_file = open(file_name, "r")
        columns, row, sum_data, mean_data, std_deviation, z_normalized, centered_data = [], [], [], [], [], [], []
        rows = 0
        for _ in range(9):
            sum_data.append(0)
            mean_data.append(0)
            std_deviation.append(0)
            centered_data.append([])
            z_normalized.append([])
            columns.append([])
            row.append([])

        for line in open_file:
            data = line.strip().split(",")[1:-1]
            for i, value in enumerate(data):
                row[i] = float(value)
                columns[i].append(float(value))
                sum_data[i] += float(value)
            rows += 1

        for i in range(9):
            mean_data[i] = sum_data[i] / rows
            for j in range(rows):
                centered_data[i].append(columns[i][j] - mean_data[i])
                std_deviation[i] += (centered_data[i][j] ** 2)
            std_deviation[i] = math.sqrt(std_deviation[i] / rows)
            z_normalized[i] = [round((columns[i][j] - mean_data[i]) / std_deviation[i], 3) for j in range(rows)]

        print("Z_Normalized data is as follows:")
        for j in range(rows):
            print(*[z_normalized[i][j] for i in range(9)])
        print("\n------------------------------------------------------------------------------------\n")

        z_normalized_np = np.array(z_normalized)
        np_trans=np.transpose(z_normalized_np)

        # Manual covariance calculation using sum of outer products with np.outer
        cov_manual = np.zeros((9, 9))
        for i in range(rows):
            column_vector = z_normalized_np[:, i]
            cov_manual += np.outer(column_vector, column_vector)
        cov_manual /= rows

        print("Manual Covariance is as follows:")
        print(cov_manual)
        print()

        # NumPy covariance calculation
        cov_numpy = np.cov(z_normalized_np, bias=True)
        print("Numpy Covariance is as follows:")
        print(cov_numpy)
        print()

        if(np.allclose(cov_manual, cov_numpy, atol=1e-4)):
            print("Matches!")
        else:
            print("Mismatch!")

        dom_eigval_man, dom_eigvec_man=power_iteration(cov_numpy)
        print(f"Manual Dominant eigenvalue = {dom_eigval_man}")
        print(f"Manual Dominant eigenvector = {dom_eigvec_man}")

        eigvals, eigvecs = np.linalg.eig(cov_numpy)
        max_eigval_index = np.argmax(eigvals)
        dom_eigval_numpy = eigvals[max_eigval_index]
        dom_eigvec_numpy = eigvecs[:, max_eigval_index]

        print(f"Numpy dominant eigenvalue = {dom_eigval_numpy}")
        print(f"Numpy dominant eigenvector = {dom_eigvec_numpy}")
        print()

        sorted_indices = np.argsort(eigvals)[::-1]
        top_two_eigvecs = eigvecs[:, sorted_indices[:2]]
        projected_data = np.dot(np_trans, top_two_eigvecs)
        projected_variance = np.var(projected_data, axis=0).sum()
        print(f"Variance = {projected_variance}")
        print()

        U = eigvecs
        lmbda = np.diag(eigvals)
        print(f"U = {U}")
        print(f"lmbda = {lmbda}")
        print()

        cov_recon = np.dot(U, np.dot(lmbda, np.transpose(U)))

        projected_data_2d = np.dot(np_trans, top_two_eigvecs)
        recon_data = np.dot(projected_data_2d, np.transpose(top_two_eigvecs))

        mse_value = mse(np_trans, recon_data)
        print(f"MSE = {mse_value}")
        print()

        sum_rem_eigvals = np.sum(eigvals[2:])
        print(f"Sum of remaining eigvals = {sum_rem_eigvals}")

        if np.isclose(mse_value, sum_rem_eigvals, atol=1e-8):
            print("MSE == sum of remaining eigenvals")
        else:
            print("MSE != sum of remaining eigenvals")

        pca_vectors, eigenvalues = pca(np_trans)
        projected_data_2d = np.dot(np_trans, pca_vectors[:, :2])
        print(pca_vectors[:, :2])
        print(projected_data_2d)
        plt.figure(figsize=(10, 8))
        plt.scatter(projected_data_2d[:, 0], projected_data_2d[:, 1])
        plt.xlabel('First Principal Component')
        plt.ylabel('Second Principal Component')
        plt.title('Data points projected onto first two principal components')
        plt.savefig('pca_projection.png')
        plt.close()
        print()

        cumulative_variance_ratio = np.cumsum(eigenvalues) / np.sum(eigenvalues)
        n_comps_var = np.argmax(cumulative_variance_ratio >= 0.95) + 1
        print(f"No. of principal components needed to preserve 95% of variance = {n_comps_var}")

        pcavec_var = pca_vectors[:, :n_comps_var]
        projdata_var = np.dot(np_trans, pcavec_var)
        print(f"Coords of initial 10 data points in new basis is as follows:")
        print(len(projdata_var), projdata_var.shape[0])
        for i in range(min(10, projdata_var.shape[0])):
            print(f"Data point {i + 1}: {projdata_var[i]}")

    except FileNotFoundError:
        print(f"Error: File '{file_name}' not found.")

    finally:
        if sys.stdout != sys.__stdout__:
            sys.stdout.close()
            sys.stdout = sys.__stdout__
        print(f"Output saved to: {output_file}")


file_name = sys.argv[1]
imported_file(file_name)