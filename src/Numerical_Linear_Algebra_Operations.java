import java.io.*;
import java.util.*;

/**
 * Numerical Linear Algebra Operations.
 * <p>
 * A small, educational utility class implementing common numerical linear
 * algebra routines (e.g., determinants, rank, transpose, and a basic QR
 * decomposition). Methods are written to be readable and self-contained.
 * </p>
 * <h2>Included functionality</h2>
 * <ul>
 * <li>Determinant via recursive Laplace expansion (educational; small matrices
 * only)</li>
 * <li>Determinant via LU decomposition with partial pivoting</li>
 * <li>Matrix transpose</li>
 * <li>Matrix rank via Gaussian elimination with partial pivoting</li>
 * <li>QR decomposition (Q and R) via Modified Gram–Schmidt</li>
 * <li>Basic linear independence check for a list of vectors</li>
 * </ul>
 * <h2>Notes on numerical stability</h2>
 * <p>
 * Some routines (notably Laplace expansion and basic Modified Gram–Schmidt) are
 * primarily educational and may be numerically fragile for ill-conditioned
 * inputs.
 * </p>
 */
public class Numerical_Linear_Algebra_Operations
{
    /**
     * Threshold used to treat very small values as zero in floating-point
     * checks.
     */
    private static final double EPS = 1e-12;

    // ----------------------------------------------------------
    /**
     * Computes the determinant of a square integer matrix using recursive
     * Laplace expansion along the first row.
     * <p>
     * This algorithm is primarily for learning purposes; it has factorial time
     * complexity ({@code O(n!)}) and should only be used for very small
     * matrices.
     * </p>
     *
     * @param arr
     *            the input {@code n x n} integer matrix
     * @return the determinant of {@code arr}
     * @throws IllegalArgumentException
     *             if {@code arr} is empty or not square
     * @throws Exception
     *             if an error occurs during recursion
     */
    public static int det_Lagrange_Expansion(int[][] arr)
        throws Exception
    {
        // Validate matrix shape (must be square and non-empty).
        if (arr.length != arr[0].length || arr.length == 0
            || arr[0].length == 0)
        {
            throw new IllegalArgumentException(
                "Matrix must be non-empty and square.");
        }

        // Base case: 1x1 determinant is the lone entry.
        if (arr.length == 1)
        {
            return arr[0][0];
        }

        // Expand along the first row.
        int[] firstRow = arr[0];
        int det = 0;

        for (int i = 0; i < firstRow.length; i++)
        {
            // Build the (n-1)x(n-1) minor by removing row 0 and column i.
            Queue<Integer> minorList = new LinkedList<>();

            for (int r = 1; r < firstRow.length; r++)
            {
                for (int c = 0; c < firstRow.length; c++)
                {
                    if (c != i)
                    {
                        minorList.add(arr[r][c]);
                    }
                }
            }

            int[][] minorMatrix = constructMinor(minorList);

            // Cofactor sign alternates by column.
            int sign = (i % 2 == 0) ? 1 : -1;
            det += sign * firstRow[i] * det_Lagrange_Expansion(minorMatrix);
        }

        return det;
    }


    // ----------------------------------------------------------
    /**
     * Constructs a square minor matrix from a queue of integers in row-major
     * order.
     * <p>
     * This helper is used by {@link #det_Lagrange_Expansion(int[][])}. The
     * queue must contain a perfect square number of entries.
     * </p>
     *
     * @param list
     *            a queue containing the minor's entries in row-major order
     * @return the reconstructed minor matrix
     * @throws IllegalArgumentException
     *             if the queue size is not a perfect square
     */
    public static int[][] constructMinor(Queue<Integer> list)
    {
        int size = list.size();
        int dim = (int)Math.sqrt(size);

        // Validate that size is a perfect square.
        if (dim * dim != size)
        {
            throw new IllegalArgumentException(
                "Minor construction failed: list size " + size
                    + " is not a perfect square.");
        }

        int[][] minor = new int[dim][dim];

        // Fill row-by-row using FIFO ordering.
        for (int r = 0; r < dim; r++)
        {
            for (int c = 0; c < dim; c++)
            {
                minor[r][c] = list.poll();
            }
        }

        return minor;
    }


    // ----------------------------------------------------------
    /**
     * Computes the determinant of a square real matrix using LU decomposition
     * with partial pivoting (Doolittle factorization).
     * <p>
     * If {@code P A = L U} with {@code P} a permutation matrix representing row
     * swaps, then:
     * </p>
     *
     * <pre>
     * det(A) = (-1)^(swapCount) * Π U[i][i]
     * </pre>
     *
     * @param matrix
     *            the input {@code n x n} matrix
     * @return the determinant of {@code matrix}
     * @throws IllegalArgumentException
     *             if {@code matrix} is null, empty, ragged, or not square
     */
    public static double det_LU(double[][] matrix)
    {
        // Basic validation: non-null, non-empty, square.
        if (matrix == null || matrix.length == 0 || matrix[0] == null)
        {
            throw new IllegalArgumentException("Matrix is null or empty.");
        }

        int n = matrix.length;
        for (int i = 0; i < n; i++)
        {
            if (matrix[i] == null || matrix[i].length != n)
            {
                throw new IllegalArgumentException(
                    "Matrix must be square (n x n).");
            }
        }

        // Copy input so we don't mutate the caller's matrix.
        double[][] a = new double[n][n];
        for (int i = 0; i < n; i++)
        {
            System.arraycopy(matrix[i], 0, a[i], 0, n);
        }

        // L has implicit 1s on diagonal; U is upper triangular.
        double[][] L = new double[n][n];
        double[][] U = new double[n][n];

        for (int i = 0; i < n; i++)
        {
            L[i][i] = 1.0;
        }

        int swapCount = 0;

        // Doolittle LU with partial pivoting.
        for (int k = 0; k < n; k++)
        {
            // Choose pivot row p >= k maximizing |a[p][k]|.
            int pivot = k;
            double maxAbs = Math.abs(a[k][k]);
            for (int i = k + 1; i < n; i++)
            {
                double v = Math.abs(a[i][k]);
                if (v > maxAbs)
                {
                    maxAbs = v;
                    pivot = i;
                }
            }

            // If pivot is effectively zero, the matrix is singular (det = 0).
            if (maxAbs < EPS)
            {
                return 0.0;
            }

            // Swap rows in the working copy and the already-built portion of L.
            if (pivot != k)
            {
                double[] tmp = a[k];
                a[k] = a[pivot];
                a[pivot] = tmp;

                // Swap only columns < k in L (the part that has been computed).
                for (int j = 0; j < k; j++)
                {
                    double t = L[k][j];
                    L[k][j] = L[pivot][j];
                    L[pivot][j] = t;
                }

                swapCount++;
            }

            // Compute row k of U.
            for (int j = k; j < n; j++)
            {
                double sum = 0.0;
                for (int t = 0; t < k; t++)
                {
                    sum += L[k][t] * U[t][j];
                }
                U[k][j] = a[k][j] - sum;
            }

            // Compute column k of L below the diagonal.
            for (int i = k + 1; i < n; i++)
            {
                double sum = 0.0;
                for (int t = 0; t < k; t++)
                {
                    sum += L[i][t] * U[t][k];
                }
                L[i][k] = (a[i][k] - sum) / U[k][k];
            }
        }

        // det(A) = (-1)^(swapCount) * product of U diagonal.
        double det = (swapCount % 2 == 0) ? 1.0 : -1.0;
        for (int i = 0; i < n; i++)
        {
            det *= U[i][i];
        }
        return det;
    }


    // ----------------------------------------------------------
    /**
     * Computes the transpose of a real matrix.
     * <p>
     * If {@code A} is {@code m x n}, the transpose {@code A^T} is {@code n x m}
     * and satisfies {@code A^T[i][j] = A[j][i]}.
     * </p>
     *
     * @param A
     *            an {@code m x n} real matrix
     * @return the transpose {@code A^T}
     * @throws IllegalArgumentException
     *             if {@code A} is null, empty, or ragged
     */
    public static double[][] transpose(double[][] A)
    {
        if (A == null || A.length == 0 || A[0] == null)
        {
            throw new IllegalArgumentException("Matrix is null or empty.");
        }

        int m = A.length;
        int n = A[0].length;

        // Ensure matrix is rectangular.
        for (int i = 0; i < m; i++)
        {
            if (A[i] == null || A[i].length != n)
            {
                throw new IllegalArgumentException(
                    "Matrix must be rectangular.");
            }
        }

        double[][] AT = new double[n][m];

        for (int i = 0; i < m; i++)
        {
            for (int j = 0; j < n; j++)
            {
                AT[j][i] = A[i][j];
            }
        }

        return AT;
    }


    // ----------------------------------------------------------
    /**
     * Returns {@code true} iff the given vectors are linearly independent.
     * <p>
     * If there are more vectors than the ambient dimension, they must be
     * linearly dependent. Otherwise, this method checks independence by
     * converting the vectors into a matrix (as rows) and comparing rank to the
     * number of vectors.
     * </p>
     *
     * @param list
     *            array of vectors (all must have the same dimension)
     * @return true if the vectors are linearly independent; false otherwise
     * @throws IllegalArgumentException
     *             if vectors are not all in the same dimension
     */
    public static boolean is_Linearly_Independent(Vector[] list)
    {
        int n = list.length;

        if (!checkDimmension(list))
        {
            throw new IllegalArgumentException(
                "Vectors must be in the same space.");
        }

        int d = list[0].size();

        // Pigeonhole principle: more vectors than dimension implies dependence.
        if (n > d)
        {
            return false;
        }

        // Square case: determinant test (fast conceptual shortcut).
        if (n == d)
        {
            double[][] matrix = new double[n][n];
            for (int r = 0; r < n; r++)
            {
                Vector v = list[r];
                for (int c = 0; c < n; c++)
                {
                    matrix[r][c] = v.get(c);
                }
            }
            return det_LU(matrix) != 0.0;
        }

        // Rectangular case: vectors are independent iff the row-rank is n.
        double[][] matrix = new double[n][d];
        for (int r = 0; r < n; r++)
        {
            Vector v = list[r];
            for (int c = 0; c < d; c++)
            {
                matrix[r][c] = v.get(c);
            }
        }

        return rank(matrix) == n;
    }


    // ----------------------------------------------------------
    /**
     * Returns {@code true} iff all vectors in {@code list} have the same
     * dimension.
     *
     * @param list
     *            vectors to validate
     * @return true if all vectors share the same dimension; false otherwise
     */
    public static boolean checkDimmension(Vector[] list)
    {
        int currentDim = list[0].size();
        for (Vector v : list)
        {
            if (v.size() != currentDim)
            {
                return false;
            }
        }
        return true;
    }


    // ----------------------------------------------------------
    /**
     * Computes the rank of a real matrix using Gaussian elimination with
     * partial pivoting.
     * <p>
     * The rank is the number of pivots found during elimination. This method
     * works for rectangular matrices and runs in {@code O(m n min(m, n))} time.
     * </p>
     *
     * @param matrix
     *            an {@code m x n} real matrix
     * @return the rank of {@code matrix}
     * @throws IllegalArgumentException
     *             if {@code matrix} is null, empty, or ragged
     */
    public static int rank(double[][] matrix)
    {
        if (matrix == null || matrix.length == 0 || matrix[0] == null)
        {
            throw new IllegalArgumentException("Matrix is null or empty.");
        }

        int n = matrix.length;
        int m = matrix[0].length;

        // Ensure rectangular matrix.
        for (int i = 0; i < n; i++)
        {
            if (matrix[i] == null || matrix[i].length != m)
            {
                throw new IllegalArgumentException(
                    "Matrix must be rectangular.");
            }
        }

        // Copy matrix so the original is not mutated.
        double[][] A = new double[n][m];
        for (int i = 0; i < n; i++)
        {
            System.arraycopy(matrix[i], 0, A[i], 0, m);
        }

        int rank = 0;
        int row = 0;

        // Iterate over columns and find pivots.
        for (int col = 0; col < m && row < n; col++)
        {
            // Choose pivot row (partial pivoting).
            int pivot = row;
            double maxAbs = Math.abs(A[row][col]);
            for (int r = row + 1; r < n; r++)
            {
                double valAbs = Math.abs(A[r][col]);
                if (valAbs > maxAbs)
                {
                    maxAbs = valAbs;
                    pivot = r;
                }
            }

            // No pivot in this column.
            if (maxAbs < EPS)
            {
                continue;
            }

            // Swap pivot row into place.
            if (pivot != row)
            {
                double[] tmp = A[row];
                A[row] = A[pivot];
                A[pivot] = tmp;
            }

            // Eliminate rows below pivot.
            double pivotVal = A[row][col];
            for (int r = row + 1; r < n; r++)
            {
                double factor = A[r][col] / pivotVal;
                if (Math.abs(factor) < EPS)
                {
                    continue;
                }
                for (int c = col; c < m; c++)
                {
                    A[r][c] -= factor * A[row][c];
                }
            }

            rank++;
            row++;
        }

        return rank;
    }


    // ----------------------------------------------------------
    /**
     * Returns the nullity of a matrix, defined as
     * {@code nullity(A) = n - rank(A)} for an {@code m x n} matrix.
     *
     * @param A
     *            an {@code m x n} matrix
     * @return the nullity of {@code A}
     * @throws IllegalArgumentException
     *             if {@code A} is null, empty, or ragged
     */
    public static int nullity(double[][] A)
    {
        // For an m x n matrix, nullity = n - rank(A).
        return A[0].length - rank(A);
    }


    // ----------------------------------------------------------
    /**
     * Computes the Q factor from the QR decomposition of {@code A} using the
     * Modified Gram–Schmidt algorithm.
     * <p>
     * This returns a thin {@code Q} of shape {@code m x n} (assuming {@code A}
     * is {@code m x n}). Columns of {@code Q} are orthonormal when {@code A}
     * has full column rank.
     * </p>
     *
     * @param A
     *            an {@code m x n} matrix (treated as a set of column vectors)
     * @return an {@code m x n} matrix {@code Q} with orthonormal columns
     * @throws IllegalArgumentException
     *             if {@code A} is null, empty, ragged, or numerically
     *             rank-deficient
     */
    public static double[][] computeQ(double[][] A)
    {
        if (A == null || A.length == 0 || A[0] == null || A[0].length == 0)
        {
            throw new IllegalArgumentException("Matrix is null or empty.");
        }

        int m = A.length;
        int n = A[0].length;

        // Ensure rectangular matrix.
        for (int r = 0; r < m; r++)
        {
            if (A[r] == null || A[r].length != n)
            {
                throw new IllegalArgumentException(
                    "Matrix must be rectangular.");
            }
        }

        Vector[] q = new Vector[n];
        Vector[] u = new Vector[n];

        for (int j = 0; j < n; j++)
        {
            u[j] = getColumn(A, j);

            // Remove components in directions q[0..j-1] (Modified
            // Gram–Schmidt).
            for (int i = 0; i < j; i++)
            {
                Vector proj = q[i].projection(u[j]);
                u[j] = u[j].subtract(proj);
            }

            double uNorm = u[j].norm();
            if (Double.isNaN(uNorm) || Math.abs(uNorm) < EPS)
            {
                throw new IllegalArgumentException(
                    "Columns are linearly dependent or numerically near-dependent.");
            }

            // Normalize to get q[j].
            Vector qj = new Vector();
            for (int k = 0; k < u[j].size(); k++)
            {
                qj.setEntry(u[j].get(k) / uNorm);
            }
            q[j] = qj;
        }

        // Build Q matrix column-by-column.
        double[][] Q = new double[m][n];
        for (int j = 0; j < n; j++)
        {
            for (int i = 0; i < m; i++)
            {
                Q[i][j] = q[j].get(i);
            }
        }

        return Q;
    }


    // ----------------------------------------------------------
    /**
     * Computes the R factor from the QR decomposition of {@code A}, given the
     * corresponding {@code Q}.
     * <p>
     * For a thin QR decomposition, {@code R = Q^T A} and is upper triangular
     * ({@code n x n}) when {@code A} is {@code m x n}.
     * </p>
     *
     * @param A
     *            the original {@code m x n} matrix
     * @param Q
     *            the {@code m x n} Q factor produced by
     *            {@link #computeQ(double[][])}
     * @return the {@code n x n} upper-triangular matrix {@code R}
     */
    public static double[][] computeR(double[][] A, double[][] Q)
    {
        int m = A.length;
        int n = A[0].length;

        double[][] R = new double[n][n];

        // Compute R = Q^T A (only upper triangle needed).
        for (int i = 0; i < n; i++)
        {
            for (int j = i; j < n; j++)
            {
                double sum = 0.0;
                for (int k = 0; k < m; k++)
                {
                    sum += Q[k][i] * A[k][j];
                }
                R[i][j] = sum;
            }
        }

        return R;
    }


    // ----------------------------------------------------------
    /**
     * Returns a column of {@code A} as a {@link Vector}.
     *
     * @param A
     *            an {@code m x n} matrix
     * @param col
     *            column index in {@code [0, n)}
     * @return the {@code col}-th column as a vector of length {@code m}
     */
    private static Vector getColumn(double[][] A, int col)
    {
        Vector v = new Vector();
        for (int i = 0; i < A.length; i++)
        {
            v.setEntry(A[i][col]);
        }
        return v;
    }


    // ----------------------------------------------------------
    /**
     * Prints a real matrix to standard output in row-major order.
     * <p>
     * Intended for quick debugging and small matrices.
     * </p>
     *
     * @param arr
     *            the matrix to print
     */
    public static void printMatrix(double[][] arr)
    {
        for (int i = 0; i < arr.length; i++)
        {
            for (int j = 0; j < arr[0].length; j++)
            {
                System.out.print(arr[i][j] + " ");
            }
            System.out.println();
        }
    }


    // ----------------------------------------------------------
    /**
     * Prints a boolean matrix to standard output in row-major order.
     * <p>
     * Intended for quick debugging and small matrices.
     * </p>
     *
     * @param arr
     *            the matrix to print
     */
    public static void printMatrix(boolean[][] arr)
    {
        for (int i = 0; i < arr.length; i++)
        {
            for (int j = 0; j < arr[0].length; j++)
            {
                System.out.print(arr[i][j] + " ");
            }
            System.out.println();
        }
    }
}
