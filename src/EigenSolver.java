import java.io.*;
import java.util.*;

/**
 * EigenSolver
 * <p>
 * Lightweight utility class for determinant computation and simple matrix I/O
 * diagnostics. This class currently includes:
 * </p>
 * <ul>
 * <li>Determinant via recursive Laplace expansion (educational; small matrices
 * only)</li>
 * <li>Determinant via LU decomposition with partial pivoting (recommended)</li>
 * <li>Console-based test harness (main method)</li>
 * </ul>
 * <p>
 * Note: The Laplace expansion implementation has factorial time complexity and
 * is not suitable for large matrices. The LU-based determinant runs in
 * {@code O(n^3)} time and is the preferred approach for most use cases.
 * </p>
 */
public class EigenSolver
{
    /**
     * Simple interactive console harness that prompts the user to enter an
     * {@code n x n} matrix and then computes its determinant using LU
     * decomposition.
     * <p>
     * This method is intended for quick manual testing and demonstration.
     * </p>
     *
     * @param args
     *            command-line arguments (unused)
     * @throws Exception
     *             if an I/O error occurs while reading input
     */
    public static void main(String[] args)
        throws Exception
    {
        // Interactive prompt for building an n×n matrix from user input.
        // This is simply a test harness to demonstrate the det(...) method.

        Scanner scanner = new Scanner(System.in);

        System.out.println("Size of Matrix?: ");
        int n = scanner.nextInt();

        double[][] Matrix = new double[n][n];

        // Populate matrix with user-provided entries.
        for (int r = 0; r < n; r++)
        {
            for (int c = 0; c < n; c++)
            {
                System.out.println("Entry at [" + r + "][" + c + "]: ");
                Matrix[r][c] = scanner.nextDouble();
            }
        }

        det_LU(Matrix);

        // Output determinant using recursive cofactor expansion.
        // System.out.println("Determinant: " + det_Lagrange_Expansion(Matrix));
    }


    // ----------------------------------------------------------
    /**
     * Computes the determinant of a square matrix using recursive Laplace
     * expansion along the first row.
     * <p>
     * Algorithm overview:
     * </p>
     * <ol>
     * <li>Validate the matrix is square and non-empty.</li>
     * <li>Handle the {@code 1 x 1} base case.</li>
     * <li>For {@code n > 1}, expand along row 0:
     * <ul>
     * <li>Construct each {@code (n-1) x (n-1)} minor by removing row 0 and
     * column {@code i}.</li>
     * <li>Apply alternating cofactor signs {@code (+ - + - ...)}.</li>
     * <li>Recursively compute each minor determinant.</li>
     * </ul>
     * </li>
     * </ol>
     * <p>
     * Time complexity is {@code O(n!)}; use only for small matrices.
     * </p>
     *
     * @param arr
     *            the input {@code n x n} matrix
     * @return the determinant of {@code arr}
     * @throws Exception
     *             if an error occurs during recursion
     * @throws IllegalArgumentException
     *             if {@code arr} is not square or is empty
     */
    public static int det_Lagrange_Expansion(int[][] arr)
        throws Exception
    {

        // Validate matrix shape (must be square and non-empty).
        if (arr.length != arr[0].length || arr.length == 0
            || arr[0].length == 0)
        {
            throw new IllegalArgumentException("MATRIX IS NOT SQUARE");
        }

        // Base Case: 1×1 determinant is the lone element.
        if (arr.length == 1)
        {
            return arr[0][0];
        }

        // Laplace expansion along the first row.
        int[] firstRow = arr[0];
        int det = 0;

        for (int i = 0; i < firstRow.length; i++)
        {
            // Build list of values for the (n−1)×(n−1) minor matrix
            // by skipping row 0 and skipping column i.
            Queue<Integer> minorList = new LinkedList<>();

            for (int r = 1; r < firstRow.length; r++)
            {
                for (int c = 0; c < firstRow.length; c++)
                {
                    if (c != i)  // exclude column being expanded
                    {
                        minorList.add(arr[r][c]);
                    }
                }
            }

            // Convert the flat minor list into a proper 2D matrix.
            int[][] minorMatrix = constructMinor(minorList);

            // Apply alternating sign pattern (+ - + - ...),
            // multiply by current cofactor, and accumulate.
            det += (i % 2 == 0
                ? firstRow[i] * det_Lagrange_Expansion(minorMatrix)
                : -1 * firstRow[i] * det_Lagrange_Expansion(minorMatrix));
        }

        return det;
    }


    /**
     * Constructs a square minor matrix from a queue of integers.
     * <p>
     * This helper is used by {@link #det_Lagrange_Expansion(int[][])} during
     * Laplace expansion. The queue is expected to contain exactly
     * {@code (n-1)^2} elements in row-major order, corresponding to the minor
     * formed by removing one row and one column from the original matrix.
     * </p>
     *
     * @param list
     *            a queue containing the minor's elements in row-major order
     * @return a reconstructed {@code (n-1) x (n-1)} minor matrix
     * @throws IllegalArgumentException
     *             if the queue size is not a perfect square
     */
    public static int[][] constructMinor(Queue<Integer> list)
    {
        // Determine the dimension of the minor (should be √size).
        int size = list.size();
        int dim = (int)Math.sqrt(size);

        // validate that size is a perfect square.
        if (dim * dim != size)
        {
            throw new IllegalArgumentException(
                "Minor construction failed: list size " + size
                    + " is not a perfect square");
        }

        int[][] minor = new int[dim][dim];

        // Fill the minor matrix row-by-row using FIFO ordering.
        for (int r = 0; r < dim; r++)
        {
            for (int c = 0; c < dim; c++)
            {
                minor[r][c] = list.poll();
            }
        }

        return minor;
    }


    /**
     * Computes the determinant of a square matrix using LU decomposition with
     * partial pivoting (Doolittle factorization).
     * <p>
     * This method performs an LU factorization with row swaps (partial
     * pivoting), then returns:
     * </p>
     * 
     * <pre>
     * det(A) = (-1)^(swapCount) * Π U[i][i]
     * </pre>
     * <p>
     * where {@code swapCount} is the number of row swaps performed and
     * {@code U[i][i]} are the diagonal entries of the upper-triangular matrix.
     * Runtime is {@code O(n^3)}.
     * </p>
     *
     * @param matrix
     *            the input {@code n x n} matrix
     * @return the determinant of {@code matrix}
     * @throws IllegalArgumentException
     *             if {@code matrix} is null, empty, ragged, or not square
     */
    public static double det_LU(double[][] matrix)
    {
        // Basic validation
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

        // Copy input so we don't mutate it
        double[][] a = new double[n][n];
        for (int i = 0; i < n; i++)
        {
            System.arraycopy(matrix[i], 0, a[i], 0, n);
        }

        // We'll store L and U in separate arrays for clarity
        double[][] L = new double[n][n];
        double[][] U = new double[n][n];

        // Initialize L's diagonal to 1
        for (int i = 0; i < n; i++)
        {
            L[i][i] = 1.0;
        }

        int swapCount = 0;
        final double EPS = 1e-12; // singularity threshold (tweak if you want)

        // Doolittle LU with partial pivoting:
        // for k = 0..n-1:
        // choose pivot row p >= k maximizing |a[p][k]|
        // swap rows in a and in L (columns < k), count swaps
        // compute U[k][j] for j>=k
        // compute L[i][k] for i>k
        for (int k = 0; k < n; k++)
        {
            // Pivot selection
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

            // If pivot is effectively zero -> determinant is 0
            if (maxAbs < EPS)
            {
                return 0.0;
            }

            // Row swap if needed
            if (pivot != k)
            {
                double[] tmp = a[k];
                a[k] = a[pivot];
                a[pivot] = tmp;

                // IMPORTANT: swap the already-built part of L (columns 0..k-1)
                for (int j = 0; j < k; j++)
                {
                    double t = L[k][j];
                    L[k][j] = L[pivot][j];
                    L[pivot][j] = t;
                }

                swapCount++;
            }

            // Build U row k: U[k][j] = a[k][j] - sum_{t<k} L[k][t]U[t][j]
            for (int j = k; j < n; j++)
            {
                double sum = 0.0;
                for (int t = 0; t < k; t++)
                {
                    sum += L[k][t] * U[t][j];
                }
                U[k][j] = a[k][j] - sum;
            }

            // Build L column k below diagonal:
            // L[i][k] = (a[i][k] - sum_{t<k} L[i][t]U[t][k]) / U[k][k]
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

        // determinant = (-1)^(swapCount) * product(diag(U))
        double det = (swapCount % 2 == 0) ? 1.0 : -1.0;
        for (int i = 0; i < n; i++)
        {
            det *= U[i][i];
        }
        return det;
    }


    /**
     * Prints a {@code double[][]} matrix to standard output in row-major order.
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


    /**
     * Prints a {@code boolean[][]} matrix to standard output in row-major
     * order.
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
