import java.io.*;
import java.util.*;

/**
 * EigenSolver A lightweight utility class demonstrating recursive determinant
 * computation via Laplace expansion along the first row. This implementation is
 * intended for educational / small-matrix use (n ≤ ~8) due to factorial
 * complexity.
 */
public class EigenSolver
{
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
     * expansion along the first row. This algorithm: 1. Validates that the
     * matrix is square. 2. Handles the 1×1 base case. 3. For n > 1: - Iterates
     * across the first row (cofactors). - Builds each (n-1)×(n-1) minor matrix
     * by removing row 0 and column i. - Applies the alternating sign rule (+ -
     * + - ...). - Recursively computes determinant of each minor. Time
     * Complexity: O(n!) — appropriate for small matrices only.
     *
     * @param arr
     *            the input n×n matrix
     * @return determinant of arr
     * @throws IllegalArgumentException
     *             if matrix is not square
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
     * Constructs a square (n−1)×(n−1) minor matrix from a queue of integers.
     * <p>
     * This helper is used by {@link #det(int[][])} during Laplace expansion.
     * The queue is expected to contain exactly (n−1)² elements in row-major
     * order, corresponding to the minor formed by removing one row and one
     * column from the original matrix.
     * </p>
     *
     * @param list
     *            a queue containing the minor's elements in row-major order
     * @return a reconstructed (n−1)×(n−1) minor matrix
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


    public static boolean is_LI(Vector[] list)
    {

    }


    public static boolean is_Square(Vector[] list)
    {
        int n = list.length;

        for (int i = 0; i < n; i++)
        {
            if (n != list[i].size())
            {
                return false;
            }
        }

        return true;
    }


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
