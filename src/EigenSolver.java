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

        if (matrix.length != matrix[0].length)
        {
            throw new IllegalArgumentException();
        }

        int n = matrix.length;

        double[][] LT = new double[n][n]; // Lower Triangle Matrix (L)
        double[][] UT = new double[n][n]; // Upper Triangle Matrix (U)

        boolean[][] LT_valueUpdated = new boolean[n][n];
        boolean[][] UT_valueUpdated = new boolean[n][n];

        for (int r = 0; r < n; r++)
        {
            for (int c = 0; c < n; c++)
            {
                if (r <= c)
                {
                    LT_valueUpdated[r][c] = true;
                }
                else
                {
                    UT_valueUpdated[r][c] = true;
                }
            }
        }

        // Fill first row with same entries as matrix
        for (int c = 0; c < n; c++)
        {
            UT[0][c] = matrix[0][c];
            UT_valueUpdated[0][c] = true;
        }

        // Diagonal entires are 1;
        for (int i = 0; i < n; i++)
        {
            LT[i][i] = 1;
            LT_valueUpdated[i][i] = true;
        }

        // pivoting
        for (int p = 1; p < matrix[0].length; p++)
        {
            LT[p][0] = matrix[p][0] / matrix[0][0];

        }

        int pivotRow = 0;  // first pivot is row 0

        for (int r = 1; r < n; r++)
        {
            double rowReduceConstant = LT[r][0];  // L[r,0] multiplier

            for (int c = 0; c < n; c++)
            {
                // use the fixed pivot row
                UT[r][c] =
                    matrix[r][c] - rowReduceConstant * matrix[pivotRow][c];
                UT_valueUpdated[r][c] = true;
            }
        }

        printMatrix(LT);
        System.out.println("WIP");
        return 0;
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
