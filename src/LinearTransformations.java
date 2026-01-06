/**
 * Represents a linear transformation {@code T(x) = A x}, where {@code A} is an
 * {@code m x n} real matrix.
 * <p>
 * This class is a lightweight wrapper around a matrix that exposes common
 * linear-map properties such as rank, nullity, injectivity, surjectivity, and
 * bijectivity. The underlying matrix is stored defensively (deep-copied) to
 * prevent external mutation.
 * </p>
 * <h2>Conventions</h2>
 * <ul>
 * <li>If {@code A} is {@code m x n}, then the transformation maps
 * {@code R^n -> R^m}.</li>
 * <li>Domain dimension is {@code n} (number of columns).</li>
 * <li>Codomain dimension is {@code m} (number of rows).</li>
 * </ul>
 * <h2>Key facts used</h2>
 * <ul>
 * <li>Injective iff {@code rank(A) = n} (requires {@code n <= m}).</li>
 * <li>Surjective iff {@code rank(A) = m} (requires {@code n >= m}).</li>
 * <li>Bijective iff {@code A} is square and full-rank.</li>
 * </ul>
 */
public class LinearTransformations
{
    /** Backing matrix for the linear map {@code T(x) = A x}. */
    private final double[][] A;

    // ----------------------------------------------------------
    /**
     * Constructs a linear transformation from the given matrix.
     *
     * @param A
     *            an {@code m x n} matrix defining {@code T(x) = A x}
     * @throws IllegalArgumentException
     *             if {@code A} is null, empty, or ragged
     */
    public LinearTransformations(double[][] A)
    {
        validateRectangular(A);
        this.A = deepCopy(A);
    }


    // ----------------------------------------------------------
    /**
     * Returns a defensive (deep) copy of the underlying matrix {@code A}.
     * <p>
     * This ensures callers cannot mutate the internal representation of this
     * transformation.
     * </p>
     *
     * @return a deep copy of {@code A}
     */
    public double[][] getMatrix()
    {
        return deepCopy(A);
    }


    // ----------------------------------------------------------
    /**
     * Returns the dimension of the domain of the transformation.
     * <p>
     * If {@code A} is {@code m x n}, then the domain is {@code R^n}.
     * </p>
     *
     * @return {@code n}, the number of columns of {@code A}
     */
    public int domainDimension()
    {
        return A[0].length;
    }


    // ----------------------------------------------------------
    /**
     * Returns the dimension of the codomain of the transformation.
     * <p>
     * If {@code A} is {@code m x n}, then the codomain is {@code R^m}.
     * </p>
     *
     * @return {@code m}, the number of rows of {@code A}
     */
    public int codomainDimension()
    {
        return A.length;
    }


    // ----------------------------------------------------------
    /**
     * Returns the rank of the underlying matrix {@code A}.
     *
     * @return {@code rank(A)}
     */
    public int rank()
    {
        return Numerical_Linear_Algebra_Operations.rank(A);
    }


    // ----------------------------------------------------------
    /**
     * Returns the nullity of the underlying matrix {@code A}.
     * <p>
     * For an {@code m x n} matrix, nullity is defined as:
     * </p>
     *
     * <pre>
     * nullity(A) = n - rank(A)
     * </pre>
     *
     * @return {@code nullity(A)}
     */
    public int nullity()
    {
        return domainDimension() - rank();
    }


    // ----------------------------------------------------------
    /**
     * Returns {@code true} if this linear transformation is injective
     * (one-to-one).
     * <p>
     * For {@code A} of size {@code m x n} representing a map
     * {@code R^n -> R^m}, injectivity holds iff {@code rank(A) = n}. A
     * necessary dimension condition is {@code n <= m}.
     * </p>
     *
     * @return true if injective; false otherwise
     */
    public boolean isInjective()
    {
        int m = codomainDimension();
        int n = domainDimension();

        // Necessary condition: cannot inject a higher-dimensional space into a
        // lower-dimensional one.
        if (n > m)
        {
            return false;
        }

        return rank() == n;
    }


    // ----------------------------------------------------------
    /**
     * Returns {@code true} if this linear transformation is surjective (onto).
     * <p>
     * For {@code A} of size {@code m x n} representing a map
     * {@code R^n -> R^m}, surjectivity holds iff {@code rank(A) = m}. A
     * necessary dimension condition is {@code n >= m}.
     * </p>
     *
     * @return true if surjective; false otherwise
     */
    public boolean isSurjective()
    {
        int m = codomainDimension();
        int n = domainDimension();

        // Necessary condition: cannot cover a higher-dimensional codomain from
        // a
        // lower-dimensional domain.
        if (n < m)
        {
            return false;
        }

        return rank() == m;
    }


    // ----------------------------------------------------------
    /**
     * Returns {@code true} if this linear transformation is bijective (both
     * injective and surjective).
     * <p>
     * A linear map is bijective iff it is both injective and surjective, which
     * for matrices means {@code A} must be square and full rank.
     * </p>
     *
     * @return true if bijective; false otherwise
     */
    public boolean isBijective()
    {
        // Fast necessary condition: bijective linear maps must be square.
        if (codomainDimension() != domainDimension())
        {
            return false;
        }

        // Use logical AND (short-circuit) rather than bitwise &.
        return isSurjective() && isInjective();
    }


    // ----------------------------------------------------------
    /**
     * Validates that a matrix is non-null, non-empty, and rectangular.
     *
     * @param M
     *            matrix to validate
     * @throws IllegalArgumentException
     *             if {@code M} is null, empty, or ragged
     */
    private static void validateRectangular(double[][] M)
    {
        if (M == null || M.length == 0 || M[0] == null || M[0].length == 0)
        {
            throw new IllegalArgumentException("Matrix is null or empty.");
        }

        int cols = M[0].length;
        for (int r = 0; r < M.length; r++)
        {
            if (M[r] == null || M[r].length != cols)
            {
                throw new IllegalArgumentException(
                    "Matrix must be rectangular.");
            }
        }
    }


    // ----------------------------------------------------------
    /**
     * Creates a deep copy of a rectangular matrix.
     *
     * @param M
     *            matrix to copy (assumed rectangular)
     * @return deep copy of {@code M}
     */
    private static double[][] deepCopy(double[][] M)
    {
        double[][] copy = new double[M.length][M[0].length];
        for (int r = 0; r < M.length; r++)
        {
            System.arraycopy(M[r], 0, copy[r], 0, M[0].length);
        }
        return copy;
    }
}
