import java.util.ArrayList;
import java.util.List;

/**
 * A simple dense vector of {@code double} values.
 * <p>
 * This class supports a small set of vector operations commonly used in
 * introductory numerical linear algebra: element access, addition/subtraction,
 * dot products, norms, and orthogonal projections.
 * </p>
 * <h2>Design notes</h2>
 * <ul>
 * <li>This implementation is <strong>mutable</strong>.</li>
 * <li>Entries are stored in a {@link List} and the vector grows only via
 * {@link #setEntry(double)}, which <em>appends</em> (it does not overwrite an
 * existing index).</li>
 * <li>The class is intended for educational and lightweight use; it is not a
 * high-performance numeric container.</li>
 * </ul>
 * <h2>Indexing</h2>
 * <p>
 * Indices are 0-based. Valid indices are {@code 0..size()-1}.
 * </p>
 */
public class Vector
{
    /** Underlying storage for vector entries in index order. */
    private final List<Double> vector;

    // ----------------------------------------------------------
    /**
     * Constructs an empty vector (dimension 0).
     */
    public Vector()
    {
        vector = new ArrayList<Double>();
    }


    // ----------------------------------------------------------
    /**
     * Returns the dimension (number of entries) of this vector.
     *
     * @return the size (dimension) of the vector
     */
    public int size()
    {
        return vector.size();
    }


    // ----------------------------------------------------------
    /**
     * Returns the entry at the specified index.
     *
     * @param index
     *            the entry index (0-based)
     * @return the value at {@code index}
     * @throws IllegalArgumentException
     *             if {@code index} is out of bounds
     */
    public double get(int index)
    {
        if (index < 0 || index >= this.size())
        {
            throw new IllegalArgumentException("Index out of bounds: " + index);
        }

        return vector.get(index);
    }


    // ----------------------------------------------------------
    /**
     * Appends an entry to the end of this vector.
     * <p>
     * This method grows the vector by one element. It does not replace an
     * existing element at an index.
     * </p>
     *
     * @param entry
     *            the value to append
     */
    public void setEntry(double entry)
    {
        vector.add(entry);
    }


    // ----------------------------------------------------------
    /**
     * Returns the vector sum {@code this + v}.
     *
     * @param v
     *            the vector to add
     * @return a new vector equal to {@code this + v}, or {@code null} if
     *             {@code v} is {@code null}
     * @throws IllegalArgumentException
     *             if {@code v} has a different dimension than this vector
     */
    public Vector add(Vector v)
    {
        if (v == null)
        {
            return null;
        }

        if (v.size() != this.size())
        {
            throw new IllegalArgumentException(
                "Vector dimensions must match: " + this.size() + " vs "
                    + v.size());
        }

        Vector sum = new Vector();
        for (int i = 0; i < this.size(); i++)
        {
            sum.setEntry(this.get(i) + v.get(i));
        }

        return sum;
    }


    // ----------------------------------------------------------
    /**
     * Returns the vector difference {@code this - v}.
     *
     * @param v
     *            the vector to subtract
     * @return a new vector equal to {@code this - v}, or {@code null} if
     *             {@code v} is {@code null}
     * @throws IllegalArgumentException
     *             if {@code v} has a different dimension than this vector
     */
    public Vector subtract(Vector v)
    {
        if (v == null)
        {
            return null;
        }

        if (v.size() != this.size())
        {
            throw new IllegalArgumentException(
                "Vector dimensions must match: " + this.size() + " vs "
                    + v.size());
        }

        Vector difference = new Vector();
        for (int i = 0; i < this.size(); i++)
        {
            difference.setEntry(this.get(i) - v.get(i));
        }

        return difference;
    }


    // ----------------------------------------------------------
    /**
     * Computes the dot product {@code this · v}.
     *
     * @param v
     *            the other vector
     * @return the dot product value
     * @throws IllegalArgumentException
     *             if {@code v} is {@code null} or has a different dimension
     */
    public double dotProduct(Vector v)
    {
        if (v == null)
        {
            throw new IllegalArgumentException("Vector must not be null.");
        }

        if (v.size() != this.size())
        {
            throw new IllegalArgumentException(
                "Vector dimensions must match: " + this.size() + " vs "
                    + v.size());
        }

        double dot = 0.0;
        for (int i = 0; i < this.size(); i++)
        {
            dot += this.get(i) * v.get(i);
        }

        return dot;
    }


    // ----------------------------------------------------------
    /**
     * Computes the Euclidean (L2) norm of this vector.
     * <p>
     * This returns {@code sqrt(this · this)}.
     * </p>
     *
     * @return the Euclidean norm
     */
    public double norm()
    {
        return Math.sqrt(dotProduct(this));
    }


    // ----------------------------------------------------------
    /**
     * Computes the orthogonal projection of {@code b} onto this vector.
     * <p>
     * The result is:
     * </p>
     *
     * <pre>
     * proj_this(b) = ((this · b) / (this · this)) * this
     * </pre>
     *
     * @param b
     *            the vector to project onto this vector
     * @return the projection of {@code b} onto this vector
     * @throws IllegalArgumentException
     *             if {@code b} is {@code null}, has a different dimension, or
     *             if this vector is the zero vector (projection undefined)
     */
    public Vector projection(Vector b)
    {
        if (b == null)
        {
            throw new IllegalArgumentException(
                "Vector to project must not be null.");
        }

        if (b.size() != this.size())
        {
            throw new IllegalArgumentException(
                "Vector dimensions must match: " + this.size() + " vs "
                    + b.size());
        }

        double denom = this.dotProduct(this);
        if (denom == 0.0)
        {
            throw new IllegalArgumentException(
                "Cannot project onto the zero vector.");
        }

        double coeff = this.dotProduct(b) / denom;

        Vector proj = new Vector();
        for (int i = 0; i < this.size(); i++)
        {
            proj.setEntry(coeff * this.get(i));
        }

        return proj;
    }
}
