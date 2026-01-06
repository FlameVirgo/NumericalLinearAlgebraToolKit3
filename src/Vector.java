import java.util.ArrayList;
import java.util.List;

public class Vector
{
    private List<Double> vector;

    public Vector()
    {
        vector = new ArrayList<Double>();
    }


    public int size()
    {
        return vector.size();
    }


    public double get(int index)
    {
        if (index < 0 || index > this.size())
        {
            throw new IllegalArgumentException();
        }

        return vector.get(index);
    }


    public static double norm(Vector v)
    {
        return Math.sqrt(v.dotProduct(v));
    }


    public Vector add(Vector v)
    {
        if (v == null)
        {
            return null;
        }

        if (v.size() != this.size())
        {
            throw new IllegalArgumentException();
        }

        Vector sum = new Vector();
        for (int i = 0; i < v.size(); i++)
        {
            sum.setEntry(this.get(i) + v.get(i));
        }

        return sum;

    }


    public void setEntry(double entry)
    {
        vector.add(entry);
    }


    public Vector subtract(Vector v)
    {
        if (v == null)
        {
            return null;
        }

        if (v.size() != this.size())
        {
            throw new IllegalArgumentException();
        }

        Vector difference = new Vector();
        for (int i = 0; i < v.size(); i++)
        {
            difference.setEntry(this.get(i) - v.get(i));
        }

        return difference;

    }


    public double dotProduct(Vector v)
    {
        if (v == null)
        {
            throw new IllegalArgumentException();
        }

        if (v.size() != this.size())
        {
            throw new IllegalArgumentException();
        }

        double dotProduct = 0;
        for (int i = 0; i < this.size(); i++)
        {
            dotProduct += (v.get(i) * this.get(i));
        }

        return dotProduct;

    }


    public Vector projection(Vector b)
    {
        double dotProductCoefficient =
            this.dotProduct(b) / this.dotProduct(this);

        Vector projection = new Vector();
        for (int i = 0; i < this.size(); i++)
        {
            projection.setEntry(dotProductCoefficient * this.get(i));
        }

        return projection;
    }

}
