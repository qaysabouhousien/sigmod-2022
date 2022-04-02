package sparkImpl;

import java.util.Objects;

public class Tuple<A,B> {
    A a;
    B b;
    public Tuple(A a, B b){
        this.a = a;
        this.b = b;
    }
    public static <A,B> Tuple<A,B> create(A a, B b){
        return new Tuple<>(a,b);
    }
    public void setA(A a) {
        this.a = a;
    }

    public A getA() {
        return a;
    }

    public void setB(B b) {
        this.b = b;
    }

    public B getB() {
        return b;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        Tuple<?, ?> tuple = (Tuple<?, ?>) o;
        return a.equals(tuple.a) && b.equals(tuple.b);
    }

    @Override
    public int hashCode() {
        return Objects.hash(a, b);
    }

    @Override
    public String toString() {
        return "Tuple{" +
                "a=" + a +
                ", b=" + b +
                '}';
    }
}
