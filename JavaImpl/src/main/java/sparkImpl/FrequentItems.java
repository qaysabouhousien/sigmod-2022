package sparkImpl;

import java.util.List;
import java.util.Set;

public class FrequentItems {
    int id;
    List<List<String>> sets;
    public FrequentItems(){}
    public FrequentItems(int id, List<List<String>> sets){
        this.id = id;
        this.sets = sets;
    }
    public List<List<String>> getSets() {
        return sets;
    }

    public void setSets(List<List<String>> sets) {
        this.sets = sets;
    }

    public int getId() {
        return id;
    }

    public void setId(int id) {
        this.id = id;
    }

    @Override
    public String toString() {
        return "FrequentItems{" +
                "id=" + id +
                ", sets=" + sets +
                '}';
    }
}
