import ner.StanfordNerService;
import org.apache.commons.lang3.time.StopWatch;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;

public class Main {

    public static String text = "Joe Smith was born in California. " +
            "In 2017, he went to Paris, France in the summer. " +
            "His flight left at 3:00pm on July 10th, 2017. " +
            "After eating some escargot for the first time, Joe said, \"That was delicious!\" " +
            "He sent a postcard to his sister Jane Smith. " +
            "After hearing about Joe's trip, Jane decided she might go to France one day.";

    public static void main(String[] args) throws IOException, ClassNotFoundException {
        StopWatch w = new StopWatch();
        w.start();
        System.out.println(w);
        StanfordNerService s = new StanfordNerService();
        w.reset();
        w.start();
        var res = s.apply("BEST ASPIRE CUBE INTEL 6885 INTEL FRAME QUAD I5 1TB AND - SWITCHING 500GB 2.90");
        System.out.println(res);
        w.stop();
        System.out.println(w);
    }
}
