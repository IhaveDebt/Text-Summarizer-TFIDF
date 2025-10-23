// TextSummarizerTFIDF.java
// Simple extractive summarizer based on TF-IDF and cosine similarity
// Compile & run: javac TextSummarizerTFIDF.java && java TextSummarizerTFIDF

import java.util.*;
import java.util.stream.*;

/**
 * TextSummarizerTFIDF
 * - Splits document into sentences, builds TF-IDF vectors, ranks sentences by centrality
 * - Simple and dependency-free
 */
public class TextSummarizerTFIDF {
    static List<String> splitSentences(String text) {
        return Arrays.stream(text.split("(?<=[.!?])\\s+"))
                .map(String::trim).filter(s->!s.isEmpty()).collect(Collectors.toList());
    }

    static List<String> tokenize(String s) {
        return Arrays.stream(s.toLowerCase().replaceAll("[^a-z0-9\\s]", " ").split("\\s+"))
                .filter(tok->!tok.isEmpty()).collect(Collectors.toList());
    }

    static Map<String, Integer> tf(List<String> tokens) {
        Map<String, Integer> m = new HashMap<>();
        for (String t : tokens) m.put(t, m.getOrDefault(t,0)+1);
        return m;
    }

    static Map<String, Double> idf(List<List<String>> docs) {
        Map<String, Integer> df = new HashMap<>();
        for (List<String> doc : docs) {
            Set<String> uniq = new HashSet<>(doc);
            for (String w : uniq) df.put(w, df.getOrDefault(w,0)+1);
        }
        int N = docs.size();
        Map<String, Double> idf = new HashMap<>();
        for (Map.Entry<String,Integer> e : df.entrySet()) {
            idf.put(e.getKey(), Math.log((double)N / (1 + e.getValue())));
        }
        return idf;
    }

    static Map<String, Double> tfidfVector(Map<String,Integer> tf, Map<String,Double> idf) {
        Map<String, Double> v = new HashMap<>();
        for (Map.Entry<String,Integer> e : tf.entrySet()) {
            v.put(e.getKey(), e.getValue() * idf.getOrDefault(e.getKey(), Math.log(1)));
        }
        return v;
    }

    static double cosine(Map<String,Double> a, Map<String,Double> b) {
        double num = 0;
        for (String k : a.keySet()) num += a.get(k) * b.getOrDefault(k, 0.0);
        double na = Math.sqrt(a.values().stream().mapToDouble(x->x*x).sum());
        double nb = Math.sqrt(b.values().stream().mapToDouble(x->x*x).sum());
        if (na==0 || nb==0) return 0;
        return num / (na*nb);
    }

    public static String summarize(String text, int sentencesToReturn) {
        List<String> sents = splitSentences(text);
        if (sents.size() <= sentencesToReturn) return text;
        List<List<String>> tokenized = sents.stream().map(TextSummarizerTFIDF::tokenize).collect(Collectors.toList());
        Map<String, Double> idf = idf(tokenized);
        List<Map<String, Double>> vectors = new ArrayList<>();
        for (List<String> toks : tokenized) {
            vectors.add(tfidfVector(tf(toks), idf));
        }
        // compute sentence centrality: average cosine to others
        double[] scores = new double[vectors.size()];
        for (int i = 0; i < vectors.size(); i++) {
            double s = 0;
            for (int j = 0; j < vectors.size(); j++) {
                if (i==j) continue;
                s += cosine(vectors.get(i), vectors.get(j));
            }
            scores[i] = s / (vectors.size() - 1);
        }
        // pick top-k sentences by score in original order
        Integer[] idx = IntStream.range(0, scores.length).boxed().toArray(Integer[]::new);
        Arrays.sort(idx, (a,b) -> Double.compare(scores[b], scores[a]));
        Set<Integer> chosen = Arrays.stream(idx).limit(sentencesToReturn).collect(Collectors.toCollection(LinkedHashSet::new));
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < sents.size(); i++) if (chosen.contains(i)) sb.append(sents.get(i)).append(" ");
        return sb.toString().trim();
    }

    public static void main(String[] args) {
        String doc = "Artificial intelligence (AI) refers to the simulation of human intelligence in machines that are programmed to think like humans and mimic their actions. " +
                "AI systems can perform tasks such as learning, reasoning, problem-solving, perception, and language understanding. " +
                "In recent years, AI has seen rapid advancements due to improvements in computational power and the availability of large datasets. " +
                "This progress has enabled the development of practical applications across many industries including healthcare, finance, and transportation. " +
                "However, AI also raises ethical and societal concerns that must be addressed responsibly.";
        System.out.println("=== Original ===\n" + doc + "\n");
        System.out.println("=== Summary ===\n" + summarize(doc, 2));
    }
}
