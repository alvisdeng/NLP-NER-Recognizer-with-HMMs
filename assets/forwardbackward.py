import argparse
import math

class HMMPredictor():
    def __init__(self):
        self.words_lines = []
        self.tags_lines = []
        self.preditions = []
        self.accuracy = 0
        self.avg_log_likelihood = 0

        self.idx_to_tag = {}
        self.idx_to_word = {}

        self.prior_table = {}
        self.trans_table = {}
        self.emit_table = {}

    def fit(self,input_file,idx_to_word_file,idx_to_tag_file):
        with open(input_file,mode="r",encoding="utf8") as f:
            for line in f:
                temp_words = []
                temp_tags = []
                splitted_line = line.strip().split(" ")
                for token in splitted_line:
                    word = token.split("_")[0]
                    tag = token.split("_")[1]
                    temp_words.append(word)
                    temp_tags.append(tag)
                self.words_lines.append(temp_words)
                self.tags_lines.append(temp_tags)
        
        with open(idx_to_word_file,mode="r",encoding="utf8") as f:
            for idx,word in enumerate(f):
                self.idx_to_word[idx] = word.strip()
        
        with open(idx_to_tag_file,mode='r',encoding="utf8") as f:
            for idx,tag in enumerate(f):
                self.idx_to_tag[idx] = tag.strip()

    def load_model(self,hmmprior_file,hmmemit_file,hmmtrans_file):
        with open (hmmprior_file,mode='r') as f:
            for idx,value in enumerate(f):
                tag = self.idx_to_tag[idx]
                self.prior_table[tag] = float(value.strip())

        with open(hmmemit_file,mode="r") as f:
            for idx,line in enumerate(f):
                tag = self.idx_to_tag[idx]
                self.emit_table[tag] = {}

                splitted_line = line.strip().split(" ")
                for j, value in enumerate(splitted_line):
                    word = self.idx_to_word[j]
                    self.emit_table[tag][word] = float(value.strip())
        
        with open(hmmtrans_file,mode="r") as f:
            for idx,line in enumerate(f):
                prev_tag = self.idx_to_tag[idx]
                self.trans_table[prev_tag] = {}

                splitted_line = line.strip().split(" ")
                for j, value in enumerate(splitted_line):
                    next_tag = self.idx_to_tag[j]
                    self.trans_table[prev_tag][next_tag] = float(value.strip())

    def log_sum_exp_tric(self,V):
        m = max(V)
        temp = 0
        for v in V:
            temp += math.exp(v-m)
        return m+math.log(temp)

    def forward_log_alpha(self,observations):
        T = len(observations)
        alpha = {}
        if T == 1:
            alpha[T] = {}
            for state in self.prior_table.keys():
                alpha[T][state] = math.log(self.prior_table[state]) + math.log(self.emit_table[state][observations[T-1]])
        else:
            alpha[1] = {}
            for state in self.prior_table.keys():
                alpha[1][state] = math.log(self.prior_table[state]) + math.log(self.emit_table[state][observations[0]])
            for idx in range(2,T+1):
                alpha[idx] = {}
                for state in self.prior_table.keys():
                    alpha[idx][state] = math.log(self.emit_table[state][observations[idx-1]])

                    V = []
                    for prev_state in self.prior_table.keys():
                        V.append(alpha[idx-1][prev_state]+math.log(self.trans_table[prev_state][state]))
                    alpha[idx][state] += self.log_sum_exp_tric(V)
        return alpha
    
    def backward_log_beta(self,observations):
        T = len(observations)
        beta = {}
        beta[T] = {}
        for state in self.prior_table.keys():
            beta[T][state] = 0
        
        for idx in range(1,T)[::-1]:
            beta[idx] = {}
            for state in self.prior_table.keys():
                V = []
                for next_state in self.prior_table.keys():
                    V.append(math.log(self.emit_table[next_state][observations[idx]])+beta[idx+1][next_state]+math.log(self.trans_table[state][next_state]))
                beta[idx][state] = self.log_sum_exp_tric(V)
        return beta
    
    def predict(self):
        total = 0
        wrong = 0
        for idx in range(len(self.words_lines)):
            observations = self.words_lines[idx]
            tags_line = self.tags_lines[idx]

            T = len(observations)
            alpha = self.forward_log_alpha(observations)
            beta = self.backward_log_beta(observations)

            self.avg_log_likelihood += self.compute_log_likelihood(alpha,T)

            temp = []
            for t in range(1,T+1):
                total += 1
                sums = []
                for idx in self.idx_to_tag.keys():
                    tag = self.idx_to_tag[idx]
                    sums.append(alpha[t][tag]+beta[t][tag])
                max_idx = sums.index(max(sums))

                pred = self.idx_to_tag[max_idx]
                tag = tags_line[t-1]

                if pred != tag:
                    wrong += 1
                temp.append(pred)
            self.preditions.append(temp)
        self.accuracy = 1-wrong/total
        self.avg_log_likelihood /= len(self.words_lines)
    
    def compute_log_likelihood(self,alpha,T):
        V = []
        for idx in self.idx_to_tag.keys():
            tag = self.idx_to_tag[idx]
            V.append(alpha[T][tag])
        return self.log_sum_exp_tric(V)

def parse_args():
    parser = argparse.ArgumentParser("Evaluating Validation Data with Forward-Backward Algorithm")
    parser.add_argument("validation_input",help="path to the validation input .txt ﬁle that will be evaluated by your forward backward algorithm")
    parser.add_argument("index_to_word",help="path to the .txt that speciﬁes the dictionary mapping from words to indices.")
    parser.add_argument("index_to_tag",help="path to the .txt that speciﬁes the dictionary mapping from tags to indices.")
    parser.add_argument("hmmprior",help="path to input .txt ﬁle which contains the estimated prior (π).")
    parser.add_argument("hmmemit",help="path to input .txt ﬁle which contains the emission probabilities (B).")
    parser.add_argument("hmmtrans",help="path to input .txt ﬁle which contains transition probabilities (A).")
    parser.add_argument("predicted_file",help="path to the output .txt ﬁle to which the predicted tags will be written.")
    parser.add_argument("metric_file",help="path to the output .txt ﬁle to which the metrics will be written.")

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    predictor = HMMPredictor()
    predictor.fit(args.validation_input,args.index_to_word,args.index_to_tag)
    predictor.load_model(args.hmmprior,args.hmmemit,args.hmmtrans)
    predictor.predict()

    accuracy = predictor.accuracy
    avg_log_likelihood = predictor.avg_log_likelihood
    
    with open(args.metric_file,mode="w") as f:
        f.write("Average Log-Likelihood: " + str(avg_log_likelihood))
        f.write("\n")
        f.write("Accuracy: " + str(accuracy))
    
    words_lines = predictor.words_lines
    predictions = predictor.preditions

    with open(args.predicted_file,mode="w") as f:
        for i in range(len(words_lines)):
            words_line = words_lines[i]
            prediction = predictions[i]

            out = [words_line[i]+"_"+prediction[i] for i in range(len(words_line))]
            out = " ".join(out)
            f.write(out)
            f.write("\n")


