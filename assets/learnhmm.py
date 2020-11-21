import argparse
from collections import Counter
from collections import defaultdict

class HMMLearner():
    def __init__(self):
        self.words_lines = []
        self.tags_lines = []
        self.idx_to_tag = {}
        self.idx_to_word = {}
    
    def fit(self,train_file,idx_to_word_file,idx_to_tag_file):
        with open(train_file,mode="r",encoding="utf8") as f:
            for idx,line in enumerate(f):
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

                if idx == 9999:
                    break
        
        with open(idx_to_tag_file,mode='r',encoding="utf8") as f:
            for idx,tag in enumerate(f):
                self.idx_to_tag[idx] = tag.strip()
        
        with open(idx_to_word_file,mode="r",encoding="utf8") as f:
            for idx,word in enumerate(f):
                self.idx_to_word[idx] = word.strip()
    
    def learn(self):
        prior_counter = Counter()
        trans_counter = defaultdict(lambda: Counter())
        emit_counter = defaultdict(lambda: Counter())

        for i in range(len(self.words_lines)):
            words_line = self.words_lines[i]
            tags_line = self.tags_lines[i]

            prior_counter[tags_line[0]] += 1
            for j in range(len(words_line)):
                word = words_line[j]
                tag = tags_line[j]
                emit_counter[tag][word] += 1

                if j < len(words_line)-1:
                    next_tag = tags_line[j+1]
                    trans_counter[tag][next_tag] += 1

        prior_table = []
        trans_table = []
        emit_table = []

        for i in self.idx_to_tag.keys():
            numerator = prior_counter[self.idx_to_tag[i]] + 1
            denominator = 0
            for j in self.idx_to_tag.keys():
                denominator += prior_counter[self.idx_to_tag[j]] + 1
            prior_table.append(numerator/denominator)

        for i in self.idx_to_tag.keys():
            tag = self.idx_to_tag[i]
            denominator = 0
            temp = []
            for j in self.idx_to_tag.keys():
                next_tag = self.idx_to_tag[j]
                denominator += trans_counter[tag][next_tag] + 1

            for k in self.idx_to_tag.keys():
                next_tag = self.idx_to_tag[k]
                numerator = trans_counter[tag][next_tag] + 1
                temp.append(numerator/denominator)
            
            trans_table.append(temp)
        
        for i in self.idx_to_tag.keys():
            tag = self.idx_to_tag[i]
            denominator = 0
            temp = []
            for j in self.idx_to_word.keys():
                word = self.idx_to_word[j]
                denominator += emit_counter[tag][word] + 1
            
            for k in self.idx_to_word.keys():
                word = self.idx_to_word[k]
                numerator = emit_counter[tag][word] + 1
                temp.append(numerator/denominator)
            
            emit_table.append(temp)

        return prior_table,trans_table,emit_table


def parse_args():
    parser = argparse.ArgumentParser("Learn HMM Parameters with Supervised Method")
    parser.add_argument("train_input",help="path to the training input .txt ﬁle")
    parser.add_argument("index_to_word",help="path to the .txt that speciﬁes the dictionary mapping from words to indices.")
    parser.add_argument("index_to_tag",help="path to the .txt that speciﬁes the dictionary mapping from tags to indices.")
    parser.add_argument("hmmprior",help="path to output .txt ﬁle to which the estimated prior (π) will be written.")
    parser.add_argument("hmmemit",help="path to output .txt ﬁle to which the emission probabilities (B) will be written.")
    parser.add_argument("hmmtrans",help="path to output .txt ﬁle to which the transition probabilities (A) will be written.")

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    hmm_learner = HMMLearner()
    hmm_learner.fit(args.train_input,args.index_to_word,args.index_to_tag)
    prior_table, trans_table, emit_table = hmm_learner.learn()

    with open(args.hmmprior,mode='w') as f:
        for value in prior_table:
            f.write(str(value))
            f.write('\n')
        f.truncate()

    with open(args.hmmemit,mode='w') as f:
        for values in emit_table:
            values = [str(i) for i in values]
            out = " ".join(values)
            f.write(out)
            f.write('\n')
        f.truncate()

    with open(args.hmmtrans,mode='w') as f:
        for values in trans_table:
            values = [str(i) for i in values]
            out = " ".join(values)
            f.write(out)
            f.write('\n')
        f.truncate()