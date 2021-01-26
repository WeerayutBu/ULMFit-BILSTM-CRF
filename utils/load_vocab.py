from collections import Counter
def sort_dict_freq(counter_dict, MINCOUNT=1):
    Dict  = {(w,c) for w, c in counter_dict.items() if c >= MINCOUNT}
    Dict  = sorted(Dict, key=lambda item: item[1], reverse=True)
    Dict  = [w[0] for w in Dict]
    return Dict


def load_conll_format_nested_ner(path, MAX_LEVEL=5, NE_TYPE='flatten'):
    file_path = path
    corpus = []
    words = Counter()
    tags = Counter()
    chars = Counter()
    sentence = []
    with open(path, encoding='utf-8') as file:
        for index, line in enumerate(file):
            line = line.strip('\n')
            token = line.split()
            len_sample = len(token)
#             print(index, line.isspace(),len_sample, line)

            if len_sample == MAX_LEVEL+1:
                word = token[0]
                words.update([word])
                chars.update(word)
                
                if NE_TYPE == 'flatten':
                    tag = token[1]
                    tags.update([tag])
                    sentence.append([word, tag])
                    
                elif NE_TYPE == 'nested':
                    tag = token[1:MAX_LEVEL+1]
                    tags.update(tag)
                    sentence.append([word, tag])
                
                else:
                    print(line)
                    raise TypeError("Tag mismatch")
            
            elif len_sample == 0 and line == '':
                corpus.append(sentence)
                sentence = []
            
            else:
                print(index)
                print(index, line.isspace(),len_sample, line)
                raise "Error tokenize error"
        
        if len(sentence) == 0:
            pass
        else:
            corpus.append(sentence)
            
        return corpus, \
               sort_dict_freq(words), \
               sort_dict_freq(chars), \
               sort_dict_freq(tags)
    
    
