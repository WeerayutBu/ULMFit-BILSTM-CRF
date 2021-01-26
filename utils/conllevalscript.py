# !wget https://raw.githubusercontent.com/WeerayutBu/conlleval/master/conlleval

import torch

# Save batch_of_results to conlleval format
# inputs :
#         sents : (batch_size, words in each sentences)
#         golds : (batch_size, words in each sentences)
#         preds : (batch_size, words in each sentences)
#         FILE  : a file to save the results
class ConllevalScript():
    def __init__(self, idx_to_word, idx_to_tag, UNK_ID):
        super().__init__()
        self.idx_to_word = idx_to_word
        self.idx_to_tag = idx_to_tag
        self.UNK_ID = UNK_ID
        
    def save_conlleval_format(self, sents, golds, preds, FILE):
        def write_one_sent(sent, gold, pred):
            sent_text = ""
            for t, g, p in zip(sent, gold, pred):
                sent_text += t+' '+g+' '+p+' '+'\n'
            return sent_text+'\n'
        
        for sent, gold, pred in zip(sents, golds, preds):
            FILE.writelines(write_one_sent(sent, gold, pred))

    # Convert list of tags idx(a sentence) to string
    def tags_idx_to_text(self, label_idx, seq_len):
        return [self.idx_to_tag[tag_idx] 
                for idx, tag_idx in enumerate(label_idx) if seq_len > idx]

    def sent_idx_to_text(self, sent_idx, seq_len):
        instances = []
        for idx, token_idx in enumerate(sent_idx):
            if seq_len > idx :
                if token_idx in self.idx_to_word:
                    instances.append(self.idx_to_word[token_idx])
                else:
                    instances.append(self.idx_to_word[self.UNK_ID])
        return instances

    # Convert batch_sentences idx to batch_sentences string
    def pred_idx_to_text(self, batch_preds):
        return [[self.idx_to_tag[idx.item() if isinstance(idx, torch.Tensor) else idx]  for idx in s] for s in batch_preds]
    
    def convert_idx_to_text(self, batch_input, batch_len, status='word'):
        # Convert results idx from prediction to tags string
        if status == 'pred':
            return self.pred_idx_to_text(batch_input)
        
        # Convert sentence idx or tags idx each instance to string
        instances = []
        for instance, instance_len in zip(batch_input, batch_len):
            if status == 'word':
                # Convert list of words idx(a sentence) to string
                instances.append(self.sent_idx_to_text(instance, instance_len))
                
            elif status == 'tag':
                # Convert list of tags idx(a sentence) to string
                instances.append(self.tags_idx_to_text(instance, instance_len))
            else:
                raise "Status mismatch"
                return False
        return instances
    '''
        FILE = open("TEST_CONLLEVAL_FORMAT.txt", "w")
        FILE.close()
        !cat TEST_CONLLEVAL_FORMAT.txt | ./conllevel.pl
    '''