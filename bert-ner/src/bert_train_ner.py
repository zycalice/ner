# https://github.com/kamalkraj/BERT-NER/blob/dev/run_ner.py
# https://github.com/billpku/NLP_In_Action/blob/master/NER_with_BERT.ipynb
# https://huggingface.co/docs/transformers/custom_datasets#tok-ner

import math
import os
# import logging

import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from utils import flatten

from transformers import BertTokenizer, AdamW, get_linear_schedule_with_warmup
from transformers.models.bert.modeling_bert import BertForTokenClassification
from tqdm import trange
from seqeval.metrics import classification_report, f1_score, accuracy_score

# logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
#                     datefmt='%m/%d/%Y %H:%M:%S',
#                     level=logging.INFO)
# logger = logging.getLogger(__name__)

# Global vars
INPUT_DATA_DIR = "../data/"

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
N_GPU = torch.cuda.device_count()

DOMAIN_UNIQUE_LABELS = {
    "conll2003": ["O", "B-MISC", "I-MISC", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC"],
    "legal": None,  # todo
    "medical": None,  # todo
    "mixed": None,  # todo
}

CLS, SEP, NON_FIRST_SUBTOKEN_LABEL = "[CLS]", "[SEP]", "X"

CURR_DOMAIN = "conll2003"

EXAMPLE_NUM = 2000


# Feature preprocessing
def readfile(filename, use_examples):
    """
    read file
    when use_examples=True, only load EXAMPLE NUM data points as toy data
    """
    data = []

    with open(filename, "r") as file:
        sent_tokens = []
        sent_labels = []
        sent_id = 0
        for line in file:
            if line == "\n":
                data.append((sent_tokens, sent_labels))
                sent_tokens = []
                sent_labels = []
                sent_id += 1
                if use_examples and sent_id == EXAMPLE_NUM:
                    return data
            else:
                line_splits = line.split()
                sent_tokens.append(line_splits[0])
                sent_labels.append(line_splits[-1])
    return data


class NerDataset(Dataset):
    def __init__(self, data, tokenizer, max_len, unique_labels):
        """
        data are outputs from function readfile, in form of lists of tuples [([x,x,x], [l,l,l]), ([x,x,x], [l,l,l])]
        unique_labels includes special characters
        """
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.token_max_len = max_len - 2
        self.data = data
        self.unique_labels = unique_labels
        self.label2idx = {t: i for i, t in enumerate(self.unique_labels)}
        self.idx2label = {self.label2idx[key]: key for key in self.label2idx.keys()}

    def __len__(self):
        return len(self.data)

    def get_unpadded_sub_tokens_labels(self, words, label):
        """
        return 1) sub_tokens in a single list as well as 2) first_sub_token masks as indicator
        """
        # use tokenizer
        sub_tokens_lists = list(map(self.tokenizer.tokenize, words))

        # get first_token masks without padding
        unpadded_first_sub_token_masks = [1 if i == 0 else 0 for w in sub_tokens_lists for i, subw in enumerate(w)]

        # fill label for non-first sub-token words
        unpadded_labels = [label[j] if i == 0 else NON_FIRST_SUBTOKEN_LABEL for j, w in enumerate(sub_tokens_lists)
                           for i, subw in enumerate(w)]

        # flatten the list of list to make all sub_tokens one single list
        unpadded_sub_tokens = list(flatten(sub_tokens_lists))

        # consider max_length and truncation
        if len(unpadded_sub_tokens) > self.token_max_len:
            unpadded_sub_tokens = unpadded_sub_tokens[:self.token_max_len]
            unpadded_first_sub_token_masks = unpadded_first_sub_token_masks[:self.token_max_len]
            unpadded_labels = unpadded_labels[:self.token_max_len]

        # append cls and sep
        unpadded_sub_tokens = [CLS] + unpadded_sub_tokens + [SEP]
        unpadded_first_sub_token_masks = [0] + unpadded_first_sub_token_masks + [0]
        unpadded_labels = [self.label2idx[CLS]] + \
                          [self.label2idx[label] for label in unpadded_labels] + \
                          [self.label2idx[SEP]]

        return unpadded_sub_tokens, unpadded_first_sub_token_masks, unpadded_labels

    def get_padded_ids_labels(self, unpadded_sub_tokens, unpadded_first_sub_token_masks, unpadded_labels):
        """
        get input_ids, attention_masks, first_sub_token_masks, and labels
        """
        # get padded input_ids
        input_ids = self.tokenizer.convert_tokens_to_ids(unpadded_sub_tokens)
        input_ids = input_ids + [0] * (self.max_len - len(unpadded_sub_tokens))  # CLS and SEP are included in seq len
        input_ids = torch.tensor(input_ids)

        # get padded attention_masks
        attention_masks = torch.zeros(self.max_len)
        attention_masks[input_ids != 0] = 1

        # get padded first_sub_token_masks
        first_sub_token_masks = torch.zeros(self.max_len)
        first_sub_token_masks[:len(unpadded_first_sub_token_masks)] = torch.tensor(unpadded_first_sub_token_masks)

        # get padded labels
        labels = unpadded_labels + [0] * (self.max_len - len(unpadded_sub_tokens))
        labels = torch.tensor(labels)

        return input_ids, attention_masks, first_sub_token_masks, labels

    def __getitem__(self, idx):
        """
        output:
        input_ids: tensor of size [1, max_len]
        attention_masks: tensor of size [1, max_len], marks padding as 0
        labels: a list of length max_len
        """
        words, text_labels = self.data[idx]

        # get features
        unpadded_sub_tokens, unpadded_first_sub_token_masks, unpadded_labels = \
            self.get_unpadded_sub_tokens_labels(words, text_labels)
        input_ids, attention_masks, first_sub_token_masks, labels = self.get_padded_ids_labels(
            unpadded_sub_tokens, unpadded_first_sub_token_masks, unpadded_labels
        )

        return {"input_ids": input_ids,
                "attention_masks": attention_masks,
                "first_sub_token_masks": first_sub_token_masks,
                "labels": labels}


# Model
def train_ner_eval_val(domain, model_name, max_len, epochs, max_grad_norm, batch_num,
                       model_output_address, evaluation_output_address,
                       full_fine_tune=True, use_examples=False, first_subtokens_loss=True):
    # add CLS, SEP and NON_FIRST LABEL to obtain the unique labels for this classification problem
    unique_labels = DOMAIN_UNIQUE_LABELS[domain] + [CLS, SEP, NON_FIRST_SUBTOKEN_LABEL]
    # load model
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertForTokenClassification.from_pretrained(model_name, num_labels=len(unique_labels))

    # load data
    train_data = readfile(INPUT_DATA_DIR + domain + "_train.txt", use_examples=use_examples)
    val_data = readfile(INPUT_DATA_DIR + domain + "_val.txt", use_examples=use_examples)
    # test_data = readfile(INPUT_DATA_DIR + domain + "_test.txt", use_examples=use_examples)

    train_dataset = NerDataset(data=train_data, tokenizer=tokenizer, max_len=max_len, unique_labels=unique_labels)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_num, shuffle=True)

    # setup model
    model.to(DEVICE)
    model.train()
    num_train_optimization_steps = int(math.ceil(len(train_data) / batch_num) / 1) * epochs
    print("***** Running training *****")
    print("  Num examples = %d" % (len(train_data)))
    print("  Batch size = %d" % batch_num)
    print("  Num steps = %d" % num_train_optimization_steps)

    # fine-tune method
    if full_fine_tune:
        # Fine tune model all layer parameters
        param_optimizer = list(model.named_parameters())
        no_decay = ['bias', 'gamma', 'beta']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
             'weight_decay_rate': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
             'weight_decay_rate': 0.0}
        ]
    else:
        # Only fine tune classifier parameters
        param_optimizer = list(model.classifier.named_parameters())
        optimizer_grouped_parameters = [{"params": [p for n, p in param_optimizer]}]
    optimizer = AdamW(optimizer_grouped_parameters, lr=3e-5)
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=0,  # Default value in run_glue.py
                                                num_training_steps=num_train_optimization_steps)

    # train loop
    for _ in trange(epochs, desc="Epoch"):
        tr_loss = 0
        num_tr_examples, num_tr_steps = 0, 0
        for data in train_dataloader:
            # add batch to gpu
            b_input_ids, b_attention_masks = data["input_ids"].to(DEVICE), data["attention_masks"].to(DEVICE)
            b_first_sub_token_masks = data["first_sub_token_masks"].to(DEVICE)
            b_labels = data["labels"].to(DEVICE)

            # customize the calculation of loss
            # forward pass but customize the way to calculate loss; # exclude [CLS], [SEP] and X in loss calculation
            # https://huggingface.co/transformers/v4.12.5/_modules/transformers/modeling_bert.html#BertForTokenClassification.forward
            if first_subtokens_loss:
                loss_fct = torch.nn.CrossEntropyLoss()
                # logits include all subtokens
                logits = model(input_ids=b_input_ids,
                               attention_mask=b_attention_masks,
                               labels=b_labels)[1]
                # loss include only the non-first subtokens
                active_loss = b_first_sub_token_masks.view(-1) == 1  # switched attention masks to first sub token masks
                active_logits = logits.view(-1, len(unique_labels))
                active_labels = torch.where(
                    active_loss, b_labels.view(-1), torch.tensor(loss_fct.ignore_index).type_as(b_labels)
                )
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = model(input_ids=b_input_ids,
                             attention_mask=b_attention_masks,
                             labels=b_labels)[0]

            # When multi gpu, average it
            if N_GPU > 1:
                loss = loss.mean()

            # backward pass
            loss.backward()

            # track train loss
            tr_loss += loss.item()
            num_tr_examples += b_input_ids.size(0)
            num_tr_steps += 1

            # gradient clipping
            torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=max_grad_norm)

            # update parameters
            optimizer.step()
            scheduler.step()
            model.zero_grad()

        # print train loss per epoch
        print("Train loss: {}".format(tr_loss / num_tr_steps))

        # evaluate while training
        eval_ner(val_data, model, tokenizer, max_len, unique_labels, batch_num, evaluation_output_address)

    # output model Make dir if not exits
    model_output_address = model_output_address + "/model/"
    if not os.path.exists(model_output_address):
        os.makedirs(model_output_address)

    # save a trained model, configuration and tokenizer
    model_to_save = model.module if hasattr(model, 'module') else model
    output_model_file = os.path.join(model_output_address, "pytorch_model.bin")
    output_config_file = os.path.join(model_output_address, "config.json")

    # save model into file
    torch.save(model_to_save.state_dict(), output_model_file)
    model_to_save.config.to_json_file(output_config_file)
    tokenizer.save_vocabulary(model_output_address)


def eval_ner(eval_data, trained_model, tokenizer, max_len, unique_labels, batch_num, evaluation_output_address):
    trained_model.eval()
    y_true = []
    y_pred = []

    print("***** Running evaluation *****")
    print("  Num examples = {}".format(len(eval_data)))
    print("  Batch size = {}".format(batch_num))

    val_dataset = NerDataset(data=eval_data, tokenizer=tokenizer, max_len=max_len, unique_labels=unique_labels)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_num, shuffle=True)
    for data in val_dataloader:
        b_input_ids, b_attention_masks = data["input_ids"].to(DEVICE), data["attention_masks"].to(DEVICE)
        b_first_sub_token_masks = data["first_sub_token_masks"].to(DEVICE)
        b_labels = data["labels"].to(DEVICE)

        with torch.no_grad():
            logits = trained_model(input_ids=b_input_ids,
                                   attention_mask=b_attention_masks)[0]

        # Get NER predict result
        # take only the actual labels for prediction, exclude [CLS], [SEP] and X
        logits = torch.argmax(F.log_softmax(logits[:, :, :len(DOMAIN_UNIQUE_LABELS[CURR_DOMAIN])], dim=2), dim=2)
        logits = logits.detach().cpu().numpy()

        # Get NER true result
        b_labels = b_labels.to('cpu').numpy()

        # Only predict the real word, mark=0, will not calculate
        # Compare the true with predicted result
        for i, mask in enumerate(b_first_sub_token_masks):
            # Real one
            sentence_y_true = []
            # Predict one
            sentence_y_pred = []

            for j, m in enumerate(mask):
                # Mark=0, meaning it's a pad word or not a first sub-token, don't compare
                if m:
                    sentence_y_true.append(val_dataset.idx2label[b_labels[i][j]])
                    sentence_y_pred.append(val_dataset.idx2label[logits[i][j]])

            y_true.append(sentence_y_true)
            y_pred.append(sentence_y_pred)

    # Get acc , recall, F1 result report
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    report = classification_report(y_true, y_pred, digits=4)

    # Save the report into file
    if not os.path.exists(evaluation_output_address):
        os.makedirs(evaluation_output_address)
    output_eval_file = os.path.join(evaluation_output_address, "eval_results_val.txt")

    # output results; append mode to continue writing outputs for different epochs
    with open(output_eval_file, "a") as writer:
        print("***** Eval results *****")
        print("f1 score: %f" % f1)
        print("Accuracy score: %f" % acc)
        print("Report\n%s" % report)

        writer.write("f1 score:\n")
        writer.write(str(f1))
        writer.write("\n\nAccuracy score:\n")
        writer.write(str(acc))
        writer.write("\n\nReport:\n")
        writer.write(report)
        writer.write("\n")


if __name__ == '__main__':
    pass

    # # test dataset
    # example_data = readfile("../data/conll2003_train.txt", use_examples=True)
    #
    # # check input
    # example_id = 3
    # print(example_data[example_id])
    # tokenizer = BertTokenizer.from_pretrained("nlpaueb/legal-bert-base-uncased", do_lower_case=True)
    # nerdataset = NerDataset(data=example_data, tokenizer=tokenizer, max_len=128,
    #                         unique_labels=DOMAIN_UNIQUE_LABELS['conll2003'])
    # example_result = nerdataset[example_id]
    # print(tokenizer.convert_ids_to_tokens(example_result['input_ids']))
    #
    # # check output
    # print(example_result)
    #
    # # check length
    # for x in example_result:
    #     print(len(example_result[x]))

    max_len_input = 128
    epochs_input = 3
    max_grad_norm_input = 1
    batch_num_input = 10

    # # test model - bert base
    # model_name_input = "bert-base-uncased"
    # train_ner_eval_val(domain=CURR_DOMAIN,
    #                    model_name=model_name_input,
    #                    max_len=max_len_input,
    #                    epochs=epochs_input,
    #                    max_grad_norm=max_grad_norm_input,
    #                    batch_num=batch_num_input,
    #                    model_output_address="./output/conll_base_all_subtokens_loss/",
    #                    evaluation_output_address="./output/conll_base_all_subtokens_loss/",
    #                    full_fine_tune=True,
    #                    use_examples=True,
    #                    first_subtokens_loss=False)

    # test model - legal bert
    model_name_input = "nlpaueb/legal-bert-base-uncased"
    train_ner_eval_val(domain=CURR_DOMAIN,
                       model_name=model_name_input,
                       max_len=max_len_input,
                       epochs=epochs_input,
                       max_grad_norm=max_grad_norm_input,
                       batch_num=batch_num_input,
                       model_output_address="./output/conll_legal_first_subtokens_loss/",
                       evaluation_output_address="./output/conll_legal_first_subtokens_loss/",
                       full_fine_tune=True,
                       use_examples=True,
                       first_subtokens_loss=True)
