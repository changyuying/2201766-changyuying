import copy
import time
import math

import torch
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from torch import nn, Tensor
from transformers import AutoTokenizer
from TransformerForQuestionAnswering import TransformerModel


max_length = 512  # 输入feature的最大长度，question和context拼接之后
doc_stride = 128  # 2个切片之间的重合token数量
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")


def generate_square_subsequent_mask(sz: int) -> Tensor:
    """Generates an upper-triangular matrix of -inf, with zeros on diag."""
    return torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)


def train(model: nn.Module, batch_size, device, input_ids, criterion, optimizer, scheduler, epoch, start_positions, end_positions) -> None:
    model.train()  # turn on train mode
    total_loss = 0.
    total_acc = 0.
    log_interval = 100
    start_time = time.time()
    src_mask = generate_square_subsequent_mask(512).to(device)

    num_batches = len(input_ids) // batch_size
    for batch, i in enumerate(range(0, input_ids.size(0) - 1, batch_size)):
        seq_len = min(batch_size, len(input_ids) - 1 - i)
        train_input_ids = input_ids[i:i + seq_len, :].transpose(0, 1).to(device)
        train_start_positions = start_positions[i:i + seq_len].to(device)
        train_end_positions = end_positions[i:i + seq_len].to(device)

        start_logits, end_logits = model(train_input_ids, src_mask)
        start_loss = criterion(start_logits, train_start_positions)
        end_loss = criterion(end_logits, train_end_positions)
        loss = (start_loss + end_loss) / 2

        acc_start = (start_logits.argmax(1) == train_start_positions).float().mean()
        acc_end = (end_logits.argmax(1) == train_end_positions).float().mean()
        acc = (acc_start + acc_end) / 2

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()

        total_loss += loss.item()
        total_acc += acc
        if batch % log_interval == 0 and batch > 0:
            lr = scheduler.get_last_lr()[0]
            ms_per_batch = (time.time() - start_time) * 1000 / log_interval
            cur_loss = total_loss / log_interval
            cur_acc = total_acc / log_interval
            print(f'| epoch {epoch:3d} | {batch:5d}/{num_batches:5d} batches | '
                  f'lr {lr:02.2f} | ms/batch {ms_per_batch:5.2f} | '
                  f'loss {cur_loss:5.2f} | accuracy {cur_acc:5.2f} | ')
            total_loss = 0
            start_time = time.time()

def evaluate(model: nn.Module, input_ids: Tensor, batch_size: int, device, criterion, start_positions, end_positions, tokenizer=None, test=False):
    model.eval()  # turn on evaluation mode
    total_loss = 0.
    total_acc = 0.
    src_mask = generate_square_subsequent_mask(512).to(device)
    with torch.no_grad():
        for i in range(0, input_ids.size(0) - 1, batch_size):
            seq_len = min(batch_size, len(input_ids) - 1 - i)
            val_input_ids = input_ids[i:i + seq_len, :].transpose(0, 1)
            val_start_positions = start_positions[i:i + seq_len]
            val_end_positions = end_positions[i:i + seq_len]

            start_logits, end_logits = model(val_input_ids, src_mask)
            start_loss = criterion(start_logits, val_start_positions)
            end_loss = criterion(end_logits, val_end_positions)
            loss = (start_loss + end_loss) / 2

            acc_start = (start_logits.argmax(1) == val_start_positions).float().mean()
            acc_end = (end_logits.argmax(1) == val_end_positions).float().mean()
            acc = (acc_start + acc_end) / 2

            total_loss += seq_len * loss.item()
            total_acc += seq_len * acc

            if test == True:
                pre_start_position = torch.tensor([start_logit.argmax() for start_logit in start_logits])
                pre_end_position = torch.tensor([end_logit.argmax() for end_logit in end_logits])
                for i in range(len(pre_start_position)):
                    quest_ans = tokenizer.decode(val_input_ids[i])
                    true_answer = tokenizer.decode(val_input_ids[i][val_start_positions[i]: val_end_positions[i] + 1])
                    pre_answer = tokenizer.decode(val_input_ids[i][pre_start_position[i]: pre_end_position[i] + 1])
                    print("Question + context: " + quest_ans)
                    print("True answer: " + true_answer)
                    print("Prediction answer: " + pre_answer)


    return total_loss / (len(input_ids) - 1), total_acc / (len(input_ids) - 1)


def data_preprocess(examples):
    # 每一个超长文本会被切片成多个输入，相邻两个输入之间会有交集
    tokenized_examples = tokenizer(
        examples["question"],
        examples["context"],
        truncation="only_second",
        max_length=max_length,
        stride=doc_stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    # 使用 overflow_to_sample_mapping 参数来映射切片片ID到原始ID
    # 比如有2个 expamples 被切成4片，那么对应是[0, 0, 1, 1]，前两片对应原来的第一个 example
    sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
    # offset_mapping 也对应4片
    # offset_mapping 参数帮助我们映射到原始输入，由于答案标注在原始输入上，所以有助于我们找到答案的起始和结束位置
    offset_mapping = tokenized_examples.pop("offset_mapping")

    # 重新标注数据
    tokenized_examples["start_positions"] = []
    tokenized_examples["end_positions"] = []

    for i, offsets in enumerate(offset_mapping):
        # 对每一片进行处理
        # 将无答案的样本标注到CLS上
        input_ids = tokenized_examples["input_ids"][i]
        cls_index = input_ids.index(tokenizer.cls_token_id)

        # 区分 question 和 context
        sequence_ids = tokenized_examples.sequence_ids(i)

        # 拿到原始的 example 下标
        sample_index = sample_mapping[i]
        answers = examples["answers"][sample_index]
        # 如果没有答案，则使用 CLS 所在的位置为答案
        if len(answers["answer_start"]) == 0:
            tokenized_examples["start_positions"].append(cls_index)
            tokenized_examples["end_positions"].append(cls_index)
        else:
            # 答案的 character 级别 start/end 位置
            start_char = answers["answer_start"][0]
            end_char = start_char + len(answers["text"][0])

            # 找到 token 级别的 index_start
            token_start_index = 0
            while sequence_ids[token_start_index] != 1:
                token_start_index += 1

            # 找到 token 级别的 index_end
            token_end_index = len(input_ids) - 1
            while sequence_ids[token_end_index] != 1:
                token_end_index -= 1

            # 检测答案是否超出文本长度，超出的话也适用 CLS 作为标注
            if not (offsets[token_start_index][0] <= start_char and offsets[token_end_index][1] >= end_char):
                tokenized_examples["start_positions"].append(cls_index)
                tokenized_examples["end_positions"].append(cls_index)
            else:
                # 如果不超出则找到答案 token 的 start 和 end 位置
                while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
                    token_start_index += 1
                tokenized_examples["start_positions"].append(token_start_index - 1)
                while offsets[token_end_index][1] >= end_char:
                    token_end_index -= 1
                tokenized_examples["end_positions"].append(token_end_index + 1)

    return tokenized_examples


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(3)

    # build the vocabulary using training set
    datasets = load_dataset("squad")
    vocab = tokenizer.get_vocab()
    tokenized_datasets = datasets.map(data_preprocess, batched=True, remove_columns=datasets["train"].column_names)
    test_data = tokenized_datasets["validation"]
    train_data, val_data = train_test_split(tokenized_datasets["train"], test_size=0.3, random_state=2021)

    # convert the text data into tensors
    # train_input_ids = torch.tensor(train_data["input_ids"], dtype=torch.long).to(device)
    # train_start_positions = torch.tensor(train_data["start_positions"], dtype=torch.long).to(device)
    # train_end_positions = torch.tensor(train_data["end_positions"], dtype=torch.long).to(device)
    #
    # val_input_ids = torch.tensor(val_data["input_ids"], dtype=torch.long).to(device)
    # val_start_positions = torch.tensor(val_data["start_positions"], dtype=torch.long).to(device)
    # val_end_positions = torch.tensor(val_data["end_positions"], dtype=torch.long).to(device)
    #
    # test_input_ids = torch.tensor(test_data["input_ids"], dtype=torch.long).to(device)
    # test_start_positions = torch.tensor(test_data["start_positions"], dtype=torch.long).to(device)
    # test_end_positions = torch.tensor(test_data["end_positions"], dtype=torch.long).to(device)


    # hyper-parameters for training and evaluation
    batch_size = 100      # batch size for training
    lr = 0.1              # learning rate
    nepoch = 1            # number of iterations
    ntokens = len(vocab)  # size of vocabulary
    emsize = 512          # embedding dimension
    d_hid = 64            # dimension of the feedforward network model in nn.TransformerEncoder
    nlayers = 6           # number of nn.TransformerEncoderLayer in nn.TransformerEncoder
    nhead = 8             # number of heads in nn.MultiheadAttention
    dropout = 0.1         # dropout probability

    # prepare data loader, model, and other things for training
    model = TransformerModel(ntokens, emsize, nhead, d_hid, nlayers, dropout).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)

    # training loop
    best_val_acc = float(0)
    best_model = None
    print('| Start training for', nepoch, 'epochs')
    print('-' * 89)
    for epoch in range(1, nepoch + 1):
        epoch_start_time = time.time()

        train_input_ids = torch.tensor(train_data["input_ids"], dtype=torch.long)
        train_start_positions = torch.tensor(train_data["start_positions"], dtype=torch.long)
        train_end_positions = torch.tensor(train_data["end_positions"], dtype=torch.long)
        train(model, batch_size, device, train_input_ids, criterion, optimizer, scheduler, epoch, train_start_positions, train_end_positions)

        val_input_ids = torch.tensor(val_data["input_ids"], dtype=torch.long).to(device)
        val_start_positions = torch.tensor(val_data["start_positions"], dtype=torch.long).to(device)
        val_end_positions = torch.tensor(val_data["end_positions"], dtype=torch.long).to(device)
        val_loss, val_acc = evaluate(model, val_input_ids, batch_size, device, criterion, val_start_positions, val_end_positions)

        elapsed = time.time() - epoch_start_time
        print('-' * 89)
        print(f'| End of epoch {epoch:3d} | time: {elapsed:5.2f}s | ' f'valid loss {val_loss:5.2f} | ' f'valid accuracy {val_acc:5.2f} |')
        print('-' * 89)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model = copy.deepcopy(model)

        scheduler.step()

    # evaluate the best model with lowest PPL on the validation set
    test_input_ids = torch.tensor(test_data["input_ids"], dtype=torch.long).to(device)
    test_start_positions = torch.tensor(test_data["start_positions"], dtype=torch.long).to(device)
    test_end_positions = torch.tensor(test_data["end_positions"], dtype=torch.long).to(device)
    test_loss, test_acc = evaluate(best_model, test_input_ids, batch_size, device, criterion, test_start_positions, test_end_positions, tokenizer=tokenizer, test=True)

    print('=' * 89)
    print(f'| End of training | test loss {test_loss:5.2f} | test accuracy {test_acc:5.2f} |')
    print('=' * 89)


if __name__ == '__main__':
    main()