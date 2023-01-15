import copy

import torch
from datasets import load_dataset, load_metric
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer
from transformers import AutoModelForQuestionAnswering, TrainingArguments, Trainer
from torch import nn
import numpy as np


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.cuda.set_device(2)

batch_size = 10
max_length = 512  # 输入feature的最大长度，question和context拼接之后
doc_stride = 128  # 2个切片之间的重合token数量
num_epochs = 10

# 加载数据集
datasets = load_dataset("squad")
"""
DatasetDict({
    train: Dataset({
        features: ['id', 'title', 'context', 'question', 'answers'],
        num_rows: 87599
    })
    validation: Dataset({
        features: ['id', 'title', 'context', 'question', 'answers'],
        num_rows: 10570
    })
})

{
    'id': '5733be284776f41900661182', 
    'title': 'University_of_Notre_Dame', 
    'context': 'Architecturally, the school has a Catholic character. Atop the Main Building\'s gold dome is a golden statue of the Virgin Mary. Immediately in front of the Main Building and facing it, is a copper statue of Christ with arms upraised with the legend "Venite Ad Me Omnes". Next to the Main Building is the Basilica of the Sacred Heart. Immediately behind the basilica is the Grotto, a Marian place of prayer and reflection. It is a replica of the grotto at Lourdes, France where the Virgin Mary reputedly appeared to Saint Bernadette Soubirous in 1858. At the end of the main drive (and in a direct line that connects through 3 statues and the Gold Dome), is a simple, modern stone statue of Mary.', 
    'question': 'To whom did the Virgin Mary allegedly appear in 1858 in Lourdes France?', 
    'answers': {
        'text': ['Saint Bernadette Soubirous'], 
        'answer_start': [515]
    }
}
"""

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModelForQuestionAnswering.from_pretrained("bert-base-uncased")

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


tokenized_datasets = datasets.map(data_preprocess, batched=True, remove_columns=datasets["train"].column_names)
# tokenized_datasets["train"]["input_ids"] = torch.tensor(np.array([tokenized_datasets["train"]["input_ids"]]))
# print(tokenized_datasets["train"]["token_type_ids"][0])
# print(len(tokenized_datasets["train"]))
"""
['input_ids', 'token_type_ids', 'attention_mask', 'start_positions', 'end_positions']
"""

test_data = tokenized_datasets["validation"]
train_data, val_data = train_test_split(tokenized_datasets["train"], test_size=0.3, random_state=2021)

# convert the text data into tensors
train_input_ids = torch.tensor(train_data["input_ids"], dtype=torch.long).to(device)
train_token_type_ids = torch.tensor(train_data["token_type_ids"], dtype=torch.long).to(device)
train_attention_mask = torch.tensor(train_data["attention_mask"], dtype=torch.long).to(device)
train_start_positions = torch.tensor(train_data["start_positions"], dtype=torch.long).to(device)
train_end_positions = torch.tensor(train_data["end_positions"], dtype=torch.long).to(device)

val_input_ids = torch.tensor(val_data["input_ids"], dtype=torch.long).to(device)
val_token_type_ids = torch.tensor(val_data["token_type_ids"], dtype=torch.long).to(device)
val_attention_mask = torch.tensor(val_data["attention_mask"], dtype=torch.long).to(device)
val_start_positions = torch.tensor(val_data["start_positions"], dtype=torch.long).to(device)
val_end_positions = torch.tensor(val_data["end_positions"], dtype=torch.long).to(device)

test_input_ids = torch.tensor(test_data["input_ids"], dtype=torch.long).to(device)
test_token_type_ids = torch.tensor(test_data["token_type_ids"], dtype=torch.long).to(device)
test_attention_mask = torch.tensor(test_data["attention_mask"], dtype=torch.long).to(device)
test_start_positions = torch.tensor(test_data["start_positions"], dtype=torch.long).to(device)
test_end_positions = torch.tensor(test_data["end_positions"], dtype=torch.long).to(device)


optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
criterion = nn.CrossEntropyLoss()
model.to(device)

model.train()
best_val_acc = 0.
for epoch in range(num_epochs):
    print('epoch:', epoch)
    total_loss = 0.
    total_acc = 0.
    log_interval = 100
    num_batches = len(train_input_ids) // batch_size
    for batch, i in enumerate(range(0, len(train_input_ids) - 1, batch_size)):
        seq_len = min(batch_size, len(train_input_ids) - 1 - i)
        inputs = {"input_ids": train_input_ids[i: i + seq_len], "token_type_ids": train_token_type_ids[i: i + seq_len], "attention_mask": train_attention_mask[i: i + seq_len]}
        outputs = model(**inputs)

        train_start_position = train_start_positions[i:i + seq_len]
        train_end_position = train_end_positions[i:i + seq_len]

        answer_start_position = outputs.start_logits
        answer_end_position = outputs.end_logits
        start_loss = criterion(answer_start_position, train_start_position)
        end_loss = criterion(answer_end_position, train_end_position)
        loss = (start_loss + end_loss) / 2

        acc_start = (answer_start_position.argmax(1) == train_start_position).float().mean()
        acc_end = (answer_end_position.argmax(1) == train_end_position).float().mean()
        acc = (acc_start + acc_end) / 2

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        total_loss += loss.item()
        total_acc += acc.item()
        if batch % log_interval == 0 and batch > 0:
            cur_loss = total_loss / log_interval
            cur_acc = total_acc / log_interval
            print(f'| epoch {epoch:3d} | {batch:5d}/{num_batches:5d} batches | '
                  f'loss {cur_loss:5.2f} | accuracy {cur_acc:5.2f} | ')
            total_loss = 0
            total_acc = 0

    model.eval()  # turn on evaluation mode
    total_loss = 0.
    total_acc = 0.
    with torch.no_grad():
        for batch, i in enumerate(range(0, len(val_input_ids) - 1, batch_size)):
            seq_len = min(batch_size, len(val_input_ids) - 1 - i)
            inputs = {"input_ids": val_input_ids[i: i + seq_len],
                      "token_type_ids": val_token_type_ids[i: i + seq_len],
                      "attention_mask": val_attention_mask[i: i + seq_len]}
            outputs = model(**inputs)

            val_start_position = val_start_positions[i:i + seq_len]
            val_end_position = val_end_positions[i:i + seq_len]

            answer_start_position = outputs.start_logits
            answer_end_position = outputs.end_logits
            start_loss = criterion(answer_start_position, val_start_position)
            end_loss = criterion(answer_end_position, val_end_position)
            loss = (start_loss + end_loss) / 2

            acc_start = (answer_start_position.argmax(1) == val_start_position).float().mean()
            acc_end = (answer_end_position.argmax(1) == val_end_position).float().mean()
            acc = (acc_start + acc_end) / 2

            total_loss += seq_len * loss.item()
            total_acc += seq_len * acc.item()

        val_loss = total_loss / (len(val_input_ids) - 1)
        val_acc = total_acc / (len(val_input_ids) - 1)

        print('-' * 89)
        print(
            f'| End of epoch {epoch:3d} | ' f'valid loss {val_loss:5.2f} | ' f'valid accuracy {val_acc:5.2f} |')
        print('-' * 89)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model = copy.deepcopy(model)

best_model.save_pretrained('./checkpoint.pt')

best_model.eval()  # turn on evaluation mode
total_loss = 0.
total_acc = 0.
with torch.no_grad():
    for batch, i in enumerate(range(0, len(test_input_ids) - 1, batch_size)):
        seq_len = min(batch_size, len(test_input_ids) - 1 - i)
        validation_input_ids = test_input_ids[i:i + seq_len, :]
        inputs = {"input_ids": test_input_ids[i: i + seq_len],
                  "token_type_ids": test_token_type_ids[i: i + seq_len],
                  "attention_mask": test_attention_mask[i: i + seq_len]}
        outputs = best_model(**inputs)

        test_start_position = test_start_positions[i:i + seq_len]
        test_end_position = test_end_positions[i:i + seq_len]

        answer_start_position = outputs.start_logits
        answer_end_position = outputs.end_logits
        start_loss = criterion(answer_start_position, test_start_position)
        end_loss = criterion(answer_end_position, test_end_position)
        loss = (start_loss + end_loss) / 2

        acc_start = (answer_start_position.argmax(1) == test_start_position).float().mean()
        acc_end = (answer_end_position.argmax(1) == test_end_position).float().mean()
        acc = (acc_start + acc_end) / 2

        total_loss += seq_len * loss.item()
        total_acc += seq_len * acc.item()

        pre_start_position = torch.tensor([start_logit.argmax() for start_logit in answer_start_position])
        pre_end_position = torch.tensor([end_logit.argmax() for end_logit in answer_end_position])
        for m in range(len(pre_start_position)):
            for inputs in range(len(validation_input_ids[m])):
                if validation_input_ids[m][inputs] == 0:
                    break
            show_input_ids = validation_input_ids[m][:inputs]
            quest_ans = tokenizer.decode(show_input_ids)
            true_answer = tokenizer.decode(show_input_ids[
                                           test_start_position[m]: test_end_position[m] + 1 if test_end_position[
                                                                                                   m] + 1 <= inputs else inputs])
            if pre_start_position[m] <= pre_end_position:
                pre_answer = tokenizer.decode(show_input_ids[
                                              pre_start_position[m]: pre_end_position[m] + 1 if pre_end_position[
                                                                                                    m] + 1 <= inputs else inputs])
            else:
                pre_answer = tokenizer.decode(show_input_ids[0:1])
            print("Question + context: " + quest_ans)
            print("True answer: " + true_answer)
            print("Prediction answer: " + pre_answer)

    test_loss = total_loss / (len(test_data["input_ids"]) - 1)
    test_acc = total_acc / (len(test_data["input_ids"]) - 1)

    print('-' * 89)
    print(f'| ' f'test loss {test_loss:5.2f} | ' f'test accuracy {test_acc:5.2f} |')
    print('-' * 89)

# args = TrainingArguments(
#     f"test-squad",
#     evaluation_strategy = "epoch",
#     learning_rate=2e-5,
#     per_device_train_batch_size=batch_size,
#     per_device_eval_batch_size=batch_size,
#     num_train_epochs=3,
#     weight_decay=0.01,
# )
#
#
# data_collator = default_data_collator
# trainer = Trainer(
#     model,
#     args,
#     train_dataset=tokenized_datasets["train"],
#     eval_dataset=tokenized_datasets["validation"],
#     data_collator=data_collator,
#     tokenizer=tokenizer,
# )

# trainer.train()