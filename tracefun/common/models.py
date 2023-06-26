import torch
from torch import nn
from transformers import AutoTokenizer, AutoModel, PreTrainedModel, AutoModelForSequenceClassification


class AvgPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.pooler = torch.nn.AdaptiveAvgPool2d((1, config.hidden_size))

    def forward(self, hidden_states):
        return self.pooler(hidden_states).view(-1, self.hidden_size)

class RelationClassifyHeader(nn.Module):
    """
    H2:
    use averaging pooling across tokens to replace first_token_pooling
    """

    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.code_pooler = AvgPooler(config)
        self.text_pooler = AvgPooler(config)

        self.dense = nn.Linear(config.hidden_size * 3, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.output_layer = nn.Linear(config.hidden_size, 2)

    def forward(self, code_hidden, text_hidden):
        pool_code_hidden = self.code_pooler(code_hidden)
        pool_text_hidden = self.text_pooler(text_hidden)
        diff_hidden = torch.abs(pool_code_hidden - pool_text_hidden)
        concated_hidden = torch.cat((pool_code_hidden, pool_text_hidden), 1)
        concated_hidden = torch.cat((concated_hidden, diff_hidden), 1)

        x = self.dropout(concated_hidden)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.output_layer(x)
        return x

class BertClassifier(PreTrainedModel):
    def __init__(self, config, code_bert):
        super().__init__(config)

        self.ctokneizer = AutoTokenizer.from_pretrained(code_bert)
        self.cbert = AutoModel.from_pretrained(code_bert)

        self.ntokenizer = self.ctokneizer
        self.nbert = self.cbert

        self.cls = RelationClassifyHeader(config)

    def forward(self, inputs):
        code_ids = inputs['code_ids']
        code_attention_mask = inputs['code_attention_mask']
        text_ids = inputs['text_ids']
        text_attention_mask = inputs['text_attention_mask']

        code_hidden = self.cbert(code_ids, attention_mask=code_attention_mask)[0]
        text_hidden = self.nbert(text_ids, attention_mask=text_attention_mask)[0]
        logits = self.cls(code_hidden=code_hidden, text_hidden=text_hidden)
        return logits

    def get_sim_score(self, text_hidden, code_hidden):
        logits = self.cls(text_hidden=text_hidden, code_hidden=code_hidden)
        sim_scores = torch.softmax(logits, 1).data.tolist()
        return [x[1] for x in sim_scores]

    def get_nl_tokenizer(self):
        return self.ntokenizer

    def get_pl_tokenizer(self):
        return self.ctokneizer

    def create_nl_embd(self, input_ids, attention_mask):
        return self.nbert(input_ids, attention_mask)

    def create_pl_embd(self, input_ids, attention_mask):
        return self.cbert(input_ids, attention_mask)

    def get_nl_sub_model(self):
        return self.nbert

    def get_pl_sub_model(self):
        return self.cbert