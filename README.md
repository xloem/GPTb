from GPTB import GPTBLMHeadModel
from transformers.models.gpt2.configuration_gpt2 import GPT2Config
```
config = GPT2Config() # pass settings or you can pull the config from some pretrained model and tweak it (transformers.pipeline('text-generation').model.config)
config.rebias = True # additional parameter for GPTB
model = GPTBLMHeadModel(config)

model.train()
model.zero_grad()
optimizer.zero_grad()
past_hidden_states = None
past_logits = None
for batch_of_tokens in data: # shape of batch_of_tokens is (batchsize, 1)
    if past_logits is not None:
        loss = torch.nn.functional.cross_entropy(past_logits.view(-1, vocab_size), batch_of_tokens.view(-1))
        loss.backward()
        optimizer.step()
        model.zero_grad()
        optimizer.zero_grad()
    past_logits, past_hidden_states, extra = model(batch_of_tokens, past_hidden_states=past_hidden_states)
