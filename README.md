# GPTb

I don't know neural networks or anything but I thought it would be a fun puzzle
to put a little time into making GPT-style models reuse state maybe
intelligently.  I guess that makes them recurrent or something?  I dunno.  I
was thinking it looked wasteful to continue passing the same tokens over and
over while generating.  I believe there is a caching mechanism inside the
transformers implementation, but I have not yet reviewed it.  It was mostly an
excuse to get a little comfortable with the model internals, which can be quite
hard for me, so I did it very fast, like jumping into cold water.

The quickest way seemed to be to just connect their outputs to their inputs.
And hence, we have GPTB.  I think it might be able to learn more effectively
per data unit and per model size than GPT2 can, because it can pass much more
complex state from token to token, but it needs much more batching to do so at
any reasonable speed, because it only processes one token in a sequence at a
time (in order to form state for the next).  Could be wrong.

The code of GPTB is just a copy-paste of code from GPT2 with tweaks to accept
only token, to accept a tensor of past state, and to output an additional
tensor of state to pass back.

Raises value around making these model implementations more modular.

Right now it only works with manual training, like:
```
from GPTB import GPTBLMHeadModel
from transformers.models.gpt2.configuration_gpt2 import GPT2Config
config = GPT2Config() # pass settings or you can pull the config from some pretrained model and tweak it
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
```
