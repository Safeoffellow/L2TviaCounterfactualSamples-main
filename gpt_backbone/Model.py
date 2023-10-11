import mindspore as ms
from mindspore import nn
import mindspore.ops as ops
import copy

from mindnlp.models.gpt2 import GPT2Model, GPT2LMHeadModel

class AttnMaskGPT2Model(GPT2Model):
    def construct(
        self,
        input_ids=None,
        past_key_values=None,    
        attention_mask=None,     
        token_type_ids=None,     
        position_ids=None,        
        head_mask=None,
        inputs_embeds=None,      
        encoder_hidden_states=None,
        encoder_attention_mask=None,
    ):
        '''
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        '''

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        
        elif input_ids is not None:
            input_shape = input_ids.shape
            input_ids = input_ids.view(-1, input_shape[-1])
            batch_size = input_ids.shape[0]
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size[:-1]
            batch_size = inputs_embeds.shape[0]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")


        if token_type_ids is not None:
            token_type_ids = token_type_ids.view(-1, input_shape[-1])
        if position_ids is not None:
            position_ids = position_ids.view(-1, input_shape[-1])
        
        if past_key_values is None:
            past_length = 0
            past_key_values = tuple([None] * len(self.h))    
        else:
            past_length = past_key_values[0][0].shape[-2]    
    
        if position_ids is None:
            position_ids = ops.arange(past_length, input_shape[-1] + past_length, dtype=ms.int64)
            position_ids = position_ids.expand_dims(0).view(-1, input_shape[-1])

        # GPT2Attention mask.
        if attention_mask is not None:

            if batch_size <= 0:
                raise ValueError("batch_size has to be defined and > 0")
            
            if attention_mask.dim() == 3:
                attention_mask = attention_mask[:, None, :, :]
            
            else:
                attention_mask = attention_mask.view(batch_size, -1)   
                # We create a 3D attention mask from a 2D tensor mask.
                # Sizes are [batch_size, 1, 1, to_seq_length]
                # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
                # this attention mask is more simple than the triangular masking of causal attention
                # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
                attention_mask = attention_mask[:, None, None, :]      

            # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
            # masked positions, this operation will create a tensor which is 0.0 for
            # positions we want to attend and -10000.0 for masked positions.
            # Since we are adding it to the raw scores before the softmax, this is
            # effectively the same as removing these entirely.zh

            attention_mask = attention_mask.astype(dtype=self.dtype)  # fp16 compatibility
            attention_mask = (1.0 - attention_mask) * -10000.0

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]

        if self.add_cross_attention and encoder_hidden_states is not None:    
            
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.shape 
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length) 

            
            if encoder_attention_mask is None:
                encoder_attention_mask = ops.ones(encoder_hidden_shape)
            encoder_attention_mask = self.invert_attention_mask(encoder_attention_mask)   
        else:
            encoder_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # head_mask has shape n_layer x batch x n_heads x N x N
        head_mask = self.get_head_mask(head_mask, self.config.n_layer)

        if inputs_embeds is None:
            inputs_embeds = self.wte(input_ids)    
        position_embeds = self.wpe(position_ids)   
        hidden_states = inputs_embeds + position_embeds 

        if token_type_ids is not None:
            token_type_embeds = self.wte(token_type_ids)
            hidden_states = hidden_states + token_type_embeds  

        hidden_states = self.drop(hidden_states)

        output_shape = input_shape + (hidden_states.shape[-1],)
        
        presents = () if self.use_cache else None
        all_self_attentions = () 
        all_cross_attentions = () 
        all_hidden_states = ()

        for i, block in enumerate(self.h):
            if self.output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            outputs = block(
                hidden_states,
                layer_past=past_key_values[i],
                attention_mask=attention_mask,
                head_mask=head_mask[i],
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                use_cache=self.use_cache,
            )

            hidden_states = outputs[0]
            if self.use_cache is True:
                presents = presents + (outputs[1],)

            if self.output_attentions:
                all_self_attentions = all_self_attentions + (outputs[2 if self.use_cache else 1],)
                if self.config.add_cross_attention:
                    all_cross_attentions = all_cross_attentions + (outputs[3 if self.use_cache else 2],)
       
        hidden_states = self.ln_f(hidden_states)

        hidden_states = hidden_states.view(output_shape)
        # Add last hidden state
        if self.output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)
        
        
        outputs = (hidden_states, presents)
        if self.output_attentions:
            outputs += (all_hidden_states, all_self_attentions)
            if self.add_cross_attention:
                outputs += (all_cross_attentions,)
        return outputs

class AttnMaskGPT2LMHeadModel(GPT2LMHeadModel):    
    def __init__(self, config):
        super().__init__(config)                 
        self.transformer = AttnMaskGPT2Model(config)
        self.lm_head = nn.Dense(config.n_embd, config.vocab_size, has_bias=False)  
        self.post_init()


    def prepare_inputs_for_generation(self, input_ids, past=None, **kwargs):
        token_type_ids = kwargs.get("token_type_ids", None)
        # only last token for inputs_ids if past is defined in kwargs
        if past:
            input_ids = input_ids[:, -1].expand_dims(-1)
            if token_type_ids is not None:
                token_type_ids = token_type_ids[:, -1].expand_dims(-1)

        attention_mask = kwargs.get("attention_mask", None)
        position_ids = kwargs.get("position_ids", None)

        if attention_mask is not None and position_ids is None and len(attention_mask.shape) < 3:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1   
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past:
                
                position_ids = position_ids[:, -1].expand_dims(-1)
        else:
            position_ids = None
        return {
            "input_ids": input_ids,
            "past_key_values": past,
            "use_cache": kwargs.get("use_cache"),
            "position_ids": position_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
        }
        

    @staticmethod
    def _update_model_kwargs_for_generation(
        outputs, model_kwargs, is_encoder_decoder: bool = False
    ):
        # update past
        if "past_key_values" in outputs:
            model_kwargs["past"] = outputs.past_key_values
        elif "mems" in outputs:
            model_kwargs["past"] = outputs.mems
        elif "past_buckets_states" in outputs:
            model_kwargs["past"] = outputs.past_buckets_states
        else:
            model_kwargs["past"] = None

        # update token_type_ids with last value
        if "token_type_ids" in model_kwargs:
            token_type_ids = model_kwargs["token_type_ids"]           
    
            last = ops.ExpandDims(token_type_ids[:,-1],-1)
            model_kwargs["token_type_ids"] = ops.cat([token_type_ids, last],axis=-1)

        # update attention mask
        if not is_encoder_decoder:
            if "attention_mask" in model_kwargs and len(model_kwargs['attention_mask'].shape) < 3:
                attention_mask = model_kwargs["attention_mask"]
            
                ones = ms.Tensor(attention_mask.shape[0],1,dtype=attention_mask.dtype)
                attention_mask = ms.ops.Concat((attention_mask, ones), axis=-1)
                model_kwargs["attention_mask"] = attention_mask
        if 'attention_mask' in model_kwargs:
            model_kwargs.pop('attention_mask')

        return model_kwargs
    
class SequencePipeline(object):
    """
    Pipeline processor of whole sequence .
    """
    def __init__(self, ptm_path:str, beam_size, empty_token, stop_token, max_length=50):
        self.model: AttnMaskGPT2LMHeadModel = AttnMaskGPT2LMHeadModel.from_pretrained(ptm_path)
        self.model.to_device(ms.context.device())
        self.beam_size = beam_size
        self.empty_token = empty_token
        self.stop_token = stop_token
        self.max_length = max_length

    def get_loss(self, x, decoder_output):   
        x = ms.Tensor(x)
        y = ms.Tensor(decoder_output)

        # batch_size = x.shape[0]

        output = self.model(x[:,:-1],y)

        loss = nn.CrossEntropyLoss()(output,y)
        return loss

    def train_batch(self, batch_dict):
        enc_in = ms.Tensor(batch_dict['enc_in'])
        gpt_context = ms.Tensor(batch_dict['gpt_context'])
        x = ms.cat((enc_in,gpt_context),-1)

        dec_out = ms.Tensor(batch_dict['dec_out'])

        lm_loss = self.get_loss(x, dec_out)

        return {
            'lm_loss': lm_loss,
            # 'level_loss': level_loss,
            # 'chain_loss': chain_loss
        }

    def decode_batch(self, batch_dict, prefix_mode=''):
        # x, batch_size, max_dec_len, decoder_len, decoder_input, decoder_output = self.extract_input(batch_dict)
        # relation_d = batch_dict['drelation']    # [bs, key_length, key_length]
        if prefix_mode != '':
            prefix_mode += '_'
        
        enc_in = ms.Tensor(batch_dict[f'{prefix_mode}enc_in'])
        gpt_context = ms.Tensor(batch_dict['ChatMindAi_context'])
        x = ms.cat((enc_in, gpt_context),-1)
        
        enc_len = x.shape[1]

        generated = self.model.generate(
            x,
            max_length=self.max_length + 50,
            num_beams=self.beam_size,
            pad_token_id=self.stop_token,          
        )
        return generated[:, enc_len:]
