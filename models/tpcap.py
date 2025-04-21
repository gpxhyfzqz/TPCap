import logging
import torch
from torch.cuda.amp import autocast as autocast
import torch.nn as nn
import random
from models.blip2 import Blip2Base, disabled_train
from models.modeling_llama import LlamaForCausalLM
from transformers import LlamaTokenizer
import pickle
import re
import os
from search import greedy_search

class TPCap(Blip2Base):
    
    def __init__(
        self,
        vit_model="eva_clip_g",
        q_former_model="https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/BLIP2/blip2_pretrained_flant5xxl.pth",
        img_size=224,
        drop_path_rate=0,
        use_grad_checkpoint=False,
        vit_precision="fp16",
        freeze_vit=True,
        freeze_qformer=True,
        num_query_token=32,
        num_query_token_txt=8,
        topn=9,
        llama_model="",
        prompt_path="prompts/prompt1.txt",
        prompt_path2="prompts/prompt2.txt",
        prompt_template='###Human: {} ###Assistant: ',
        max_txt_len=160,
        end_sym='\n',
        low_resource=False,
        device_8bit=0,
    ):
        super().__init__()

        self.low_resource = low_resource
        self.topn = topn
        print('topn:', self.topn)

        ##### Image 
        print('Loading VIT')
        self.visual_encoder, self.ln_vision = self.init_vision_encoder(
            vit_model, img_size, drop_path_rate, use_grad_checkpoint, vit_precision
        )
        if freeze_vit:
            for name, param in self.visual_encoder.named_parameters():
                param.requires_grad = False
            self.visual_encoder = self.visual_encoder.eval()
            self.visual_encoder.train = disabled_train
            for name, param in self.ln_vision.named_parameters():
                param.requires_grad = False
            self.ln_vision = self.ln_vision.eval()
            self.ln_vision.train = disabled_train
            logging.info("freeze vision encoder")
        print('Loading VIT Done')

        print('Loading mm_proj and prep linear')
        model_path = 'llava-pretrain-vicuna-7b-v1.3/'   ### your llava pretrain projector path
        mm_projector_weights = torch.load(os.path.join(model_path, 'mm_projector.bin'), map_location='cpu')
        mm_projector_weights = {k: v.to(torch.float16) for k, v in mm_projector_weights.items()}
        self.mm_proj = torch.nn.Linear(in_features=mm_projector_weights['model.mm_projector.weight'].shape[1], out_features=mm_projector_weights['model.mm_projector.weight'].shape[0])
        self.mm_proj.weight.data = mm_projector_weights['model.mm_projector.weight'].clone()  # copy weights
        if 'bias' in mm_projector_weights:
            self.mm_proj.bias.data = mm_projector_weights['model.mm_projector.bias'].clone()  # copy bias
        for param in self.mm_proj.parameters():
            param.requires_grad = False        
        self.linear = nn.Linear(768, 1024)
        print('Loading mm_proj and prep linear Done')

        print('Loading Q-Former')
        self.Qformer, self.query_tokens = self.init_Qformer(
            num_query_token, self.visual_encoder.num_features
        )
        self.Qformer.cls = None
        self.Qformer.bert.embeddings.word_embeddings = None
        self.Qformer.bert.embeddings.position_embeddings = None
        for layer in self.Qformer.bert.encoder.layer:
            layer.output = None
            layer.intermediate = None
        self.load_from_pretrained(url_or_filename=q_former_model)

        if freeze_qformer:
            for name, param in self.Qformer.named_parameters():
                param.requires_grad = False
            self.Qformer = self.Qformer.eval()
            self.Qformer.train = disabled_train
            self.query_tokens.requires_grad = True
            logging.info("freeze Qformer")
        print('Loading Q-Former Done')


        ##### Text
        print('Loading Multi-Modal Purification')
        self.bert_tokenizer = self.init_tokenizer()
        self.Mp, self.query_tokens_txt = self.init_MP(
            num_query_token_txt, self.Qformer.config.hidden_size
        )
        self.Mp.resize_token_embeddings(len(self.bert_tokenizer))
        self.Mp.cls = None
        self.load_from_pretrained(url_or_filename=q_former_model)
        if freeze_qformer:
            for name, param in self.Mp.named_parameters():
                param.requires_grad = False
            self.Mp = self.Mp.eval()
            self.Mp.train = disabled_train
            self.query_tokens_txt.requires_grad = True
            logging.info("freeze Qformer")
        print('query_tokens_txt', self.query_tokens_txt.shape)
        print('Loading Multi-Modal Purification Done')


        ##### Caption generation 
        print('Loading LLAMA')
        self.llama_tokenizer = LlamaTokenizer.from_pretrained(llama_model, use_fast=False)
        self.llama_tokenizer.pad_token = self.llama_tokenizer.eos_token

        if self.low_resource:
            self.llama_model = LlamaForCausalLM.from_pretrained(
                llama_model,
                torch_dtype=torch.float16,
                load_in_8bit=True,
                device_map={'': device_8bit}
            )
        else:
            self.llama_model = LlamaForCausalLM.from_pretrained(
                llama_model,
                torch_dtype=torch.float16,
            )

        for name, param in self.llama_model.named_parameters():
            param.requires_grad = False
        print('Loading LLAMA Done')

        self.max_txt_len = max_txt_len
        self.end_sym = end_sym

        if prompt_path:
            with open(prompt_path, 'r') as f:
                raw_prompts = f.read().splitlines()
            filted_prompts = [raw_prompt for raw_prompt in raw_prompts if "<ImageHere>" in raw_prompt]
            self.prompt_list = [prompt_template.format(p) for p in filted_prompts]
            print('Load {} training prompts'.format(len(self.prompt_list)))
            print('Prompt Example \n{}'.format(random.choice(self.prompt_list)))
        else:
            self.prompt_list = []
        if prompt_path2:
            with open(prompt_path2, 'r') as f:
                raw_prompts = f.read().splitlines()
            filted_prompts = [raw_prompt for raw_prompt in raw_prompts if "<ImageHere>" in raw_prompt]
            self.prompt_list2 = [prompt_template.format(p) for p in filted_prompts]
            print('Load {} training prompts'.format(len(self.prompt_list2)))
            print('Prompt Example \n{}'.format(random.choice(self.prompt_list2)))
        else:
            self.prompt_list2 = []


    def vit_to_cpu(self):
        self.ln_vision.to("cpu")
        self.ln_vision.float()
        self.visual_encoder.to("cpu")
        self.visual_encoder.float()


    def prompt_wrap(self, img_embeds, atts_img, prompt_list):
        if prompt_list:
            batch_size = img_embeds.shape[0]
            emb_lists = []
            for i in range(batch_size):
                prompt = random.choice(prompt_list)
                p_before, p_after = prompt.split("<ImageHere>", 1)
                self.llama_tokenizer.padding_side = "right"
                p_before_tokens = self.llama_tokenizer(
                    p_before, return_tensors="pt", add_special_tokens=False).to(img_embeds.device)
                p_after_tokens = self.llama_tokenizer(
                    p_after, return_tensors="pt", add_special_tokens=False).to(img_embeds.device)        
                p_before_embeds = self.llama_model.model.embed_tokens(p_before_tokens.input_ids)
                p_after_embeds = self.llama_model.model.embed_tokens(p_after_tokens.input_ids)
                img_embeds_i = img_embeds[i].unsqueeze(0)
                wrapped_embed_i = torch.cat([p_before_embeds, img_embeds_i, p_after_embeds], dim=1)
                emb_lists.append(wrapped_embed_i)  

            emb_lens = [emb.shape[1] for emb in emb_lists]
            pad_emb = self.llama_model.model.embed_tokens(torch.tensor(self.llama_tokenizer.pad_token_id, device=img_embeds.device))
            wrapped_embs = pad_emb.expand(len(emb_lens), max(emb_lens), -1).clone()
            wrapped_atts = torch.zeros([len(emb_lens), max(emb_lens)], dtype=torch.int, device=img_embeds.device)
            for i, emb in enumerate(emb_lists):
                wrapped_embs[i, :emb_lens[i]] = emb
                wrapped_atts[i, :emb_lens[i]] = 1
            return wrapped_embs, wrapped_atts
        else:
            return img_embeds, atts_img


    def pre_name(self, caption):
        caption = re.sub(
            r"([_!,'\"()*#:;~])",
            " ",
            caption.lower(),
        )
        caption = re.sub(
            r"\s{2,}",
            " ",
            caption,
        )
        caption = caption.rstrip("\n")
        caption = caption.strip(" ")
        return caption



    def encode_img(self, image):
        device = image.device
        if self.low_resource:
            self.vit_to_cpu()
            image = image.to("cpu")

        with self.maybe_autocast():
            image_embeds = self.ln_vision(self.visual_encoder(image)).to(device)
            image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(device)

            query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
            query_outputs_img = self.Qformer.bert(
                query_embeds=query_tokens,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )
            query_output_img = query_outputs_img.last_hidden_state
            query_output_img_atts = torch.ones(query_output_img.size()[:-1], dtype=torch.long).to(device)

            align_txt = self.linear(query_output_img)
            align_txt = self.mm_proj(align_txt)
            prompt_embeds, atts_prompt = self.prompt_wrap(align_txt, query_output_img_atts, self.prompt_list2)
            bos = torch.ones([align_txt.shape[0], 1],
                device=align_txt.device) * self.llama_tokenizer.bos_token_id
            bos = bos.long()
            bos_embeds = self.llama_model.model.embed_tokens(bos)
            align_txt = torch.cat([bos_embeds, prompt_embeds], dim=1)
            sentence_ = greedy_search(embeddings = align_txt, tokenizer = self.llama_tokenizer, model = self.llama_model)
            print(sentence_)
            text = self.bert_tokenizer(
                    sentence_,
                    truncation=True,
                    padding="longest",
                    max_length=self.max_txt_len,
                    return_tensors="pt",
                ).to(image.device)

            query_tokens_txt = self.query_tokens_txt.expand(image_embeds.shape[0], -1, -1)
            query_atts_txt = torch.ones(query_tokens_txt.size()[:-1], dtype=torch.long).to(
                image_embeds.device
            )

            query_output_img_atts = torch.ones(query_output_img.size()[:-1], dtype=torch.long).to(device)
            query_output_img_atts = torch.cat([query_atts_txt, query_output_img_atts], dim=1)


            attention_mask = text.attention_mask
            attention_mask = attention_mask.to(device)
            text.input_ids = text.input_ids.to(device)
            query_outputs_txt = self.Mp.bert(
                text.input_ids,
                query_embeds=query_tokens_txt,
                attention_mask=attention_mask,
                encoder_hidden_states=query_output_img,
                encoder_attention_mask=query_output_img_atts,
                return_dict=True,
            )
            query_output_txt = query_outputs_txt.last_hidden_state[:, : query_tokens_txt.size(1), :]

            query_output_all = torch.cat([query_output_img, query_output_txt], dim=1) 
            qform_all_proj = self.linear(query_output_all)
            qform_all_proj = self.mm_proj(qform_all_proj)
            atts_qform_all_proj = torch.ones(qform_all_proj.size()[:-1], dtype=torch.long).to(device)
        return qform_all_proj, atts_qform_all_proj


    def forward(self, samples):
        ##### Image
        image = samples["image"]
        qform_all_proj, atts_qform_all_proj = self.encode_img(image)
        if self.prompt_list:
            prompt_embeds, atts_prompt = self.prompt_wrap(qform_all_proj, atts_qform_all_proj, self.prompt_list) #(self, img_embeds, batch_names, atts_img, prompt_list):

        ##### Caption generation
        self.llama_tokenizer.padding_side = "right"
        text = [t + self.end_sym for t in samples["text_input"]]
        text_tokens = self.llama_tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=self.max_txt_len,
            add_special_tokens=False
        ).to(image.device)


        bos = torch.ones([qform_all_proj.shape[0], 1],
                         dtype=text_tokens.input_ids.dtype,
                         device=text_tokens.input_ids.device) * self.llama_tokenizer.bos_token_id
        bos_embeds = self.llama_model.model.embed_tokens(bos)
        atts_bos = atts_qform_all_proj[:, :1]


        targets = text_tokens.input_ids.masked_fill(
            text_tokens.input_ids == self.llama_tokenizer.pad_token_id, -100
        )
        empty_targets = (
            torch.ones([qform_all_proj.shape[0], 1 + prompt_embeds.shape[1]], 
                       dtype=torch.long).to(image.device).fill_(-100)  
        )
        targets = torch.cat([empty_targets, targets], dim=1)
        text_embeds = self.llama_model.model.embed_tokens(text_tokens.input_ids)
        
        inputs_embeds = torch.cat([bos_embeds, prompt_embeds, text_embeds], dim=1)
        attention_mask = torch.cat([atts_bos, atts_prompt, text_tokens.attention_mask], dim=1)

        with self.maybe_autocast():
            outputs = self.llama_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                return_dict=True,
                labels=targets,
            )
        loss = outputs.loss

        return {"output": outputs[0], "loss": loss}
