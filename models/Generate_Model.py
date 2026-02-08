from torch import nn
import torch
from models.Temporal_Model import *
from models.Prompt_Learner import *
from models.Text import class_descriptor_5_only_face, class_descriptor_daisee
from models.Adapter import Adapter
from clip import clip
import itertools
import math

class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.wq = nn.Linear(dim, dim, bias=qkv_bias)
        self.wk = nn.Linear(dim, dim, bias=qkv_bias)
        self.wv = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, y):
        # x: Query source (e.g., Face) [B, 1, C] or [B, C]
        # y: Key/Value source (e.g., Body) [B, 1, C] or [B, C]
        
        # Ensure dimensions are [B, 1, C]
        if x.dim() == 2: x = x.unsqueeze(1)
        if y.dim() == 2: y = y.unsqueeze(1)
            
        B, N, C = x.shape
        
        q = self.wq(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = self.wk(y).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = self.wv(y).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x.squeeze(1)

class GenerateModel(nn.Module):

    def __init__(self, input_text, clip_model, args):

        super().__init__()

        self.args = args

        

        self.is_ensemble = any(isinstance(i, list) for i in input_text)

        

        if self.is_ensemble:

            self.num_classes = len(input_text)

            self.num_prompts_per_class = len(input_text[0])

            self.input_text = list(itertools.chain.from_iterable(input_text))

            print(f"=> Using Prompt Ensembling with {self.num_prompts_per_class} prompts per class.")

        else:

            self.input_text = input_text



        # [LUỒNG 3.2.1: TEXT BRANCH]

        # Khởi tạo Prompt Learner (CoOp)

        self.prompt_learner = PromptLearner(self.input_text, clip_model, args)

        self.tokenized_prompts = self.prompt_learner.tokenized_prompts

        self.text_encoder = TextEncoder(clip_model)

        self.dtype = clip_model.dtype

        self.image_encoder = clip_model.visual



        # [LUỒNG 3.2.2: VISUAL ADAPTER]

        # Khởi tạo Face Adapter (Efficient Fine-tuning)

        self.face_adapter = Adapter(c_in=512, reduction=4)



        # For MI Loss

        if args.dataset.strip() == 'DAISEE':

            hand_crafted_prompts = class_descriptor_daisee

        else:

            hand_crafted_prompts = class_descriptor_5_only_face

            

        self.tokenized_hand_crafted_prompts = torch.cat([clip.tokenize(p) for p in hand_crafted_prompts])

        with torch.no_grad():

            embedding = clip_model.token_embedding(self.tokenized_hand_crafted_prompts).type(self.dtype)

        self.register_buffer("hand_crafted_prompt_embeddings", embedding)



        # [LUỒNG 3.2.3: TEMPORAL MODULE]

        if hasattr(args, 'temporal_type') and args.temporal_type == 'cls':

            print("=> Using Temporal_Transformer_Cls (Baseline style)")

            TemporalClass = Temporal_Transformer_Cls

        else:

            print("=> Using Temporal_Transformer_AttnPool (Proposed style)")

            TemporalClass = Temporal_Transformer_AttnPool



        self.temporal_net = TemporalClass(num_patches=16,

                                                     input_dim=512,

                                                     depth=args.temporal_layers,

                                                     heads=8,

                                                     mlp_dim=1024,

                                                     dim_head=64)

        

        self.temporal_net_body = TemporalClass(num_patches=16,

                                                     input_dim=512,

                                                     depth=args.temporal_layers,

                                                     heads=8,

                                                     mlp_dim=1024,

                                                     dim_head=64)

        self.clip_model_ = clip_model

        

        # [UPGRADE] Cross Attention Fusion instead of simple concat + FC

        print("=> Using Cross-Attention Fusion")

        self.cross_attn = CrossAttention(dim=512)

        self.project_fc = nn.Identity()



        # [UPGRADE] MoCo: Momentum Encoder & Queue

        print("=> Initializing MoCo components...")

        self.m = 0.999 # Momentum

        self.K = 65536 # Queue size

        self.T = 0.07  # Temperature



        # Create Momentum Encoder (Key Encoder) - Copy of Query Encoder

        # We need to copy image_encoder, face_adapter, temporal_nets, cross_attn

        # For simplicity, we create a deep copy of the VISUAL components

        # Note: In practice, we update params manually.

        import copy

        self.image_encoder_k = copy.deepcopy(self.image_encoder)

        self.face_adapter_k = copy.deepcopy(self.face_adapter)

        self.temporal_net_k = copy.deepcopy(self.temporal_net)

        self.temporal_net_body_k = copy.deepcopy(self.temporal_net_body)

        self.cross_attn_k = copy.deepcopy(self.cross_attn)



        # Freeze Key Encoder params

        for param in self.image_encoder_k.parameters(): param.requires_grad = False

        for param in self.face_adapter_k.parameters(): param.requires_grad = False

        for param in self.temporal_net_k.parameters(): param.requires_grad = False

        for param in self.temporal_net_body_k.parameters(): param.requires_grad = False

        for param in self.cross_attn_k.parameters(): param.requires_grad = False



        # Create Queue

        self.register_buffer("queue", torch.randn(512, self.K))

        self.queue = nn.functional.normalize(self.queue, dim=0)

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))



    @torch.no_grad()

    def _momentum_update_key_encoder(self):

        """

        Momentum update of the key encoder

        """

        for param_q, param_k in zip(self.image_encoder.parameters(), self.image_encoder_k.parameters()):

            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

        for param_q, param_k in zip(self.face_adapter.parameters(), self.face_adapter_k.parameters()):

            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

        for param_q, param_k in zip(self.temporal_net.parameters(), self.temporal_net_k.parameters()):

            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

        for param_q, param_k in zip(self.temporal_net_body.parameters(), self.temporal_net_body_k.parameters()):

            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

        for param_q, param_k in zip(self.cross_attn.parameters(), self.cross_attn_k.parameters()):

            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)



    @torch.no_grad()

    def _dequeue_and_enqueue(self, keys):

        # Gather keys before updating queue (if multi-gpu, not needed for single GPU/MPS but good practice)

        # keys = concat_all_gather(keys) 

        

        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)

        assert self.K % batch_size == 0  # for simplicity



        # Replace the keys at ptr (dequeue and enqueue)

        self.queue[:, ptr:ptr + batch_size] = keys.T

        ptr = (ptr + batch_size) % self.K  # move pointer



        self.queue_ptr[0] = ptr



    def forward_key(self, image_face, image_body):

        # Compute Key Features using Momentum Encoders

        with torch.no_grad(): # No gradient to keys

            # Shuffle BN? Not strictly needed for ViT but good practice (skipped for simplicity on MPS)

            

            # 1. Face

            n, t, c, h, w = image_face.shape

            image_face_reshaped = image_face.contiguous().view(-1, c, h, w)

            k_face = self.image_encoder_k(image_face_reshaped.type(self.dtype))

            k_face = self.face_adapter_k(k_face)

            k_face = k_face.contiguous().view(n, t, -1)

            k_face = self.temporal_net_k(k_face)

            

            # 2. Body

            image_body_reshaped = image_body.contiguous().view(-1, c, h, w)

            k_body = self.image_encoder_k(image_body_reshaped.type(self.dtype))

            k_body = k_body.contiguous().view(n, t, -1)

            k_body = self.temporal_net_body_k(k_body)

            

            # 3. Fusion

            k_feat = self.cross_attn_k(k_face, k_body)

            k_feat = nn.functional.normalize(k_feat, dim=1)

            

            return k_feat



    def forward(self, image_face, image_body):

        # [LUỒNG 7: FORWARD PASS]

        

        ################# Visual Part (Query) #################

        # 1. Face Part

        n, t, c, h, w = image_face.shape

        image_face_reshaped = image_face.contiguous().view(-1, c, h, w)

        image_face_features = self.image_encoder(image_face_reshaped.type(self.dtype))

        

        # Apply Face Adapter Only if Enabled

        if not hasattr(self.args, 'use_adapter') or self.args.use_adapter == 'True':

            image_face_features = self.face_adapter(image_face_features) 

            

        image_face_features = image_face_features.contiguous().view(n, t, -1)

        video_face_features = self.temporal_net(image_face_features)

        

        # 2. Body Part

        n, t, c, h, w = image_body.shape

        image_body_reshaped = image_body.contiguous().view(-1, c, h, w)

        image_body_features = self.image_encoder(image_body_reshaped.type(self.dtype))

        image_body_features = image_body_features.contiguous().view(n, t, -1)

        video_body_features = self.temporal_net_body(image_body_features)



        # 3. Cross Attention Fusion

        video_features = self.cross_attn(video_face_features, video_body_features)

        video_features = video_features / video_features.norm(dim=-1, keepdim=True)



        ################# MoCo Logic (Only during training) #################

        moco_logits = None

        moco_labels = None

        if self.training:

            # Update momentum encoder

            self._momentum_update_key_encoder()

            

            # Compute key features

            k = self.forward_key(image_face, image_body)

            

            # Compute logits: Einstein sum is more intuitive

            # Positive logits: Nx1

            l_pos = torch.einsum('nc,nc->n', [video_features, k]).unsqueeze(-1)

            # Negative logits: NxK

            l_neg = torch.einsum('nc,ck->nk', [video_features, self.queue.clone().detach()])



            # Logits: Nx(1+K)

            moco_logits = torch.cat([l_pos, l_neg], dim=1)

            moco_logits /= self.T # Apply temperature

            

            # Labels: positives are the 0-th

            moco_labels = torch.zeros(moco_logits.shape[0], dtype=torch.long).to(self.args.device)

            

            # Update queue

            self._dequeue_and_enqueue(k)



        ################# Text Part ###################

        # 4. Learnable prompts (Context Vectors + Class Names)

        prompts = self.prompt_learner()

        tokenized_prompts = self.tokenized_prompts

        text_features = self.text_encoder(prompts, tokenized_prompts)

        text_features = text_features / text_features.norm(dim=-1, keepdim=True)



        # 5. Hand-crafted prompts (Only for MI Loss, not for prediction)

        hand_crafted_prompts = self.hand_crafted_prompt_embeddings

        tokenized_hand_crafted_prompts = self.tokenized_hand_crafted_prompts.to(hand_crafted_prompts.device)

        hand_crafted_text_features = self.text_encoder(hand_crafted_prompts, tokenized_hand_crafted_prompts)

        hand_crafted_text_features = hand_crafted_text_features / hand_crafted_text_features.norm(dim=-1, keepdim=True)



        ################# Classification ###################

        # 6. Calculate Similarity (Video <-> Text)

        if self.is_ensemble:

            text_features = text_features.view(self.num_classes, self.num_prompts_per_class, -1)

            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

            logits = torch.einsum('bd,cpd->bcp', video_features, text_features)

            output = torch.mean(logits, dim=2) / self.args.temperature

        else:

            output = video_features @ text_features.t() / self.args.temperature



        # Return moco_logits and moco_labels for loss calculation in Trainer

        return output, text_features, hand_crafted_text_features, video_features, moco_logits, moco_labels