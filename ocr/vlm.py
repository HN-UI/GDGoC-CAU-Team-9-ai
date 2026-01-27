import torch
import torch.nn as nn
from PIL import Image
from transformers import AutoModel, Gemma3ForCausalLM, AutoProcessor, AutoTokenizer, BitsAndBytesConfig

'''
SigLIP2 + Gemma 3 1B 모델을 결합한 VLM 모델입니다.

Gemma 3 270M, 1B는 경량화 모델이지만, 기본적으로 멀티모달 구조가 아니기 때문에 이미지를 입력으로 받을 수 없습니다.
따라서 비전 인코더를 추가하여 이미지를 임베딩 토큰으로 변환한 후, 프로젝터를 통해 Gemma 3의 언어 모델이 처리할 수 있는 차원으로 매칭해야 합니다.
이를 위해 SigLIP2의 비전 인코더를 사용하였으며, Gemma 3의 언어 모델과 차원 수를 맞추기 위해 선형 프로젝터를 추가하였습니다.
'''

class VLM(nn.Module):
    def __init__(self, vision_model_id, language_model_id):
        super().__init__()
        self.template = ""

        # vision encoder (frozen)
        self.vision_encoder = AutoModel.from_pretrained(
            vision_model_id, local_files_only=True,
        )
        self.vision_processor = AutoProcessor.from_pretrained(
            vision_model_id, local_files_only=True
        )

        # language model (frozen)
        self.language_model = Gemma3ForCausalLM.from_pretrained(
            language_model_id, local_files_only=True,
        )
        self.language_processor = AutoTokenizer.from_pretrained(
            language_model_id, local_files_only=True
        )

        # projector (for matching dimensions)
        self.projector = nn.Linear(
            self.vision_encoder.config.vision_config.hidden_size,
            self.language_model.config.hidden_size,
            device=self.language_model.device,
            dtype=self.language_model.dtype
        )

        # for alignment
        self.image_token = nn.Parameter(
            torch.randn(1, 1, self.language_model.config.hidden_size)
        )

        # freeze vision encoder and language model
        for p in self.vision_encoder.parameters():
            p.requires_grad = False

        for p in self.language_model.parameters():
            p.requires_grad = False

    def preprocess(self, image: Image.Image = None, text: str = None):
        '''
        Preprocesses the input image and text.

        Args:
            image: Input image
            text: Input text prompt

        Returns:
            pixel_values: Preprocessed image tensor
            input_ids: Token IDs for the text input
        '''
        image_tokens, text_tokens = None, None

        # preprocess images
        if image is not None:
            image_tokens = self.vision_processor(
                images=[image], 
                return_tensors="pt"
            ).to(device=self.language_model.device, dtype=self.language_model.dtype) if image is not None else None

        # preprocess text
        if text is not None:
            messages = [
                [
                    {
                        "role": "system",
                        "content": [{"type": "text", "text": "You are an OCR specialist."},]
                    },
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": text},]
                    },
                ],
            ]

            text_tokens = self.language_processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            ).to(self.language_model.device)

        return image_tokens, text_tokens

    def forward(self, image: Image.Image = None, text: str = None):
        image_tokens, text_tokens = self.preprocess(image, text)

        # convert text ids to embeddings
        text_embeds = self.language_model.get_input_embeddings()(text_tokens["input_ids"])

        if image_tokens is not None:
            image_embeddings = self.vision_encoder.get_image_features(**image_tokens)

            projected_patches = self.projector(image_embeddings)
            projected_patches = projected_patches + self.image_token

            # concat image and text embeddings
            bos = text_embeds[:, :1]
            rest = text_embeds[:, 1:]
            inputs_embeds = torch.cat([bos, projected_patches, rest], dim=1)

            # create an attention mask
            batch_size, num_patches, _ = projected_patches.shape
            image_mask = torch.ones(
                (batch_size, num_patches), 
                device=self.language_model.device,
                dtype=text_tokens["attention_mask"].dtype
            )
            attention_mask = torch.cat([image_mask, text_tokens["attention_mask"]], dim=1)
        else:
            inputs_embeds = text_embeds
            attention_mask = text_tokens["attention_mask"]

        # generate response
        generated_ids = self.language_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            max_new_tokens=64,
        )

        return self.language_processor.batch_decode(generated_ids, skip_special_tokens=True)