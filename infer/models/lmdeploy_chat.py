from lmdeploy import pipeline, TurbomindEngineConfig, GenerationConfig, ChatTemplateConfig
from lmdeploy.vl.constants import IMAGE_TOKEN
import base64
import os

def encode_image_base64(image_path):
    with open(image_path, "rb") as image_file:
        encoded = base64.b64encode(image_file.read()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"
    
def load_model(model_name, model_args, use_accel=False):
    model_path = model_args.get('model_path_or_name')
    tp = model_args.get('tp', 8)
    model_components = {}

    if os.environ.get('LMDEPLOY_USE_MODELSCOPE'):
       from modelscope import AutoModelForCausalLM, AutoTokenizer
    else:
       from transformers import AutoTokenizer, AutoModelForCausalLM 
    if use_accel:
        model_components['use_accel'] = True
        # model_components['chat_template'] = get_chat_template_from_config(model_path)
        model_components['tokenizer'] = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model_components['model'] = pipeline(model_path,backend_config=TurbomindEngineConfig(tp=tp,session_len=32768))
        # model_components['model'] = LLM(model=model_path, tokenizer=model_path, gpu_memory_utilization=0.95, tensor_parallel_size=tp, trust_remote_code=True, disable_custom_all_reduce=True, enforce_eager=True)
        model_components['model_name'] = model_name
    else:
        print ('use_accel1 pending...')
    return model_components

def infer(prompts, text_only, **kwargs): 
    model = kwargs.get('model')
    tokenizer = kwargs.get('tokenizer', None)
    # chat_template = kwargs.get('chat_template', None)
    model_name = kwargs.get('model_name', None)
    use_accel = kwargs.get('use_accel', False)
    max_new_tokens = 2048
    gen_config = GenerationConfig(max_new_tokens=max_new_tokens, temperature=0.0, top_p=0.95)

    responses = []

    for prompt in prompts:
        messages = []

        if "conversations" in prompt:  # few-shot setting
            for idx, data in enumerate(prompt["conversations"]):
               content = []
               text = data['prompt']
               if 'deepseek' in (model_name or '').lower():
                   text = f'{IMAGE_TOKEN}' + data['prompt']
               content.append(dict(type='text', text=text))

               if not text_only:
                    if 'deepseek' in (model_name or '').lower():
                        for image_path in data.get("image", []):
                            base64_image = encode_image_base64(image_path)
                            content.append({
                                "type": "image",
                                "image": {
                                    "url": base64_image,
                                    "detail": "low"
                                }
                            })
                    else:
                        for image_path in data.get("image", []):
                            base64_image = encode_image_base64(image_path)
                            content.append({
                                "type": "image_url",
                                "image_url": {
                                    "url": base64_image,
                                    "detail": "low"
                                }
                            })

               messages.append(dict(role='user', content=content))

               if idx != len(prompt["conversations"]) - 1 and "response" in data:
                    messages.append(dict(role='assistant', content=data["response"]))

        else:
            # Zero-shot
            content = []
            text = prompt['prompt']
            if 'deepseek' in (model_name or '').lower():
                text = f'{IMAGE_TOKEN}' + prompt['prompt']
            content.append(dict(type='text', text=text))
            if not text_only:
                for image_path in prompt.get("image", []):
                    base64_image = encode_image_base64(image_path)
                    content.append({
                        "type": "image_url",
                        "image_url": {
                            "url": base64_image,
                            "detail": "low"
                        }
                    })
            messages.append(dict(role='user', content=content))
        
        if use_accel:
            out = model(messages, gen_config=gen_config)
            responses.append(out.text)
        else:
            responses.append("")

    return responses


if __name__ == '__main__':
    pass