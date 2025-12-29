import requests
from utils.vl_utils import make_interleave_content
import json

class ChoicesNotRetrievedError(Exception):
    def __init__(self, message="Exceeded maximum retries without getting 'choices' in the response."):
        self.message = message
        super().__init__(self.message)


def load_model(model_name="GPT4", base_url="", api_key="", model="gpt-4-turbo-preview"):
    model_components = {}
    model_components['model_name'] = model_name
    model_components['model'] = model
    model_components['base_url'] = base_url
    model_components['api_key'] = api_key
    return model_components

def request_with_interleave_content(interleave_content, base_url="", api_key="", model="", model_name=None, timeout=60):
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
        
    payload = {
        "model": model,
        "messages": interleave_content,
        "max_tokens": 8192
        }
    
    # 最大重试次数
    max_retries = 5
    retry_count = 0
    
    while retry_count < max_retries:
        try:
            response = requests.post(base_url, headers=headers, json=payload, stream=False)
            response.raise_for_status()
            response = response.json()

            if "choices" in response:
                print("request success")
                break
            else:
                print(f"Failed to obtain the 'choices' field in the {retry_count + 1}th request. Retrying...")
        except requests.RequestException as e:
            # print(response.json())
            print(f"An error occurred during the request: {e}")
        except ValueError as e:
            print(f"The response content is not in valid JSON format: {e}")
        
        retry_count += 1

    if retry_count >= max_retries and "choices" not in response:
        print("Exceeded maximum retries without getting 'choices' in the response.")
        raise ChoicesNotRetrievedError()
    
    response = response["choices"][0]["message"]["content"]
    return response

def infer(prompts, text_only, **kwargs):
    model = kwargs.get('model')
    base_url = kwargs.get('base_url')
    api_key = kwargs.get('api_key')
    model_name = kwargs.get('model_name', None)

    def get_image_tokens(images):
        return ["<|image|>" + img if "<|image|>" not in img else img for img in images]

    def build_user_message(prompt, text_only):
        question = prompt["prompt"]
        images = get_image_tokens(prompt.get("image", []))
        content = make_interleave_content([question] if text_only else [question] + images)
        return {"role": "user", "content": content}
    
    responses = []
    for prompt in prompts:
        messages = []

        if "conversations" in prompt:  # few-shot setting
            for idx, data in enumerate(prompt["conversations"]):
                messages.append(build_user_message(data, text_only))
                if idx != len(prompt["conversations"]) - 1 and "response" in data:
                    messages.append({"role": "assistant", "content": data["response"]})
        else:
            messages.append(build_user_message(prompt, text_only))


        response = request_with_interleave_content(messages, base_url, api_key, model, model_name)
        responses.append(response)

    return responses