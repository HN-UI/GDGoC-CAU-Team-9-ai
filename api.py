import mimetypes
from google import genai
from google.genai import types
import time

class Gemma:
    def __init__(self, api_key: str, model='gemma-3-4b-it'):
        self.client = genai.Client(api_key=api_key)
        self.model = model
        self.last_latency = None
        self.last_response = None

    # 이미지 입력 생성
    def image_part_from_file(self, path: str):
        mime_type, _ = mimetypes.guess_type(path)
        if mime_type is None:
            mime_type = "image/jpeg"  # fallback

        with open(path, "rb") as f:
            data = f.read()

        return types.Part.from_bytes(data=data, mime_type=mime_type)

    # 
    def generate_content(self, contents: list, max_output_tokens: int = 800):
        if not contents:
            raise ValueError("Contents must be provided.")

        t0 = time.time()
        response = self.client.models.generate_content(
            model=self.model,
            contents=contents,
            config=types.GenerateContentConfig(
                max_output_tokens=max_output_tokens,
                temperature=0.2,
            ),
        )
        self.last_latency = time.time() - t0
        self.last_response = response
        return response.text

    def print_delay(self):
        if self.last_latency is None:
            print("No previous response.")
        else:
            print(f"Response Time: {self.last_latency:.3f} seconds")

MY_KEY = "AIzaSyC7Zgp9nWiFTkweTy2gzNXx-8xbogIxYHQ"
gemma = Gemma(api_key=MY_KEY)
img = gemma.image_part_from_file("menu.png")  # 같은 폴더에 있다고 했으니 OK

prompt = '''
Extract menu item names from the image.

Rules:
1) Include only standalone dishes sold as menu items.
2) Exclude options, sizes, add-ons, toppings, and price descriptions.
3) Output ONLY valid JSON in the format:
{ "items": ["..."] }

'''

text = gemma.generate_content([img, prompt])
print(text)