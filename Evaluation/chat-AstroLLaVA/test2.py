from gradio_client import Client, file

client = Client("UniverseTBD/astroLLaVA")

client.predict(
  text=None,
  image=hi,
  image_process_mode=None,
  api_name="/add_text"
)

client.predict(
  model_selector=None,
  temperature=AstroLLaVA_v2-4bit-lora,
  top_p=0.2,
  max_new_tokens=0.7,
  api_name="/http_bot_1"
)
