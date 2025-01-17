from gradio_client import Client

def prompt_model():
    # Initialize the client with the Space name
    client = Client("UniverseTBD/astroLLaVA")

    # Get user input
    text_prompt = input("Enter your text prompt: ")
    temperature = float(input("Enter temperature (default 0.2): ") or 0.2)
    top_p = float(input("Enter top_p (default 0.7): ") or 0.7)
    max_new_tokens = int(input("Enter max_new_tokens (default 512): ") or 512)

    # Send the prompt to the model
    result = client.predict(
        model_selector=None,
        temperature=temperature,
        top_p=top_p,
        max_new_tokens=max_new_tokens,
        api_name="/http_bot"
    )

    # Print the response
    print("Model Response:", result)

if __name__ == "__main__":
    prompt_model()
