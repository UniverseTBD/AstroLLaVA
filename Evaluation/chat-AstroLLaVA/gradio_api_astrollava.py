from gradio_client import Client, handle_file

def main():
    client = Client("UniverseTBD/astroLLaVA")

    print("Choose a command to run:")
    print("1. Upvote the last response")
    print("2. Downvote the last response")
    print("3. Flag the last response")
    print("4. Regenerate response")
    print("5. HTTP bot interaction")
    print("6. Clear history")
    print("7. Add text with image")
    
    choice = int(input("Enter the number of your choice: "))

    if choice == 1:
        result = client.predict(model_selector=None, api_name="/upvote_last_response")
        print("Result:", result)
    elif choice == 2:
        result = client.predict(model_selector=None, api_name="/downvote_last_response")
        print("Result:", result)
    elif choice == 3:
        result = client.predict(model_selector=None, api_name="/flag_last_response")
        print("Result:", result)
    elif choice == 4:
        image_process_mode = input("Enter image process mode (Crop, Resize, Pad, Default): ") or "Default"
        result = client.predict(image_process_mode=image_process_mode, api_name="/regenerate")
        print("Result:", result)
    elif choice == 5:
        temperature = float(input("Enter temperature (default 0.2): ") or 0.2)
        top_p = float(input("Enter top_p (default 0.7): ") or 0.7)
        max_new_tokens = int(input("Enter max_new_tokens (default 512): ") or 512)
        result = client.predict(model_selector=None, temperature=temperature, top_p=top_p, max_new_tokens=max_new_tokens, api_name="/http_bot")
        print("Result:", result)
    elif choice == 6:
        result = client.predict(api_name="/clear_history")
        print("Result:", result)
    elif choice == 7:
        text = input("Enter the text: ")
        image_path = input("Enter the path to the image file: ")
        image_process_mode = input("Enter image process mode (Crop, Resize, Pad, Default): ") or "Default"
        result = client.predict(text=text, image=handle_file(image_path), image_process_mode=image_process_mode, api_name="/add_text")
        print("Result:", result)
    else:
        print("Invalid choice. Please run the script again.")

if __name__ == "__main__":
    main()
