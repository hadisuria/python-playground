from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load pre-trained GPT-2 model and tokenizer
model_name = "gpt2-medium"  # You can choose different sizes: "gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)


# Input text
exit_text = "exit"
input_text = ""
i = 1

while input_text != exit_text:
  # Input text
  print("\n")
  print("Iteration: " + str(i))
  print("\n")
  input_text = input("Enter short story: ")
  print("\n\n")

  if input_text != exit_text:
    # Tokenize input text
    input_ids = tokenizer.encode(input_text, return_tensors="pt")

    # Generate text
    max_length = 100  # Maximum length of ge`nerated text
    output = model.generate(input_ids, max_length=max_length, num_return_sequences=1, no_repeat_ngram_size=2, top_k=50, temperature=0.7)

    # Decode generated text
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

    print("\n\n")
    print("Generated text:")
    print(generated_text)
    print("\n")

  i += 1
