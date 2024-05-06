import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

def predict_gender(names):
    # Load fine-tuned model and tokenizer
    model_name = "bert-base-uncased"  # You can choose different pre-trained models
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)

    # Encode names
    encoded_inputs = tokenizer(names, padding=True, truncation=True, return_tensors="pt")

    # Predict gender
    with torch.no_grad():
        outputs = model(**encoded_inputs)
        predicted_labels = torch.argmax(outputs.logits, dim=1)

    # Decode predictions
    predicted_genders = ["Female" if label.item() == 1 else "Male" for label in predicted_labels]
    return predicted_genders

def main():
    # # Test names
    # names = ["John", "Mary", "Michael", "Anna", "David", "Jennifer", "Joseph", "Jessica", "Robert", "Elizabeth"]

    # # Predict gender for names
    # predicted_genders = predict_gender(names)

    # # Display results
    # for name, gender in zip(names, predicted_genders):
    #     print(f"{name} is likely to be {gender}.")

  # Input text
  exit_text = "exit"
  input_text = ""
  i = 1

  while input_text != exit_text:
    # Input text
    print("\n")
    print("Iteration: " + str(i))
    print("\n")
    input_text = input("Enter name: ")
    print("\n\n")

    if input_text != exit_text:
      # Tokenize input text
      names = [input_text]

      # Generate text
      max_length = 100  # Maximum length of ge`nerated text
      predicted_genders = predict_gender(names)

      # Decode generated text
      for name, gender in zip(names, predicted_genders):
        print(f"{name} is likely to be {gender}.")

      # print("\n\n")
      # print("Generated text:")
      # print(generated_text)
      # print("\n")

    i += 1



if __name__ == "__main__":
    main()

