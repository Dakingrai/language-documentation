import pdb
from openai import OpenAI
from scripts import utils
import os


def prompt_gpt(prompt, prompt_input, trans_dir, file_path=""):
    results = []
    for each in prompt_input:
        content = prompt + "\n Now write the translation. If you are not sure what the translation should be, then give your best guess. Do not say that you do not speak Kulung. If your translation is wrong, that is fine, but you have to provide a translation." + each[0]
        result = {}
        result['gold_kulung'] = each[1]
        result['gold_english'] = each[2]
        message_text = [
                {"role": "system", "content": "You are a helpful assistant designed to assist a linguist in translating sentences between languages."},
                {"role": "user", "content": content}
            ]

        client = OpenAI()

        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=message_text
        )
        
        response = completion.choices[0].message.content

        if trans_dir == "english_kulung":
            result['gpt4_kulung'] = response
        else:
            result['gpt4_english'] = response
        results.append(result)

    utils.save_data(results, f"{file_path}/results.txt")

def prompt_qwen(prompt, prompt_input, trans_dir, file_path=""):
  model_name = "../../downloads/huggingface/models/qwen-7b"
  device = "cuda" # the device to load the model onto
  model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto"
    )
  tokenizer = AutoTokenizer.from_pretrained(model_name)
  results = []
  for each in prompt_input:
    result = {}
    result['gold_kulung'] = each[1]
    result['gold_english'] = each[2]
    content = prompt + each[0]

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": content}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(device)
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=512
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    print(response)
    if trans_dir == "english_kulung":
      result['gpt4_kulung'] = response
    else:
      result['gpt4_english'] = response
    results.append(result)
    pdb.set_trace()


def load_train_input(trans_dir, results_dir):
  tolsmo_grammar_input = "To help with the translation, here is a grammar book for the Kulung language. \n"
  for i in range(3, 5): #30 page159 18 is the number for parallel sentences
    file = open("data/grammar/tolsmo_grammar_" + str(i) + ".txt", "r")
    content = file.read()
    tolsmo_grammar_input += content
    file.close()
  
  if trans_dir == "kulung_english":
    english_kulung_dictionary_input = "To help with the translation, here is a dictionary for translating english words to kulung. \n"
  else:
    english_kulung_dictionary_input = "This a dictionary for translating kulung words to english. \n"
  eng_kulung = utils.load_data("data/english_kulung.json")
  test_data = load_test_input(trans_dir) # for filtering out the irrelevant keys

  for key, value in eng_kulung['english_kulung'].items():
    multiple_keys = key.split(", ")
    for each_key in multiple_keys:
      if trans_dir == "kulung_english":
        english_kulung_dictionary_input += " " + each_key + " : " + ', '.join(value) + " \n"
      else:
        english_kulung_dictionary_input += " " + ', '.join(value) + " : " + each_key + " \n"

  english_kulung_dictionary_input = english_kulung_dictionary_input[:10000]
  
  parallel_sentences_input = "To help with the translation, here is a list of parallel sentences in English and Kulung. \n"
  parallel_sentences = utils.load_data("data/parallel_sentences.json")[:50]
  for each in parallel_sentences:
    if trans_dir == "kulung_english":
      parallel_sentences_input += "Kulung: " + each['Kulung'] + "\n English: " + each['English'] + "\n"
    else:
      parallel_sentences_input += "English: " + each['English'] + "\n Kulung: " + each['Kulung'] + "\n"
      
  
  full_input = tolsmo_grammar_input+ english_kulung_dictionary_input + parallel_sentences_input
  utils.save_data(tolsmo_grammar_input, f"{results_dir}/grammar.txt")
  utils.save_data(english_kulung_dictionary_input, f"{results_dir}/dictionary.txt")
  utils.save_data(parallel_sentences_input, f'{results_dir}/parallel_sentences.txt')
  utils.save_data(full_input, f"{results_dir}/full_input.txt")
  return full_input

def load_test_input(trans_dir):
  parallel_sentences = utils.load_data("data/parallel_sentences.json")[100:]
  prompt_input = []
  for each in parallel_sentences:
    if trans_dir == "kulung_english":
      prompt_input.append(("Kulung: " + each['Kulung'] + "\n English: ", each['Kulung'], each['English']))
    else:
      prompt_input.append(("English: " + each['English'] + "\n Kulung: ", each['Kulung'], each['English']))
  return prompt_input


def main(
    trans_dir:str = "english_kulung",
    results_dir:str = "results/english_kulung/gpt-4-o-mini"
):
    if not os.path.exists(results_dir):
        print("Creating directory: ", results_dir)
        os.makedirs(results_dir)

    full_input = load_train_input(trans_dir, results_dir)
    prompt_input = load_test_input(trans_dir)
    prompt_gpt(full_input, prompt_input, trans_dir, results_dir)

if __name__ == "__main__":
  main()