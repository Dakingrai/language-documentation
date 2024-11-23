
import pdb
from scripts import utils
import evaluate

def main(
    trans_dir:str = "english_kulung",
    data:str = "results/english_kulung/gpt-4-o-mini/results.txt"
):
    results = utils.load_data(data)
    cleaned_results = []
    gold_results = []
    for each in results:
        if trans_dir == "english_kulung":
            gold_results.append(each['gold_kulung'])
            cleaned_results.append(each['gpt4_kulung'].replace("Kulung:", "").strip())
        else:
            gold_results.append(each['gold_english'])
            cleaned_results.append(each['gpt4_english'].replace("English:", "").strip())
    
    chrf = evaluate.load("chrf")
    chrf_score = chrf.compute(predictions=cleaned_results, references=gold_results)
    print(chrf_score)

if __name__ == "__main__":
    main()