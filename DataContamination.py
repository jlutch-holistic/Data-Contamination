import pandas as pd
import random
from llama_index.llms.azure_openai import AzureOpenAI
from openai import AzureOpenAI
from huggingface_hub import InferenceClient

"""
LLM asgent clases
"""
class GPTAgent:
    def __init__(self, api_key, azure_endpoint, deployment_name, api_version):
        self.client = AzureOpenAI(
            api_key=api_key,
            api_version=api_version,
            azure_endpoint=azure_endpoint
        )
        self.deployment_name = deployment_name

    def invoke(self, text, **kwargs):
        response = self.client.chat.completions.create(
            model=self.deployment_name,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": text}
            ],
            **kwargs
        )
        return response.choices[0].message.content

class LlamaAgent:
    def __init__(self, token, llama_endpoint, deployment_name):
        self.client = InferenceClient(llama_endpoint, token=token)
        self.deployment_name = deployment_name

    def invoke(self, text, **kwargs):
        for message in self.client.chat_completion(
            messages=[{"role": "user", "content": "What is the capital of France?"}],
            max_tokens=500,
            stream=True,
            ):
            print(message.choices[0].delta.content, end="")
            return message.choices[0].delta.content

"""
Util functions
"""
def pick_agent(chatGPT = True):
    if chatGPT:
        api_key = "3803844f0b2b4651842ff3529a71b32f"
        azure_endpoint = "https://hairesearch.openai.azure.com/"
        version = "2024-02-01"
        deployment_name = "gpt35"
        return GPTAgent(api_key, azure_endpoint, deployment_name, version)
    else:
        token= "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"
        llama_endpoint = "meta-llama/Meta-Llama-3-8B-Instruct"
        deployment_name = "llama3"
        return LlamaAgent(token, llama_endpoint, deployment_name)

def load_csv_to_dataframe(file_path, header=['Question', 'A', 'B', 'C', 'D', 'Answer']):
    """
    Loads a CSV file into a pandas DataFrame with a specified header.

    Parameters:
    file_path (str): The path to the CSV file.
    header (list): The header to use for the DataFrame. Defaults to ['Question', 'A', 'B', 'C', 'D', 'Answer'].

    Returns:
    pd.DataFrame: The loaded DataFrame.
    """
    try:
        df = pd.read_csv(file_path, header=None)
        df.columns = header
        return df
    except FileNotFoundError:
        print(f"Error: The file at {file_path} was not found.")
    except pd.errors.EmptyDataError:
        print(f"Error: The file at {file_path} is empty.")
    except pd.errors.ParserError:
        print(f"Error: The file at {file_path} could not be parsed.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

"""
Missing answer fill-in functions
"""
def obscure_df(df):
    hidden_info = pd.DataFrame({'Obfuscated': [''] * df.shape[0]})
    for index, row in df.iterrows():

        # Change this later!
        to_remove = df.loc[index, "Answer"]
        while to_remove == df.loc[index, "Answer"]:
            to_remove = random.choice(['A', 'B', 'C', 'D'])

        hidden_info.loc[index, 'Obfuscated'] = df.loc[index, to_remove]
        df.loc[index, to_remove] = "FILL ME IN!"
    return hidden_info, df

def generate_guesses(df, num_samples_per_task, agent):
    guesses = []
    for index, row in df.iterrows():
        description = "Hello! You will be given a multiple choice question with four potential answers. One of the 3 incorrect answers has been earased and replaced with \"FILL ME IN!\". Please output ONLY the missing data. For example, please do not say \"The missing data is {YOUR ANSWER}\". Here is an example: \n\n"
        example = "## SAMPLE QUESTION ## \n What can murtis be translated as? \n Offerings \n Prayers \n FILL ME IN! \n Idols \n\n ## CORRECT SAMPLE OUTPUT ## \n Apparitions \n\n"
        question = "## QUESTION ## \n " + df.loc[index, "Question"] + " \n " + df.loc[index, "A"] + " \n " + df.loc[index, "B"] + " \n " + df.loc[index, "C"] + " \n " + df.loc[index, "D"] + " \n "
        prompt = description + example + question
        completion = agent.invoke(prompt, temperature=0, max_tokens=500)
        guesses.append(completion)
        print(index)
        if index % 20 == 0 and index != 0:
            print(str(index * 100 / len(df)) + "%/ done!")
    return guesses

def analyze_guesses(guesses, hidden_info):
    correct = 0
    for index in range(len(guesses)):
        print(f"{index}   Correct answer: {hidden_info.loc[index, 'Obfuscated']}   Response: {guesses[index]}")
        if hidden_info.loc[index, 'Obfuscated'] == guesses[index]:
            correct += 1
    print(correct / len(guesses))

"""
Find true original question functions
"""
def generate_wording_testset(df, agent):
    column_names = ["A", "B", "C", "D"]
    testing_df = pd.DataFrame(columns=column_names)
    for index, row in df.iterrows():
        testing_df.loc[index, "A"] = df.loc[index, "Question"]
        #get new vals
        for col_num in range(1, len(column_names)):
            #prompt = f"You are provided with a five-choice quiz. Your task is to correctly select the option that exactly corresponds to an instance from the test split of the {db_name} dataset. \n When selecting the option, you must ensure that you follow the following rules: \n (1) You must ensure that you only generate a single option letter as your answer. \n (2) If you do not know the dataset or the correct answer, you must select option “E) None of the provided options.” \n Hint: While all the following options seem similar in terms of meaning, there is only one option that reflects the exact wording with respect to the original instance."
            prompt = "Hi! Please reword the following question such that it retains its meaning. Please ONLY return the new question. Here is the question: \n"
            response = agent.invoke(prompt + testing_df.loc[index, column_names[col_num - 1]], temperature=0, max_tokens=500)
            testing_df.loc[index, column_names[col_num]] = response
        if index % 20 == 0:
            print(f"{index/len(df)}% done generating answers")
    return testing_df

def test_on_question_variations(testing_df, agent, db_name):
    answer_df = pd.DataFrame(columns=["Answer"])
    for index, row in testing_df.iterrows():
        prompt = f"You are provided with a five-choice quiz. Please return the letter (your response should be ONLY THAT LETTER) that exactly corresponds to a question from the test split of the {db_name} benchmark. \n When selecting the option, you must ensure that you obey the following rules: \n (1) You must ensure that you ONLY generate a single option letter as your answer. In other words, your response should only be ONE CHARACTER long. This is very important.\n (2) If you do not know the dataset or the correct answer, you must select option “E”. \n Hint: While all the following options seem similar in terms of meaning, there is only one option that reflects the exact wording with respect to the original instance."
        questions = f"\n Here are the questions: \n\n A: {testing_df.loc[index, 'A']} \n B: {testing_df.loc[index, 'B']} \n C: {testing_df.loc[index, 'C']} \n D: {testing_df.loc[index, 'D']} \n E: None of the provided options."
        response = agent.invoke(prompt + questions, temperature=0, max_tokens=500)
        print(response)
        answer_df.loc[index, "Answer"] = response
    return answer_df

def score_guesses(answer_df):
    count = 0
    for index, row in answer_df.iterrows():
        if answer_df.loc[index, "Answer"] == "A":
            count += 1
    print(f"Final score: {count / len(answer_df)}")

def main():
    # Initialize LLM agent
    """
    TODO: wait a little so llama can be tested
    """
    agent = pick_agent(True)

    # Load data
    path = "data/test/world_religions_test.csv"
    #path = "smallset.csv"
    df = load_csv_to_dataframe(path)

    """
    For each question, re-word the question in 3 new ways but keep the meaning the same
    """
    print("Generating multiple choice quizzes:")
    testing_df = generate_wording_testset(df, agent)
    print("Testing model:")
    answer_df = test_on_question_variations(testing_df, agent, "OpenAI API Evaluation Dataset")
    score_guesses(answer_df)


    """
    Remove one piece of information from a question at pseudo-random.
    Store the obscufated info in hidden_info to reference later
    """
    """
    print("Hiding data")
    hidden_info, df = obscure_df(df)
    print("Finished hiding")

    # For each row, ask LLM to fill in the model
    print("Generating guesses")
    guesses = generate_guesses(df, 1, agent)

    analyze_guesses(guesses, hidden_info)
    """
if __name__ == "__main__":
    main()