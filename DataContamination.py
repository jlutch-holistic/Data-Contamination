import pandas as pd
import random
from llama_index.llms.azure_openai import AzureOpenAI
from openai import AzureOpenAI


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

def pick_agent(chatGPT = True):
	if chatGPT:
		api_key = "3803844f0b2b4651842ff3529a71b32f"
		azure_endpoint = "https://hairesearch.openai.azure.com/"
		version = "2024-02-01"
		deployment_name = "gpt35"
		return GPTAgent(api_key, azure_endpoint, deployment_name, version)
	else:
		return None

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
		type = 'instruct' # Might be worth it to try out with 'standard' and 'write as well'
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

def main():
	# Initialize LLM agent
	
	agent = pick_agent()

	# Load data
	path = "data/test/world_religions_test.csv"
	df = load_csv_to_dataframe(path)

	#Drop the answer column
	#df.drop(columns="Answer", inplace=True)

	"""
	Remove one piece of information from a question at pseudo-random.
	Store the obscufated info in hidden_info to reference later
	"""
	print("Hiding data")
	hidden_info, df = obscure_df(df)
	print("Finished hiding")

	# For each row, ask LLM to fill in the model
	print("Generating guesses")
	guesses = generate_guesses(df, 1, agent)

	analyze_guesses(guesses, hidden_info)


	# Then, tally up to how many are the same


if __name__ == "__main__":
	main()