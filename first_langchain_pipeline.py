from langchain_community.llms import Cohere, HuggingFacePipeline, HuggingFaceHub
import os
import getpass

def get_environment_vars():
    os.environ["COHERE_API_KEY"] = getpass.getpass("Cohere API Key:")
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = getpass.getpass("HuggingFace API Key:")

def define_hf_model(repo_name: str,
                    temp: float,
                    max_length: int):
    llm = HuggingFaceHub(
                model_kwargs={"temperature": temp, "max_length": max_length},
                repo_id=repo_name
                )
    return llm

def get_cohere_model():
    cohere_llm = Cohere()
    return cohere_llm

if __name__ == "__main__":
    # get_environment_vars()
    prompt = "In which country is Tokyo?"
    # hf_llm = define_hf_model(repo_name= "google/flan-t5-xxl",
    #                    temp= 0.5,
    #                    max_length= 64)
    cohere_llm = get_cohere_model()
    completion = cohere_llm.invoke(prompt)
    print(completion)